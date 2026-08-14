// PagedAttentionKernel.swift
//
// Swift dispatch wrapper for the CBv2 paged-attention decode kernels
// (`pagedattention.metal`, shipped as a bundle resource). Decode is a
// two-pass flash-decoding dispatch: pass A computes unnormalized partials
// over PTOK-token partitions (one threadgroup per (sequence, kv head,
// partition)); pass B merges the partials per (sequence, query head) with
// sinks folded into the denominator. See the .metal header for the full
// contract and attribution.
//
// Kernel instances are JIT-compiled by MLX and cached both by MLX (per
// kernel name) and here (per configuration). Every distinct configuration
// gets a distinct kernel NAME: MLX's custom-kernel library cache is keyed
// by name and re-JITs whenever the generated source for a name changes, so
// sharing one name across dtypes/head-dims would thrash the pipeline cache.
//
// PERFORMANCE NOTE (2026-07, kernel-opt track): the original real-model
// ~100x slowdown (GPT-OSS-20B at ~2 s/token, 30 s step-watchdog kills —
// benchmarks/reports/gptoss-20b-mxfp4q8-main.md) was NOT kernel math: MLX
// retains kernel-INPUT data handles until the command buffer completes, so
// per-layer slab slice-updates failed donation and copied the full multi-
// GiB slab (~370 GiB of copies per token). Fixed by writing the slabs IN
// PLACE (fused decode write in pass A + `bulkWrite` for chunks) with a
// per-group write-fence chain for ordering; measured 25.5 s -> 13.6 ms per
// 24-layer step at the 16 GiB pool (docs/engine-v2/kernel-research.md §3).

import Foundation
import MLX
import MLXFast

/// Configuration key for one compiled paged-attention kernel variant.
struct PagedAttentionKernelKey: Hashable {
    enum Pass: Hashable {
        case part
        case merge
        /// Bulk in-place KV write (prefill chunks / prefix adoption).
        case write
    }
    var pass: Pass
    var dtype: DType
    var headDim: Int
    var pageSize: Int
    var gqa: Int
    var simdgroups: Int
    var hasSinks: Bool
    var hasSoftcap: Bool
    /// Pass A only: fuse the decode KV write into the kernel.
    var hasWrite: Bool = false

    var kernelName: String {
        let d: String
        switch dtype {
        case .float16: d = "f16"
        case .float32: d = "f32"
        case .bfloat16: d = "bf16"
        default: d = "dt\(dtype)"
        }
        let p: String
        switch pass {
        case .part: p = "part"
        case .merge: p = "merge"
        case .write: p = "write"
        }
        return "cbv2_paged_\(p)_\(d)_d\(headDim)_s\(pageSize)_g\(gqa)"
            + "_n\(simdgroups)_sink\(hasSinks ? 1 : 0)_cap\(hasSoftcap ? 1 : 0)"
            + (hasWrite ? "_w1" : "")
    }
}

/// Loads the MSL source for the paged-attention kernels from the package
/// bundle exactly once.
enum PagedAttentionMSL {
    static let header: String = {
        guard
            let url = Bundle.module.url(
                forResource: "pagedattention", withExtension: "metal")
        else {
            fatalError(
                "[PagedAttentionKernel] missing bundle resource pagedattention.metal "
                    + "— check Package.swift resources for MLXLMCommon")
        }
        guard let source = try? String(contentsOf: url, encoding: .utf8) else {
            fatalError("[PagedAttentionKernel] unable to read pagedattention.metal")
        }
        return source
    }()

    /// Bodies of the auto-generated kernel functions. They reference the
    /// thread attributes and `_shape` helpers by name so MLX includes them
    /// in the generated signature (MLX scans the body source for tokens).
    ///
    /// Two pass-A variants: the decode path fuses the per-row KV write
    /// (HAS_WRITE=true, `knew`/`vnew` inputs); the KV-borrowing path
    /// dispatches the write-free variant, where `q` stands in for the
    /// never-read `knew`/`vnew` parameters (dead code under
    /// HAS_WRITE=false).
    static let partBody: String = """
            const int kvh = kcache_shape[1];
            const int maxp = tables_shape[1];
            // grid z extent == the partial buffers' partition capacity
            // (output `_shape` params are not injected by MLX, only inputs').
            const int maxpart = threadgroups_per_grid.z;
            // Advance the slab group's write-fence chain: this dispatch
            // wrote the step's K/V tiles in place, and later slab readers
            // consume the fence to order after it.
            if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0
                && thread_position_in_grid.z == 0) {
                fence[0] = wfence[0] + 1;
            }
            threadgroup float q_smem[HPT * D];
            threadgroup float red_smem[NSG * HPT * (D + 2)];
            cbv2::paged_attention_part_impl<T, D, S, GQA, HPT, NSG, PTOK, true, HAS_SOFTCAP>(
                q, knew, vnew, kcache, vcache, tables, seqinfo, params,
                kvh, maxp, maxpart, q_smem, red_smem, partials, meta,
                threadgroup_position_in_grid,
                thread_position_in_threadgroup,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
        """

    static let partBodyNoWrite: String = """
            const int kvh = kcache_shape[1];
            const int maxp = tables_shape[1];
            // grid z extent == the partial buffers' partition capacity
            // (output `_shape` params are not injected by MLX, only inputs').
            const int maxpart = threadgroups_per_grid.z;
            threadgroup float q_smem[HPT * D];
            threadgroup float red_smem[NSG * HPT * (D + 2)];
            cbv2::paged_attention_part_impl<T, D, S, GQA, HPT, NSG, PTOK, false, HAS_SOFTCAP>(
                q, q, q, kcache, vcache, tables, seqinfo, params,
                kvh, maxp, maxpart, q_smem, red_smem, partials, meta,
                threadgroup_position_in_grid,
                thread_position_in_threadgroup,
                simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
        """

    /// Bulk in-place KV write: scatters `[KVH, N, D]` tiles into the slabs
    /// and emits the next fence in the group's write chain.
    static let writeBody: String = """
            const int kvh = ktile_shape[0];
            const int n = ktile_shape[1];
            const int d = ktile_shape[2];
            cbv2::paged_kv_write_impl<T, S>(
                ktile, vtile, slots, prev, kcache, vcache,
                kvh, n, d, fence,
                thread_position_in_grid);
        """

    static let mergeBody: String = """
            const int heads = partials_shape[1];
            const int maxpart = partials_shape[2];
            cbv2::paged_attention_merge_impl<T, D, PTOK, HAS_SINKS>(
                partials, meta, seqinfo, sinks, heads, maxpart, out,
                threadgroup_position_in_grid,
                thread_index_in_simdgroup);
        """
}

/// Batched single-token (decode) paged attention.
public enum PagedAttentionKernel {

    /// Head dims with a validated shared-memory/register budget. The
    /// threadgroup-memory budget additionally depends on the GQA factor,
    /// but the head split (`headsPerThreadgroup`) keeps every supported
    /// head dim within it at any fleet GQA — Gemma-4 global layers
    /// (headDim 512, GQA 8) run split 2-heads-per-threadgroup.
    public static let supportedHeadDims: Set<Int> = [64, 128, 256, 512]

    /// Tokens per flash-decoding partition. Must be a multiple of the page
    /// size. 256 gives ceil(len/256) partitions per (row, kv head) —
    /// enough threadgroups to saturate the GPU at B=1 (the per-token online
    /// softmax is ALU-serialized within a simdgroup, so parallelism comes
    /// from partition count) without bloating the partial buffers.
    public static let partitionTokens = 256

    // MARK: - Threadgroup-memory budget (single source of truth)
    //
    // These constants mirror the part kernel's threadgroup allocations —
    // `PagedAttentionMSL.partBody` above plus the RSTRIDE layout in
    // pagedattention.metal (which carries a matching keep-in-sync comment):
    //
    //   threadgroup float q_smem[HPT * D];               // staged queries
    //   threadgroup float red_smem[NSG * HPT * (D + 2)]; // merge records:
    //                                                    // acc[D], m, l
    //
    // HPT is the HEAD SPLIT (`headsPerThreadgroup`), NOT the full GQA
    // factor — sizing these by GQA is exactly the bug that made Gemma-4
    // global layers (d512, GQA 8) a 32,832 B process fatal. Both buffers
    // are float32 regardless of the slab dtype T (K/V rows are converted
    // to float on load), the merge pass and the bulk-write kernel allocate
    // NO threadgroup memory, and neither the HAS_SOFTCAP, HAS_SINKS nor
    // HAS_WRITE variant adds any — so the budget is a function of
    // (headDim, gqa, simdgroups) alone. `CBv2PagedEligibilityTests`
    // asserts the generated bodies still match this model.

    /// Metal's per-threadgroup memory cap (`setThreadgroupMemoryLength`
    /// limit on Apple GPUs). A dispatch over this limit is an UNCATCHABLE
    /// process fatal ("Threadgroup memory size (...) exceeds the maximum
    /// (32768)"), not a thrown error — eligibility must be refused
    /// statically, before any dispatch.
    public static let threadgroupMemoryLimit = 32 * 1024

    /// Per-(simdgroup, query head) merge record trailer: the shader's
    /// RSTRIDE is `D + 2` floats — acc[D] plus (m, l).
    public static let mergeRecordMetaFloats = 2

    /// Candidate simdgroup counts for the part kernel, largest first.
    static let simdgroupCandidates = [8, 4, 2, 1]

    /// Per-thread float32 budget for the pass-A value accumulator
    /// (`acc[HPT][D/32]`): caps the head split so registers never explode
    /// at large head dims (32 floats == the level the fleet's validated
    /// d64/GQA-8 shape already runs at).
    static let maxAccumulatorFloatsPerThread = 32

    /// Query heads per pass-A threadgroup (the kernel's HPT template
    /// parameter): the largest divisor of `gqa` whose per-thread value
    /// accumulator stays within `maxAccumulatorFloatsPerThread`. A kv
    /// head's GQA query heads split across `gqa / hpt` threadgroups; the
    /// split is per-head, so no head's arithmetic changes. d64/d128 fleet
    /// shapes keep hpt == gqa (identical to the unsplit kernel); d512 at
    /// GQA 8 (Gemma-4 global layers) runs hpt == 2.
    public static func headsPerThreadgroup(headDim: Int, gqa: Int) -> Int {
        let cap = max(1, maxAccumulatorFloatsPerThread * 32 / headDim)
        var hpt = min(gqa, cap)
        while gqa % hpt != 0 { hpt -= 1 }
        return hpt
    }

    /// Exact threadgroup bytes the part kernel allocates for one
    /// threadgroup: (q_smem + red_smem) float32 elements, sized by the
    /// HEAD-SPLIT group (`headsPerThreadgroup`), not the full GQA factor.
    public static func partThreadgroupBytes(
        headDim: Int, gqa: Int, simdgroups: Int
    ) -> Int {
        let hpt = headsPerThreadgroup(headDim: headDim, gqa: gqa)
        let qSmem = hpt * headDim
        let redSmem = simdgroups * hpt * (headDim + mergeRecordMetaFloats)
        return (qSmem + redSmem) * MemoryLayout<Float32>.size
    }

    /// Largest simdgroup count whose staging + merge buffers fit the Metal
    /// threadgroup-memory cap, or nil when even NSG=1 exceeds it — the
    /// configuration is statically ineligible for the paged kernels.
    static func simdgroupsPerThreadgroup(headDim: Int, gqa: Int) -> Int? {
        simdgroupCandidates.first {
            partThreadgroupBytes(headDim: headDim, gqa: gqa, simdgroups: $0)
                <= threadgroupMemoryLimit
        }
    }

    /// Static eligibility of one attention shape for the paged decode
    /// kernels: nil when eligible, else a human-readable reason. Callers
    /// building paged state (`PagedKVBackend.init`) MUST refuse ineligible
    /// shapes before any dispatch (see `threadgroupMemoryLimit`).
    public static func ineligibilityReason(headDim: Int, gqa: Int) -> String? {
        guard supportedHeadDims.contains(headDim) else {
            return "paged kernel does not support headDim \(headDim); "
                + "supported: \(supportedHeadDims.sorted())"
        }
        guard gqa >= 1 else {
            return "invalid GQA factor \(gqa)"
        }
        guard simdgroupsPerThreadgroup(headDim: headDim, gqa: gqa) != nil else {
            let bytes = partThreadgroupBytes(headDim: headDim, gqa: gqa, simdgroups: 1)
            return "headDim \(headDim) with GQA \(gqa) needs \(bytes) B of "
                + "threadgroup memory even at 1 simdgroup, over the "
                + "\(threadgroupMemoryLimit) B Metal limit — the paged part "
                + "kernel would trap at first dispatch"
        }
        return nil
    }

    private final class KernelCache: @unchecked Sendable {
        private var kernels: [PagedAttentionKernelKey: MLXFast.MLXFastKernel] = [:]
        private let lock = NSLock()

        func kernel(for key: PagedAttentionKernelKey) -> MLXFast.MLXFastKernel {
            lock.lock()
            defer { lock.unlock() }
            if let k = kernels[key] { return k }
            let k: MLXFast.MLXFastKernel
            switch key.pass {
            case .part where key.hasWrite:
                // `wfence` is the slab group's write fence: the graph edge
                // (and the encoder barrier it induces) orders this dispatch
                // after every prior in-place write of the group; the
                // `fence` output advances the chain for later readers.
                k = MLXFast.metalKernel(
                    name: key.kernelName,
                    inputNames: [
                        "q", "knew", "vnew", "kcache", "vcache", "tables", "seqinfo",
                        "params", "wfence",
                    ],
                    outputNames: ["partials", "meta", "fence"],
                    source: PagedAttentionMSL.partBody,
                    header: PagedAttentionMSL.header,
                    ensureRowContiguous: true
                )
            case .part:
                k = MLXFast.metalKernel(
                    name: key.kernelName,
                    inputNames: [
                        "q", "kcache", "vcache", "tables", "seqinfo", "params", "wfence",
                    ],
                    outputNames: ["partials", "meta"],
                    source: PagedAttentionMSL.partBodyNoWrite,
                    header: PagedAttentionMSL.header,
                    ensureRowContiguous: true
                )
            case .merge:
                k = MLXFast.metalKernel(
                    name: key.kernelName,
                    inputNames: ["partials", "meta", "seqinfo", "sinks"],
                    outputNames: ["out"],
                    source: PagedAttentionMSL.mergeBody,
                    header: PagedAttentionMSL.header,
                    ensureRowContiguous: true
                )
            case .write:
                k = MLXFast.metalKernel(
                    name: key.kernelName,
                    inputNames: ["ktile", "vtile", "slots", "prev", "kcache", "vcache"],
                    outputNames: ["fence"],
                    source: PagedAttentionMSL.writeBody,
                    header: PagedAttentionMSL.header,
                    ensureRowContiguous: true
                )
            }
            kernels[key] = k
            return k
        }
    }

    private static let cache = KernelCache()

    static func kernel(for key: PagedAttentionKernelKey) -> MLXFast.MLXFastKernel {
        cache.kernel(for: key)
    }

    /// Shared dummy sinks (the generated signature keeps a `device` input
    /// even when HAS_SINKS is false). Read-only after creation; MLXArray is
    /// not Sendable but this one is never mutated (engine-thread discipline).
    nonisolated(unsafe) private static let zeroSinks: MLXArray = {
        let z = MLXArray.zeros([8], dtype: .float32)
        eval(z)
        return z
    }()

    /// Dispatch decode attention for `B` rows.
    ///
    /// - Parameters:
    ///   - queries: `[B, queryHeads, 1, headDim]` or `[B, queryHeads, headDim]`,
    ///     any float dtype (converted to the slab dtype if needed).
    ///   - newKeys/newValues: this step's K/V tiles `[B, kvHeads, headDim]`
    ///     (any float dtype). When non-nil the kernel writes them IN PLACE
    ///     into the slabs at each row's `{writePage, writeSlot}` (seqinfo
    ///     fields 3/4) before attending — the fused decode write. Pass nil
    ///     for KV-borrowing dispatches (the rows were written by their
    ///     owning layer).
    ///   - kSlab/vSlab: pool slabs `[P, kvHeads, pageSize, headDim]`.
    ///   - tables: `[B, maxPages]` int32, `maxPages >= 8`.
    ///   - seqinfo: `[B, 8]` int32 rows `{attendStart, attendLen, tableLen,
    ///     writePage, writeSlot, 0…}` — attend fields describe the range
    ///     INCLUDING the newly written position.
    ///   - maxAttendLength: max over rows of the attended length (host-side
    ///     Swift Int — sizes the partial buffers, never a device sync).
    ///   - sinks: optional per-query-head sink logits `[queryHeads]`.
    ///   - params: `[8]` float32 `{softcap, scale, 0…}` (cache it per layer —
    ///     it is constant across steps).
    ///   - softcap: whether params[0] is an active softcap.
    ///   - writeFence: the slab group's write fence (`[1]` int32,
    ///     `PagedKVGroup.writeFence`). The graph edge orders this dispatch
    ///     after every prior in-place write of the group (see
    ///     pagedattention.metal, "In-place slab writes").
    /// - Returns: attention `[B, queryHeads, headDim]` in the slab dtype,
    ///   plus — when the fused write ran — the group's NEXT write fence,
    ///   which the caller MUST store back into the group.
    public static func decode(
        queries: MLXArray,
        newKeys: MLXArray? = nil,
        newValues: MLXArray? = nil,
        kSlab: MLXArray,
        vSlab: MLXArray,
        tables: MLXArray,
        seqinfo: MLXArray,
        maxAttendLength: Int,
        sinks: MLXArray?,
        params: MLXArray,
        softcap: Bool,
        pageSize: Int,
        writeFence: MLXArray,
        stream: StreamOrDevice = .default
    ) -> (out: MLXArray, nextWriteFence: MLXArray?) {
        var q = queries
        if q.ndim == 4 {
            precondition(q.dim(2) == 1, "decode kernel requires L == 1")
            q = q.squeezed(axis: 2)
        }
        precondition(q.ndim == 3, "queries must be [B, QH, D]")
        precondition(
            (newKeys == nil) == (newValues == nil),
            "newKeys/newValues must be passed together")

        let dtype = kSlab.dtype
        let b = q.dim(0)
        let queryHeads = q.dim(1)
        let headDim = q.dim(2)
        let kvHeads = kSlab.dim(1)
        precondition(kSlab.dim(3) == headDim && vSlab.dim(3) == headDim)
        precondition(kSlab.dim(2) == pageSize)
        precondition(queryHeads % kvHeads == 0, "GQA requires QH % KVH == 0")
        precondition(supportedHeadDims.contains(headDim), "unsupported head dim \(headDim)")
        precondition(tables.dim(0) == b && seqinfo.dim(0) == b)
        precondition(tables.dim(1) >= 8, "pad tables to >= 8 columns for a stable signature")
        precondition(partitionTokens % pageSize == 0, "PTOK must be a page multiple")
        precondition(maxAttendLength >= 1)

        let gqa = queryHeads / kvHeads
        guard let nsg = simdgroupsPerThreadgroup(headDim: headDim, gqa: gqa) else {
            preconditionFailure(
                "paged decode dispatched for a statically ineligible shape: "
                    + (ineligibilityReason(headDim: headDim, gqa: gqa)
                        ?? "headDim \(headDim), GQA \(gqa)"))
        }
        let maxParts = (maxAttendLength + partitionTokens - 1) / partitionTokens

        if q.dtype != dtype { q = q.asType(dtype) }

        let hasWrite = newKeys != nil
        var partInputs: [MLXArray] = [q]
        if let newKeys, let newValues {
            precondition(
                newKeys.ndim == 3 && newKeys.dim(0) == b && newKeys.dim(1) == kvHeads
                    && newKeys.dim(2) == headDim,
                "newKeys must be [B, KVH, D]")
            partInputs.append(newKeys.dtype == dtype ? newKeys : newKeys.asType(dtype))
            partInputs.append(
                newValues.dtype == dtype ? newValues : newValues.asType(dtype))
        }
        partInputs.append(contentsOf: [kSlab, vSlab, tables, seqinfo, params, writeFence])

        let hpt = headsPerThreadgroup(headDim: headDim, gqa: gqa)
        let splits = gqa / hpt
        let partKey = PagedAttentionKernelKey(
            pass: .part, dtype: dtype, headDim: headDim, pageSize: pageSize, gqa: gqa,
            simdgroups: nsg, hasSinks: false, hasSoftcap: softcap, hasWrite: hasWrite)
        let tg = 32 * nsg
        let partOut = kernel(for: partKey)(
            partInputs,
            template: [
                ("T", dtype),
                ("D", headDim),
                ("S", pageSize),
                ("GQA", gqa),
                ("HPT", hpt),
                ("NSG", nsg),
                ("PTOK", partitionTokens),
                ("HAS_SOFTCAP", softcap),
            ],
            grid: (kvHeads * splits * tg, b, maxParts),
            threadGroup: (tg, 1, 1),
            outputShapes: hasWrite
                ? [[b, queryHeads, maxParts, headDim], [b, queryHeads, maxParts, 2], [1]]
                : [[b, queryHeads, maxParts, headDim], [b, queryHeads, maxParts, 2]],
            outputDTypes: hasWrite ? [.float32, .float32, .int32] : [.float32, .float32],
            stream: stream
        )

        let mergeKey = PagedAttentionKernelKey(
            pass: .merge, dtype: dtype, headDim: headDim, pageSize: pageSize, gqa: gqa,
            simdgroups: 1, hasSinks: sinks != nil, hasSoftcap: false)
        let outputs = kernel(for: mergeKey)(
            [partOut[0], partOut[1], seqinfo, sinks ?? zeroSinks],
            template: [
                ("T", dtype),
                ("D", headDim),
                ("PTOK", partitionTokens),
                ("HAS_SINKS", sinks != nil),
            ],
            grid: (queryHeads * 32, b, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[b, queryHeads, headDim]],
            outputDTypes: [dtype],
            stream: stream
        )
        return (outputs[0], hasWrite ? partOut[2] : nil)
    }

    /// Bulk in-place KV write (prefill chunks / prefix adoption): scatter
    /// `keys`/`values` `[KVH, N, D]` (slab dtype) into the slabs at the
    /// host-computed physical `slots` (`[N]` int32, `page * pageSize +
    /// slot` per token) and return the next fence in the group's write
    /// chain. Callers MUST route every subsequent read of the group's
    /// slabs through the returned fence (see pagedattention.metal,
    /// "In-place slab writes") — `PagedKVPool` owns that discipline.
    public static func bulkWrite(
        kSlab: MLXArray,
        vSlab: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        slots: MLXArray,
        prevFence: MLXArray,
        pageSize: Int,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        let dtype = kSlab.dtype
        precondition(keys.ndim == 3 && values.ndim == 3, "tiles must be [KVH, N, D]")
        precondition(keys.dtype == dtype && values.dtype == dtype, "tiles in slab dtype")
        let kvHeads = keys.dim(0)
        let n = keys.dim(1)
        let headDim = keys.dim(2)
        precondition(kSlab.dim(1) == kvHeads && kSlab.dim(3) == headDim)
        precondition(kSlab.dim(2) == pageSize)
        precondition(slots.dim(0) >= max(n, 8) && n > 0, "pad slots to >= 8 entries")

        let key = PagedAttentionKernelKey(
            pass: .write, dtype: dtype, headDim: headDim, pageSize: pageSize, gqa: 0,
            simdgroups: 0, hasSinks: false, hasSoftcap: false)
        let outputs = kernel(for: key)(
            [keys, values, slots, prevFence, kSlab, vSlab],
            template: [
                ("T", dtype),
                ("S", pageSize),
            ],
            grid: (headDim, kvHeads, n),
            threadGroup: (min(headDim, 256), 1, 1),
            outputShapes: [[1]],
            outputDTypes: [.int32],
            stream: stream
        )
        return outputs[0]
    }
}
