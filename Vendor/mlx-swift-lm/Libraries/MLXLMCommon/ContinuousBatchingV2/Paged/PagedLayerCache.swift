// PagedLayerCache.swift
//
// Batch-facing per-layer cache for the paged backend (WS-C): the
// `CBv2AttendingLayerCache` that owns BOTH the KV update and the attention
// dispatch, so models and the scheduler never see the storage layout.
//
// Attention paths (numerically pinned — one path per phase, never switching
// mask representation across steps):
//   - decode (`L == 1`, any B, including B == 1): the paged Metal kernel.
//     Per-row window clamping is start-offset arithmetic on absolute
//     positions computed host-side from Swift Ints; masks do not exist.
//   - prefill chunk (`B == 1`, `L > 1`): per-request SDPA over gathered
//     pages with an explicit BOOL mask built from absolute positions
//     (always `.array`, never `.none`/`.causal`, so the path cannot drift
//     — see MLX #3384). Models with an attention-logit softcap use the
//     composed reference path instead (SDPA cannot express softcap).
//
// Rows with different retained lengths are handled internally: the caller
// never pads and never builds masks.

import Foundation
import MLX
import MLXFast

public final class PagedLayerCache: CBv2AttendingLayerCache {
    public let layerIndex: Int
    public let kind: CBv2LayerKind
    let pool: PagedKVPool
    /// Optional attention-logit soft cap. Not part of the contract's
    /// `updateAndAttend` signature, so it is layer-cache configuration
    /// (from model config) instead.
    public let attentionSoftcap: Float?

    private var pagedRows: [PagedSequenceKV] = []

    // Per-row absolute RoPE offsets `[B]` (int32, device array). REBUILT from
    // host integers only on membership changes (`setRows`); ADVANCED
    // on-device (`+ L`) inside `updateAndAttend`, mirroring `CBv2LayerCache`.
    // The step loop therefore never uploads a fresh host array per layer per
    // step, and never syncs — the model reads this each step and feeds it
    // into the forward graph, which the step's `asyncEval` collapses so the
    // lazy `+ L` chain cannot grow O(steps).
    private var cachedPositionOffsets: MLXArray = MLXArray([] as [Int32])
    /// Times `positionOffsets` was rebuilt from host integers (test hook);
    /// must only move on batch membership changes, never inside the step loop.
    private(set) var positionOffsetsHostRebuilds = 0

    // Device block-table cache: rebuilt only when a row's page table
    // changes (page allocated / rollback / composition change), not on
    // every step. Fingerprinted by the pool-issued row SERIAL (never
    // reused), not ObjectIdentifier — a heap address can be recycled after
    // a finished request's row deallocates, and a same-shape decode would
    // then attend the FINISHED request's page ids (cross-request read).
    private var cachedTables: MLXArray?
    private var cachedTablesFingerprint: [(serial: UInt64, tableVersion: Int)] = []
    /// Times the device tables were rebuilt (test/telemetry hook).
    private(set) var tablesRebuildCount = 0

    // Kernel params `{softcap, scale, 0…}` are constant across steps for a
    // layer; cache the device array keyed on the scale actually passed.
    private var cachedParams: MLXArray?
    private var cachedParamsScale: Float?

    // Sinks prepared for the kernel (fp32, >= 8 elements). Models pass the
    // same parameter array every step, so key the cache on object identity.
    private var cachedSinks: MLXArray?
    private var cachedSinksSource: ObjectIdentifier?

    public init(
        layerIndex: Int, kind: CBv2LayerKind, pool: PagedKVPool,
        attentionSoftcap: Float? = nil
    ) {
        self.layerIndex = layerIndex
        self.kind = kind
        self.pool = pool
        self.attentionSoftcap = attentionSoftcap
    }

    // MARK: - Rows

    public var rows: [CBv2SequenceKV] { pagedRows }

    /// Set the current batch rows (row order == batch row order). O(B);
    /// join = append a row object, leave = drop it — no storage moves.
    /// (Contract `CBv2AttendingLayerCache.setRows` — the canonical binding.)
    public func setRows(_ rows: [CBv2SequenceKV]) {
        precondition(
            kind.sharesKVWithLayer == nil || rows.isEmpty,
            "KV-shared layers own no rows; attention borrows via attendBorrowing")
        pagedRows = rows.map { row in
            guard let paged = row as? PagedSequenceKV else {
                fatalError("[PagedLayerCache] rows must be PagedSequenceKV from the same backend")
            }
            precondition(
                paged.groupKey == PagedKVGroupKey(kind),
                "row group \(paged.groupKey) does not match layer kind")
            return paged
        }
        rebuildPositionOffsets()
    }

    /// Per-row absolute RoPE offsets `[B]` (device int32). Read BEFORE
    /// `updateAndAttend` for the step — it holds the PRE-update offsets of
    /// the tokens about to be processed (snapshot semantics). KV-shared
    /// layers own no rows, so their own value is empty (they reuse the
    /// source layer's pre-update capture).
    public var positionOffsets: MLXArray { cachedPositionOffsets }

    private func rebuildPositionOffsets() {
        positionOffsetsHostRebuilds += 1
        CBv2CoreInstrumentation.recordPositionOffsetsHostRebuild()
        cachedPositionOffsets = MLXArray(pagedRows.map { Int32($0.absoluteOffset) })
    }

    // MARK: - Attention

    public func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        precondition(kind.sharesKVWithLayer == nil, "shared layers must call attendBorrowing")
        let b = queries.dim(0)
        let l = queries.dim(2)
        precondition(b == pagedRows.count, "queries batch \(b) != rows \(pagedRows.count)")
        // Sinks apply only to layers that declare them (same gate as
        // CBv2AttentionV1) — a sink-free layer must ignore a passed array.
        let effectiveSinks = kind.hasSinks ? sinks : nil

        let output: MLXArray
        if l == 1 {
            // Decode: reserve each row's destination host-side; the kernel
            // writes the tiles IN PLACE before attending (fused write —
            // the slabs are never versioned through MLX ops).
            let targets = pagedRows.map { $0.prepareDecodeWrite() }
            output = dispatchDecode(
                queries: queries,
                newKeys: keys.squeezed(axis: 2),
                newValues: values.squeezed(axis: 2),
                writeTargets: targets,
                rows: pagedRows, scale: scale, sinks: effectiveSinks)
        } else {
            precondition(b == 1, "prefill chunks are per-request [1, chunk]")
            let row = pagedRows[0]
            row.write(keys: keys.squeezed(axis: 0), values: values.squeezed(axis: 0))
            output = prefillAttend(
                queries: queries, row: row, scale: scale, sinks: effectiveSinks)
        }
        // Advance offsets ON-DEVICE (uniform L for every row in the call:
        // decode is [B,1], prefill is [1,chunk]) — the rows just advanced
        // their absolute counters by exactly L, so the cached device array
        // tracks them without a per-step host rebuild.
        cachedPositionOffsets = cachedPositionOffsets + Int32(l)
        return output
    }

    public func attendBorrowing(
        source: CBv2AttendingLayerCache,
        queries: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        precondition(kind.sharesKVWithLayer != nil, "attendBorrowing requires a KV-shared layer")
        guard let src = source as? PagedLayerCache else {
            fatalError("[PagedLayerCache] can only borrow from another PagedLayerCache")
        }
        precondition(
            kind.attention == src.kind.attention,
            "KV-shared layer must share the source layer's attention type")
        let l = queries.dim(2)
        // Same sink gate as updateAndAttend, keyed on THIS layer's kind.
        let effectiveSinks = kind.hasSinks ? sinks : nil
        if l == 1 {
            return dispatchDecode(
                queries: queries, rows: src.pagedRows, scale: scale, sinks: effectiveSinks,
                tableProvider: src)
        } else {
            precondition(queries.dim(0) == 1 && src.pagedRows.count == 1)
            return prefillAttend(
                queries: queries, row: src.pagedRows[0], scale: scale, sinks: effectiveSinks)
        }
    }

    // MARK: - Decode path (paged kernel)

    /// One fused kernel dispatch for the batch. `newKeys`/`newValues`
    /// (`[B, kvHeads, D]`) with per-row `writeTargets` make pass A write
    /// the step's tiles in place before attending; both are nil on the
    /// KV-borrowing path (the owning layer already wrote).
    private func dispatchDecode(
        queries: MLXArray,
        newKeys: MLXArray? = nil,
        newValues: MLXArray? = nil,
        writeTargets: [(page: Int32, slot: Int)]? = nil,
        rows: [PagedSequenceKV], scale: Float, sinks: MLXArray?,
        tableProvider: PagedLayerCache? = nil
    ) -> MLXArray {
        precondition((newKeys == nil) == (writeTargets == nil))
        let provider = tableProvider ?? self
        let group = pool.group(rows[0].groupKey)
        let tables = provider.deviceTables(rows: rows)

        var info = [Int32]()
        info.reserveCapacity(rows.count * 8)
        var maxAttendLength = 1
        for (i, row) in rows.enumerated() {
            let (start, length) = row.decodeAttendRange
            info.append(Int32(start))
            info.append(Int32(length))
            // Windowed rows report the FULL ring length, not table.count —
            // a partially-allocated ring (prefix-adopted row) would else
            // wrap at the wrong divisor and alias wrong pages (Codex P2).
            info.append(Int32(row.decodeTableLength))
            if let targets = writeTargets {
                info.append(targets[i].page)
                info.append(Int32(targets[i].slot))
            } else {
                info.append(contentsOf: [0, 0])
            }
            info.append(contentsOf: [0, 0, 0])
            maxAttendLength = max(maxAttendLength, length)
        }
        let seqinfo = MLXArray(info, [rows.count, 8])

        let (out, nextFence) = PagedAttentionKernel.decode(
            queries: queries,
            newKeys: newKeys,
            newValues: newValues,
            kSlab: group.kSlab,
            vSlab: group.vSlab,
            tables: tables,
            seqinfo: seqinfo,
            maxAttendLength: maxAttendLength,
            sinks: preparedSinks(sinks),
            params: params(scale: scale),
            softcap: attentionSoftcap != nil,
            pageSize: pool.config.pageSize,
            writeFence: group.writeFence
        )
        // The fused write advanced the group's write-fence chain: later
        // slab readers (KV-borrowing layers, next steps' dispatches,
        // gathers) consume it and order after this dispatch's writes.
        if let nextFence {
            group.writeFence = nextFence
        }
        // [B, QH, D] -> [B, QH, 1, D], back in the model dtype.
        var result = out.expandedDimensions(axis: 2)
        if result.dtype != queries.dtype {
            result = result.asType(queries.dtype)
        }
        return result
    }

    /// Kernel params `{softcap, scale, 0…}`, cached across steps.
    private func params(scale: Float) -> MLXArray {
        if let cached = cachedParams, cachedParamsScale == scale {
            return cached
        }
        var values: [Float] = [attentionSoftcap ?? 1.0, scale]
        values.append(contentsOf: [Float](repeating: 0, count: 8 - values.count))
        let params = MLXArray(values)
        cachedParams = params
        cachedParamsScale = scale
        return params
    }

    /// Sinks prepared for the merge kernel: fp32, padded to >= 8 elements
    /// so the generated signature keeps the `device` address space.
    private func preparedSinks(_ sinks: MLXArray?) -> MLXArray? {
        guard let sinks else { return nil }
        if let cached = cachedSinks, cachedSinksSource == ObjectIdentifier(sinks) {
            return cached
        }
        var s = sinks.asType(.float32).reshaped([-1])
        if s.dim(0) < 8 {
            s = concatenated([s, MLXArray.zeros([8 - s.dim(0)], dtype: .float32)])
        }
        cachedSinks = s
        cachedSinksSource = ObjectIdentifier(sinks)
        return s
    }

    /// Device `[B, maxPages]` int32 block tables, rebuilt only when some
    /// row's identity (pool serial) or page table changed since the last
    /// dispatch. Internal (not private) for regression tests.
    func deviceTables(rows: [PagedSequenceKV]) -> MLXArray {
        let fingerprint = rows.map { (serial: $0.serial, tableVersion: $0.tableVersion) }
        if let cached = cachedTables,
            fingerprint.count == cachedTablesFingerprint.count,
            zip(fingerprint, cachedTablesFingerprint).allSatisfy({ $0 == $1 })
        {
            return cached
        }
        tablesRebuildCount += 1
        // Pad to >= 8 columns so the kernel signature's address space is
        // stable across batch shapes (see PagedAttentionKernel).
        let maxPages = max(8, rows.map { $0.table.count }.max() ?? 0)
        var flat = [Int32](repeating: 0, count: rows.count * maxPages)
        for (i, row) in rows.enumerated() {
            flat.replaceSubrange(
                (i * maxPages) ..< (i * maxPages + row.table.count), with: row.table)
        }
        let tables = MLXArray(flat, [rows.count, maxPages])
        cachedTables = tables
        cachedTablesFingerprint = fingerprint
        return tables
    }

    // MARK: - Prefill path (per-request SDPA over gathered pages)

    private func prefillAttend(
        queries: MLXArray, row: PagedSequenceKV, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        let l = queries.dim(2)
        let end = row.absoluteOffset
        let retained = row.retainedCount
        let (rawK, rawV) = row.attendableViews()
        let k = rawK.dtype == queries.dtype ? rawK : rawK.asType(queries.dtype)
        let v = rawV.dtype == queries.dtype ? rawV : rawV.asType(queries.dtype)

        // Absolute positions: queries are the LAST l tokens written; keys
        // are the retained tail. Bool mask = causal AND in-window.
        let qStart = end - l
        let kStart = end - retained
        let qpos = MLXArray(Int32(qStart) ..< Int32(end)).expandedDimensions(axis: 1)
        let kpos = MLXArray(Int32(kStart) ..< Int32(end)).expandedDimensions(axis: 0)
        var mask = kpos .<= qpos
        if case .slidingWindow(let window) = kind.attention {
            mask = mask & (kpos .> (qpos - Int32(window)))
        }

        if let softcap = attentionSoftcap {
            // SDPA cannot express logit softcapping — composed path, pinned
            // for softcap configs.
            return PagedAttentionReference.composedAttention(
                queries: queries, keys: k, values: v, scale: scale,
                boolMask: mask, sinks: sinks, softcap: softcap)
        }

        // MLX SDPA requires the sink dtype to promote to the output dtype
        // (fp16 queries + fp32 sinks trap), so match the query dtype here.
        // The decode kernel is unaffected: it consumes sinks in fp32.
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: k, values: v, scale: scale,
            mask: .array(mask), sinks: sinks?.asType(queries.dtype))
    }
}

// MARK: - Legacy KVCache conformance

/// Same shape as `CBv2LayerCache`'s conformance: lets the paged cache
/// travel through existing `[KVCache]` model plumbing (the models' v2
/// branches downcast to `CBv2AttendingLayerCache`); every legacy mutation
/// path TRAPS — v2-adapted models must call `updateAndAttend`.
extension PagedLayerCache: KVCache {
    /// Legacy scalar offset: max row offset. Host integers only — no sync.
    public var offset: Int {
        pagedRows.reduce(0) { max($0, $1.absoluteOffset) }
    }

    public var maxSize: Int? {
        switch kind.attention {
        case .full: return nil
        case .slidingWindow(let window): return window
        }
    }

    /// The paged slabs are pool-owned persistent buffers; per-step writes
    /// are materialized transitively by the engine's step asyncEval. The
    /// on-device `positionOffsets` advance chain is surfaced here (same
    /// convention as `CBv2LayerCache.innerState`) so it rides the step's
    /// eval set and cannot grow O(steps).
    public func innerState() -> [MLXArray] { [cachedPositionOffsets] }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(
            "PagedLayerCache.update(keys:values:) is unsupported — v2-adapted models must call updateAndAttend (layer \(layerIndex))"
        )
    }

    public var state: [MLXArray] {
        get { [] }
        set {
            fatalError("PagedLayerCache has no serializable state (layer \(layerIndex))")
        }
    }

    public var metaState: [String] {
        get { [] }
        set {
            fatalError("PagedLayerCache has no metaState (layer \(layerIndex))")
        }
    }

    public var isTrimmable: Bool { false }

    @discardableResult
    public func trim(_ n: Int) -> Int { 0 }

    public func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        fatalError(
            "PagedLayerCache.makeMask is unsupported — v2 attention owns its masks (layer \(layerIndex))"
        )
    }

    public func copy() -> any KVCache {
        fatalError(
            "PagedLayerCache.copy is unsupported — v2 rows are engine-owned (layer \(layerIndex))")
    }
}
