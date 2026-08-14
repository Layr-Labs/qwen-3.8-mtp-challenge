// PagedKVBackend.swift
//
// `CBv2KVBackend` factory over a `PagedKVPool` (WS-C). Swappable with the
// WS-A contiguous backend behind the same contract: the scheduler and
// models never see the difference.
//
// Eligibility is validated at construction (engine build time), per the
// contract: unsupported head dims, shapes over the paged kernel's
// threadgroup-memory budget (`PagedAttentionKernel.ineligibilityReason` —
// dispatching one is an uncatchable Metal fatal; the kernel's head split
// keeps every supported head dim within budget, incl. Gemma-4 global
// layers at headDim 512 / GQA 8), quant schemes, or malformed KV-sharing
// throw `CBv2KVError.backendIneligible` before any request is admitted.
// Attention sinks ARE supported (they are a kernel parameter here).
//
// Admission model: the worst-case page count for a request's `maxLength`
// is reserved UP FRONT — at admission via `reserve(layerKinds:maxLength:)`,
// or (when no admission-time reservation was taken) lazily by
// `makeSequenceState`, which reconciles against any prior reservation so the
// pages are charged exactly once. `reserve` throws `capacityExhausted` when
// the pool cannot honor the demand, so an admission controller can reject or
// queue a request that would otherwise fail at materialization (Codex P2:
// several same-step admissions must not over-commit the pool). Physical
// pages materialize lazily as tokens are written (`bytesInUse` stays
// truthful); `CBv2SequenceKV.update` therefore never fails mid-decode. See
// PagedKVPool.swift for the rationale.

import Foundation
import MLX

public final class PagedKVBackend: CBv2KVBackend {
    public let pool: PagedKVPool
    /// The model's per-layer structure this backend was built for.
    public let layerKinds: [CBv2LayerKind]

    public init(layerKinds: [CBv2LayerKind], config: PagedKVPoolConfig) throws {
        for (index, kind) in layerKinds.enumerated() {
            if let source = kind.sharesKVWithLayer {
                guard source >= 0, source < layerKinds.count,
                    layerKinds[source].sharesKVWithLayer == nil
                else {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index) shares KV with invalid layer \(source)")
                }
                let src = layerKinds[source]
                guard src.kvHeads == kind.kvHeads, src.headDim == kind.headDim,
                    src.attention == kind.attention
                else {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index) KV-shares with structurally different layer "
                            + "\(source)")
                }
            }
            guard kind.kvHeads > 0, kind.queryHeads % kind.kvHeads == 0 else {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index): queryHeads \(kind.queryHeads) not a multiple "
                        + "of kvHeads \(kind.kvHeads)")
            }
            // Kernel-level static eligibility (head dim support + the part
            // kernel's threadgroup-memory budget). Checked for EVERY layer
            // that will dispatch paged attention — including KV-shared
            // layers, which borrow storage but launch with their own GQA.
            // One over-budget layer makes the whole model ineligible.
            if let reason = PagedAttentionKernel.ineligibilityReason(
                headDim: kind.headDim, gqa: kind.queryHeads / kind.kvHeads)
            {
                throw CBv2KVError.backendIneligible(reason: "layer \(index): \(reason)")
            }
            if case .slidingWindow(let window) = kind.attention, window <= 0 {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index): invalid sliding window \(window)")
            }
        }
        self.layerKinds = layerKinds
        self.pool = try PagedKVPool(layerKinds: layerKinds, config: config)
    }

    // MARK: - Admission-time reservation (Codex P2)

    /// Reserve the worst-case page demand for a request of `maxLength` tokens
    /// BEFORE it is admitted, so several same-step admissions cannot
    /// over-commit the pool. Charges the pool up front (reflected in
    /// `bytesReserved`) and throws `capacityExhausted` when it cannot fit —
    /// the admission controller then rejects or queues the request instead of
    /// accepting one that would only fail at `makeSequenceState`.
    ///
    /// A subsequent `makeSequenceState` for the SAME request must be told the
    /// pages are already held (`reserved: true`) so it does not double-charge;
    /// `release`/`makeSequenceState(adopting:)`/finish balance the hold via
    /// the per-row `reservedPages` bookkeeping exactly once. Balance an
    /// admission that never materializes with `unreserve(layerKinds:maxLength:)`.
    public func reserve(layerKinds: [CBv2LayerKind], maxLength: Int) throws {
        precondition(maxLength > 0)
        try pool.reserve(pageNeeds(layerKinds: layerKinds, maxLength: maxLength))
    }

    /// Release an admission-time `reserve` that never reached
    /// `makeSequenceState` (rejected, superseded, or shut down).
    public func unreserve(layerKinds: [CBv2LayerKind], maxLength: Int) {
        pool.unreserve(pageNeeds(layerKinds: layerKinds, maxLength: maxLength))
    }

    // MARK: - CBv2KVBackend

    public func makeSequenceState(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        try makeSequenceState(
            layerKinds: layerKinds, promptLength: promptLength, maxLength: maxLength,
            reserved: false)
    }

    /// `reserved: true` skips the pool reservation because `reserve(...)`
    /// already charged the pages at admission — the per-row `reservedPages`
    /// still governs allocation and `release` still unreserves them, so the
    /// hold is charged and released exactly once.
    public func makeSequenceState(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int, reserved: Bool
    ) throws -> [CBv2SequenceKV?] {
        precondition(maxLength >= promptLength && maxLength > 0)
        let needs = pageNeeds(layerKinds: layerKinds, maxLength: maxLength)
        if !reserved {
            try pool.reserve(needs)
        }
        var states: [CBv2SequenceKV?] = []
        states.reserveCapacity(layerKinds.count)
        for kind in layerKinds {
            if kind.sharesKVWithLayer != nil {
                states.append(nil)
            } else {
                let reserved = PagedKVPool.pageDemand(
                    kind: kind, maxLength: maxLength, config: pool.config)
                states.append(
                    PagedSequenceKV(
                        pool: pool, kind: kind, maxLength: maxLength, reservedPages: reserved))
            }
        }
        return states
    }

    /// Adopt a donated prefix. Snapshots are written into fresh pages via
    /// the in-place bulk-write kernel (off the hot decode path; the writes
    /// ride the group's fence chain and evaluate with the row's first
    /// consuming step). Reconciled
    /// adoption semantics (contract `makeSequenceState(adopting:)`): the
    /// engine already sliced the prefix down by `cbv2RequiredRecompute`, so
    /// every non-nil entry carries the same uniform offset; windowed layers
    /// are fast-forwarded to that offset (counter only, no storage) and the
    /// engine replays the trailing tokens through all layers.
    public func makeSequenceState(
        adopting prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        layerKinds: [CBv2LayerKind], maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        precondition(prefix.count == layerKinds.count, "prefix/layer count mismatch")
        var matched = 0
        for (index, entry) in prefix.enumerated() {
            guard let entry else { continue }
            precondition(
                layerKinds[index].sharesKVWithLayer == nil,
                "prefix donated to a KV-shared layer \(index)")
            precondition(
                matched == 0 || matched == entry.offset,
                "non-uniform prefix offsets (\(entry.offset) vs \(matched))")
            matched = entry.offset
        }
        let states = try makeSequenceState(
            layerKinds: layerKinds, promptLength: 0, maxLength: maxLength)
        for (index, state) in states.enumerated() {
            guard let state = state as? PagedSequenceKV else { continue }
            if let snapshot = prefix[index] {
                precondition(
                    state.windowSize == nil,
                    "prefix donated to a windowed layer \(index)")
                var keys = snapshot.keys
                var values = snapshot.values
                if keys.ndim == 4 {
                    keys = keys.squeezed(axis: 0)
                    values = values.squeezed(axis: 0)
                }
                precondition(
                    keys.dim(1) == snapshot.offset,
                    "full-attention prefix snapshot must cover [0, offset)")
                state.write(keys: keys, values: values)
            } else if state.windowSize != nil, matched > 0 {
                state.fastForward(to: matched)
            }
        }
        return states
    }

    public func release(_ state: [CBv2SequenceKV?]) {
        for entry in state {
            guard let entry else { continue }
            guard let paged = entry as? PagedSequenceKV else {
                fatalError("[PagedKVBackend] release of a foreign sequence state")
            }
            paged.releaseStorage()
        }
    }

    public var bytesInUse: Int { pool.bytesInUse }
    public var bytesCapacity: Int { pool.bytesCapacity }
    /// Admission-relevant bytes (worst-case reservations of live requests).
    public var bytesReserved: Int { pool.bytesReserved }
    /// Paged snapshots are views over the SHARED slabs; the donor's pages
    /// are recycled once its state is released, so donated snapshots must
    /// be device-materialized before the prefix cache indexes them.
    public var requiresMaterializedSnapshots: Bool { true }

    // MARK: - Helpers

    func pageNeeds(layerKinds: [CBv2LayerKind], maxLength: Int) -> [PagedKVGroupKey: Int] {
        var needs: [PagedKVGroupKey: Int] = [:]
        for kind in layerKinds where kind.sharesKVWithLayer == nil {
            let pages = PagedKVPool.pageDemand(
                kind: kind, maxLength: maxLength, config: pool.config)
            needs[PagedKVGroupKey(kind), default: 0] += pages
        }
        return needs
    }

    /// One layer cache per model layer (KV-shared layers get a borrowing
    /// cache with no rows). `attentionSoftcap` comes from model config —
    /// it is not part of the contract's per-call surface.
    public func makeLayerCaches(attentionSoftcap: Float? = nil) -> [PagedLayerCache] {
        layerKinds.enumerated().map { index, kind in
            PagedLayerCache(
                layerIndex: index, kind: kind, pool: pool,
                attentionSoftcap: attentionSoftcap)
        }
    }
}
