// LayerCacheV2.swift
//
// The per-layer batch-facing cache object v2 models interact with.
//
// `CBv2LayerCache` conforms to BOTH:
//  - `CBv2AttendingLayerCache` — the v2 surface: `updateAndAttend` owns the
//    KV update AND the attention computation (backends are swappable), and
//  - the legacy `KVCache` protocol — so it can travel through existing
//    `[KVCache]` plumbing. `update(keys:values:)` TRAPS: v2-adapted models
//    must call `updateAndAttend` (via the hook in `attentionWithCacheUpdate`).
//
// Batch membership is object membership: join = `appendRow`, leave =
// `removeRow`. There is no shared frontier, no left padding, no batch-wide
// trim — a row's KV state and positions cannot be affected by its batchmates.

import Foundation
import MLX

/// Per-layer, batch-facing cache + attention dispatcher for the v2 engine.
public final class CBv2LayerCache: CBv2AttendingLayerCache {

    public let layerIndex: Int
    public let kind: CBv2LayerKind

    /// Ordered per-row sequence states (row order == batch row order).
    /// Empty for KV-shared layers (`kind.sharesKVWithLayer != nil`), which
    /// own no storage and borrow via `attendBorrowing`.
    public private(set) var rows: [CBv2SequenceKV]

    /// Per-row absolute RoPE offsets `[B]` (int32, device array).
    ///
    /// REBUILT from host integers only on membership changes; ADVANCED
    /// on-device (`+ L`) inside `updateAndAttend`. The step loop therefore
    /// never uploads fresh host arrays and never syncs (`.item()`) — the
    /// engine loop's per-step `asyncEval` (over `innerState()`) collapses
    /// the lazy `+ L` chain so it cannot grow O(steps) (DAR-325).
    ///
    /// NOTE: models must read this BEFORE calling `updateAndAttend` for the
    /// step (it holds the offsets of the tokens about to be processed), and
    /// KV-shared layers must reuse the SOURCE layer's pre-update capture —
    /// the same discipline as `gemma4CapturePositionOffset`.
    public var positionOffsets: MLXArray { cachedPositionOffsets }

    private var cachedPositionOffsets: MLXArray

    /// Times `positionOffsets` was rebuilt from host integers. Tests assert
    /// this only moves on membership changes — never inside the step loop.
    public private(set) var positionOffsetsHostRebuilds = 0

    /// Optional attention-logit soft cap (`cap * tanh(qk / cap)` before
    /// softmax, Gemma-2 style). Construction-time configuration from model
    /// config — identical plumbing on both backends (`PagedLayerCache` takes
    /// the same parameter); never part of the per-call contract surface.
    public let attentionSoftcap: Float?

    /// Vision span context for the CURRENT span-containing prefill chunk,
    /// bound by the engine immediately before that chunk's graph build and
    /// unbound immediately after (`CBv2SpanMaskBinding`). ALWAYS nil outside
    /// that window, so decode, text chunks, and other requests' chunks take
    /// the untouched pinned paths. The engine only ever binds it while the
    /// cache's rows are exactly the one vision request's row (prefill is
    /// per-request [1, chunk]), so the span mask can never leak onto a
    /// batchmate's computation.
    private(set) var boundSpanContext: CBv2SpanChunkContext?

    public init(
        layerIndex: Int, kind: CBv2LayerKind, rows: [CBv2SequenceKV] = [],
        attentionSoftcap: Float? = nil
    ) {
        precondition(
            kind.sharesKVWithLayer == nil || rows.isEmpty,
            "CBv2LayerCache: KV-shared layers own no rows")
        self.layerIndex = layerIndex
        self.kind = kind
        self.rows = rows
        self.attentionSoftcap = attentionSoftcap
        self.cachedPositionOffsets = Self.buildPositionOffsets(rows)
    }

    // MARK: - Membership (the ONLY places positionOffsets is host-rebuilt)

    public func appendRow(_ row: CBv2SequenceKV) {
        precondition(
            kind.sharesKVWithLayer == nil, "CBv2LayerCache: cannot add rows to a KV-shared layer")
        rows.append(row)
        rebuildPositionOffsets()
    }

    public func removeRow(at index: Int) {
        rows.remove(at: index)
        rebuildPositionOffsets()
    }

    /// Replace the whole row set (batch recomposition). Also the correct way
    /// to re-sync `positionOffsets` after out-of-band row mutation
    /// (e.g. rollback during speculative verification).
    public func setRows(_ newRows: [CBv2SequenceKV]) {
        precondition(
            kind.sharesKVWithLayer == nil || newRows.isEmpty,
            "CBv2LayerCache: KV-shared layers own no rows")
        rows = newRows
        rebuildPositionOffsets()
    }

    // MARK: - CBv2AttendingLayerCache

    public func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        precondition(
            kind.sharesKVWithLayer == nil,
            "CBv2LayerCache: KV-shared layer \(layerIndex) must use attendBorrowing")
        let output = CBv2AttentionV1.updateAndAttend(
            rows: rows, kind: kind,
            queries: queries, keys: keys, values: values,
            scale: scale, sinks: sinks, softcap: attentionSoftcap,
            spanContext: boundSpanContext)
        // Advance offsets ON-DEVICE (uniform: decode is [B,1], prefill is
        // [1,chunk] — L is the same for every row in the call).
        cachedPositionOffsets = cachedPositionOffsets + Int32(queries.dim(2))
        return output
    }

    public func attendBorrowing(
        source: CBv2AttendingLayerCache,
        queries: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        precondition(
            kind.sharesKVWithLayer != nil,
            "CBv2LayerCache: attendBorrowing called on a storage-owning layer")
        precondition(
            kind.sharesKVWithLayer == source.layerIndex,
            "CBv2LayerCache: layer \(layerIndex) shares KV with \(kind.sharesKVWithLayer!), not \(source.layerIndex)"
        )
        return CBv2AttentionV1.attendBorrowing(
            sourceRows: source.rows, sourceKind: source.kind, kind: kind,
            queries: queries, scale: scale, sinks: sinks, softcap: attentionSoftcap,
            spanContext: boundSpanContext)
    }

    // MARK: - Private

    private func rebuildPositionOffsets() {
        positionOffsetsHostRebuilds += 1
        CBv2CoreInstrumentation.recordPositionOffsetsHostRebuild()
        cachedPositionOffsets = Self.buildPositionOffsets(rows)
    }

    private static func buildPositionOffsets(_ rows: [CBv2SequenceKV]) -> MLXArray {
        MLXArray(rows.map { Int32($0.absoluteOffset) })
    }
}

// MARK: - Vision span-mask binding

extension CBv2LayerCache: CBv2SpanMaskBinding {
    public func bindSpanContext(_ context: CBv2SpanChunkContext?) {
        boundSpanContext = context
    }
}

// MARK: - Legacy KVCache conformance

extension CBv2LayerCache: KVCache {
    /// Legacy scalar offset: max row offset. Host integers only — no sync.
    public var offset: Int {
        rows.reduce(0) { max($0, $1.absoluteOffset) }
    }

    public var maxSize: Int? {
        switch kind.attention {
        case .full: return nil
        case .slidingWindow(let window): return window
        }
    }

    /// The engine loop evaluates cache inner state each step (asyncEval) to
    /// collapse lazy chains: per-row storage plus the positionOffsets chain.
    public func innerState() -> [MLXArray] {
        var arrays = [cachedPositionOffsets]
        for row in rows {
            if let provider = row as? CBv2InnerStateProviding {
                arrays.append(contentsOf: provider.cbv2InnerState())
            }
        }
        return arrays
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(
            "CBv2LayerCache.update(keys:values:) is unsupported — v2-adapted models must call updateAndAttend (layer \(layerIndex))"
        )
    }

    public var state: [MLXArray] {
        get { [] }
        set {
            fatalError("CBv2LayerCache has no serializable state (layer \(layerIndex))")
        }
    }

    public var metaState: [String] {
        get { [] }
        set {
            fatalError("CBv2LayerCache has no metaState (layer \(layerIndex))")
        }
    }

    public var isTrimmable: Bool { false }

    @discardableResult
    public func trim(_ n: Int) -> Int { 0 }

    public func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        fatalError(
            "CBv2LayerCache.makeMask is unsupported — v2 attention owns its masks (layer \(layerIndex))"
        )
    }

    public func copy() -> any KVCache {
        fatalError(
            "CBv2LayerCache.copy is unsupported — v2 rows are engine-owned (layer \(layerIndex))")
    }
}
