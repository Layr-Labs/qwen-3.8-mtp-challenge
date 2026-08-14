// CBv2InvariantSuiteTests.swift — WS-G: executable encodings of the report 10
// §4 invariants (docs/research/batching-engine-2026-07/10-…md) at the unit
// level. One test (or one documented N/A-by-construction assertion) per
// invariant; docs/engine-v2/TESTPLAN.md maps the full matrix.
//
// Invariants covered here:
//   1  Position invariance under batch composition (unit level)
//   2  Mask/KV congruence — N/A-by-construction, asserted structurally
//   3  Fixed update-order convention (post-update retention boundaries)
//   4  Padding provably inert — no padding exists; rollback scrubbing tested
//   5  One numerically-pinned attention path per (model, phase)
//   6  Sliding window == storage eviction keyed to absolute positions
//   7  Graph/metadata hygiene — zero host syncs from the KV/attention layer
//   8  Explicit sync points — snapshot() temporal order; windowed caches are
//      never coerced into full-history prefixes
//   9  Sinks are a first-class attention capability (static eligibility +
//      numerical effect)
//  10  B=1/batched semantic parity — covered by CBv2ParityTests (cross-ref)

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLMCommon

final class CBv2InvariantSuiteTests: XCTestCase {

    private let kvHeads = 2
    private let headDim = 4

    private func kvChunk(_ n: Int, fill: Float? = nil, seed: UInt64 = 7) -> (MLXArray, MLXArray) {
        if let fill {
            let shape = [1, kvHeads, n, headDim]
            return (
                MLXArray.full(shape, values: MLXArray(fill)),
                MLXArray.full(shape, values: MLXArray(fill))
            )
        }
        MLXRandom.seed(seed)
        let k = MLXRandom.normal([1, kvHeads, n, headDim])
        let v = MLXRandom.normal([1, kvHeads, n, headDim])
        return (k, v)
    }

    /// A chunk whose entries encode their absolute positions, so eviction and
    /// ordering can be asserted exactly.
    private func positionCodedChunk(from start: Int, count: Int) -> (MLXArray, MLXArray) {
        let values = (start ..< start + count).map { Float($0) }
        let base = MLXArray(values).reshaped(1, 1, count, 1)
        let k = broadcast(base, to: [1, kvHeads, count, headDim])
        return (k, k)
    }

    private func firstPositions(_ array: MLXArray) -> [Float] {
        // [1, kvHeads, n, headDim] → position code of each temporal slot.
        array[0, 0, 0..., 0].asArray(Float.self)
    }

    // MARK: - Invariant 1: position invariance under batch composition

    /// A row's absolute offset (its RoPE position source) never moves when
    /// other rows join or leave, and the layer cache's positionOffsets are
    /// captured pre-update in row order.
    func testInvariant1_PositionCountersUnaffectedByMembership() {
        let kind = CBv2LayerKind(
            attention: .full, headDim: headDim, kvHeads: kvHeads, queryHeads: 4)
        let cache = HarnessLayerCache(layerIndex: 0, kind: kind)

        let rowA = HarnessFullSequenceKV()
        _ = rowA.update(keys: kvChunk(5).0, values: kvChunk(5).1)
        cache.rows = [rowA]
        XCTAssertEqual(cache.positionOffsets.asArray(Int32.self), [5])

        // A second row joins with a different length — row A's offset and its
        // slot in positionOffsets must not move.
        let rowB = HarnessFullSequenceKV()
        _ = rowB.update(keys: kvChunk(9).0, values: kvChunk(9).1)
        cache.rows = [rowA, rowB]
        XCTAssertEqual(cache.positionOffsets.asArray(Int32.self), [5, 9])
        XCTAssertEqual(rowA.absoluteOffset, 5)

        // Row B leaves; row A still untouched.
        cache.rows = [rowA]
        XCTAssertEqual(cache.positionOffsets.asArray(Int32.self), [5])
        XCTAssertEqual(rowA.absoluteOffset, 5)
    }

    /// KV-shared layers must consume the SAME captured offset as their source
    /// layer (Gemma-4's 20 shared layers). Encoded via attendBorrowing: the
    /// borrowed keys are the source's retained keys — byte-identical.
    func testInvariant1_SharedLayerBorrowsSourceOffsetAndKV() {
        let sourceKind = CBv2LayerKind(
            attention: .full, headDim: headDim, kvHeads: kvHeads, queryHeads: 4)
        let sharedKind = CBv2LayerKind(
            attention: .full, sharesKVWithLayer: 0,
            headDim: headDim, kvHeads: kvHeads, queryHeads: 4)

        let source = HarnessLayerCache(layerIndex: 0, kind: sourceKind)
        let shared = HarnessLayerCache(layerIndex: 1, kind: sharedKind)

        let row = HarnessFullSequenceKV()
        source.rows = [row]

        MLXRandom.seed(3)
        let q = MLXRandom.normal([1, 4, 1, headDim])
        let (k, v) = kvChunk(1, seed: 4)

        // Source layer updates (offset 0 → 1) …
        let sourceOut = source.updateAndAttend(
            queries: q, keys: k, values: v, scale: 1.0, sinks: nil)
        // … then the shared layer borrows: same single key/value, no second
        // update (offset must remain 1).
        let sharedOut = shared.attendBorrowing(
            source: source, queries: q, scale: 1.0, sinks: nil)

        XCTAssertEqual(row.absoluteOffset, 1, "borrowing must not advance the source offset")
        XCTAssertTrue(
            allClose(sourceOut, sharedOut, rtol: 0, atol: 0).item(Bool.self),
            "shared layer attended different KV than its source")
    }

    // MARK: - Invariant 2: mask/KV congruence (N/A by construction)

    /// v2 decode builds no masks at all — congruence cannot be violated. This
    /// test asserts the structural facts that make the old bug class
    /// unrepresentable: (a) decode attends with mask mode "none" for every
    /// layer kind, (b) each row's attention sees exactly its own retained KV
    /// length, never a batch-wide mask axis.
    func testInvariant2_DecodeIsMaskFreeAndPerRow() throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let engine = CBv2HarnessEngine(model: model)
        _ = try engine.join(prompt: makePromptTokens(length: 20, seed: 21), maxTokens: 8)
        _ = try engine.join(prompt: makePromptTokens(length: 5, seed: 22), maxTokens: 8)

        // Clear prefill-time logs, then decode.
        let decodeStart = engine.layerCaches.map { $0.maskModeLog.count }
        for _ in 0 ..< 4 { engine.decodeStep() }

        for (i, cache) in engine.layerCaches.enumerated() {
            let decodeLog = cache.maskModeLog[decodeStart[i]...]
            XCTAssertFalse(decodeLog.isEmpty)
            XCTAssertTrue(
                decodeLog.allSatisfy { $0 == "none" },
                "layer \(i) used a mask during decode: \(Array(decodeLog))")
        }

        // Rows retain different lengths (mixed prompts) — legal precisely
        // because no shared mask axis exists.
        let retained = engine.layerCaches[0].rows.map { $0.retainedCount }
        XCTAssertEqual(Set(retained).count, 2, "rows should have distinct retained lengths")
    }

    // MARK: - Invariant 3: fixed update-order convention

    /// update() returns POST-update retention: the current token's K/V is
    /// always attendable (>= vs > convention drift was compiled-decode bug
    /// 29602e0), and the windowed prefill retention keeps window-1 old
    /// entries so every chunk query has its full window available.
    func testInvariant3_PostUpdateRetentionConvention() {
        let window = 8
        let kv = HarnessWindowedSequenceKV(window: window)

        // Prefill chunk of 6 (< window): everything retained, including the
        // newest token.
        let (k6, v6) = positionCodedChunk(from: 0, count: 6)
        let (r1, _) = kv.update(keys: k6, values: v6)
        XCTAssertEqual(r1.dim(2), 6)
        XCTAssertEqual(firstPositions(r1).last, 5, "newest token must be attendable")

        // Second chunk of 6 (straddles the window): retained = min(6, window-1)
        // old + 6 new = 13 → but old is 6 ≤ window-1=7 so 12 total.
        let (k2, v2) = positionCodedChunk(from: 6, count: 6)
        let (r2, _) = kv.update(keys: k2, values: v2)
        XCTAssertEqual(r2.dim(2), 12)
        XCTAssertEqual(firstPositions(r2).first, 0)
        XCTAssertEqual(firstPositions(r2).last, 11)

        // Third chunk of 4: old (12) trimmed to window-1=7 THEN 4 appended.
        let (k3, v3) = positionCodedChunk(from: 12, count: 4)
        let (r3, _) = kv.update(keys: k3, values: v3)
        XCTAssertEqual(r3.dim(2), window - 1 + 4)
        XCTAssertEqual(firstPositions(r3).first, 5, "oldest retained must be pos 12-7=5")
        XCTAssertEqual(firstPositions(r3).last, 15)

        // Decode step: append then trim to exactly `window`, keeping the
        // RECENT end (positions 9…16).
        let (k4, v4) = positionCodedChunk(from: 16, count: 1)
        let (r4, _) = kv.update(keys: k4, values: v4)
        XCTAssertEqual(r4.dim(2), window)
        XCTAssertEqual(firstPositions(r4), (9 ... 16).map(Float.init))
        XCTAssertEqual(kv.absoluteOffset, 17)
    }

    // MARK: - Invariant 4: rollback scrubs unconfirmed tail state

    /// After rollback(n), the rolled-back K/V must be unreachable: attention
    /// output is identical to a sequence that never saw those tokens, even if
    /// the tail was poisoned with sentinel values first.
    func testInvariant4_RollbackScrubsTailState() {
        let clean = HarnessFullSequenceKV()
        let dirty = HarnessFullSequenceKV()

        let (k, v) = positionCodedChunk(from: 0, count: 6)
        _ = clean.update(keys: k, values: v)
        _ = dirty.update(keys: k, values: v)

        // dirty speculates 3 extra tokens, then gets poisoned + rolled back.
        let (sk, sv) = positionCodedChunk(from: 6, count: 3)
        _ = dirty.update(keys: sk, values: sv)
        dirty.poisonTail(3, value: .nan)  // worst case: NaN in stale KV
        dirty.rollback(3)

        XCTAssertEqual(dirty.absoluteOffset, clean.absoluteOffset)
        XCTAssertEqual(dirty.retainedCount, clean.retainedCount)

        // Next step on both must be bit-identical (NaN never attended).
        let (nk, nv) = positionCodedChunk(from: 6, count: 1)
        let (ck, _) = clean.update(keys: nk, values: nv)
        let (dk, _) = dirty.update(keys: nk, values: nv)
        XCTAssertTrue(
            allClose(ck, dk, rtol: 0, atol: 0).item(Bool.self),
            "rolled-back tail leaked into subsequent state")
        XCTAssertFalse(
            any(isNaN(dk)).item(Bool.self), "poisoned KV survived rollback")

        // Windowed rollback must un-append across its retention too.
        let windowed = HarnessWindowedSequenceKV(window: 4)
        let (wk, wv) = positionCodedChunk(from: 0, count: 3)
        _ = windowed.update(keys: wk, values: wv)
        let (wk2, wv2) = positionCodedChunk(from: 3, count: 1)
        _ = windowed.update(keys: wk2, values: wv2)
        windowed.rollback(1)
        XCTAssertEqual(windowed.absoluteOffset, 3)
        let (wk3, wv3) = positionCodedChunk(from: 3, count: 1)
        let (wr, _) = windowed.update(keys: wk3, values: wv3)
        XCTAssertEqual(firstPositions(wr), [0, 1, 2, 3])
    }

    // MARK: - Invariant 5: one numerically-pinned attention path per phase

    /// The mask mode is a pure function of (phase): prefill (L>1) always
    /// "array", decode (L==1) always "none" — it never drifts across steps
    /// for the same logical computation (MLX #3384 bug class).
    func testInvariant5_AttentionPathPinnedPerPhase() throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let engine = CBv2HarnessEngine(model: model)

        // Prompt > window and > chunk so prefill emits several chunks.
        _ = try engine.join(prompt: makePromptTokens(length: 40, seed: 23), maxTokens: 24)
        for _ in 0 ..< 10 { engine.decodeStep() }
        _ = try engine.join(prompt: makePromptTokens(length: 3, seed: 24), maxTokens: 24)
        for _ in 0 ..< 10 { engine.decodeStep() }

        // The engine's decode caches log ONLY decode-phase calls (prefill
        // runs on per-request caches inside join()). Decode must be
        // uniformly "none" across all steps, before and after the join.
        for cache in engine.layerCaches {
            XCTAssertFalse(cache.maskModeLog.isEmpty)
            XCTAssertEqual(
                Set(cache.maskModeLog), ["none"],
                "decode mask mode drifted on layer \(cache.layerIndex): \(Set(cache.maskModeLog))")
        }
    }

    // MARK: - Invariant 6: window eviction keyed to absolute positions

    /// Boundary exactness at window, window±1, wrap, and multi-chunk cross;
    /// the RECENT end is kept and temporal order is preserved throughout.
    func testInvariant6_WindowEvictionBoundaries() {
        let window = 4
        let kv = HarnessWindowedSequenceKV(window: window)

        // Fill to exactly `window` one token at a time.
        for position in 0 ..< window {
            let (k, v) = positionCodedChunk(from: position, count: 1)
            let (r, _) = kv.update(keys: k, values: v)
            XCTAssertEqual(r.dim(2), position + 1, "no eviction below the window")
        }
        XCTAssertEqual(kv.retainedCount, window)

        // window + 1: oldest (pos 0) evicted, recent end kept.
        let (k1, v1) = positionCodedChunk(from: window, count: 1)
        let (r1, _) = kv.update(keys: k1, values: v1)
        XCTAssertEqual(r1.dim(2), window)
        XCTAssertEqual(firstPositions(r1), [1, 2, 3, 4])

        // Long decode run: retention stays pinned at `window`, contents are
        // always the most recent `window` absolute positions in temporal
        // order (wrap after wrap after wrap).
        for position in (window + 1) ..< (4 * window) {
            let (k, v) = positionCodedChunk(from: position, count: 1)
            let (r, _) = kv.update(keys: k, values: v)
            XCTAssertEqual(r.dim(2), window)
            XCTAssertEqual(
                firstPositions(r),
                ((position - window + 1) ... position).map(Float.init),
                "wrong window contents at absolute position \(position)")
        }
        XCTAssertEqual(kv.absoluteOffset, 4 * window)
    }

    // MARK: - Invariant 7: zero host syncs from the KV/attention layer

    /// A 10-step decode performs no GPU→CPU sync originating from the layer
    /// caches or sequence KV (the mocks route any would-be sync through
    /// CBv2HarnessSyncCounter). The only host readbacks are the engine's
    /// per-step token boundary syncs.
    func testInvariant7_NoHostSyncsInDecodeHotPath() throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let engine = CBv2HarnessEngine(model: model)
        _ = try engine.join(prompt: makePromptTokens(length: 10, seed: 25), maxTokens: 16)
        _ = try engine.join(prompt: makePromptTokens(length: 20, seed: 26), maxTokens: 16)

        CBv2HarnessSyncCounter.reset()
        for _ in 0 ..< 10 { engine.decodeStep() }

        XCTAssertEqual(
            CBv2HarnessSyncCounter.layerCacheSyncs, 0,
            "KV/attention layer performed host syncs during decode")
        XCTAssertEqual(
            CBv2HarnessSyncCounter.boundarySyncs, 10,
            "expected exactly one boundary sync (token readback) per step")
    }

    // MARK: - Invariant 8: explicit state sync points

    /// snapshot() must return temporal order (oldest → newest) with the true
    /// absolute offset — including for windowed storage after wrap — and a
    /// windowed cache must never be coerced into a full-history prefix
    /// (the backend rejects donated prefixes for windowed layers).
    func testInvariant8_SnapshotTemporalOrderAndWindowedPrefixExclusion() throws {
        let window = 4
        let kv = HarnessWindowedSequenceKV(window: window)
        for position in 0 ..< 7 {
            let (k, v) = positionCodedChunk(from: position, count: 1)
            _ = kv.update(keys: k, values: v)
        }
        let snap = kv.snapshot()
        XCTAssertEqual(snap.offset, 7)
        XCTAssertEqual(firstPositions(snap.keys), [3, 4, 5, 6], "snapshot must be temporal order")

        // Full-attention snapshot round-trips through prefix adoption…
        let full = HarnessFullSequenceKV()
        let (fk, fv) = positionCodedChunk(from: 0, count: 6)
        _ = full.update(keys: fk, values: fv)
        let fullSnap = full.snapshot()

        let backend = HarnessKVBackend()
        let kinds = [
            CBv2LayerKind(attention: .full, headDim: headDim, kvHeads: kvHeads, queryHeads: 4),
            CBv2LayerKind(
                attention: .slidingWindow(window), headDim: headDim, kvHeads: kvHeads,
                queryHeads: 4),
        ]
        // …while the windowed layer's prefix slot is nil (recomputed).
        let adopted = try backend.makeSequenceState(
            adopting: [(fullSnap.keys, fullSnap.values, fullSnap.offset), nil],
            layerKinds: kinds, maxLength: 128)
        XCTAssertEqual(adopted[0]?.absoluteOffset, 6)
        XCTAssertEqual(
            adopted[1]?.absoluteOffset, 6,
            "windowed layer fast-forwards to the uniform adopted offset")
        XCTAssertEqual(adopted[1]?.retainedCount, 0, "windowed storage must start empty")
    }

    // MARK: - Invariant 9: sinks are a first-class capability

    /// A backend that cannot honor sinks must be statically ineligible for
    /// sink models: engine build throws, no silent sink-dropping.
    func testInvariant9_SinklessBackendStaticallyIneligible() {
        let model = TinyTestModel.make(seed: 0xC0FFEE, withSinks: true)
        let backend = HarnessKVBackend(supportsSinks: false)

        XCTAssertThrowsError(
            try backend.makeSequenceState(
                layerKinds: model.layerKinds, promptLength: 8, maxLength: 64)
        ) { error in
            guard case CBv2KVError.backendIneligible = error else {
                return XCTFail("expected backendIneligible, got \(error)")
            }
        }
    }

    /// Sinks must actually change the attention numerics (denominator-only
    /// fold) — guarding against a path that accepts sinks and drops them.
    func testInvariant9_SinksAffectAttentionOutput() {
        let kind = CBv2LayerKind(
            attention: .full, hasSinks: true, headDim: headDim, kvHeads: kvHeads, queryHeads: 4)

        MLXRandom.seed(31)
        let q = MLXRandom.normal([1, 4, 1, headDim])
        let (k, v) = kvChunk(3, seed: 32)
        let sinks = MLXArray([Float](repeating: 4.0, count: 4))

        let withSinks = HarnessLayerCache(
            layerIndex: 0, kind: kind, rows: [HarnessFullSequenceKV()]
        ).updateAndAttend(queries: q, keys: k, values: v, scale: 1.0, sinks: sinks)

        let withoutKind = CBv2LayerKind(
            attention: .full, hasSinks: false, headDim: headDim, kvHeads: kvHeads, queryHeads: 4)
        let withoutSinks = HarnessLayerCache(
            layerIndex: 0, kind: withoutKind, rows: [HarnessFullSequenceKV()]
        ).updateAndAttend(queries: q, keys: k, values: v, scale: 1.0, sinks: nil)

        XCTAssertFalse(
            allClose(withSinks, withoutSinks, atol: 1e-6).item(Bool.self),
            "sinks had no effect — they are being silently dropped")

        // Denominator-only: sinks uniformly scale every attention weight by
        // D/(D + Σexp(sink)) < 1, so the output magnitude must strictly
        // decrease, never grow.
        let normWith = sum(withSinks * withSinks).item(Float.self)
        let normWithout = sum(withoutSinks * withoutSinks).item(Float.self)
        XCTAssertLessThan(normWith, normWithout, "sinks must only dilute the softmax")
    }

    // MARK: - Invariant 10: cross-reference

    /// B=1 / batched / (later) compiled semantic parity is a whole-pipeline
    /// property — pinned by CBv2ParityTests (legacy vs v2, token-exact) and
    /// CBv2InvarianceTests (solo vs batched, token-exact). This placeholder
    /// makes the mapping executable so the suite fails loudly if those
    /// classes are ever renamed away.
    func testInvariant10_ParitySuitesExist() {
        XCTAssertNotNil(CBv2ParityTests.self)
        XCTAssertNotNil(CBv2InvarianceTests.self)
    }
}
