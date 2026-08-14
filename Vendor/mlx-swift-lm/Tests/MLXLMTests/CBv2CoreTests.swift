// CBv2CoreTests.swift
//
// Workstream A (core runtime) unit tests for ContinuousBatchingV2:
//  - Full/Windowed/Quantized per-sequence KV: append/rollback/snapshot
//    round-trips, windowed eviction at exact boundaries, temporal order
//    after wrap, multi-chunk window crossings.
//  - AttentionV1 parity: per-row output at B=3 mixed lengths {7, 130, 950}
//    equals running each row alone, with/without window, with sinks, GQA.
//  - Reference correctness: v2 path vs a float32 full-history reference
//    with explicit causal∧window masking.
//  - positionOffsets correctness under join/leave churn.
//  - No host syncs / no host rebuilds during a 10-step decode.
//
// No model weights required — everything runs on tiny random tensors.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

// MARK: - Helpers

/// K/V whose element [0, h, t, d] encodes the token's absolute position (plus
/// small head/dim offsets) so storage order is exactly checkable.
private func positionCoded(from: Int, count: Int, kvHeads: Int = 2, headDim: Int = 4) -> MLXArray {
    let positions = MLXArray((0 ..< count).map { Float(from + $0) }).reshaped([1, 1, count, 1])
    let heads = MLXArray((0 ..< kvHeads).map { Float($0) * 0.25 }).reshaped([1, kvHeads, 1, 1])
    let dims = MLXArray((0 ..< headDim).map { Float($0) * 0.001 }).reshaped([1, 1, 1, headDim])
    return broadcast(positions + heads + dims, to: [1, kvHeads, count, headDim])
}

/// Expected buffer content for the given absolute positions, matching
/// `positionCoded`'s encoding.
private func expectedPositions(_ positions: [Int], kvHeads: Int = 2, headDim: Int = 4) -> MLXArray {
    guard !positions.isEmpty else {
        return MLXArray.zeros([1, kvHeads, 0, headDim])
    }
    let pos = MLXArray(positions.map { Float($0) }).reshaped([1, 1, positions.count, 1])
    let heads = MLXArray((0 ..< kvHeads).map { Float($0) * 0.25 }).reshaped([1, kvHeads, 1, 1])
    let dims = MLXArray((0 ..< headDim).map { Float($0) * 0.001 }).reshaped([1, 1, 1, headDim])
    return broadcast(pos + heads + dims, to: [1, kvHeads, positions.count, headDim])
}

private func expectClose(
    _ a: MLXArray, _ b: MLXArray, rtol: Double = 1e-5, atol: Double = 1e-6,
    _ label: String = "", sourceLocation: Testing.SourceLocation = #_sourceLocation
) {
    #expect(a.shape == b.shape, "shape mismatch \(label): \(a.shape) vs \(b.shape)",
        sourceLocation: sourceLocation)
    let close = allClose(a, b, rtol: rtol, atol: atol).item(Bool.self)
    #expect(close, "values differ \(label)", sourceLocation: sourceLocation)
}

/// Float32 full-history reference attention with explicit causal∧window
/// masking and denominator-only sinks — the gold semantics the storage-
/// eviction design must reproduce.
private func referenceAttention(
    queries: MLXArray,  // [1, qHeads, L, D]
    keys: MLXArray,  // [1, kvHeads, T, D] — FULL history, never evicted
    values: MLXArray,
    queryPositions: [Int],
    scale: Float,
    window: Int?,
    sinks: MLXArray?  // [qHeads]
) -> MLXArray {
    let q = queries.asType(.float32)
    let k = keys.asType(.float32)
    let v = values.asType(.float32)
    let (qHeads, L, D) = (q.dim(1), q.dim(2), q.dim(3))
    let (kvHeads, T) = (k.dim(1), k.dim(2))
    let repeats = qHeads / kvHeads

    let q5 = (q * scale).reshaped([1, kvHeads, repeats, L, D])
    let k5 = expandedDimensions(k, axis: 2)
    let v5 = expandedDimensions(v, axis: 2)
    var scores = q5.matmul(k5.transposed(0, 1, 2, 4, 3))  // [1, kvHeads, repeats, L, T]

    var allowed: [Bool] = []
    allowed.reserveCapacity(L * T)
    for i in 0 ..< L {
        let p = queryPositions[i]
        for j in 0 ..< T {
            var ok = j <= p
            if let window { ok = ok && j > p - window }
            allowed.append(ok)
        }
    }
    let mask = MLXArray(allowed).reshaped([L, T])
    scores = MLX.where(mask, scores, MLXArray(Float(-1e9)))

    let scores4 = scores.reshaped([1, qHeads, L, T])
    let weights: MLXArray
    if let sinks {
        let sinkLogits = sinks.asType(.float32).reshaped([1, qHeads, 1, 1])
        let m = maximum(scores4.max(axis: -1, keepDims: true), sinkLogits)
        let e = exp(scores4 - m)
        let denom = e.sum(axis: -1, keepDims: true) + exp(sinkLogits - m)
        weights = e / denom
    } else {
        weights = softmax(scores4, axis: -1, precise: true)
    }
    let out = weights.reshaped([1, kvHeads, repeats, L, T]).matmul(v5)
    return out.reshaped([1, qHeads, L, D])
}

// MARK: - FullSequenceKV

@Suite("CBv2Core: FullSequenceKV")
struct CBv2CoreFullSequenceKVTests {

    @Test func appendRollbackSnapshotRoundTrip() {
        let row = CBv2FullSequenceKV(promptLength: 5, maxLength: 64, kvHeads: 2, headDim: 4)
        #expect(row.absoluteOffset == 0)
        #expect(row.byteCount == 0)  // lazy allocation

        let (k1, v1) = row.update(
            keys: positionCoded(from: 0, count: 5), values: positionCoded(from: 100, count: 5))
        expectClose(k1, expectedPositions([0, 1, 2, 3, 4]), "keys after first append")
        expectClose(v1, expectedPositions([100, 101, 102, 103, 104]), "values after first append")
        #expect(row.absoluteOffset == 5)
        #expect(row.retainedCount == 5)
        #expect(row.byteCount > 0)

        let (k2, _) = row.update(
            keys: positionCoded(from: 5, count: 3), values: positionCoded(from: 105, count: 3))
        expectClose(k2, expectedPositions(Array(0 ..< 8)), "keys after second append")

        row.rollback(2)
        #expect(row.absoluteOffset == 6)
        let snap = row.snapshot()
        #expect(snap.offset == 6)
        expectClose(snap.keys, expectedPositions(Array(0 ..< 6)), "snapshot keys after rollback")

        // Re-append over the rolled-back tail: stale state must never resurface.
        let (k3, _) = row.update(
            keys: positionCoded(from: 60, count: 2), values: positionCoded(from: 160, count: 2))
        expectClose(k3, expectedPositions([0, 1, 2, 3, 4, 5, 60, 61]), "keys after re-append")
    }

    @Test func growthByDoublingPreservesContent() {
        let row = CBv2FullSequenceKV(promptLength: 0, maxLength: 4096, kvHeads: 2, headDim: 4)
        // Initial capacity is 256; push well past it in uneven chunks.
        var written = 0
        for chunk in [100, 200, 300, 111] {
            _ = row.update(
                keys: positionCoded(from: written, count: chunk),
                values: positionCoded(from: written, count: chunk))
            written += chunk
        }
        let snap = row.snapshot()
        #expect(snap.offset == written)
        expectClose(snap.keys, expectedPositions(Array(0 ..< written)), "keys after growth")
    }

    @Test func snapshotAdoptionRoundTrip() {
        let donor = CBv2FullSequenceKV(promptLength: 7, maxLength: 64, kvHeads: 2, headDim: 4)
        _ = donor.update(
            keys: positionCoded(from: 0, count: 7), values: positionCoded(from: 50, count: 7))
        let snap = donor.snapshot()

        let adopter = CBv2FullSequenceKV(
            promptLength: snap.offset, maxLength: 64, kvHeads: 2, headDim: 4)
        _ = adopter.update(keys: snap.keys, values: snap.values)
        #expect(adopter.absoluteOffset == 7)
        expectClose(adopter.snapshot().keys, snap.keys, "adopted keys")
    }
}

// MARK: - WindowedSequenceKV

@Suite("CBv2Core: WindowedSequenceKV")
struct CBv2CoreWindowedSequenceKVTests {

    /// Decode-style single-token appends across the exact window boundaries
    /// (window-1, window, window+1) and multiple wraps: the returned views
    /// must always be the last min(offset, window) positions in temporal order.
    @Test func singleTokenEvictionBoundariesAndWrap() {
        let window = 8
        let row = CBv2WindowedSequenceKV(window: window, kvHeads: 2, headDim: 4)
        for t in 0 ..< (2 * window + 3) {
            let (k, v) = row.update(
                keys: positionCoded(from: t, count: 1),
                values: positionCoded(from: 1000 + t, count: 1))
            let expectedStart = max(0, t + 1 - window)
            expectClose(
                k, expectedPositions(Array(expectedStart ..< (t + 1))), "keys at t=\(t)")
            expectClose(
                v, expectedPositions(Array((1000 + expectedStart) ..< (1000 + t + 1))),
                "values at t=\(t)")
            #expect(row.absoluteOffset == t + 1)
            #expect(row.retainedCount == min(t + 1, window))
        }
        // Ring holds exactly `window` slots.
        #expect(row.byteCount == 2 * 2 * window * 4 * MemoryLayout<Float32>.size)
    }

    /// Multi-token chunks crossing the window: the returned views must be
    /// `min(retained, window-1)` history entries plus the chunk, so every
    /// chunk token can attend its full window.
    @Test func multiChunkWindowCrossing() {
        let window = 8
        let row = CBv2WindowedSequenceKV(window: window, kvHeads: 2, headDim: 4)

        // Chunk 1: 5 tokens, no history.
        let (k1, _) = row.update(
            keys: positionCoded(from: 0, count: 5), values: positionCoded(from: 0, count: 5))
        expectClose(k1, expectedPositions(Array(0 ..< 5)), "chunk 1 return")
        #expect(row.retainedCount == 5)

        // Chunk 2: 6 tokens; history = min(5, 7) = 5 → return positions 0..<11.
        let (k2, _) = row.update(
            keys: positionCoded(from: 5, count: 6), values: positionCoded(from: 5, count: 6))
        expectClose(k2, expectedPositions(Array(0 ..< 11)), "chunk 2 return")
        #expect(row.absoluteOffset == 11)
        #expect(row.retainedCount == window)
        expectClose(
            row.snapshot().keys, expectedPositions(Array(3 ..< 11)), "retained ring after chunk 2")

        // Chunk 3: 4 tokens; history = min(8, 7) = 7 → return positions 4..<15.
        let (k3, _) = row.update(
            keys: positionCoded(from: 11, count: 4), values: positionCoded(from: 11, count: 4))
        expectClose(k3, expectedPositions(Array(4 ..< 15)), "chunk 3 return")
        expectClose(
            row.snapshot().keys, expectedPositions(Array(7 ..< 15)), "retained ring after chunk 3")
    }

    /// A single chunk larger than the window: only the trailing `window`
    /// tokens are retained afterwards.
    @Test func chunkLargerThanWindow() {
        let window = 4
        let row = CBv2WindowedSequenceKV(window: window, kvHeads: 2, headDim: 4)
        let (k, _) = row.update(
            keys: positionCoded(from: 0, count: 10), values: positionCoded(from: 0, count: 10))
        expectClose(k, expectedPositions(Array(0 ..< 10)), "oversized chunk return")
        #expect(row.retainedCount == window)
        expectClose(
            row.snapshot().keys, expectedPositions(Array(6 ..< 10)), "retained after oversize")
    }

    @Test func rollbackBeforeWrapIsFullRecovery() {
        let window = 8
        let row = CBv2WindowedSequenceKV(window: window, kvHeads: 2, headDim: 4)
        _ = row.update(
            keys: positionCoded(from: 0, count: 6), values: positionCoded(from: 0, count: 6))
        row.rollback(2)
        #expect(row.absoluteOffset == 4)
        #expect(row.retainedCount == 4)
        expectClose(row.snapshot().keys, expectedPositions(Array(0 ..< 4)), "post-rollback keys")

        // Fresh tokens replace the rolled-back ones; stale data never resurfaces.
        let (k, _) = row.update(
            keys: positionCoded(from: 40, count: 1), values: positionCoded(from: 40, count: 1))
        expectClose(k, expectedPositions([0, 1, 2, 3, 40]), "post-rollback re-append")
    }

    /// Rollback after the ring wrapped: the speculative writes destroyed the
    /// oldest in-window entries, so the window shrinks (and NEVER re-exposes
    /// destroyed slots as valid data).
    @Test func rollbackAfterWrapShrinksWindow() {
        let window = 4
        let row = CBv2WindowedSequenceKV(window: window, kvHeads: 2, headDim: 4)
        for t in 0 ..< 6 {
            _ = row.update(
                keys: positionCoded(from: t, count: 1), values: positionCoded(from: t, count: 1))
        }
        // offset 6, retained = positions [2, 6).
        row.rollback(2)
        #expect(row.absoluteOffset == 4)
        #expect(row.retainedCount == 2)  // positions [2, 4); 0..2 were destroyed
        expectClose(row.snapshot().keys, expectedPositions([2, 3]), "post-wrap rollback keys")

        // Refill: retained recovers as fresh tokens land.
        let (k, _) = row.update(
            keys: positionCoded(from: 40, count: 2), values: positionCoded(from: 40, count: 2))
        expectClose(k, expectedPositions([2, 3, 40, 41]), "refilled window")
        #expect(row.retainedCount == 4)
    }

    /// Non-zero initial offset (prefix-cache adoption): absolute positions
    /// keep counting from the adopted offset.
    @Test func initialOffsetAlignsAbsolutePositions() {
        let row = CBv2WindowedSequenceKV(window: 4, kvHeads: 2, headDim: 4, initialOffset: 10)
        #expect(row.absoluteOffset == 10)
        #expect(row.retainedCount == 0)
        let (k, _) = row.update(
            keys: positionCoded(from: 10, count: 3), values: positionCoded(from: 10, count: 3))
        expectClose(k, expectedPositions([10, 11, 12]), "adopted windowed keys")
        #expect(row.snapshot().offset == 13)
    }
}

// MARK: - QuantizedSequenceKV

@Suite("CBv2Core: QuantizedSequenceKV")
struct CBv2CoreQuantizedSequenceKVTests {

    @Test func roundTripReturnsDequantizedViews() {
        MLXRandom.seed(7)
        let row = CBv2QuantizedSequenceKV(
            promptLength: 8, maxLength: 128, kvHeads: 2, headDim: 64, groupSize: 64, bits: 8)
        let keys = MLXRandom.normal([1, 2, 8, 64]).asType(.float16)
        let values = MLXRandom.normal([1, 2, 8, 64]).asType(.float16)
        let (k, v) = row.update(keys: keys, values: values)
        #expect(k.shape == [1, 2, 8, 64])
        #expect(k.dtype == .float16)
        // 8-bit affine quantization error is small.
        expectClose(k, keys, rtol: 0.05, atol: 0.05, "dequantized keys")
        expectClose(v, values, rtol: 0.05, atol: 0.05, "dequantized values")
        #expect(row.byteCount > 0)
        // Quantized storage is smaller than fp16 storage for the same capacity.
        let fp16Row = CBv2FullSequenceKV(promptLength: 8, maxLength: 128, kvHeads: 2, headDim: 64)
        _ = fp16Row.update(keys: keys, values: values)
        #expect(row.byteCount < fp16Row.byteCount)
    }

    @Test func appendRollbackSnapshotBookkeeping() {
        MLXRandom.seed(8)
        let row = CBv2QuantizedSequenceKV(
            promptLength: 4, maxLength: 512, kvHeads: 2, headDim: 64, groupSize: 64, bits: 8)
        let first = MLXRandom.normal([1, 2, 4, 64]).asType(.float16)
        _ = row.update(keys: first, values: first)
        // Growth past initial capacity (4 + 256) in a second big chunk.
        let second = MLXRandom.normal([1, 2, 300, 64]).asType(.float16)
        let (k2, _) = row.update(keys: second, values: second)
        #expect(k2.dim(2) == 304)
        #expect(row.absoluteOffset == 304)

        row.rollback(300)
        #expect(row.absoluteOffset == 4)
        let snap = row.snapshot()
        #expect(snap.offset == 4)
        expectClose(snap.keys, first, rtol: 0.05, atol: 0.05, "snapshot after rollback")
    }

    @Test func groupSizeResolvesToHeadDimCompatible() {
        // headDim 96 is not divisible by 64 → nearest compatible is 32.
        let row = CBv2QuantizedSequenceKV(
            promptLength: 2, maxLength: 16, kvHeads: 1, headDim: 96, groupSize: 64, bits: 8)
        #expect(row.groupSize == 32)
    }
}

// MARK: - ContiguousKVBackend

@Suite("CBv2Core: ContiguousKVBackend")
struct CBv2CoreContiguousBackendTests {

    private let layerKinds: [CBv2LayerKind] = [
        CBv2LayerKind(attention: .full, headDim: 4, kvHeads: 2, queryHeads: 4),
        CBv2LayerKind(attention: .slidingWindow(8), headDim: 4, kvHeads: 2, queryHeads: 4),
        CBv2LayerKind(
            attention: .full, sharesKVWithLayer: 0, headDim: 4, kvHeads: 2, queryHeads: 4),
        CBv2LayerKind(attention: .full, hasSinks: true, headDim: 4, kvHeads: 2, queryHeads: 4),
    ]

    @Test func makeSequenceStateShapesAndAccounting() throws {
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 1 << 24, kvDType: .float32))
        let state = try backend.makeSequenceState(
            layerKinds: layerKinds, promptLength: 4, maxLength: 64)
        #expect(state.count == 4)
        #expect(state[0] is CBv2FullSequenceKV)
        #expect(state[1] is CBv2WindowedSequenceKV)
        #expect(state[2] == nil)  // KV-shared layer owns no storage
        #expect(state[3] is CBv2FullSequenceKV)

        #expect(backend.bytesInUse == 0)  // allocation is lazy
        _ = state[0]!.update(
            keys: positionCoded(from: 0, count: 4), values: positionCoded(from: 0, count: 4))
        #expect(backend.bytesInUse == state[0]!.byteCount)
        #expect(backend.bytesInUse > 0)

        backend.release(state)
        #expect(backend.bytesInUse == 0)
    }

    @Test func admissionThrowsCapacityExhausted() {
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 64))
        do {
            _ = try backend.makeSequenceState(
                layerKinds: layerKinds, promptLength: 4, maxLength: 64)
            Issue.record("expected capacityExhausted")
        } catch CBv2KVError.capacityExhausted(let needed, let available) {
            #expect(needed > available)
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }

    /// Admission RESERVES (PR#62 review): rows report `byteCount == 0` until
    /// their first update, so without a reservation several same-step
    /// admissions would each see `bytesInUse == 0` and collectively blow past
    /// `bytesCapacity`. The estimate is charged at admission and freed on
    /// release.
    @Test func sameStepAdmissionsReserveAndCannotExceedCapacity() throws {
        // One full layer, kvHeads 1, headDim 4, fp16 ⇒ initial slots
        // min(64, 4 + 256) = 64 ⇒ 64 * 1 * 4 * 2 * 2 = 1024 bytes/admission.
        let kinds = [CBv2LayerKind(attention: .full, headDim: 4, kvHeads: 1, queryHeads: 2)]
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1500))

        let first = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 4, maxLength: 64)
        #expect(backend.bytesInUse == 0)  // allocation is lazy
        #expect(backend.bytesReserved == 1024)  // but the estimate is reserved

        // Second same-step admission: 1024 more against a 1024 reservation
        // exceeds the 1500 budget even though bytesInUse is still 0.
        #expect(throws: CBv2KVError.self) {
            try backend.makeSequenceState(layerKinds: kinds, promptLength: 4, maxLength: 64)
        }

        // Releasing the first frees its reservation; a new admission fits.
        backend.release(first)
        #expect(backend.bytesReserved == 0)
        let third = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 4, maxLength: 64)
        backend.release(third)
    }

    /// `bytesReserved` stays truthful: once a row's actual allocation grows
    /// past its initial reservation (doubling), reservation trues up to the
    /// real byte count.
    @Test func bytesReservedTruesUpToActualAllocation() throws {
        let kinds = [CBv2LayerKind(attention: .full, headDim: 4, kvHeads: 1, queryHeads: 2)]
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 20))
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 1, maxLength: 100_000)
        let reservation = backend.bytesReserved
        #expect(reservation > 0)
        #expect(backend.bytesInUse == 0)  // lazy

        // Grow the row well past its initial estimate (buffer doubles).
        for i in 0 ..< 1000 {
            _ = state[0]!.update(
                keys: positionCoded(from: i, count: 1, kvHeads: 1, headDim: 4),
                values: positionCoded(from: i, count: 1, kvHeads: 1, headDim: 4))
        }
        #expect(backend.bytesInUse > reservation, "row grew past its initial reservation")
        #expect(
            backend.bytesReserved == backend.bytesInUse,
            "reservation trues up to actual bytes once exceeded")
        backend.release(state)
        #expect(backend.bytesReserved == 0)
    }

    @Test func invalidShareSourceIsRejected() {
        let bad = [
            CBv2LayerKind(
                attention: .full, sharesKVWithLayer: 5, headDim: 4, kvHeads: 2, queryHeads: 4)
        ]
        let backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 24))
        #expect(throws: CBv2KVError.self) {
            try backend.makeSequenceState(layerKinds: bad, promptLength: 1, maxLength: 8)
        }
    }

    @Test func quantizedConfigProducesQuantizedFullLayers() throws {
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 1 << 24, quantization: (groupSize: 64, bits: 8)))
        let kinds = [
            CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4),
            CBv2LayerKind(attention: .slidingWindow(8), headDim: 64, kvHeads: 2, queryHeads: 4),
        ]
        let state = try backend.makeSequenceState(layerKinds: kinds, promptLength: 4, maxLength: 64)
        #expect(state[0] is CBv2QuantizedSequenceKV)
        #expect(state[1] is CBv2WindowedSequenceKV)  // windowed stays unquantized
        backend.release(state)
    }

    @Test func adoptingPrefixCopiesFullLayersAndOffsetsWindowedRows() throws {
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 1 << 24, kvDType: .float32))

        // Donor prefix of 12 tokens on the full-attention layers.
        let donor = CBv2FullSequenceKV(promptLength: 12, maxLength: 64, kvHeads: 2, headDim: 4)
        _ = donor.update(
            keys: positionCoded(from: 0, count: 12), values: positionCoded(from: 200, count: 12))
        let donated = donor.snapshot()

        let prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?] = [
            (donated.keys, donated.values, donated.offset),
            nil,  // windowed: recomputed by the scheduler
            nil,  // shared: no storage
            (donated.keys, donated.values, donated.offset),
        ]
        let state = try backend.makeSequenceState(
            adopting: prefix, layerKinds: layerKinds, maxLength: 64)

        let adopted = state[0] as! CBv2FullSequenceKV
        #expect(adopted.absoluteOffset == 12)
        expectClose(adopted.snapshot().keys, donated.keys, "adopted full-layer keys")

        // Windowed rows start at the UNIFORM adopted offset (the engine
        // slices the matched prefix by cbv2RequiredRecompute before
        // adopting, then replays [offset, prompt) through all layers).
        let windowed = state[1] as! CBv2WindowedSequenceKV
        #expect(windowed.absoluteOffset == 12)
        #expect(windowed.retainedCount == 0)
        #expect(state[2] == nil)
        backend.release(state)
    }
}

// MARK: - AttentionV1 parity (solo vs batched)

/// Deterministic per-request Q/K/V streams shared between the solo and the
/// batched runs (the SAME MLXArray objects feed both paths).
private struct RequestStream {
    var prefillChunks: [(q: MLXArray, k: MLXArray, v: MLXArray)] = []
    var decodeSteps: [(q: MLXArray, k: MLXArray, v: MLXArray)] = []

    init(
        seed: UInt64, promptLength: Int, decodeSteps: Int, chunkSize: Int,
        queryHeads: Int, kvHeads: Int, headDim: Int
    ) {
        MLXRandom.seed(seed)
        var remaining = promptLength
        while remaining > 0 {
            let n = min(chunkSize, remaining)
            prefillChunks.append(
                (
                    q: MLXRandom.normal([1, queryHeads, n, headDim]).asType(.float16),
                    k: MLXRandom.normal([1, kvHeads, n, headDim]).asType(.float16),
                    v: MLXRandom.normal([1, kvHeads, n, headDim]).asType(.float16)
                ))
            remaining -= n
        }
        for _ in 0 ..< decodeSteps {
            self.decodeSteps.append(
                (
                    q: MLXRandom.normal([1, queryHeads, 1, headDim]).asType(.float16),
                    k: MLXRandom.normal([1, kvHeads, 1, headDim]).asType(.float16),
                    v: MLXRandom.normal([1, kvHeads, 1, headDim]).asType(.float16)
                ))
        }
    }
}

@Suite("CBv2Core: AttentionV1 parity", .serialized)
struct CBv2CoreAttentionParityTests {

    private let queryHeads = 4
    private let kvHeads = 2  // GQA everywhere
    private let headDim = 64
    private let scale: Float = 0.125
    private let lengths = [7, 130, 950]
    private let decodeSteps = 10
    private let chunkSize = 512
    private let maxLength = 2048

    private func makeRow(kind: CBv2LayerKind, promptLength: Int) -> CBv2SequenceKV {
        switch kind.attention {
        case .full:
            return CBv2FullSequenceKV(
                promptLength: promptLength, maxLength: maxLength, kvHeads: kvHeads,
                headDim: headDim)
        case .slidingWindow(let window):
            return CBv2WindowedSequenceKV(window: window, kvHeads: kvHeads, headDim: headDim)
        }
    }

    /// Prefill a row through a single-row layer cache (exactly how the
    /// engine runs per-request [1, chunk] prefill), returning the row.
    private func prefill(kind: CBv2LayerKind, stream: RequestStream, sinks: MLXArray?)
        -> (row: CBv2SequenceKV, outputs: [MLXArray])
    {
        let promptLength = stream.prefillChunks.reduce(0) { $0 + $1.q.dim(2) }
        let row = makeRow(kind: kind, promptLength: promptLength)
        let cache = CBv2LayerCache(layerIndex: 0, kind: kind, rows: [row])
        var outputs: [MLXArray] = []
        for chunk in stream.prefillChunks {
            outputs.append(
                cache.updateAndAttend(
                    queries: chunk.q, keys: chunk.k, values: chunk.v, scale: scale, sinks: sinks))
        }
        return (row, outputs)
    }

    private func runParity(kind: CBv2LayerKind, sinks: MLXArray?) {
        let streams = lengths.enumerated().map { index, length in
            RequestStream(
                seed: UInt64(1000 + index), promptLength: length, decodeSteps: decodeSteps,
                chunkSize: chunkSize, queryHeads: queryHeads, kvHeads: kvHeads, headDim: headDim)
        }

        // Solo: each request runs alone (prefill + decode at B == 1).
        var soloDecodeOutputs: [[MLXArray]] = []
        for stream in streams {
            let (row, _) = prefill(kind: kind, stream: stream, sinks: sinks)
            let cache = CBv2LayerCache(layerIndex: 0, kind: kind, rows: [row])
            var outputs: [MLXArray] = []
            for step in stream.decodeSteps {
                outputs.append(
                    cache.updateAndAttend(
                        queries: step.q, keys: step.k, values: step.v, scale: scale, sinks: sinks))
            }
            soloDecodeOutputs.append(outputs)
        }

        // Batched: same prefill flow, then all three rows decode together.
        let rows = streams.map { prefill(kind: kind, stream: $0, sinks: sinks).row }
        let batch = CBv2LayerCache(layerIndex: 0, kind: kind)
        for row in rows { batch.appendRow(row) }
        for stepIndex in 0 ..< decodeSteps {
            let q = concatenated(streams.map { $0.decodeSteps[stepIndex].q }, axis: 0)
            let k = concatenated(streams.map { $0.decodeSteps[stepIndex].k }, axis: 0)
            let v = concatenated(streams.map { $0.decodeSteps[stepIndex].v }, axis: 0)
            let out = batch.updateAndAttend(
                queries: q, keys: k, values: v, scale: scale, sinks: sinks)
            for (rowIndex, _) in streams.enumerated() {
                expectClose(
                    out[rowIndex ..< (rowIndex + 1)],
                    soloDecodeOutputs[rowIndex][stepIndex],
                    rtol: 1e-4, atol: 1e-4,
                    "row \(rowIndex) step \(stepIndex) (window: \(kind.attention), sinks: \(sinks != nil))"
                )
            }
        }
    }

    @Test func fullAttentionParity() {
        runParity(
            kind: CBv2LayerKind(
                attention: .full, headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads),
            sinks: nil)
    }

    @Test func slidingWindowParity() {
        runParity(
            kind: CBv2LayerKind(
                attention: .slidingWindow(512), headDim: headDim, kvHeads: kvHeads,
                queryHeads: queryHeads),
            sinks: nil)
    }

    @Test func slidingWindowWithSinksParity() {
        MLXRandom.seed(99)
        let sinks = MLXRandom.normal([queryHeads]).asType(.float16)
        runParity(
            kind: CBv2LayerKind(
                attention: .slidingWindow(512), hasSinks: true, headDim: headDim,
                kvHeads: kvHeads, queryHeads: queryHeads),
            sinks: sinks)
    }
}

// MARK: - Reference correctness (v2 path vs float32 full-history reference)

@Suite("CBv2Core: reference correctness", .serialized)
struct CBv2CoreReferenceTests {

    private let queryHeads = 4
    private let kvHeads = 2
    private let headDim = 64
    private let scale: Float = 0.125

    /// Run chunked prefill + decode through the v2 path and compare EVERY
    /// output against a float32 reference that keeps the full history and
    /// masks explicitly — validating that storage eviction + pinned mask
    /// modes reproduce exact sliding-window semantics.
    private func runReference(window: Int?, sinks: MLXArray?, chunks: [Int], decodeSteps: Int) {
        let kind = CBv2LayerKind(
            attention: window.map { .slidingWindow($0) } ?? .full,
            hasSinks: sinks != nil,
            headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads)
        let promptLength = chunks.reduce(0, +)
        let row: CBv2SequenceKV =
            window.map {
                CBv2WindowedSequenceKV(window: $0, kvHeads: kvHeads, headDim: headDim)
            }
            ?? CBv2FullSequenceKV(
                promptLength: promptLength, maxLength: 1024, kvHeads: kvHeads, headDim: headDim)
        let cache = CBv2LayerCache(layerIndex: 0, kind: kind, rows: [row])

        MLXRandom.seed(4242)
        var historyK: MLXArray? = nil
        var historyV: MLXArray? = nil
        var position = 0

        func step(_ n: Int, _ label: String) {
            let q = MLXRandom.normal([1, queryHeads, n, headDim]).asType(.float16)
            let k = MLXRandom.normal([1, kvHeads, n, headDim]).asType(.float16)
            let v = MLXRandom.normal([1, kvHeads, n, headDim]).asType(.float16)
            historyK = historyK.map { concatenated([$0, k], axis: 2) } ?? k
            historyV = historyV.map { concatenated([$0, v], axis: 2) } ?? v

            let out = cache.updateAndAttend(
                queries: q, keys: k, values: v, scale: scale, sinks: sinks)
            let expected = referenceAttention(
                queries: q, keys: historyK!, values: historyV!,
                queryPositions: Array(position ..< (position + n)),
                scale: scale, window: window, sinks: sinks)
            expectClose(
                out.asType(.float32), expected, rtol: 1e-2, atol: 1e-2, label)
            position += n
        }

        for (index, chunk) in chunks.enumerated() {
            step(chunk, "prefill chunk \(index) (window: \(String(describing: window)))")
        }
        for stepIndex in 0 ..< decodeSteps {
            step(1, "decode step \(stepIndex) (window: \(String(describing: window)))")
        }
    }

    @Test func windowedMatchesFullHistoryReference() {
        // Chunks cross the window boundary in every configuration: partially
        // filled ring, exact fill, wrap, oversize accumulation.
        runReference(window: 8, sinks: nil, chunks: [5, 6, 9], decodeSteps: 6)
    }

    @Test func fullAttentionMatchesReference() {
        runReference(window: nil, sinks: nil, chunks: [5, 6, 9], decodeSteps: 6)
    }

    @Test func sinksFoldIntoDenominatorOnly() {
        MLXRandom.seed(31)
        let sinks = (MLXRandom.normal([queryHeads]) + 1.0).asType(.float16)
        runReference(window: 8, sinks: sinks, chunks: [5, 6], decodeSteps: 4)
    }

    /// Gemma-4 style KV-shared layer: attendBorrowing must attend the source
    /// rows' CURRENT retained KV without updating it.
    @Test func borrowedAttentionMatchesSourceKV() {
        let window = 8
        let sourceKind = CBv2LayerKind(
            attention: .slidingWindow(window), headDim: headDim, kvHeads: kvHeads,
            queryHeads: queryHeads)
        let sharedKind = CBv2LayerKind(
            attention: .slidingWindow(window), sharesKVWithLayer: 0, headDim: headDim,
            kvHeads: kvHeads, queryHeads: queryHeads)

        MLXRandom.seed(77)
        let rows = (0 ..< 2).map { _ in
            CBv2WindowedSequenceKV(window: window, kvHeads: kvHeads, headDim: headDim)
        }
        let source = CBv2LayerCache(layerIndex: 0, kind: sourceKind, rows: rows)
        let shared = CBv2LayerCache(layerIndex: 5, kind: sharedKind)

        // Prefill each source row alone (per-request prefill), 12 tokens each
        // so the ring has wrapped.
        var histories: [(k: MLXArray, v: MLXArray)] = []
        for row in rows {
            let soloCache = CBv2LayerCache(layerIndex: 0, kind: sourceKind, rows: [row])
            let k = MLXRandom.normal([1, kvHeads, 12, headDim]).asType(.float16)
            let v = MLXRandom.normal([1, kvHeads, 12, headDim]).asType(.float16)
            let q = MLXRandom.normal([1, queryHeads, 12, headDim]).asType(.float16)
            _ = soloCache.updateAndAttend(queries: q, keys: k, values: v, scale: scale, sinks: nil)
            histories.append((k, v))
        }

        // Shared layer decode at B == 2 borrows the source KV.
        let q = MLXRandom.normal([2, queryHeads, 1, headDim]).asType(.float16)
        let out = shared.attendBorrowing(source: source, queries: q, scale: scale, sinks: nil)
        #expect(out.shape == [2, queryHeads, 1, headDim])

        for rowIndex in 0 ..< 2 {
            let expected = referenceAttention(
                queries: q[rowIndex ..< (rowIndex + 1)],
                keys: histories[rowIndex].k, values: histories[rowIndex].v,
                queryPositions: [11],  // borrowing does NOT advance positions
                scale: scale, window: window, sinks: nil)
            expectClose(
                out[rowIndex ..< (rowIndex + 1)].asType(.float32), expected,
                rtol: 1e-2, atol: 1e-2, "borrowed row \(rowIndex)")
        }
        // Borrowing never mutates source state.
        for row in rows {
            #expect(row.absoluteOffset == 12)
        }
    }
}

// MARK: - positionOffsets churn + step-loop hygiene

@Suite("CBv2Core: layer cache offsets & hygiene", .serialized)
struct CBv2CoreLayerCacheTests {

    private let kind = CBv2LayerKind(attention: .full, headDim: 4, kvHeads: 2, queryHeads: 4)

    private func prefilledRow(_ tokens: Int) -> CBv2SequenceKV {
        let row = CBv2FullSequenceKV(promptLength: tokens, maxLength: 128, kvHeads: 2, headDim: 4)
        _ = row.update(
            keys: positionCoded(from: 0, count: tokens),
            values: positionCoded(from: 0, count: tokens))
        return row
    }

    private func decodeStep(_ cache: CBv2LayerCache, batch: Int) -> MLXArray {
        let q = MLXRandom.normal([batch, 4, 1, 4]).asType(.float16)
        let k = MLXRandom.normal([batch, 2, 1, 4]).asType(.float16)
        let v = MLXRandom.normal([batch, 2, 1, 4]).asType(.float16)
        return cache.updateAndAttend(queries: q, keys: k, values: v, scale: 0.5, sinks: nil)
    }

    @Test func positionOffsetsTrackJoinLeaveChurn() {
        MLXRandom.seed(3)
        let rowA = prefilledRow(5)
        let rowB = prefilledRow(3)

        let cache = CBv2LayerCache(layerIndex: 0, kind: kind)
        cache.appendRow(rowA)
        cache.appendRow(rowB)
        #expect(cache.positionOffsets.asArray(Int32.self) == [5, 3])
        #expect(cache.positionOffsetsHostRebuilds == 2)
        #expect(cache.offset == 5)  // legacy scalar: max row offset

        // Decode advances offsets on-device — no host rebuild.
        _ = decodeStep(cache, batch: 2)
        #expect(cache.positionOffsets.asArray(Int32.self) == [6, 4])
        #expect(cache.positionOffsetsHostRebuilds == 2)

        // Join mid-flight.
        let rowC = prefilledRow(9)
        cache.appendRow(rowC)
        #expect(cache.positionOffsets.asArray(Int32.self) == [6, 4, 9])
        #expect(cache.positionOffsetsHostRebuilds == 3)

        // Leave.
        cache.removeRow(at: 0)
        #expect(cache.positionOffsets.asArray(Int32.self) == [4, 9])
        #expect(cache.positionOffsetsHostRebuilds == 4)

        _ = decodeStep(cache, batch: 2)
        #expect(cache.positionOffsets.asArray(Int32.self) == [5, 10])
        #expect(cache.rows.count == 2)
        #expect(cache.offset == 10)
    }

    /// A 10-step decode performs ZERO host syncs and ZERO host rebuilds of
    /// positionOffsets from CBv2 core code (report 10 §4 invariant 7 + the
    /// engine-thread discipline: graph-build only in the step loop).
    @Test func decodeLoopPerformsNoHostSyncsOrRebuilds() {
        MLXRandom.seed(5)
        let cache = CBv2LayerCache(layerIndex: 0, kind: kind)
        cache.appendRow(prefilledRow(4))
        cache.appendRow(prefilledRow(2))

        // Pre-generate step tensors OUTSIDE the measured loop (MLXRandom
        // sampling itself is graph work, but keep the loop pure anyway).
        let steps = (0 ..< 10).map { _ in
            (
                q: MLXRandom.normal([2, 4, 1, 4]).asType(.float16),
                k: MLXRandom.normal([2, 2, 1, 4]).asType(.float16),
                v: MLXRandom.normal([2, 2, 1, 4]).asType(.float16)
            )
        }

        let syncsBefore = CBv2CoreInstrumentation.hostSyncs
        let rebuildsBefore = cache.positionOffsetsHostRebuilds

        var outputs: [MLXArray] = []
        for step in steps {
            outputs.append(
                cache.updateAndAttend(
                    queries: step.q, keys: step.k, values: step.v, scale: 0.5, sinks: nil))
            // Engine-loop discipline: schedule without blocking; collapses
            // the lazy positionOffsets chain via innerState.
            asyncEval(outputs.last!, cache.innerState())
        }

        #expect(CBv2CoreInstrumentation.hostSyncs - syncsBefore == 0)
        #expect(cache.positionOffsetsHostRebuilds - rebuildsBefore == 0)

        // Materialize at the end (outside the step loop) and sanity-check.
        eval(outputs)
        #expect(cache.positionOffsets.asArray(Int32.self) == [14, 12])
    }

    @Test func innerStateExposesOffsetsAndRowBuffers() {
        let cache = CBv2LayerCache(layerIndex: 0, kind: kind)
        cache.appendRow(prefilledRow(3))
        cache.appendRow(prefilledRow(4))
        // 1 positionOffsets array + 2 arrays (K, V) per row.
        #expect(cache.innerState().count == 1 + 2 * 2)
        #expect(cache.maxSize == nil)
        #expect(cache.isTrimmable == false)
        #expect(cache.trim(5) == 0)
    }
}

// MARK: - Legacy attention multi-row guard (PR#62 review)

/// `attentionWithCacheUpdate`'s CBv2 compatibility branch is B == 1 ONLY: a
/// non-v2-adapted model applies scalar RoPE via `KVCache.offset` (the MAX row
/// offset) BEFORE this call, so shorter rows would be silently mis-rotated at
/// B > 1. The helper must fail loudly there while keeping B == 1 legal.
@Suite("CBv2Core: legacy attention multi-row guard")
struct CBv2LegacyAttentionGuardTests {

    @Test func singleRowIsAllowed() {
        #expect(cbv2LegacyAttentionBatchViolation(batch: 1, layerIndex: 0) == nil)
    }

    @Test func multiRowIsRejectedWithDiagnostic() {
        let violation = cbv2LegacyAttentionBatchViolation(batch: 3, layerIndex: 7)
        let message = try! #require(violation)
        #expect(message.contains("B=3"))
        #expect(message.contains("layer 7"))
        #expect(message.contains("updateAndAttend"))
    }

    /// B == 1 flows through the guarded compatibility path and drives
    /// `updateAndAttend` (no precondition trip, KV advanced).
    @Test func singleRowDrivesUpdateAndAttendThroughCompatPath() {
        let kind = CBv2LayerKind(attention: .full, headDim: 4, kvHeads: 1, queryHeads: 1)
        let row = CBv2FullSequenceKV(promptLength: 1, maxLength: 8, kvHeads: 1, headDim: 4)
        let cache = CBv2LayerCache(layerIndex: 0, kind: kind, rows: [row])
        let q = MLXRandom.normal([1, 1, 1, 4])
        let k = MLXRandom.normal([1, 1, 1, 4])
        let v = MLXRandom.normal([1, 1, 1, 4])
        let out = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: 0.5, mask: .causal)
        eval(out)
        #expect(out.shape == [1, 1, 1, 4])
        #expect(row.absoluteOffset == 1)
    }
}

// MARK: - Legacy attention custom-mask guard (PR#62 review)

/// `attentionWithCacheUpdate`'s CBv2 branch DISCARDS the `mask` parameter
/// (v2 derives its own causal/window masks), so a custom `.array` mask must
/// fail loudly in ALL build configurations — the previous debug-only
/// `assertionFailure` compiled out of release builds and let the wrong mask
/// ship silently. The helper carries the exact condition; the call site
/// trips `preconditionFailure` on a non-nil violation.
@Suite("CBv2Core: legacy attention custom-mask guard")
struct CBv2CustomMaskGuardTests {

    @Test func noneAndCausalMasksAreAllowed() {
        #expect(cbv2CustomMaskViolation(mask: .none, layerIndex: 0) == nil)
        #expect(cbv2CustomMaskViolation(mask: .causal, layerIndex: 0) == nil)
    }

    @Test func customArrayMaskIsRejectedWithDiagnostic() {
        let mask = MLXFast.ScaledDotProductAttentionMaskMode.array(
            MLXArray.zeros([1, 1, 1, 4]))
        let violation = cbv2CustomMaskViolation(mask: mask, layerIndex: 3)
        let message = try! #require(violation)
        #expect(message.contains("layer 3"))
        #expect(message.contains("custom array mask"))
        #expect(message.contains("updateAndAttend"))
    }
}
