// CBv2PagedKernelTests.swift
//
// WS-C kernel correctness: paged-attention decode kernel parity vs the
// composed fp32 reference (rel-err <= 1e-2 fp16), batch-composition
// invariance (bitwise), prefill fallback parity, KV-shared borrowing, and
// a 200-step greedy decode token-match. No model weights required.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

@Suite("CBv2PagedKernel", .serialized)
struct CBv2PagedKernelTests {

    // MARK: - Fixtures

    /// One attention layer + its paged plumbing plus an independent
    /// contiguous "mirror" of everything written, used to compute
    /// references without touching the pool.
    final class Fixture {
        let kind: CBv2LayerKind
        let backend: PagedKVBackend
        let cache: PagedLayerCache
        var states: [[CBv2SequenceKV?]] = []
        var rows: [PagedSequenceKV] = []
        var mirrorK: [MLXArray] = []
        var mirrorV: [MLXArray] = []

        init(
            kind: CBv2LayerKind, maxPrefillChunk: Int = 64,
            attentionSoftcap: Float? = nil
        ) throws {
            self.kind = kind
            self.backend = try PagedKVBackend(
                layerKinds: [kind],
                config: PagedKVPoolConfig(
                    capacityBytes: 64 << 20,
                    maxPrefillChunk: maxPrefillChunk,
                    nominalMaxSequenceLength: 4096))
            self.cache = backend.makeLayerCaches(attentionSoftcap: attentionSoftcap)[0]
        }

        deinit {
            states.forEach { backend.release($0) }
        }

        @discardableResult
        func addRow(tokens: Int, maxLength: Int = 2048) throws -> Int {
            let state = try backend.makeSequenceState(
                layerKinds: [kind], promptLength: tokens, maxLength: maxLength)
            let row = state[0] as! PagedSequenceKV
            states.append(state)
            rows.append(row)
            var k = MLXArray.zeros([kind.kvHeads, 0, kind.headDim], dtype: .float16)
            var v = k
            var remaining = tokens
            while remaining > 0 {
                let n = min(remaining, 64)
                let ck = MLXRandom.normal([kind.kvHeads, n, kind.headDim], dtype: .float16)
                let cv = MLXRandom.normal([kind.kvHeads, n, kind.headDim], dtype: .float16)
                row.write(keys: ck, values: cv)
                k = concatenated([k, ck], axis: 1)
                v = concatenated([v, cv], axis: 1)
                remaining -= n
            }
            mirrorK.append(k)
            mirrorV.append(v)
            cache.setRows(rows)
            return rows.count - 1
        }

        /// Reference decode for row `i` given this step's (q, k, v) —
        /// window clamping recomputed independently from the mirror.
        func referenceDecode(
            rowIndex i: Int, q: MLXArray, newK: MLXArray, newV: MLXArray,
            sinks: MLXArray?, scale: Float, softcap: Float? = nil
        ) -> MLXArray {
            mirrorK[i] = concatenated([mirrorK[i], newK], axis: 1)
            mirrorV[i] = concatenated([mirrorV[i], newV], axis: 1)
            let t = mirrorK[i].dim(1)
            var start = 0
            if case .slidingWindow(let w) = kind.attention {
                start = max(0, t - w)
            }
            let k = mirrorK[i][0..., start ..< t, 0...].expandedDimensions(axis: 0)
            let v = mirrorV[i][0..., start ..< t, 0...].expandedDimensions(axis: 0)
            return PagedAttentionReference.composedAttention(
                queries: q.expandedDimensions(axis: 0), keys: k, values: v,
                scale: scale, sinks: sinks, softcap: softcap)
        }
    }

    private func assertClose(
        _ got: MLXArray, _ want: MLXArray, rtol: Float = 1e-2, atol: Float = 2e-3,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(got.shape == want.shape, sourceLocation: sourceLocation)
        let ok = allClose(
            got.asType(.float32), want.asType(.float32), rtol: Double(rtol), atol: Double(atol)
        ).item(Bool.self)
        if !ok {
            let diff = abs(got.asType(.float32) - want.asType(.float32)).max().item(Float.self)
            Issue.record(
                "arrays differ, max abs err \(diff)", sourceLocation: sourceLocation)
        }
    }

    // MARK: - Kernel parity vs composed reference

    @Test(arguments: [
        (headDim: 64, kvHeads: 2, queryHeads: 8, window: Int?.none, sinks: false),
        (headDim: 64, kvHeads: 2, queryHeads: 8, window: Int?.none, sinks: true),
        (headDim: 64, kvHeads: 2, queryHeads: 16, window: Int?(20), sinks: true),
        (headDim: 128, kvHeads: 4, queryHeads: 8, window: Int?.none, sinks: false),
        (headDim: 128, kvHeads: 4, queryHeads: 8, window: Int?(33), sinks: false),
        (headDim: 256, kvHeads: 2, queryHeads: 4, window: Int?.none, sinks: false),
        (headDim: 512, kvHeads: 2, queryHeads: 4, window: Int?.none, sinks: false),
        // Gemma-4-26B global-layer shape (d512, GQA 8) — exercises the
        // head split (HPT=2, 4 threadgroups per kv head).
        (headDim: 512, kvHeads: 2, queryHeads: 16, window: Int?.none, sinks: false),
        (headDim: 512, kvHeads: 1, queryHeads: 8, window: Int?(24), sinks: false),
    ])
    func decodeKernelParity(
        _ shape: (headDim: Int, kvHeads: Int, queryHeads: Int, window: Int?, sinks: Bool)
    ) throws {
        MLXRandom.seed(42)
        let attention: CBv2LayerKind.Attention =
            shape.window.map { .slidingWindow($0) } ?? .full
        let kind = CBv2LayerKind(
            attention: attention, hasSinks: shape.sinks, headDim: shape.headDim,
            kvHeads: shape.kvHeads, queryHeads: shape.queryHeads)
        let fixture = try Fixture(kind: kind)

        // Mixed row lengths, including page-boundary and sub-page cases.
        for tokens in [4, 16, 33, 100] {
            try fixture.addRow(tokens: tokens)
        }
        let b = fixture.rows.count
        let scale = Float(1.0 / Double(shape.headDim).squareRoot())
        let sinks: MLXArray? =
            shape.sinks
            ? MLXRandom.normal([shape.queryHeads], dtype: .float32) : nil

        // A few decode steps so positions advance past page boundaries.
        for _ in 0 ..< 3 {
            let q = MLXRandom.normal(
                [b, shape.queryHeads, 1, shape.headDim], dtype: .float16)
            let k = MLXRandom.normal(
                [b, shape.kvHeads, 1, shape.headDim], dtype: .float16)
            let v = MLXRandom.normal(
                [b, shape.kvHeads, 1, shape.headDim], dtype: .float16)
            let out = fixture.cache.updateAndAttend(
                queries: q, keys: k, values: v, scale: scale, sinks: sinks)
            #expect(out.shape == [b, shape.queryHeads, 1, shape.headDim])
            for i in 0 ..< b {
                let ref = fixture.referenceDecode(
                    rowIndex: i, q: q[i], newK: k[i], newV: v[i],
                    sinks: sinks, scale: scale)
                assertClose(out[i].expandedDimensions(axis: 0), ref)
            }
        }
    }

    @Test func decodeKernelSoftcapParity() throws {
        MLXRandom.seed(7)
        let kind = CBv2LayerKind(
            attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
        let fixture = try Fixture(kind: kind, attentionSoftcap: 30.0)
        try fixture.addRow(tokens: 50)
        let scale: Float = 0.125

        let q = MLXRandom.normal([1, 4, 1, 64], dtype: .float16)
        let k = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
        let v = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
        let out = fixture.cache.updateAndAttend(
            queries: q, keys: k, values: v, scale: scale, sinks: nil)
        let ref = fixture.referenceDecode(
            rowIndex: 0, q: q[0], newK: k[0], newV: v[0],
            sinks: nil, scale: scale, softcap: 30.0)
        assertClose(out[0].expandedDimensions(axis: 0), ref)
    }

    // MARK: - Batch-composition invariance

    @Test func decodeBatchCompositionInvariance() throws {
        MLXRandom.seed(11)
        let kind = CBv2LayerKind(
            attention: .full, headDim: 64, kvHeads: 2, queryHeads: 8)
        let scale: Float = 0.125

        // Two fixtures: the probe row solo, and the SAME content batched
        // with two batchmates. Identical content => bitwise-identical
        // outputs, or the backend leaks batch composition. The 600-token
        // batchmate spans multiple flash-decoding partitions, so the
        // batched dispatch launches MORE partition threadgroups than the
        // solo one — the probe row's math must not notice.
        let solo = try Fixture(kind: kind)
        let batched = try Fixture(kind: kind)

        MLXRandom.seed(100)
        try solo.addRow(tokens: 37)
        MLXRandom.seed(100)
        try batched.addRow(tokens: 37)
        try batched.addRow(tokens: 600)
        try batched.addRow(tokens: 90)

        MLXRandom.seed(200)
        let q = MLXRandom.normal([1, 8, 1, 64], dtype: .float16)
        let k = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
        let v = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
        let qB = concatenated(
            [q, MLXRandom.normal([2, 8, 1, 64], dtype: .float16)], axis: 0)
        let kB = concatenated(
            [k, MLXRandom.normal([2, 2, 1, 64], dtype: .float16)], axis: 0)
        let vB = concatenated(
            [v, MLXRandom.normal([2, 2, 1, 64], dtype: .float16)], axis: 0)

        let outSolo = solo.cache.updateAndAttend(
            queries: q, keys: k, values: v, scale: scale, sinks: nil)
        let outBatched = batched.cache.updateAndAttend(
            queries: qB, keys: kB, values: vB, scale: scale, sinks: nil)

        #expect(
            arrayEqual(outSolo[0], outBatched[0]).item(Bool.self),
            "row output depends on batchmates — invariance violation")
    }

    // MARK: - Prefill fallback

    @Test(arguments: [Int?.none, Int?(24)])
    func prefillChunkParity(window: Int?) throws {
        MLXRandom.seed(3)
        let attention: CBv2LayerKind.Attention =
            window.map { .slidingWindow($0) } ?? .full
        let kind = CBv2LayerKind(
            attention: attention, headDim: 64, kvHeads: 2, queryHeads: 4)
        let fixture = try Fixture(kind: kind, maxPrefillChunk: 32)
        try fixture.addRow(tokens: 0)
        let row = fixture.rows[0]
        let scale: Float = 0.125

        var mirrorK = MLXArray.zeros([1, 2, 0, 64], dtype: .float16)
        var mirrorV = mirrorK
        for chunk in [9, 32, 16] {
            let q = MLXRandom.normal([1, 4, chunk, 64], dtype: .float16)
            let k = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)
            let v = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)
            let out = fixture.cache.updateAndAttend(
                queries: q, keys: k, values: v, scale: scale, sinks: nil)
            mirrorK = concatenated([mirrorK, k], axis: 2)
            mirrorV = concatenated([mirrorV, v], axis: 2)

            // Reference: full mirror + explicit causal/window bool mask.
            let t = mirrorK.dim(2)
            let qpos = MLXArray(Int32(t - chunk) ..< Int32(t)).expandedDimensions(axis: 1)
            let kpos = MLXArray(Int32(0) ..< Int32(t)).expandedDimensions(axis: 0)
            var mask = kpos .<= qpos
            if let window {
                mask = mask & (kpos .> (qpos - Int32(window)))
            }
            let ref = PagedAttentionReference.composedAttention(
                queries: q, keys: mirrorK, values: mirrorV, scale: scale,
                boolMask: mask, sinks: nil)
            assertClose(out, ref)
            #expect(row.absoluteOffset == t)
        }

        // A decode step right after chunked prefill must agree too
        // (prefill -> decode transition over the same ring).
        let q = MLXRandom.normal([1, 4, 1, 64], dtype: .float16)
        let k = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
        let v = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
        let out = fixture.cache.updateAndAttend(
            queries: q, keys: k, values: v, scale: scale, sinks: nil)
        mirrorK = concatenated([mirrorK, k], axis: 2)
        mirrorV = concatenated([mirrorV, v], axis: 2)
        let t = mirrorK.dim(2)
        var start = 0
        if let window { start = max(0, t - window) }
        let ref = PagedAttentionReference.composedAttention(
            queries: q,
            keys: mirrorK[0..., 0..., start ..< t, 0...],
            values: mirrorV[0..., 0..., start ..< t, 0...],
            scale: scale, sinks: nil)
        assertClose(out, ref)
    }

    @Test func prefillWithSinksParity() throws {
        MLXRandom.seed(13)
        let kind = CBv2LayerKind(
            attention: .full, hasSinks: true, headDim: 64, kvHeads: 2, queryHeads: 4)
        let fixture = try Fixture(kind: kind)
        try fixture.addRow(tokens: 0)
        let sinks = MLXRandom.normal([4], dtype: .float32)
        let scale: Float = 0.125

        let chunk = 12
        let q = MLXRandom.normal([1, 4, chunk, 64], dtype: .float16)
        let k = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)
        let v = MLXRandom.normal([1, 2, chunk, 64], dtype: .float16)
        let out = fixture.cache.updateAndAttend(
            queries: q, keys: k, values: v, scale: scale, sinks: sinks)

        let qpos = MLXArray(Int32(0) ..< Int32(chunk)).expandedDimensions(axis: 1)
        let kpos = MLXArray(Int32(0) ..< Int32(chunk)).expandedDimensions(axis: 0)
        let ref = PagedAttentionReference.composedAttention(
            queries: q, keys: k, values: v, scale: scale,
            boolMask: kpos .<= qpos, sinks: sinks)
        assertClose(out, ref)
    }

    // MARK: - KV-shared borrowing (Gemma-style)

    @Test func borrowingMatchesSourceStorage() throws {
        MLXRandom.seed(21)
        let owner = CBv2LayerKind(
            attention: .full, headDim: 64, kvHeads: 2, queryHeads: 8)
        var shared = owner
        shared.sharesKVWithLayer = 0
        let kinds = [owner, shared]
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: PagedKVPoolConfig(capacityBytes: 32 << 20))
        let caches = backend.makeLayerCaches()
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 256)
        let row = state[0] as! PagedSequenceKV
        caches[0].setRows([row])
        let scale: Float = 0.125

        var mirrorK = MLXArray.zeros([2, 0, 64], dtype: .float16)
        var mirrorV = mirrorK
        // Prefill through the owner, then decode; borrow at each decode.
        let pk = MLXRandom.normal([1, 2, 30, 64], dtype: .float16)
        let pv = MLXRandom.normal([1, 2, 30, 64], dtype: .float16)
        _ = caches[0].updateAndAttend(
            queries: MLXRandom.normal([1, 8, 30, 64], dtype: .float16),
            keys: pk, values: pv, scale: scale, sinks: nil)
        mirrorK = concatenated([mirrorK, pk.squeezed(axis: 0)], axis: 1)
        mirrorV = concatenated([mirrorV, pv.squeezed(axis: 0)], axis: 1)

        for _ in 0 ..< 2 {
            let k = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
            let v = MLXRandom.normal([1, 2, 1, 64], dtype: .float16)
            _ = caches[0].updateAndAttend(
                queries: MLXRandom.normal([1, 8, 1, 64], dtype: .float16),
                keys: k, values: v, scale: scale, sinks: nil)
            mirrorK = concatenated([mirrorK, k.squeezed(axis: 0)], axis: 1)
            mirrorV = concatenated([mirrorV, v.squeezed(axis: 0)], axis: 1)

            // The shared layer attends the owner's K/V with its own queries.
            let qShared = MLXRandom.normal([1, 8, 1, 64], dtype: .float16)
            let out = caches[1].attendBorrowing(
                source: caches[0], queries: qShared, scale: scale, sinks: nil)
            let ref = PagedAttentionReference.composedAttention(
                queries: qShared,
                keys: mirrorK.expandedDimensions(axis: 0),
                values: mirrorV.expandedDimensions(axis: 0),
                scale: scale, sinks: nil)
            assertClose(out, ref)
        }
        backend.release(state)
    }

    // MARK: - Position offsets

    @Test func positionOffsetsMatchAbsolutePositions() throws {
        let kind = CBv2LayerKind(
            attention: .full, headDim: 64, kvHeads: 2, queryHeads: 4)
        let fixture = try Fixture(kind: kind)
        try fixture.addRow(tokens: 12)
        try fixture.addRow(tokens: 40)
        let offsets = fixture.cache.positionOffsets
        #expect(offsets.shape == [2])
        #expect(offsets[0].item(Int32.self) == 12)
        #expect(offsets[1].item(Int32.self) == 40)
    }

    // MARK: - End-to-end greedy decode token match

    /// 200 greedy decode steps on a tiny random attention "model": the
    /// paged kernel path and the composed contiguous reference each feed
    /// their own argmax back. Correctness bar: >= 99.5% token match.
    @Test func greedyDecodeTokenMatch200Steps() throws {
        MLXRandom.seed(1234)
        let vocab = 97
        let headDim = 64
        let kvHeads = 2
        let queryHeads = 4

        let embedQ = MLXRandom.normal([vocab, queryHeads * headDim], dtype: .float16)
        let embedK = MLXRandom.normal([vocab, kvHeads * headDim], dtype: .float16)
        let embedV = MLXRandom.normal([vocab, kvHeads * headDim], dtype: .float16)
        let unembed = MLXRandom.normal([queryHeads * headDim, vocab], dtype: .float16)
        let scale = Float(1.0 / Double(headDim).squareRoot())

        let kind = CBv2LayerKind(
            attention: .full, headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads)
        let fixture = try Fixture(kind: kind)
        try fixture.addRow(tokens: 0, maxLength: 512)

        func project(_ token: Int) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
            let q = embedQ[token].reshaped([1, queryHeads, 1, headDim])
            let k = embedK[token].reshaped([1, kvHeads, 1, headDim])
            let v = embedV[token].reshaped([1, kvHeads, 1, headDim])
            return (q, k, v)
        }
        func logits(_ attnOut: MLXArray) -> MLXArray {
            matmul(attnOut.reshaped([1, queryHeads * headDim]).asType(.float32),
                   unembed.asType(.float32))
        }

        var pagedToken = 1
        var refToken = 1
        var mirrorK = MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16)
        var mirrorV = mirrorK
        var matches = 0
        let steps = 200

        for _ in 0 ..< steps {
            // Paged path.
            let p = project(pagedToken)
            let outPaged = fixture.cache.updateAndAttend(
                queries: p.q, keys: p.k, values: p.v, scale: scale, sinks: nil)
            let nextPaged = argMax(logits(outPaged), axis: -1).item(Int32.self)

            // Composed contiguous reference path.
            let r = project(refToken)
            mirrorK = concatenated([mirrorK, r.k], axis: 2)
            mirrorV = concatenated([mirrorV, r.v], axis: 2)
            let outRef = PagedAttentionReference.composedAttention(
                queries: r.q, keys: mirrorK, values: mirrorV, scale: scale, sinks: nil)
            let nextRef = argMax(logits(outRef), axis: -1).item(Int32.self)

            if nextPaged == nextRef { matches += 1 }
            pagedToken = Int(nextPaged)
            refToken = Int(nextRef)
        }
        #expect(
            Double(matches) >= 0.995 * Double(steps),
            "greedy token match \(matches)/\(steps) below 99.5%")
    }
}
