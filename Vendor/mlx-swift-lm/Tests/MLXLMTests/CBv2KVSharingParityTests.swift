// CBv2KVSharingParityTests.swift — regression: KV-BORROWING layers over a
// SLIDING-WINDOW source must stay exact under CHUNKED prefill.
//
// The bug class (Gemma-4 shape: most KV-shared layers are sliding-window):
// a windowed source layer's own `update()` returns PRE-eviction history +
// chunk (`window - 1 + n` entries) so the chunk's earliest queries see their
// full window, but the ring has already evicted those entries — so a
// borrowing layer that attends `snapshot()` (the POST-eviction ring,
// ≤ window entries) under-attends: with prompt > window, the first query of
// every chunk ≥ 2 saw 1 key instead of `window`. The fix gives borrowing
// layers the same step-scoped pre-eviction view the source attended
// (`CBv2WindowedSequenceKV.borrowableViews()`), mirroring the paged
// backend's design (exact by absolute-position arithmetic).
//
// The parity fixture: TinyTestModel's KV-sharing variant (layer 0 full,
// layer 1 sliding(16) source, layer 2 sliding(16) borrowing layer 1,
// layer 3 full DOWNSTREAM of the borrow so wrong borrowed attention
// corrupts persistent KV, as in Gemma-4's interleaving), prompt > window,
// CHUNKED prefill asserted token-exact against an UNCHUNKED B=1 reference
// — on BOTH backends.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class CBv2KVSharingParityTests: XCTestCase {

    // MARK: - Mechanism: the windowed pre-eviction borrow view

    func testWindowedBorrowableViewsMatchUpdateReturnAcrossChunks() {
        let window = 8
        let kv = CBv2WindowedSequenceKV(window: window, kvHeads: 1, headDim: 4)

        func chunk(_ n: Int, seed: Float) -> MLXArray {
            MLXArray(0 ..< n).asType(.float32).reshaped(1, 1, n, 1) * 0.1 + seed
        }
        func expand(_ x: MLXArray) -> MLXArray {
            broadcast(x, to: [1, 1, x.dim(2), 4])
        }

        // Chunk 1 (fresh): returned == chunk; borrowable == returned.
        let (k1, _) = kv.update(keys: expand(chunk(6, seed: 1)), values: expand(chunk(6, seed: 1)))
        XCTAssertEqual(k1.dim(2), 6)
        XCTAssertEqual(kv.borrowableViews().keys.dim(2), 6)

        // Chunk 2: returned = history(min(6, window-1)=6) + 6 = 12 entries,
        // while the post-eviction ring retains only `window` = 8. The
        // borrowable view must be the PRE-eviction 12, bit-identical to what
        // update() returned to the source layer's own attention.
        let (k2, v2) = kv.update(keys: expand(chunk(6, seed: 2)), values: expand(chunk(6, seed: 2)))
        XCTAssertEqual(k2.dim(2), 12)
        XCTAssertEqual(kv.snapshot().keys.dim(2), window, "ring evicts down to the window")
        let borrowed = kv.borrowableViews()
        XCTAssertEqual(borrowed.keys.dim(2), 12)
        XCTAssertTrue(allClose(borrowed.keys, k2).item(Bool.self))
        XCTAssertTrue(allClose(borrowed.values, v2).item(Bool.self))

        // Decode update: the retained ring IS the window — borrowable falls
        // back to the snapshot view.
        _ = kv.update(keys: expand(chunk(1, seed: 3)), values: expand(chunk(1, seed: 3)))
        XCTAssertEqual(kv.borrowableViews().keys.dim(2), window)
        XCTAssertEqual(kv.snapshot().keys.dim(2), window)

        // A stale chunk view must never survive a rollback.
        _ = kv.update(keys: expand(chunk(4, seed: 4)), values: expand(chunk(4, seed: 4)))
        XCTAssertEqual(kv.borrowableViews().keys.dim(2), window - 1 + 4)
        kv.rollback(2)
        XCTAssertEqual(kv.borrowableViews().keys.dim(2), kv.snapshot().keys.dim(2))
    }

    // MARK: - Parity fixture (chunked vs unchunked B=1, both backends)

    private enum BackendKind {
        case contiguous
        case paged
    }

    /// Bind one request's per-layer state into per-layer attending caches
    /// and run: chunked prefill over prompt[0 ..< n-1], then greedy decode
    /// starting from the last prompt token. Returns the greedy tokens.
    private func greedyRun(
        _ kind: BackendKind, model: TinyTestModel,
        prompt: [Int], decodeSteps: Int, chunkSize: Int
    ) throws -> [Int] {
        let layerKinds = model.layerKinds
        let maxLength = prompt.count + decodeSteps + 1

        let backend: CBv2KVBackend
        let caches: [CBv2AttendingLayerCache]
        switch kind {
        case .contiguous:
            backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 26))
            caches = layerKinds.enumerated().map { index, layerKind in
                CBv2LayerCache(layerIndex: index, kind: layerKind)
            }
        case .paged:
            let paged: PagedKVBackend
            do {
                paged = try PagedKVBackend(
                    layerKinds: layerKinds,
                    config: PagedKVPoolConfig(
                        capacityBytes: 32 << 20,
                        maxPrefillChunk: 64,
                        nominalMaxSequenceLength: 256))
            } catch let error as CBv2KVError {
                throw XCTSkip("paged backend unavailable on this hardware: \(error)")
            }
            backend = paged
            caches = paged.makeLayerCaches()
        }

        let kv = try backend.makeSequenceState(
            layerKinds: layerKinds, promptLength: prompt.count, maxLength: maxLength)
        defer { backend.release(kv) }
        for (i, layerKind) in layerKinds.enumerated() where layerKind.sharesKVWithLayer == nil {
            caches[i].setRows([kv[i]!])
        }

        // Chunked prefill of prompt[0 ..< n-1]; the last prompt token is the
        // first decode input (the harness/legacy step discipline).
        let prefill = Array(prompt.dropLast())
        var index = 0
        while index < prefill.count {
            let slice = Array(prefill[index ..< min(index + chunkSize, prefill.count)])
            let tokens = MLXArray(slice.map(Int32.init)).reshaped(1, slice.count)
            _ = model.forward(tokens: tokens, caches: caches)
            index += slice.count
        }

        var current = prompt.last!
        var generated: [Int] = []
        for _ in 0 ..< decodeSteps {
            let logits = model.forward(
                tokens: MLXArray([Int32(current)]).reshaped(1, 1), caches: caches)
            current = Int(argMax(logits[0..., -1, 0...], axis: -1).asArray(Int32.self)[0])
            generated.append(current)
        }
        return generated
    }

    private func runChunkedBorrowingParity(_ kind: BackendKind) throws {
        // headDim 64: eligible for the paged kernel; used for BOTH backends
        // so the two suites exercise identical model shapes.
        let model = TinyTestModel.make(seed: 0xC0FFEE, headDim: 64, withKVSharing: true)
        XCTAssertEqual(model.layerKinds[2].sharesKVWithLayer, 1)

        // Prompt (49) > window (16): chunks 2 and 3 of the 16-token prefill
        // start past the window, so the borrowing layer's early chunk
        // queries need the source's pre-eviction history.
        let prompt = makePromptTokens(length: 49, seed: 4242)
        let decodeSteps = 8

        let unchunked = try greedyRun(
            kind, model: model, prompt: prompt, decodeSteps: decodeSteps, chunkSize: 64)
        let chunked = try greedyRun(
            kind, model: model, prompt: prompt, decodeSteps: decodeSteps, chunkSize: 16)

        XCTAssertEqual(
            chunked, unchunked,
            "KV-borrowing over a windowed source must be invariant to prefill chunking")
    }

    func testChunkedBorrowingParity_Contiguous() throws {
        try runChunkedBorrowingParity(.contiguous)
    }

    func testChunkedBorrowingParity_Paged() throws {
        try runChunkedBorrowingParity(.paged)
    }
}
