// Tests for Scheduler's per-layer KV-cache factory selection under
// `KVQuantizationConfig`. These exercise the pure type-selection logic via
// `Scheduler.cacheFactoryTypeName(for:quantConfig:)`, which only inspects the
// concrete cache class — so no Metal device / metallib is required.
//
// Regression intent: KV quantization must replace ONLY plain full-attention
// `KVCacheSimple` layers. `ChunkedKVCache` (a `KVCacheSimple` subclass) and
// non-`KVCacheSimple` caches must stay on their fp16 batched paths so models
// that downcast their per-layer cache back to a concrete type are not broken.

import Foundation
import XCTest

@testable import MLXLMCommon

final class CBCacheFactorySelectionTests: XCTestCase {

    private let kernel = KVQuantizationConfig(
        groupSize: 128, bits: 8, mode: .affine, cacheKind: .kernel)
    private let dequant = KVQuantizationConfig(
        groupSize: 64, bits: 8, mode: .affine, cacheKind: .dequant)

    func testQuantOffKeepsFp16BatchedCaches() {
        XCTAssertEqual(
            Scheduler.cacheFactoryTypeName(for: KVCacheSimple(), quantConfig: nil),
            "BatchKVCache")
        XCTAssertEqual(
            Scheduler.cacheFactoryTypeName(
                for: RotatingKVCache(maxSize: 1024, keep: 0, step: 1024),
                quantConfig: nil),
            "BatchRotatingKVCache")
    }

    func testKVCacheSimpleIsQuantizedByKind() {
        XCTAssertEqual(
            Scheduler.cacheFactoryTypeName(for: KVCacheSimple(), quantConfig: kernel),
            "QuantizedBatchKVCache")
        XCTAssertEqual(
            Scheduler.cacheFactoryTypeName(for: KVCacheSimple(), quantConfig: dequant),
            "DequantBatchKVCache")
    }

    func testSlidingLayersAreQuantizedRotatingWhenQuantOn() {
        // Sliding-window layers are quantized too (rotating variants) so the
        // capacity win covers the whole cache, not just full-attention layers.
        XCTAssertEqual(
            Scheduler.cacheFactoryTypeName(
                for: RotatingKVCache(maxSize: 1024, keep: 0, step: 1024),
                quantConfig: kernel),
            "QuantizedBatchRotatingKVCache")
        XCTAssertEqual(
            Scheduler.cacheFactoryTypeName(
                for: RotatingKVCache(maxSize: 1024, keep: 0, step: 1024),
                quantConfig: dequant),
            "DequantBatchRotatingKVCache")
    }

    func testChunkedKVCacheIsNotQuantized() {
        // ChunkedKVCache subclasses KVCacheSimple but has bespoke chunked
        // semantics; quantizing it would drop that behavior. It must stay fp16.
        XCTAssertEqual(
            Scheduler.cacheFactoryTypeName(
                for: ChunkedKVCache(chunkSize: 256), quantConfig: kernel),
            "BatchKVCache")
    }

    func testRecurrentAndArrayCachesAreNotQuantized() {
        XCTAssertEqual(
            Scheduler.cacheFactoryTypeName(for: ArraysCache(size: 4), quantConfig: kernel),
            "ArraysCache")
        XCTAssertEqual(
            Scheduler.cacheFactoryTypeName(
                for: MambaCache(), quantConfig: kernel),
            "MambaCache")
    }
}
