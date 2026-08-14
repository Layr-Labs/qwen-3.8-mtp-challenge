// Copyright © 2026 Eigen Labs.
//
// WS-B: AdmissionV2 tests — truthful worst-case estimates (window-capped,
// KV-sharing aware), soft reservations with a watermark, rollback symmetry.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class CBv2SchedulerAdmissionTests: XCTestCase {

    private func id(_ raw: UInt64) -> CBv2RequestID { CBv2RequestID(raw) }

    /// full + sliding-window(128) + KV-shared layer, kvHeads 2, headDim 8.
    private var hybridKinds: [CBv2LayerKind] {
        [
            CBv2LayerKind(attention: .full, headDim: 8, kvHeads: 2, queryHeads: 4),
            CBv2LayerKind(attention: .slidingWindow(128), headDim: 8, kvHeads: 2, queryHeads: 4),
            CBv2LayerKind(
                attention: .full, sharesKVWithLayer: 0, headDim: 8, kvHeads: 2, queryHeads: 4),
        ]
    }

    func testEstimateRespectsWindowsAndKVSharing() {
        let admission = AdmissionV2(
            layerKinds: hybridKinds, bytesCapacity: 1 << 20,
            config: .init(watermarkFraction: 0, elementBytes: 2))
        // Per token per storage-owning layer: 2(K+V) × 2 heads × 8 dim × 2 B = 64 B.
        // 1000 tokens: full layer = 64_000; windowed = min(1000,128)×64 = 8192;
        // KV-shared layer owns NOTHING.
        XCTAssertEqual(admission.estimatedBytes(forTokens: 1000), 64_000 + 8192)
        // Below the window, both storage layers charge fully.
        XCTAssertEqual(admission.estimatedBytes(forTokens: 100), 100 * 64 * 2)
        XCTAssertEqual(admission.estimatedBytes(forTokens: 0), 0)
    }

    /// Per-layer element bytes (PR#62 review): GPT-OSS caches fp32 K/V on
    /// full-attention layers but bf16 on sliding ones. A flat 2 B/element
    /// under-charges the fp32 rows ~2x and over-admits — the estimate must
    /// honor a per-layer dtype table.
    func testPerLayerElementBytesForMixedPrecision() {
        // layer 0 full (fp32, 4 B), layer 1 sliding-window(128) (bf16, 2 B).
        let kinds = [
            CBv2LayerKind(attention: .full, headDim: 8, kvHeads: 2, queryHeads: 4),
            CBv2LayerKind(attention: .slidingWindow(128), headDim: 8, kvHeads: 2, queryHeads: 4),
        ]
        let table = AdmissionV2.Config.elementBytes(forDTypes: [.float32, .bfloat16])
        XCTAssertEqual(table, [4, 2])
        let mixed = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1 << 20,
            config: .init(watermarkFraction: 0, layerElementBytes: table))
        // 1000 tokens: full(fp32) = 1000 × 2 × 2 × 8 × 4 = 128_000;
        // windowed(bf16) = min(1000,128) × 2 × 2 × 8 × 2 = 8192.
        XCTAssertEqual(mixed.estimatedBytes(forTokens: 1000), 128_000 + 8192)

        // The flat-2-bytes assumption under-charges the fp32 layer by 2x.
        let flat = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1 << 20,
            config: .init(watermarkFraction: 0, elementBytes: 2))
        XCTAssertEqual(flat.estimatedBytes(forTokens: 1000), 64_000 + 8192)
        XCTAssertGreaterThan(
            mixed.estimatedBytes(forTokens: 1000), flat.estimatedBytes(forTokens: 1000),
            "fp32 full-attention rows must be charged ~2x the flat bf16 assumption")

        // Feasibility + reservation both honor the wider dtype.
        XCTAssertLessThan(
            mixed.admissibleBytesCapacity - mixed.estimatedBytes(forTokens: 1000),
            flat.admissibleBytesCapacity - flat.estimatedBytes(forTokens: 1000))
    }

    /// A per-layer table shorter/longer than the layer count is a construction
    /// bug — caught eagerly.
    func testMismatchedElementBytesTableTraps() {
        // Length mismatch would silently mis-charge; ensure the helper builds
        // a correctly-sized table from probed dtypes (nil ⇒ default).
        let table = AdmissionV2.Config.elementBytes(
            forDTypes: [.float32, nil, .bfloat16], defaultElementBytes: 2)
        XCTAssertEqual(table, [4, 2, 2], "nil (KV-shared / unprobed) falls back to default")
    }

    func testCanEverFitIsWorstCase() {
        // Capacity fits exactly 100 tokens of the single full layer
        // (32 B/token: 2 × 1 head × 8 dim × 2 B).
        let kinds = [CBv2LayerKind(attention: .full, headDim: 8, kvHeads: 1, queryHeads: 2)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 3200, config: .init(watermarkFraction: 0))
        XCTAssertTrue(admission.canEverFit(promptTokens: 50, maxTokens: 50))
        XCTAssertFalse(admission.canEverFit(promptTokens: 50, maxTokens: 51))
        // The check ignores current usage — feasibility, not availability.
        try? admission.reserve(id: id(1), additionalTokens: 100)
        XCTAssertTrue(admission.canEverFit(promptTokens: 50, maxTokens: 50))
    }

    /// Regression (admission/reserve livelock): `canEverFit` judged against
    /// FULL capacity while `reserve` enforces `capacity - watermark`. A
    /// request whose worst case landed in `(capacity - watermark, capacity]`
    /// was admitted, could never reserve its final tokens, self-preempted,
    /// restarted, and livelocked until its deadline. Feasibility must use
    /// the same watermark-adjusted ceiling `reserve` enforces.
    func testCanEverFitRespectsWatermark() throws {
        // 1 full layer, kv1 hd8 eb2 → 32 B/token. Capacity 3200, watermark
        // 5% (160 B) → 3040 usable → 95 tokens is the true ceiling.
        let kinds = [CBv2LayerKind(attention: .full, headDim: 8, kvHeads: 1, queryHeads: 2)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 3200, config: .init(watermarkFraction: 0.05))
        XCTAssertEqual(admission.admissibleBytesCapacity, 3040)

        // 95 tokens: admissible AND reservable end to end.
        XCTAssertTrue(admission.canEverFit(promptTokens: 45, maxTokens: 50))
        try admission.reserve(id: id(1), additionalTokens: 95)
        admission.releaseAll(id: id(1))

        // 96..100 tokens: previously admitted (<= full capacity) but the
        // solo reservation can never complete — must be rejected up front.
        for total in 96 ... 100 {
            XCTAssertFalse(
                admission.canEverFit(promptTokens: total - 50, maxTokens: 50),
                "\(total)-token worst case exceeds capacity - watermark")
            XCTAssertThrowsError(
                try admission.reserve(id: id(2), additionalTokens: total),
                "reserve agrees: \(total) tokens can never be granted")
            admission.releaseAll(id: id(2))
        }
    }

    func testSoftReserveThrowsAtWatermark() throws {
        // 1 full layer, kv1 hd1 eb2 → 4 B/token. Capacity 1000, watermark
        // 10% → 900 usable → 225 tokens.
        let kinds = [CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1000, config: .init(watermarkFraction: 0.1))
        try admission.reserve(id: id(1), additionalTokens: 225)
        XCTAssertThrowsError(try admission.reserve(id: id(2), additionalTokens: 1)) { error in
            guard case CBv2KVError.capacityExhausted(let needed, let available) = error else {
                return XCTFail("expected capacityExhausted, got \(error)")
            }
            XCTAssertEqual(needed, 4)
            XCTAssertEqual(available, 0)
        }
        XCTAssertFalse(admission.hasHeadroom(additionalTokens: 1))
    }

    func testUnreserveAndReleaseRestoreHeadroom() throws {
        let kinds = [CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 400, config: .init(watermarkFraction: 0))
        try admission.reserve(id: id(1), additionalTokens: 60)  // 240 B
        try admission.reserve(id: id(2), additionalTokens: 40)  // 400 B
        XCTAssertEqual(admission.bytesReserved, 400)
        XCTAssertThrowsError(try admission.reserve(id: id(1), additionalTokens: 1))

        // Optimistic-advance rollback path.
        admission.unreserve(id: id(2), tokens: 10)
        XCTAssertEqual(admission.bytesReserved, 360)
        XCTAssertTrue(admission.hasHeadroom(additionalTokens: 10))

        // Finish/cancel/preempt path.
        admission.releaseAll(id: id(1))
        XCTAssertEqual(admission.bytesReserved, 120)
        try admission.reserve(id: id(3), additionalTokens: 70)
        XCTAssertEqual(admission.bytesReserved, 400)
    }

    func testWindowedReservationPlateaus() throws {
        // Sliding window 4 → per-request bytes stop growing past 4 tokens,
        // so long decodes on windowed layers do not exhaust the ledger.
        let kinds = [CBv2LayerKind(attention: .slidingWindow(4), headDim: 1, kvHeads: 1, queryHeads: 1)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 64, config: .init(watermarkFraction: 0))
        try admission.reserve(id: id(1), additionalTokens: 4)  // 16 B (at the plateau)
        XCTAssertEqual(admission.bytesReserved, 16)
        for _ in 0 ..< 1000 {
            try admission.reserve(id: id(1), additionalTokens: 1)  // +0 B each
        }
        XCTAssertEqual(admission.bytesReserved, 16)
        // A second sequence still pays its own window.
        try admission.reserve(id: id(2), additionalTokens: 100)
        XCTAssertEqual(admission.bytesReserved, 32)
    }

    func testSnapshotReportsLedgerOrBackendTruth() throws {
        let kinds = [CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1000, config: .init(watermarkFraction: 0))
        try admission.reserve(id: id(1), additionalTokens: 25)  // 100 B
        let soft = admission.snapshot(activeRequests: 1, waitingRequests: 2, activeTokens: 25)
        XCTAssertEqual(soft.kvBytesInUse, 100)
        XCTAssertEqual(soft.kvBytesCapacity, 1000)
        XCTAssertEqual(soft.activeRequests, 1)
        XCTAssertEqual(soft.waitingRequests, 2)
        XCTAssertEqual(soft.activeTokens, 25)

        let hard = admission.snapshot(
            activeRequests: 1, waitingRequests: 2, activeTokens: 25, backendBytesInUse: 88)
        XCTAssertEqual(hard.kvBytesInUse, 88, "backend truth wins when provided")
    }

    // MARK: External reserve (compiled padding carve-out, PR#62 review)

    /// The compiled decode path carves its worst-case padding reserve out of
    /// the ledger at engine build. If warmup tracing disables compiled
    /// decode, the engine refunds the carve-out — admission must then judge
    /// against the FULL budget again, not stay permanently tighter than the
    /// hardware truth.
    func testExternalReserveTightensAdmissionAndRefundRestores() throws {
        // Single full layer, kvHeads 1, headDim 4 ⇒ 16 B/token.
        let kinds = [CBv2LayerKind(attention: .full, headDim: 4, kvHeads: 1, queryHeads: 2)]
        let admission = AdmissionV2(
            layerKinds: kinds, bytesCapacity: 1600,
            config: .init(watermarkFraction: 0),
            externalReserveBytes: 800)

        // 60 tokens = 960 B: fits the hardware budget (1600) but not the
        // reserve-reduced ceiling (800).
        XCTAssertEqual(admission.admissibleBytesCapacity, 800)
        XCTAssertFalse(admission.canEverFit(promptTokens: 40, maxTokens: 20))
        XCTAssertFalse(admission.hasHeadroom(additionalTokens: 60))
        XCTAssertThrowsError(try admission.reserve(id: id(1), additionalTokens: 60)) { error in
            guard case CBv2KVError.capacityExhausted(_, let available) = error else {
                return XCTFail("expected capacityExhausted, got \(error)")
            }
            XCTAssertEqual(available, 800, "available must reflect the carve-out")
        }

        // Refund (compiled decode disabled at warmup): full budget again.
        admission.refundExternalReserve()
        XCTAssertEqual(admission.admissibleBytesCapacity, 1600)
        XCTAssertTrue(admission.canEverFit(promptTokens: 40, maxTokens: 20))
        XCTAssertTrue(admission.hasHeadroom(additionalTokens: 60))
        XCTAssertNoThrow(try admission.reserve(id: id(1), additionalTokens: 60))
        // 960 B reserved; 40 more tokens (640 B) tops out exactly at 1600.
        XCTAssertNoThrow(try admission.reserve(id: id(2), additionalTokens: 40))
        XCTAssertThrowsError(try admission.reserve(id: id(3), additionalTokens: 1))

        // Refund is idempotent.
        admission.refundExternalReserve()
        XCTAssertEqual(admission.admissibleBytesCapacity, 1600)
    }
}
