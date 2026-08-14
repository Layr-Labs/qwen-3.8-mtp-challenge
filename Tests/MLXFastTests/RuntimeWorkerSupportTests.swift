import Dispatch
import Foundation
import MLX
import MLXFastCore
import MLXLMCommon
@testable import MLXFastRuntimeWorkerSupport
import Testing

// MARK: - Hybrid cache-position validation

/// Synthetic stand-in for `Qwen35TextModel.newCache`: one cache per layer,
/// `KVCacheSimple` on the 16 full-attention layers (index % 4 == 3) and
/// `MambaCache` on the 48 gated-delta layers. Constructing these allocates no
/// MLXArray and loads no model, so the hybrid contract is testable on any
/// machine.
private func syntheticQwenHybridCaches(
    attentionOffset: Int,
    recurrentOffset: Int = 0,
    layerCount: Int = MLXFastConstants.numHiddenLayers,
    homogeneous: Bool = false
) -> [BaseKVCache] {
    let interval = MLXFastConstants.fullAttentionInterval
    return (0..<layerCount).map { index in
        let isFullAttention = homogeneous || index % interval == interval - 1
        let cache: BaseKVCache = isFullAttention ? KVCacheSimple() : MambaCache()
        cache.offset = isFullAttention ? attentionOffset : recurrentOffset
        return cache
    }
}

/// REGRESSION (native box-3 `generate-golden`, branch tip 13ff24b):
/// `verifyQwenCachePosition` was renamed from `verifyLagunaCachePosition`
/// without changing its rule, so it still demanded that ALL 64 layer caches
/// report the same offset. Laguna's stack is homogeneous KV; Qwen 3.6 is
/// hybrid, and the gated-delta caches never advance past 0. The first prefill
/// passed (everything at 0) and the very next forward died with
/// "Qwen KV cache layer offsets are inconsistent".
@Test
func qwenHybridCachePositionAcceptsRecurrentCachesPinnedAtZero() throws {
    // Fresh cache before the first forward.
    try QwenRuntime.verifyQwenCachePosition(
        positionOffset: 0,
        cache: syntheticQwenHybridCaches(attentionOffset: 0)
    )
    // After a 512-token prefill: 16 KV caches at 512, 48 recurrent caches at 0.
    // This is the exact shape that used to throw.
    try QwenRuntime.verifyQwenCachePosition(
        positionOffset: 512,
        cache: syntheticQwenHybridCaches(attentionOffset: 512)
    )
    // And after each subsequent teacher-forced decode step.
    for step in 1...4 {
        try QwenRuntime.verifyQwenCachePosition(
            positionOffset: 512 + step,
            cache: syntheticQwenHybridCaches(attentionOffset: 512 + step)
        )
    }
}

@Test
func qwenHybridCachePositionRejectsDivergentFullAttentionOffsets() {
    var caches = syntheticQwenHybridCaches(attentionOffset: 512)
    // Layer 7 is full attention; make it lag the rest of the KV stack.
    caches[7].offset = 511
    #expect(throws: MLXFastError.self) {
        try QwenRuntime.verifyQwenCachePosition(positionOffset: 512, cache: caches)
    }
}

@Test
func qwenHybridCachePositionRejectsRecurrentCachesThatCountPositions() {
    // A gated-delta cache that advanced means the stack is not the pinned
    // hybrid tower; the offset it reports is not a position this check can
    // reconcile with the caller's.
    #expect(throws: MLXFastError.self) {
        try QwenRuntime.verifyQwenCachePosition(
            positionOffset: 512,
            cache: syntheticQwenHybridCaches(attentionOffset: 512, recurrentOffset: 512)
        )
    }
}

@Test
func qwenHybridCachePositionRejectsWrongTopologyOrLayerCount() {
    // A homogeneous all-KV stack is lockstep-consistent and would have
    // satisfied the old rule, but it is not this tower.
    #expect(throws: MLXFastError.self) {
        try QwenRuntime.verifyQwenCachePosition(
            positionOffset: 512,
            cache: syntheticQwenHybridCaches(attentionOffset: 512, homogeneous: true)
        )
    }
    // Wrong layer count, right topology.
    #expect(throws: MLXFastError.self) {
        try QwenRuntime.verifyQwenCachePosition(
            positionOffset: 512,
            cache: syntheticQwenHybridCaches(attentionOffset: 512, layerCount: 40)
        )
    }
    #expect(throws: MLXFastError.self) {
        try QwenRuntime.verifyQwenCachePosition(positionOffset: 0, cache: [])
    }
}

@Test
func qwenHybridCachePositionStillRejectsAStaleOrReusedCache() {
    // The fail-loudly contract the Laguna check existed for is preserved: the
    // caller's position must equal what the KV stack actually holds.
    #expect(throws: MLXFastError.self) {
        try QwenRuntime.verifyQwenCachePosition(
            positionOffset: 513,
            cache: syntheticQwenHybridCaches(attentionOffset: 512)
        )
    }
    #expect(throws: MLXFastError.self) {
        try QwenRuntime.verifyQwenCachePosition(
            positionOffset: -1,
            cache: syntheticQwenHybridCaches(attentionOffset: 0)
        )
    }
}

/// The gate and the cache check must read the same schedule; a tower whose
/// interval changed would otherwise pass one and fail the other.
@Test
func qwenFullAttentionIntervalMatchesThePinnedLayerSchedule() {
    #expect(MLXFastConstants.fullAttentionInterval == 4)
    #expect(MLXFastConstants.numHiddenLayers % MLXFastConstants.fullAttentionInterval == 0)
    let fullAttentionLayers = (0..<MLXFastConstants.numHiddenLayers).filter {
        $0 % MLXFastConstants.fullAttentionInterval
            == MLXFastConstants.fullAttentionInterval - 1
    }
    #expect(fullAttentionLayers.count == 16)
    #expect(fullAttentionLayers.first == 3)
    #expect(fullAttentionLayers.last == 63)
}

@Test
func lagunaCorrectnessSelectsGreedyTokenWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }

    #expect(try QwenCorrectness.greedyToken(
        from: MLXArray([Float(0.1), 2.0, 1.0], [3])
    ) == 1)
    #expect(try QwenCorrectness.greedyToken(
        from: MLXArray([Float(1), 2, 3, 2], [2, 2])
    ) == 0)
}

// The worker's orphan self-reaper is what frees the ~21.6 GB model residency
// when the harness parent dies while the worker is not blocked on protocol
// stdin (most importantly during the minutes-long model load).
@Test
func runtimeWorkerOrphanReaperFiresOnceTheParentIsGone() {
    let fired = DispatchSemaphore(value: 0)
    let thread = QwenRuntime.startRuntimeWorkerOrphanReaper(
        pollIntervalSeconds: 0.01,
        isOrphaned: { true },
        onOrphaned: { fired.signal() }
    )
    defer { thread.cancel() }
    #expect(fired.wait(timeout: .now() + 2) == .success)
}

@Test
func runtimeWorkerOrphanReaperStaysQuietWhileTheParentIsAlive() {
    let fired = DispatchSemaphore(value: 0)
    let thread = QwenRuntime.startRuntimeWorkerOrphanReaper(
        pollIntervalSeconds: 0.01,
        isOrphaned: { false },
        onOrphaned: { fired.signal() }
    )
    defer { thread.cancel() }
    #expect(fired.wait(timeout: .now() + 0.3) == .timedOut)
}

@Test
func phaseStartAllocatorResetLeavesExactlyEmptyCacheWhenRuntimeTestsAreEnabled() throws {
    guard ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1" else {
        return
    }
    Memory.cacheLimit = 32 << 30
    do {
        let scratch = MLXArray(Array(repeating: Float(1), count: 1 << 20), [1024, 1024])
        eval(scratch + scratch)
    }
    try QwenRuntime.resetRuntimeWorkerAllocatorForPhaseStart()
    #expect(Memory.cacheMemory == 0)
    #expect(Memory.cacheLimit == QwenRuntime.trustedRuntimeWorkerPhaseStartCacheLimitBytes)
}

@Test
func traceProtocolCarriesRequestedTopKAndExpectedTokenDiagnostics() throws {
    let request = RuntimeWorkerRequest(
        id: 4,
        kind: "correctness_step",
        token: 7,
        topK: 16,
        expectedToken: 23
    )
    let requestObject = try #require(
        JSONSerialization.jsonObject(
            with: JSONEncoder().encode(request)
        ) as? [String: Any]
    )
    #expect(requestObject["top_k"] as? Int == 16)
    #expect(requestObject["expected_token"] as? Int == 23)

    let response = RuntimeWorkerResponse(
        id: 4,
        nonce: "nonce",
        ok: true,
        token: 1,
        topLogits: [
            CorrectnessTraceLogit(token: 1, logit: 9),
        ],
        expectedTokenLogit: 3.5,
        expectedTokenRank: 19,
        topLogitMargin: 0.25
    )
    let decoded = try JSONDecoder().decode(
        RuntimeWorkerResponse.self,
        from: JSONEncoder().encode(response)
    )
    #expect(decoded.expectedTokenLogit == 3.5)
    #expect(decoded.expectedTokenRank == 19)
    #expect(decoded.topLogitMargin == 0.25)
}

@Test
func workerComputesExpectedTokenDiagnosticsOutsideReturnedTopSubset() throws {
    let diagnostics = try QwenRuntime.correctnessLogitDiagnostics(
        values: (0..<20).map { Double(20 - $0) },
        topK: 12,
        expectedToken: 19
    )
    #expect(diagnostics.topLogits.count == 12)
    #expect(!diagnostics.topLogits.contains { $0.token == 19 })
    #expect(diagnostics.expectedTokenLogit == 1)
    #expect(diagnostics.expectedTokenRank == 20)
    #expect(diagnostics.topLogitMargin == 1)
}
