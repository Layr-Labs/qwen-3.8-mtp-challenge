// CBv2CompiledDecodeTests.swift
//
// Compiled [B, 1] decode — unit + engine-level parity coverage.
//
// The safety bar is TOKEN-EXACT parity: an EngineV2 with compiled decode
// enabled must produce the same greedy tokens as the same engine with the
// compiled path disabled, across the full invariance matrix (solo, batched,
// padded buckets, mid-join churn, window crossing, sinks, KV-sharing,
// seeded sampling). Unit tests additionally pin the compiled layer cache's
// overflow-bin attention against the eager `CBv2LayerCache` numerics and
// the fallback/self-heal machinery.
//
// No model downloads; TinyTestModel weights are seeded random.

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLMCommon

final class CBv2CompiledDecodeTests: XCTestCase {

    override func setUp() {
        super.setUp()
        MLXRandom.seed(0xDEC0DE)
    }

    // MARK: - Helpers

    private func makeEngine(
        model: TinyTestModel,
        compiled: Bool,
        kvCapacity: Int = 4096,
        buckets: [Int] = [1, 2, 4],
        maxConcurrent: Int = 4,
        prefillChunkSize: Int = 16
    ) -> EngineV2 {
        EngineV2(
            model: model,
            layerKinds: model.layerKinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: model.layerKinds),
            sampler: CBv2DefaultSampler(fallbackSeed: 7),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: maxConcurrent, maxBatchedTokensPerStep: 256,
                prefillChunkSize: prefillChunkSize, maxWaiting: 16),
            compiledDecodeConfig: CBv2CompiledDecodeConfig(
                enabled: compiled, buckets: buckets, kvCapacity: kvCapacity))
    }

    private func request(
        id: UInt64, prompt: [Int], maxTokens: Int,
        sampling: CBv2SamplingParams = CBv2SamplingParams(temperature: 0)
    ) -> CBv2Request {
        CBv2Request(id: CBv2RequestID(id), promptTokens: prompt, sampling: sampling,
            maxTokens: maxTokens)
    }

    /// Run the same request set on a compiled and an eager engine (same
    /// weights) and assert token-exact equality per request.
    private func assertCompiledMatchesEager(
        model: TinyTestModel,
        prompts: [[Int]], maxTokens: [Int],
        kvCapacity: Int = 4096,
        buckets: [Int] = [1, 2, 4],
        sampling: CBv2SamplingParams = CBv2SamplingParams(temperature: 0),
        expectCompiledSteps: Bool = true,
        label: String = #function
    ) async throws {
        var results: [Bool: [CBv2SchedCollected]] = [:]
        var stats: CBv2CompiledDecodeStats?
        for compiled in [false, true] {
            let engine = makeEngine(
                model: model, compiled: compiled, kvCapacity: kvCapacity, buckets: buckets)
            var streams: [AsyncStream<CBv2Event>] = []
            for (i, prompt) in prompts.enumerated() {
                streams.append(
                    try engine.submit(
                        request(
                            id: UInt64(i + 1), prompt: prompt, maxTokens: maxTokens[i],
                            sampling: sampling)))
            }
            var collected: [CBv2SchedCollected] = []
            for stream in streams { collected.append(await cbv2SchedCollect(stream)) }
            await engine.shutdown()
            results[compiled] = collected
            if compiled { stats = engine.compiledDecodeStats }
        }
        for i in prompts.indices {
            XCTAssertEqual(
                results[true]![i].tokens, results[false]![i].tokens,
                "\(label): request \(i) diverged between compiled and eager decode")
            XCTAssertEqual(results[true]![i].finishReason, results[false]![i].finishReason)
        }
        let compiledStats = try XCTUnwrap(stats, "\(label): compiled engine must build the path")
        XCTAssertNil(compiledStats.disabledReason, "\(label): compiled decode disabled itself")
        if expectCompiledSteps {
            XCTAssertGreaterThan(
                compiledStats.compiledSteps, 0,
                "\(label): no step took the compiled path — fallbacks: \(compiledStats.fallbacks)")
        }
    }

    // MARK: - Engine-level parity (the invariance matrix, both modes)

    func testSoloParity() async throws {
        try await assertCompiledMatchesEager(
            model: TinyTestModel.make(seed: 0xC0FFEE),
            prompts: [makePromptTokens(length: 9, seed: 1)], maxTokens: [24])
    }

    func testBatchedPairParity() async throws {
        try await assertCompiledMatchesEager(
            model: TinyTestModel.make(seed: 0xC0FFEE),
            prompts: [
                makePromptTokens(length: 9, seed: 1),
                makePromptTokens(length: 21, seed: 2),
            ],
            maxTokens: [24, 16])
    }

    /// B=3 uses bucket 4 — one inert scratch lane pads the batch.
    func testPaddedBucketParity() async throws {
        try await assertCompiledMatchesEager(
            model: TinyTestModel.make(seed: 0xC0FFEE),
            prompts: [
                makePromptTokens(length: 9, seed: 1),
                makePromptTokens(length: 21, seed: 2),
                makePromptTokens(length: 33, seed: 3),
            ],
            maxTokens: [24, 20, 16])
    }

    func testFullBucketParity() async throws {
        try await assertCompiledMatchesEager(
            model: TinyTestModel.make(seed: 0xC0FFEE),
            prompts: (0 ..< 4).map { makePromptTokens(length: 7 + 9 * $0, seed: UInt64($0 + 1)) },
            maxTokens: [24, 20, 16, 12])
    }

    /// Generations long enough to cross the sliding window (16) — the
    /// modular ring masks must keep the RECENT end after wrap.
    func testWindowCrossingParity() async throws {
        try await assertCompiledMatchesEager(
            model: TinyTestModel.make(seed: 0xC0FFEE),
            prompts: [
                makePromptTokens(length: 5, seed: 4),
                makePromptTokens(length: 11, seed: 5),
            ],
            maxTokens: [48, 40])
    }

    func testSinksParity() async throws {
        try await assertCompiledMatchesEager(
            model: TinyTestModel.make(seed: 0xC0FFEE, withSinks: true),
            prompts: [
                makePromptTokens(length: 9, seed: 6),
                makePromptTokens(length: 25, seed: 7),
                makePromptTokens(length: 14, seed: 8),
            ],
            maxTokens: [24, 20, 24])
    }

    /// Gemma-4-style cross-layer KV sharing: the borrowing layer attends
    /// the source layer's post-update compiled state with the source's
    /// pre-update RoPE offsets.
    func testKVSharingParity() async throws {
        try await assertCompiledMatchesEager(
            model: TinyTestModel.make(seed: 0xC0FFEE, withKVSharing: true),
            prompts: [
                makePromptTokens(length: 9, seed: 9),
                makePromptTokens(length: 26, seed: 10),
            ],
            maxTokens: [28, 24])
    }

    /// Seeded stochastic sampling: the sampler consumes compiled logits
    /// outside the graph; keyed RNG must make both modes identical.
    func testSeededSamplingParity() async throws {
        try await assertCompiledMatchesEager(
            model: TinyTestModel.make(seed: 0xC0FFEE),
            prompts: [
                makePromptTokens(length: 9, seed: 11),
                makePromptTokens(length: 17, seed: 12),
            ],
            maxTokens: [20, 20],
            sampling: CBv2SamplingParams(temperature: 0.8, topP: 0.9, seed: 1234))
    }

    /// Requests join and finish at different times: bucket transitions
    /// 1 → 2 → 3(padded) and back, chain breaks, re-established compiled
    /// mode after every membership change.
    func testMidStreamJoinChurnParity() async throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let prompts = [
            makePromptTokens(length: 9, seed: 21),
            makePromptTokens(length: 15, seed: 22),
            makePromptTokens(length: 21, seed: 23),
        ]
        let maxTokens = [40, 12, 20]

        var perMode: [Bool: [[Int]]] = [:]
        var stats: CBv2CompiledDecodeStats?
        for compiled in [false, true] {
            let engine = makeEngine(model: model, compiled: compiled)
            // Stagger joins: submit 0, wait for some progress, submit 1,
            // then 2. Request 1 finishes early (small budget).
            let s0 = try engine.submit(request(id: 1, prompt: prompts[0], maxTokens: maxTokens[0]))
            _ = await cbv2SchedWait { engine.capacity().stepsExecuted > 4 }
            let s1 = try engine.submit(request(id: 2, prompt: prompts[1], maxTokens: maxTokens[1]))
            _ = await cbv2SchedWait { engine.capacity().stepsExecuted > 10 }
            let s2 = try engine.submit(request(id: 3, prompt: prompts[2], maxTokens: maxTokens[2]))
            let c0 = await cbv2SchedCollect(s0)
            let c1 = await cbv2SchedCollect(s1)
            let c2 = await cbv2SchedCollect(s2)
            await engine.shutdown()
            perMode[compiled] = [c0.tokens, c1.tokens, c2.tokens]
            if compiled { stats = engine.compiledDecodeStats }
        }
        // Join timing is wall-clock dependent, so cross-mode comparison of
        // BATCHED runs is not deterministic here — instead pin against solo
        // eager baselines (batch-composition invariance must hold in both
        // modes regardless of join timing).
        var solos: [[Int]] = []
        for (i, prompt) in prompts.enumerated() {
            let engine = makeEngine(model: model, compiled: false)
            let collected = await cbv2SchedCollect(
                try engine.submit(request(id: 100, prompt: prompt, maxTokens: maxTokens[i])))
            await engine.shutdown()
            solos.append(collected.tokens)
        }
        for mode in [false, true] {
            for i in prompts.indices {
                XCTAssertEqual(
                    perMode[mode]![i], solos[i],
                    "mode compiled=\(mode): request \(i) diverged from solo baseline under churn")
            }
        }
        let compiledStats = try XCTUnwrap(stats)
        XCTAssertGreaterThan(compiledStats.compiledSteps, 0)
    }

    // MARK: - Fallback + self-healing

    /// Rows past `kvCapacity` must fall back to eager MID-REQUEST (compiled
    /// steps first, eager once the row outgrows the bucket capacity) with
    /// token-exact output — exercises the compiled→eager transition, the
    /// stale-positionOffsets invalidation of the eager bank, and the eager
    /// buffer growth past the compiled padding.
    ///
    /// Geometry: the contiguous backend pre-allocates
    /// `min(maxLength, prompt + 256)` slots, and a row only binds while its
    /// buffer fits `kvCapacity` — so the crossing needs
    /// `prompt + 256 < kvCapacity < prompt + maxTokens`.
    func testCapacityOverflowFallsBackMidRequestExactly() async throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let prompt = makePromptTokens(length: 9, seed: 31)
        try await assertCompiledMatchesEager(
            model: model, prompts: [prompt], maxTokens: [300], kvCapacity: 270,
            label: "capacity overflow")
    }

    /// A request whose contiguous buffer is ALREADY larger than
    /// `kvCapacity` (long maxTokens ⇒ up-front allocation past the bucket
    /// shape) must run fully eager — bind refuses to shrink storage — with
    /// unchanged results.
    func testOversizedAllocationStaysEagerExactly() async throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let prompt = makePromptTokens(length: 9, seed: 36)
        try await assertCompiledMatchesEager(
            model: model, prompts: [prompt], maxTokens: [40], kvCapacity: 24,
            expectCompiledSteps: false,
            label: "oversized allocation")
    }

    /// Quantized-KV rows can never take the compiled path, so the engine
    /// must VETO the compiled build up front (PR#62 review): previously it
    /// built + warmed against fp16 scratch, charged the admission padding
    /// reserve, and then `laneInfo` rejected every LIVE quantized row —
    /// permanent eager fallback with a permanently tighter admission
    /// ceiling for zero benefit. Asserts: no executor, no reserve (ceiling
    /// identical to a no-compiled engine), eager decode still completes.
    /// (Token parity with the unquantized run is NOT expected —
    /// quantization changes numerics — so only behavior is asserted here.)
    func testQuantizedBackendVetoesCompiledBuildAndReserve() async throws {
        // headDim 64: the quantized cache requires headDim divisible by a
        // supported group size.
        let model = TinyTestModel.make(seed: 0xC0FFEE, headDim: 64, fullAttentionOnly: true)
        func makeEngine(quantized: Bool, compiled: Bool) -> EngineV2 {
            EngineV2(
                model: model,
                layerKinds: model.layerKinds,
                backend: CBv2ContiguousKVBackend(
                    config: .init(
                        bytesCapacity: 1 << 28,
                        quantization: quantized ? (groupSize: 32, bits: 8) : nil)),
                cacheProvider: CBv2LayerCacheBank(layerKinds: model.layerKinds),
                sampler: CBv2DefaultSampler(fallbackSeed: 7),
                schedulerConfig: CBv2SchedulerConfig(
                    maxConcurrentRequests: 2, maxBatchedTokensPerStep: 256,
                    prefillChunkSize: 16, maxWaiting: 8),
                compiledDecodeConfig: CBv2CompiledDecodeConfig(enabled: compiled))
        }
        let quantizedCompiled = makeEngine(quantized: true, compiled: true)
        let quantizedEager = makeEngine(quantized: true, compiled: false)
        let fp16Compiled = makeEngine(quantized: false, compiled: true)

        // The veto: no executor at all — not a built executor that stays
        // `.ready` while falling back on every step.
        XCTAssertNil(
            quantizedCompiled.compiledDecodeStats,
            "quantized-KV backend must veto the compiled build entirely")
        XCTAssertNotNil(
            fp16Compiled.compiledDecodeStats,
            "control: the fp16 backend must still build the compiled path")

        // No padding reserve: the admission ceiling matches a no-compiled
        // engine exactly…
        XCTAssertEqual(
            quantizedCompiled.admissionForTesting.admissibleBytesCapacity,
            quantizedEager.admissionForTesting.admissibleBytesCapacity,
            "vetoed compiled decode must not charge the admission padding reserve")
        // …while an eligible fp16 engine really does carve one out.
        XCTAssertLessThan(
            fp16Compiled.admissionForTesting.admissibleBytesCapacity,
            quantizedCompiled.admissionForTesting.admissibleBytesCapacity,
            "control: an eligible engine charges a nonzero padding reserve")

        // The vetoed engine still serves, end to end, on the eager path.
        let collected = await cbv2SchedCollect(
            try quantizedCompiled.submit(
                request(id: 1, prompt: makePromptTokens(length: 9, seed: 32), maxTokens: 12)))
        XCTAssertEqual(collected.finishReason, .length)
        await quantizedCompiled.shutdown()
        await quantizedEager.shutdown()
        await fp16Compiled.shutdown()
    }

    /// Scratch lanes must recycle before a full-attention scratch buffer
    /// can overflow. A {2}-only ladder keeps lane 1 scratch-padded for
    /// every solo request; scratch offsets accumulate ACROSS requests
    /// (real lanes rebind, the scratch lane persists), so three sequential
    /// requests at kvCapacity 32 cross the recycle threshold while every
    /// real row stays comfortably inside it. Parity holds throughout.
    func testScratchLaneRecyclingUnderPersistentPadding() async throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let prompts = (0 ..< 3).map { makePromptTokens(length: 5, seed: 33 + UInt64($0)) }
        var perMode: [Bool: [[Int]]] = [:]
        var stats: CBv2CompiledDecodeStats?
        for compiled in [false, true] {
            let engine = makeEngine(
                model: model, compiled: compiled, kvCapacity: 32, buckets: [2])
            var tokens: [[Int]] = []
            for (i, prompt) in prompts.enumerated() {
                let collected = await cbv2SchedCollect(
                    try engine.submit(
                        request(id: UInt64(i + 1), prompt: prompt, maxTokens: 20)))
                tokens.append(collected.tokens)
            }
            await engine.shutdown()
            perMode[compiled] = tokens
            if compiled { stats = engine.compiledDecodeStats }
        }
        XCTAssertEqual(perMode[true], perMode[false])
        let compiledStats = try XCTUnwrap(stats)
        XCTAssertGreaterThan(compiledStats.compiledSteps, 0)
        XCTAssertGreaterThan(
            compiledStats.scratchResets, 0,
            "~60 accumulated padded steps at kvCapacity 32 must recycle the scratch lane")
    }

    /// Single-token prompts join decode with NO prefill and NO allocated
    /// buffers — the bind path must allocate compiled-shape storage from
    /// scratch (`keys == nil`) and stay token-exact.
    func testSingleTokenPromptParity() async throws {
        try await assertCompiledMatchesEager(
            model: TinyTestModel.make(seed: 0xC0FFEE),
            prompts: [[7], [42]], maxTokens: [24, 24],
            label: "single-token prompt")
    }

    /// Prefix-cache donation AFTER compiled decode steps (snapshot views
    /// over padded buffers) and adoption INTO compiled decode
    /// (fast-forwarded windowed rows at the adopted offset) must both stay
    /// token-exact vs the eager engine.
    func testPrefixAdoptionRoundTripParity() async throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        // Long enough that matched-prefix − window recompute (16) leaves a
        // non-trivial adopted offset: 41 tokens ⇒ matched 40, effective 24.
        let prompt = makePromptTokens(length: 41, seed: 51)
        var perMode: [Bool: [[Int]]] = [:]
        var perModeHits: [Bool: [Int]] = [:]
        for compiled in [false, true] {
            let prefixCache = PrefixCacheV2(
                config: .init(blockSize: 8, modelName: "cbv2-compiled-adopt"))
            let engine = EngineV2(
                model: model,
                layerKinds: model.layerKinds,
                backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28)),
                cacheProvider: CBv2LayerCacheBank(layerKinds: model.layerKinds),
                sampler: CBv2DefaultSampler(fallbackSeed: 7),
                schedulerConfig: CBv2SchedulerConfig(
                    maxConcurrentRequests: 2, maxBatchedTokensPerStep: 256,
                    prefillChunkSize: 8, maxWaiting: 8, enablePrefixCache: true),
                prefixCache: prefixCache,
                compiledDecodeConfig: CBv2CompiledDecodeConfig(enabled: compiled))
            // Cold run donates on finish (async, on the donation queue);
            // wait for the entry to land before the warm run looks it up.
            let cold = await cbv2SchedCollect(
                try engine.submit(request(id: 1, prompt: prompt, maxTokens: 20)))
            let donated = await cbv2SchedWait { prefixCache.stats().entryCount >= 1 }
            XCTAssertTrue(donated, "cold run must donate its prefix")
            let warm = await cbv2SchedCollect(
                try engine.submit(request(id: 2, prompt: prompt, maxTokens: 20)))
            await engine.shutdown()
            perMode[compiled] = [cold.tokens, warm.tokens]
            perModeHits[compiled] = [
                cold.usage?.prefixCacheHitTokens ?? 0, warm.usage?.prefixCacheHitTokens ?? 0,
            ]
        }
        XCTAssertEqual(perMode[true], perMode[false], "adoption round trip diverged")
        XCTAssertEqual(
            perModeHits[true], perModeHits[false],
            "prefix-cache hit accounting diverged between modes")
        XCTAssertGreaterThan(
            perModeHits[true]![1], 0, "warm request must adopt a donated prefix")
    }

    /// An eager bank built WITH an attention softcap must veto compiled
    /// decode (the compiled path has no softcap; silently skipping it would
    /// drift from eager numerics) even when the compiled config forgot to
    /// mirror it.
    func testSoftcappedBankVetoesCompiledDecode() async throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let engine = EngineV2(
            model: model,
            layerKinds: model.layerKinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28)),
            cacheProvider: CBv2LayerCacheBank(
                layerKinds: model.layerKinds, attentionSoftcap: 50),
            sampler: CBv2DefaultSampler(fallbackSeed: 7),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 2, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 8),
            compiledDecodeConfig: CBv2CompiledDecodeConfig(enabled: true))
        let collected = await cbv2SchedCollect(
            try engine.submit(
                request(id: 1, prompt: makePromptTokens(length: 9, seed: 61), maxTokens: 8)))
        await engine.shutdown()
        XCTAssertEqual(collected.finishReason, .length)
        XCTAssertNil(
            engine.compiledDecodeStats,
            "softcapped eager caches must veto the compiled path entirely")
    }

    /// A provider that makes NO softcap claim (nil) must fail-safe veto
    /// compiled decode: the engine cannot verify eager/compiled numeric
    /// agreement, so it must not build the compiled path (Codex P2 round-3:
    /// the veto is a protocol obligation, not a CBv2LayerCacheBank cast).
    func testNoClaimProviderFailSafeVetoesCompiledDecode() async throws {
        final class NoClaimProvider: CBv2LayerCacheProvider {
            let inner: CBv2LayerCacheBank
            init(layerKinds: [CBv2LayerKind]) {
                self.inner = CBv2LayerCacheBank(layerKinds: layerKinds)
            }
            var uniformAttentionSoftcap: Float?? { nil }
            func layerCaches(rowStates: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache] {
                inner.layerCaches(rowStates: rowStates)
            }
        }
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let engine = EngineV2(
            model: model,
            layerKinds: model.layerKinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28)),
            cacheProvider: NoClaimProvider(layerKinds: model.layerKinds),
            sampler: CBv2DefaultSampler(fallbackSeed: 7),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 2, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 8),
            compiledDecodeConfig: CBv2CompiledDecodeConfig(enabled: true))
        let collected = await cbv2SchedCollect(
            try engine.submit(
                request(id: 1, prompt: makePromptTokens(length: 9, seed: 61), maxTokens: 8)))
        await engine.shutdown()
        XCTAssertEqual(collected.finishReason, .length)
        XCTAssertNil(
            engine.compiledDecodeStats,
            "a no-claim provider must fail-safe veto the compiled path")
    }

    /// The compiled padding reserve must be carved out of the admission
    /// ledger (real padded bytes may exceed what token admission budgets),
    /// and a budget too small for the reserve must refuse to build the
    /// compiled path rather than over-commit.
    func testPaddingReserveGatesAdmissionAndBuild() throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let reserve = CBv2CompiledDecode.paddingReserveBytes(
            layerKinds: model.layerKinds, kvCapacity: 4096,
            bucketSizes: [1, 2, 4], maxConcurrentRequests: 4)
        // TinyTestModel: full layer = 2·2·4096·16·4 bytes; window ring =
        // 2·2·16·16·4 bytes; 4 real rows + 1 scratch lane (bucket 4).
        let fullPerRow = 2 * 2 * 4096 * 16 * 4
        let ringPerRow = 2 * 2 * 16 * 16 * 4
        XCTAssertEqual(reserve, 4 * fullPerRow + 1 * (fullPerRow + ringPerRow))

        // Budget below 2× the reserve ⇒ compiled decode refuses to build.
        let tight = EngineV2(
            model: model,
            layerKinds: model.layerKinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: reserve + (1 << 20))),
            cacheProvider: CBv2LayerCacheBank(layerKinds: model.layerKinds),
            sampler: CBv2DefaultSampler(fallbackSeed: 7),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 8),
            compiledDecodeConfig: CBv2CompiledDecodeConfig(enabled: true))
        XCTAssertNil(
            tight.compiledDecodeStats,
            "a KV budget the padding reserve would halve must refuse the compiled path")

        // Ample budget ⇒ builds, and the admission ledger is reduced by
        // exactly the reserve while heartbeats keep the hardware truth.
        let ample = EngineV2(
            model: model,
            layerKinds: model.layerKinds,
            backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: 1 << 28)),
            cacheProvider: CBv2LayerCacheBank(layerKinds: model.layerKinds),
            sampler: CBv2DefaultSampler(fallbackSeed: 7),
            schedulerConfig: CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 256,
                prefillChunkSize: 16, maxWaiting: 8),
            compiledDecodeConfig: CBv2CompiledDecodeConfig(enabled: true))
        XCTAssertNotNil(ample.compiledDecodeStats)
        XCTAssertEqual(ample.capacity().kvBytesCapacity, 1 << 28)
    }

    /// PR#62 review: the padding reserve is charged at engine BUILD, but
    /// warmup tracing can disable compiled decode (a model structure that
    /// resists tracing). The engine then serves eagerly forever — the
    /// reserve must be REFUNDED, or admission stays permanently tighter
    /// than the hardware truth and otherwise-fitting requests are rejected.
    func testTraceFailureRefundsCompiledPaddingReserve() async throws {
        /// Opts into compiled decode but host-syncs on EVERY forward:
        /// harmless on the eager path (including the warmup dtype probe),
        /// fatal inside `compile`'s trace — warmup disables with
        /// "trace failed". This is the GPT-OSS-style lazy-host-probe
        /// failure class, made permanent.
        final class TraceHostileModel: CBv2SteppableModel, CBv2CompiledSteppableModel {
            let inner: TinyTestModel
            init(_ inner: TinyTestModel) { self.inner = inner }
            func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
                let logits = inner.forward(tokens: tokens, caches: caches)
                _ = logits.sum().item(Float.self)  // host sync — kills the trace
                return logits
            }
        }

        let inner = TinyTestModel.make(seed: 0xC0FFEE)
        let reserve = CBv2CompiledDecode.paddingReserveBytes(
            layerKinds: inner.layerKinds, kvCapacity: 4096,
            bucketSizes: [1], maxConcurrentRequests: 4)
        let capacityBytes = 2 * reserve + (1 << 20)

        // A request whose worst case fits the FULL budget but NOT the
        // reserve-reduced one (target: capacity - reserve/2).
        let probe = AdmissionV2(
            layerKinds: inner.layerKinds, bytesCapacity: capacityBytes,
            config: .init(watermarkFraction: 0))
        let target = capacityBytes - reserve / 2
        var lo = 1
        var hi = 8_000_000
        while lo < hi {
            let mid = (lo + hi) / 2
            if probe.estimatedBytes(forTokens: mid) < target { lo = mid + 1 } else { hi = mid }
        }
        let prompt = [1, 2, 3]
        let bigRequest = request(id: 42, prompt: prompt, maxTokens: lo - prompt.count)

        func engine(model: CBv2SteppableModel) -> EngineV2 {
            EngineV2(
                model: model,
                layerKinds: inner.layerKinds,
                backend: CBv2ContiguousKVBackend(config: .init(bytesCapacity: capacityBytes)),
                cacheProvider: CBv2LayerCacheBank(layerKinds: inner.layerKinds),
                sampler: CBv2DefaultSampler(fallbackSeed: 7),
                schedulerConfig: CBv2SchedulerConfig(
                    maxConcurrentRequests: 4, maxBatchedTokensPerStep: 256,
                    prefillChunkSize: 16, maxWaiting: 8),
                compiledDecodeConfig: CBv2CompiledDecodeConfig(
                    enabled: true, buckets: [1], kvCapacity: 4096))
        }

        // Control: with compiled decode LIVE the reserve is rightly held,
        // so the same request is rejected at submit (proves the request
        // really straddles the carve-out boundary).
        let control = engine(model: inner)
        XCTAssertNotNil(control.compiledDecodeStats, "control must build the compiled path")
        XCTAssertThrowsError(try control.submit(bigRequest)) { error in
            guard case CBv2KVError.capacityExhausted = error else {
                return XCTFail("expected capacityExhausted while the reserve is held, got \(error)")
            }
        }
        await control.shutdown()

        // Trace-hostile engine: warmup disables compiled decode → the
        // reserve is refunded → the same request is admitted.
        let hostile = engine(model: TraceHostileModel(inner))
        let disabled = await cbv2SchedWait(timeoutSeconds: 30) {
            hostile.compiledDecodeStats?.disabledReason != nil
        }
        XCTAssertTrue(disabled, "warmup must disable compiled decode for the trace-hostile model")
        XCTAssertTrue(
            hostile.compiledDecodeStats?.disabledReason?.contains("trace failed") == true,
            "disable reason should be the trace failure: \(String(describing: hostile.compiledDecodeStats?.disabledReason))"
        )
        // Engine-queue barrier: the refund runs in the same start() block as
        // warmup; sync so the submit below observes it deterministically.
        _ = hostile.loopForTesting.pausedIDsSnapshot()

        let stream = try hostile.submit(bigRequest)  // must NOT throw post-refund
        hostile.cancel(bigRequest.id)
        let collected = await cbv2SchedCollect(stream)
        XCTAssertEqual(collected.finishReason, .cancelled)
        await hostile.shutdown()
    }

    // MARK: - Steady-state discipline (DAR-325 class)

    /// A stable chained batch must not rebind lanes per step: rebinds track
    /// membership changes only, and warmup happens exactly once per bucket.
    func testStableChainRebindsOnlyOnMembership() async throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        let engine = makeEngine(model: model, compiled: true)
        let prompts = [
            makePromptTokens(length: 9, seed: 41),
            makePromptTokens(length: 13, seed: 42),
        ]
        let s0 = try engine.submit(request(id: 1, prompt: prompts[0], maxTokens: 60))
        let s1 = try engine.submit(request(id: 2, prompt: prompts[1], maxTokens: 60))
        _ = await cbv2SchedCollect(s0)
        _ = await cbv2SchedCollect(s1)
        await engine.shutdown()

        let stats = try XCTUnwrap(engine.compiledDecodeStats)
        XCTAssertGreaterThanOrEqual(stats.compiledSteps, 50)
        // Membership events here: two joins + two finishes (+ self-heal
        // after any eager interleave). ~120 compiled steps with per-step
        // rebinds would blow far past this bound.
        XCTAssertLessThanOrEqual(
            stats.laneRebinds, 24,
            "stable chained decode must not rebind lanes per step (\(stats.laneRebinds))")
        XCTAssertEqual(stats.warmup.map(\.bucket), [1, 2, 4], "one warmup trace per bucket")
        for entry in stats.warmup {
            XCTAssertGreaterThan(entry.seconds, 0)
        }
    }

    /// The EAGER decode path must submit the layer caches' inner state (the
    /// on-device `positionOffsets` `+ L` chain plus KV buffers) into the
    /// step's `asyncEval` EVERY step, so the offset chain can never grow
    /// O(steps) of unevaluated graph (DAR-325; legacy BatchKVCache had this
    /// exact bug). The engine exposes a per-step counter; without the fix it
    /// stays 0 across a long eager decode. Bounded-graph proxy: the counter
    /// tracks ~1 evaluation per eager step, and the tokens stay correct.
    func testEagerDecodeEvaluatesOffsetChainEveryStep() async throws {
        let model = TinyTestModel.make(seed: 0xC0FFEE)
        // compiled:false ⇒ every step runs the eager caches (whose offsets
        // advance lazily on-device); this is the path the guard protects.
        let engine = makeEngine(model: model, compiled: false)
        let budget = 40
        let collected = await cbv2SchedCollect(
            try engine.submit(
                request(id: 1, prompt: makePromptTokens(length: 8, seed: 3), maxTokens: budget)))
        await engine.shutdown()

        XCTAssertEqual(collected.finishReason, .length)
        XCTAssertEqual(collected.tokens.count, budget, "long decode must complete correctly")
        // ~1 prefill step + ~budget decode steps, each evaluating the offset
        // chain. Well above a conservative floor; exactly 0 without the fix.
        XCTAssertGreaterThanOrEqual(
            engine.offsetChainEvalSteps, 30,
            "eager decode must evaluate the offset/KV inner state every step (DAR-325)")
    }

    // MARK: - Unit: compiled layer cache vs eager attention numerics

    /// Drive `CBv2CompiledLayerCache` EAGERLY (outside compile) against
    /// `CBv2LayerCache` on identical rows and inputs; outputs must agree to
    /// float tolerance for full, windowed (pre/post wrap), and
    /// fast-forwarded rows.
    func testCompiledCacheMatchesEagerAttention() throws {
        let kvHeads = 2
        let headDim = 8
        let queryHeads = 4
        let scale = Float(1.0 / Double(headDim).squareRoot())
        let window = 6
        let kinds: [CBv2LayerKind] = [
            CBv2LayerKind(attention: .full, headDim: headDim, kvHeads: kvHeads,
                queryHeads: queryHeads),
            CBv2LayerKind(attention: .slidingWindow(window), headDim: headDim, kvHeads: kvHeads,
                queryHeads: queryHeads),
        ]

        // Two rows at different offsets; row 1's windowed state has wrapped.
        let histories = [4, 9]
        var eagerRows: [[CBv2SequenceKV]] = [[], []]  // [layer][row]
        var compiledRows: [[CBv2SequenceKV]] = [[], []]
        for history in histories {
            let full = CBv2FullSequenceKV(
                promptLength: history, maxLength: 32, kvHeads: kvHeads, headDim: headDim)
            let fullC = CBv2FullSequenceKV(
                promptLength: history, maxLength: 32, kvHeads: kvHeads, headDim: headDim)
            let windowed = CBv2WindowedSequenceKV(
                window: window, kvHeads: kvHeads, headDim: headDim)
            let windowedC = CBv2WindowedSequenceKV(
                window: window, kvHeads: kvHeads, headDim: headDim)
            let k = MLXRandom.normal([1, kvHeads, history, headDim])
            let v = MLXRandom.normal([1, kvHeads, history, headDim])
            _ = full.update(keys: k, values: v)
            _ = fullC.update(keys: k, values: v)
            _ = windowed.update(keys: k, values: v)
            _ = windowedC.update(keys: k, values: v)
            eagerRows[0].append(full)
            eagerRows[1].append(windowed)
            compiledRows[0].append(fullC)
            compiledRows[1].append(windowedC)
        }

        let counters = CBv2CompiledLaneCounters(lanes: 2)
        let statics = CBv2CompiledLaneStatics(lanes: 2, windows: [window])
        for (lane, history) in histories.enumerated() {
            counters.bind(lane: lane, offset: history)
            statics.bind(lane: lane, oldestByWindow: [max(0, history - window)])
        }

        for (layer, kind) in kinds.enumerated() {
            let eager = CBv2LayerCache(layerIndex: layer, kind: kind, rows: eagerRows[layer])
            let compiled = CBv2CompiledLayerCache(
                layerIndex: layer, kind: kind, kvCapacity: 32,
                counters: counters, statics: statics)
            for (lane, row) in compiledRows[layer].enumerated() {
                let storage: (keys: MLXArray, values: MLXArray)
                switch row {
                case let full as CBv2FullSequenceKV:
                    storage = try XCTUnwrap(
                        full.compiledStorage(
                            capacity: 32, keysDType: .float32, valuesDType: .float32))
                case let windowed as CBv2WindowedSequenceKV:
                    storage = try XCTUnwrap(
                        windowed.compiledStorage(keysDType: .float32, valuesDType: .float32))
                default:
                    XCTFail("unexpected row type")
                    return
                }
                compiled.bindLane(lane, keys: storage.keys, values: storage.values)
            }

            let q = MLXRandom.normal([2, queryHeads, 1, headDim])
            let k = MLXRandom.normal([2, kvHeads, 1, headDim])
            let v = MLXRandom.normal([2, kvHeads, 1, headDim])
            let eagerOut = eager.updateAndAttend(
                queries: q, keys: k, values: v, scale: scale, sinks: nil)
            let compiledOut = compiled.updateAndAttend(
                queries: q, keys: k, values: v, scale: scale, sinks: nil)
            XCTAssertTrue(
                allClose(eagerOut, compiledOut, atol: 1e-5).item(Bool.self),
                "layer \(layer) (\(kind.attention)): compiled attention diverged from eager")
        }
    }

    /// Fast-forwarded windowed rows (prefix adoption) must mask every
    /// unwritten ring slot: a fresh row at offset 40 attends ONLY the
    /// just-written token.
    func testCompiledWindowedMaskAfterFastForward() throws {
        let kvHeads = 1
        let headDim = 4
        let window = 8
        let kind = CBv2LayerKind(
            attention: .slidingWindow(window), headDim: headDim, kvHeads: kvHeads, queryHeads: 1)

        let makeRow = { () -> CBv2WindowedSequenceKV in
            let row = CBv2WindowedSequenceKV(window: window, kvHeads: kvHeads, headDim: headDim)
            row.fastForward(to: 40)
            return row
        }
        let eagerRow = makeRow()
        let compiledRow = makeRow()

        let counters = CBv2CompiledLaneCounters(lanes: 1)
        let statics = CBv2CompiledLaneStatics(lanes: 1, windows: [window])
        counters.bind(lane: 0, offset: 40)
        statics.bind(lane: 0, oldestByWindow: [compiledRow.compiledOldestValidPosition])

        let compiled = CBv2CompiledLayerCache(
            layerIndex: 0, kind: kind, kvCapacity: 64, counters: counters, statics: statics)
        // Poison the scratch ring with garbage the mask must exclude.
        let storage = try XCTUnwrap(
            compiledRow.compiledStorage(keysDType: .float32, valuesDType: .float32))
        storage.keys._updateInternal(MLXArray.full(storage.keys.shape, values: MLXArray(7.5)))
        storage.values._updateInternal(MLXArray.full(storage.values.shape, values: MLXArray(-3.25)))
        compiled.bindLane(0, keys: storage.keys, values: storage.values)

        let eager = CBv2LayerCache(layerIndex: 0, kind: kind, rows: [eagerRow])

        let q = MLXRandom.normal([1, 1, 1, headDim])
        let k = MLXRandom.normal([1, kvHeads, 1, headDim])
        let v = MLXRandom.normal([1, kvHeads, 1, headDim])
        let eagerOut = eager.updateAndAttend(queries: q, keys: k, values: v, scale: 0.5, sinks: nil)
        let compiledOut = compiled.updateAndAttend(
            queries: q, keys: k, values: v, scale: 0.5, sinks: nil)
        XCTAssertTrue(
            allClose(eagerOut, compiledOut, atol: 1e-5).item(Bool.self),
            "fast-forwarded lane attended poisoned (unwritten) ring slots")
    }

    // MARK: - Independent K/V dtype specialization (PR#62 review)

    /// The compiled trace is specialized on K and V dtypes INDEPENDENTLY: a
    /// row that caches K and V at different precisions must bind only when
    /// BOTH match the traced dtypes, and cleanly fall back (nil) otherwise.
    /// Before the fix the probe recorded only K's dtype and reused it for V,
    /// so a V-dtype mismatch bound silently and could mistrace.
    func testFullRowCompiledStorageChecksKAndVIndependently() throws {
        let row = CBv2FullSequenceKV(promptLength: 2, maxLength: 16, kvHeads: 1, headDim: 4)
        // Synthetic layer: K cached fp16, V cached fp32 (K != V dtype).
        _ = row.update(
            keys: MLXArray.zeros([1, 1, 2, 4], dtype: .float16),
            values: MLXArray.zeros([1, 1, 2, 4], dtype: .float32))

        // Traced (fp16 K, fp32 V) matches → binds.
        XCTAssertNotNil(
            row.compiledStorage(capacity: 16, keysDType: .float16, valuesDType: .float32),
            "matching K and V dtypes must bind")
        // The pre-fix bug: reusing K's dtype for V (fp16) must now be rejected
        // as a V mismatch → eager fallback, never a silent mistrace.
        XCTAssertNil(
            row.compiledStorage(capacity: 16, keysDType: .float16, valuesDType: .float16),
            "V-dtype mismatch must fall back, not bind K's dtype for V")
        // Symmetric: a K mismatch also falls back.
        XCTAssertNil(
            row.compiledStorage(capacity: 16, keysDType: .float32, valuesDType: .float32))
    }

    /// A fresh full row allocates K and V at their SEPARATE traced dtypes.
    func testFreshFullRowAllocatesSeparateKAndVDTypes() throws {
        let row = CBv2FullSequenceKV(promptLength: 0, maxLength: 16, kvHeads: 1, headDim: 4)
        let storage = try XCTUnwrap(
            row.compiledStorage(capacity: 16, keysDType: .bfloat16, valuesDType: .float32))
        XCTAssertEqual(storage.keys.dtype, .bfloat16)
        XCTAssertEqual(storage.values.dtype, .float32)
    }

    /// Windowed rows likewise check K and V dtypes independently.
    func testWindowedRowCompiledStorageChecksKAndVIndependently() {
        let row = CBv2WindowedSequenceKV(window: 8, kvHeads: 1, headDim: 4)
        _ = row.update(
            keys: MLXArray.zeros([1, 1, 1, 4], dtype: .float16),
            values: MLXArray.zeros([1, 1, 1, 4], dtype: .float32))
        XCTAssertNotNil(row.compiledStorage(keysDType: .float16, valuesDType: .float32))
        XCTAssertNil(
            row.compiledStorage(keysDType: .float16, valuesDType: .float16),
            "V-dtype mismatch must fall back")
    }
}
