// Copyright © 2026 Eigen Labs.
//
// Shared mocks/fixtures for the WS-B (scheduler + engine loop) test suites.
// Everything here runs WITHOUT model weights: the scripted model computes
// next-token = (input + 1) % vocab with tiny MLX ops, per row, so outputs
// are deterministic and batch-composition invariant by construction.
//
// All types are prefixed `CBv2Sched` to avoid symbol collisions with other
// workstreams' fixtures when suites merge into this target at integration.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

// MARK: - Deterministic RNG (scheduler simulations)

struct CBv2SchedSplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

// MARK: - Scripted capacity oracle

final class CBv2SchedMockCapacity: CBv2StepCapacity {
    var tokenLimit: Int
    private(set) var reserved: [CBv2RequestID: Int] = [:]
    private(set) var releaseAllCalls: [CBv2RequestID] = []
    var totalReserved: Int { reserved.values.reduce(0, +) }

    init(tokenLimit: Int = .max) { self.tokenLimit = tokenLimit }

    func reserve(id: CBv2RequestID, additionalTokens: Int) throws {
        guard totalReserved + additionalTokens <= tokenLimit else {
            throw CBv2KVError.capacityExhausted(
                needed: additionalTokens, available: tokenLimit - totalReserved)
        }
        reserved[id, default: 0] += additionalTokens
    }

    func unreserve(id: CBv2RequestID, tokens: Int) {
        let value = max(0, (reserved[id] ?? 0) - tokens)
        if value == 0 { reserved.removeValue(forKey: id) } else { reserved[id] = value }
    }

    func releaseAll(id: CBv2RequestID) {
        releaseAllCalls.append(id)
        reserved.removeValue(forKey: id)
    }

    func hasHeadroom(additionalTokens: Int) -> Bool {
        totalReserved + additionalTokens <= tokenLimit
    }
}

// MARK: - Mock per-sequence KV + backend

final class CBv2SchedMockSequenceKV: CBv2SequenceKV {
    private(set) var absoluteOffset = 0
    private(set) var retainedCount = 0
    private(set) var rollbackCalls: [Int] = []
    var byteCount: Int { retainedCount * 2 }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let n = keys.dim(2)
        absoluteOffset += n
        retainedCount += n
        return (keys, values)
    }

    func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        (MLXArray(Int32(0)), MLXArray(Int32(0)), absoluteOffset)
    }

    func rollback(_ n: Int) {
        rollbackCalls.append(n)
        absoluteOffset = max(0, absoluteOffset - n)
        retainedCount = max(0, retainedCount - n)
    }
}

/// Lock-protected counters: the engine thread mutates while tests poll.
final class CBv2SchedMockBackend: CBv2KVBackend, @unchecked Sendable {
    private let lock = NSLock()
    var capacity: Int

    private var _liveStates = 0
    private var _makeCalls = 0
    private var _releasedStates: [[CBv2SequenceKV?]] = []

    var liveStates: Int {
        lock.lock()
        defer { lock.unlock() }
        return _liveStates
    }
    var makeCalls: Int {
        lock.lock()
        defer { lock.unlock() }
        return _makeCalls
    }
    var releasedStates: [[CBv2SequenceKV?]] {
        lock.lock()
        defer { lock.unlock() }
        return _releasedStates
    }

    /// When set, makeSequenceState throws capacityExhausted while
    /// `liveStates >= maxLiveStates` (simulates a full pool so requeue-on-
    /// capacity can be exercised deterministically).
    var maxLiveStates: Int? = nil

    init(capacity: Int = 1 << 30) { self.capacity = capacity }

    func makeSequenceState(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        lock.lock()
        _makeCalls += 1
        if let cap = maxLiveStates, _liveStates >= cap {
            lock.unlock()
            throw CBv2KVError.capacityExhausted(needed: 1, available: 0)
        }
        _liveStates += 1
        lock.unlock()
        return layerKinds.map { $0.sharesKVWithLayer == nil ? CBv2SchedMockSequenceKV() : nil }
    }

    func makeSequenceState(
        adopting prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        layerKinds: [CBv2LayerKind], maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        try makeSequenceState(layerKinds: layerKinds, promptLength: 0, maxLength: maxLength)
    }

    func release(_ state: [CBv2SequenceKV?]) {
        lock.lock()
        _liveStates -= 1
        _releasedStates.append(state)
        lock.unlock()
    }

    var bytesInUse: Int { 0 }
    var bytesCapacity: Int { capacity }
}

// MARK: - Mock layer caches

final class CBv2SchedMockLayerCache: CBv2AttendingLayerCache {
    let layerIndex: Int
    let kind: CBv2LayerKind
    private(set) var rows: [CBv2SequenceKV]
    var positionOffsets: MLXArray { MLXArray(rows.map { Int32($0.absoluteOffset) }) }

    init(layerIndex: Int, kind: CBv2LayerKind, rows: [CBv2SequenceKV]) {
        self.layerIndex = layerIndex
        self.kind = kind
        self.rows = rows
    }

    func setRows(_ rows: [CBv2SequenceKV]) {
        self.rows = rows
    }

    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        queries
    }

    func attendBorrowing(
        source: CBv2AttendingLayerCache, queries: MLXArray, scale: Float, sinks: MLXArray?
    ) -> MLXArray {
        queries
    }
}

final class CBv2SchedMockCacheProvider: CBv2LayerCacheProvider, @unchecked Sendable {
    var uniformAttentionSoftcap: Float?? { .some(nil) }
    let layerKinds: [CBv2LayerKind]
    private let lock = NSLock()
    private var _callRowCounts: [Int] = []
    var callRowCounts: [Int] {
        lock.lock()
        defer { lock.unlock() }
        return _callRowCounts
    }

    init(layerKinds: [CBv2LayerKind]) { self.layerKinds = layerKinds }

    func layerCaches(rowStates: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache] {
        lock.lock()
        _callRowCounts.append(rowStates.count)
        lock.unlock()
        return layerKinds.enumerated().map { index, kind in
            CBv2SchedMockLayerCache(
                layerIndex: index, kind: kind,
                rows: rowStates.compactMap { $0[index] })
        }
    }
}

// MARK: - Scripted model (no weights)

/// next-token = (input + 1) % vocab, as one-hot logits. Per-row, so a
/// request's output can never depend on its batchmates.
final class CBv2SchedScriptedModel: CBv2SteppableModel, @unchecked Sendable {
    let vocab: Int
    private let lock = NSLock()
    private var _forwardShapes: [[Int]] = []
    /// Host-side sleep inside graph build — simulates a wedged step for the
    /// watchdog test (the engine thread blocks mid-`forward`).
    var forwardDelay: TimeInterval = 0

    var forwardShapes: [[Int]] {
        lock.lock()
        defer { lock.unlock() }
        return _forwardShapes
    }

    init(vocab: Int = 64) { self.vocab = vocab }

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        lock.lock()
        _forwardShapes.append(tokens.shape)
        let delay = forwardDelay
        lock.unlock()
        if delay > 0 { Thread.sleep(forTimeInterval: delay) }
        let identity = eye(vocab)
        let next = (tokens + 1) % vocab
        return take(identity, next, axis: 0)  // [B, L, vocab] one-hot
    }
}

// MARK: - Scripted detokenizer

final class CBv2SchedScriptedDetokenizer: CBv2IncrementalDetokenizer {
    /// Simulates WS-E's stop-string holdback: seeing this token flips
    /// `matchedStopString`.
    let stopTrigger: Int?
    private(set) var matchedStopString = false
    private(set) var pushed: [Int] = []

    init(stopTrigger: Int?) { self.stopTrigger = stopTrigger }

    func push(_ tokens: [Int]) -> String {
        pushed.append(contentsOf: tokens)
        if let stopTrigger, tokens.contains(stopTrigger) { matchedStopString = true }
        return tokens.map { "<\($0)>" }.joined()
    }

    func flush() -> String { "" }
}

final class CBv2SchedScriptedDetokFactory: CBv2DetokenizerFactory, @unchecked Sendable {
    /// Applied to detokenizers made for requests that declare stop strings.
    var stopTrigger: Int?

    init(stopTrigger: Int? = nil) { self.stopTrigger = stopTrigger }

    func makeDetokenizer(stopStrings: [String]) -> CBv2IncrementalDetokenizer {
        CBv2SchedScriptedDetokenizer(stopTrigger: stopStrings.isEmpty ? nil : stopTrigger)
    }
}

// MARK: - Scripted prefix cache

/// Scripted `CBv2PrefixCache`: claims a fixed `matched`-token hit for every
/// prompt longer than `matched` (full-attention layers get dummy snapshots;
/// windowed/KV-shared layers stay nil, per the contract). Tracks the
/// lookup-pin balance so tests can assert every hit is released.
final class CBv2SchedScriptedPrefixCache: CBv2PrefixCache, @unchecked Sendable {
    let matched: Int
    private let lock = NSLock()
    private var _lookups = 0
    private var _endAdoptions = 0

    init(matched: Int) { self.matched = matched }

    var lookups: Int {
        lock.lock()
        defer { lock.unlock() }
        return _lookups
    }
    var endAdoptions: Int {
        lock.lock()
        defer { lock.unlock() }
        return _endAdoptions
    }

    func lookup(tokens: [Int], layerKinds: [CBv2LayerKind])
        -> (matched: Int, prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?])?
    {
        guard tokens.count > matched, matched > 0 else { return nil }
        lock.lock()
        _lookups += 1
        lock.unlock()
        return (
            matched,
            layerKinds.map { kind in
                guard kind.sharesKVWithLayer == nil, case .full = kind.attention else {
                    return nil
                }
                let snap = MLXArray.zeros([1, kind.kvHeads, matched, kind.headDim])
                return (keys: snap, values: snap, offset: matched)
            }
        )
    }

    func donate(tokens: [Int], state: [CBv2SequenceKV?], layerKinds: [CBv2LayerKind]) {}

    func donate(
        tokens: [Int],
        snapshots: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        layerKinds: [CBv2LayerKind]
    ) {}

    func endAdoption(tokens: [Int], matched: Int) {
        lock.lock()
        _endAdoptions += 1
        lock.unlock()
    }

    func evict(toFit byteBudget: Int) {}

    var bytesInUse: Int { 0 }
}

// MARK: - Builders

enum CBv2SchedFixtures {
    // Tests mint ids from the main thread / one test at a time; the unsafe
    // opt-out just silences strict-concurrency for this test-only counter.
    nonisolated(unsafe) static var nextID: UInt64 = 0

    static func request(
        prompt: [Int], maxTokens: Int, priority: Int = 0,
        stopTokens: Set<Int> = [], stopStrings: [String] = []
    ) -> CBv2Request {
        nextID += 1
        return CBv2Request(
            id: CBv2RequestID(nextID), promptTokens: prompt, maxTokens: maxTokens,
            stopTokens: stopTokens, stopStrings: stopStrings, priority: priority)
    }

    /// One full-attention layer, kvHeads 1, headDim 4 → 16 bytes/token at
    /// fp16 (AdmissionV2 math stays easy to reason about in tests).
    static func tinyLayerKinds(layers: Int = 1) -> [CBv2LayerKind] {
        (0 ..< layers).map { _ in
            CBv2LayerKind(attention: .full, headDim: 4, kvHeads: 1, queryHeads: 2)
        }
    }
}

/// A full EngineV2 stack over mocks — no weights anywhere.
struct CBv2SchedHarness {
    let engine: EngineV2
    let model: CBv2SchedScriptedModel
    let backend: CBv2SchedMockBackend
    let provider: CBv2SchedMockCacheProvider
    let detokFactory: CBv2SchedScriptedDetokFactory

    init(
        vocab: Int = 64,
        layerKinds: [CBv2LayerKind] = CBv2SchedFixtures.tinyLayerKinds(),
        backendCapacity: Int = 1 << 30,
        schedulerConfig: CBv2SchedulerConfig = CBv2SchedulerConfig(),
        loopConfig: CBv2EngineLoopConfig = CBv2EngineLoopConfig(),
        admissionConfig: AdmissionV2.Config = .init(watermarkFraction: 0),
        stopTrigger: Int? = nil,
        prefixCache: CBv2PrefixCache? = nil
    ) {
        let model = CBv2SchedScriptedModel(vocab: vocab)
        let backend = CBv2SchedMockBackend(capacity: backendCapacity)
        let provider = CBv2SchedMockCacheProvider(layerKinds: layerKinds)
        let detokFactory = CBv2SchedScriptedDetokFactory(stopTrigger: stopTrigger)
        self.model = model
        self.backend = backend
        self.provider = provider
        self.detokFactory = detokFactory
        self.engine = EngineV2(
            model: model,
            layerKinds: layerKinds,
            backend: backend,
            cacheProvider: provider,
            // Deterministic greedy stub: these suites assert exact scripted
            // tokens ((input + 1) % vocab), independent of sampling params.
            sampler: CBv2GreedySampler(),
            detokenizerFactory: detokFactory,
            schedulerConfig: schedulerConfig,
            loopConfig: loopConfig,
            admissionConfig: admissionConfig,
            prefixCache: prefixCache)
    }
}

// MARK: - Event collection helpers

struct CBv2SchedCollected {
    var tokens: [Int] = []
    var text = ""
    var finishReason: CBv2FinishReason?
    var usage: CBv2Usage?
}

func cbv2SchedCollect(
    _ stream: AsyncStream<CBv2Event>, timeoutSeconds: TimeInterval = 20
) async -> CBv2SchedCollected {
    let task = Task { () -> CBv2SchedCollected in
        var collected = CBv2SchedCollected()
        for await event in stream {
            switch event {
            case .delta(let text, let tokens, _):
                collected.text += text
                collected.tokens.append(contentsOf: tokens)
            case .finished(let reason, let usage):
                collected.finishReason = reason
                collected.usage = usage
                return collected
            }
        }
        return collected
    }
    let watchdog = Task {
        try? await Task.sleep(nanoseconds: UInt64(timeoutSeconds * 1_000_000_000))
        task.cancel()
    }
    let result = await task.value
    watchdog.cancel()
    return result
}

/// Poll until `condition` holds (engine loop runs on its own queue).
func cbv2SchedWait(
    timeoutSeconds: TimeInterval = 10, intervalSeconds: TimeInterval = 0.005,
    for condition: @escaping () -> Bool
) async -> Bool {
    let deadline = Date().addingTimeInterval(timeoutSeconds)
    while Date() < deadline {
        if condition() { return true }
        try? await Task.sleep(nanoseconds: UInt64(intervalSeconds * 1_000_000_000))
    }
    return condition()
}

// MARK: - Pure-scheduler simulation harness

/// Mimics exactly what the engine loop does after `plan()`: mark pending
/// samples for rows that computed through their last known token, then
/// confirm them (one step "late" collapses to immediate in a pure sim).
enum CBv2SchedSim {
    /// Returns the ids that sampled this step.
    @discardableResult
    static func confirm(
        _ scheduler: SchedulerV2, plan: CBv2StepPlan,
        nextToken: (CBv2ScheduledRequest) -> Int = { rec in
            (rec.tokens.last ?? 0) + 1
        }
    ) -> [CBv2RequestID] {
        var sampled: [CBv2RequestID] = []
        for (id, _) in plan.assignments {
            guard let rec = scheduler.record(for: id) else { continue }
            if rec.numComputedTokens == rec.effectiveTokenCount {
                scheduler.markPendingSamples(ids: [id])
                scheduler.recordSampled(id: id, token: nextToken(rec))
                sampled.append(id)
            }
        }
        return sampled
    }
}
