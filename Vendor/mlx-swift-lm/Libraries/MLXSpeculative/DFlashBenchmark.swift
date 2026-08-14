// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// One DFlash benchmark measurement. Prefill is reported separately so
/// speculative speedup can be computed over the generation phase only.
public struct DFlashBenchmarkResult: Sendable {
    /// Number of tokens generated, excluding the prompt.
    public let generatedTokens: Int
    /// Seconds from prompt prefill start to prompt prefill end.
    public let prefillSeconds: Double
    /// Seconds from prefill end to the last emitted token.
    public let generationSeconds: Double
    /// Per-round count of accepted drafter tokens. `nil` for the no-drafter
    /// baseline; populated with one entry per DFlash round otherwise.
    public let acceptLengths: [Int]?
    /// Optional diagnostic phase timing for DFlash rounds. This is only
    /// populated when `measureDFlashThroughput(..., collectPhaseTimings: true)`
    /// or `collectVerifySubphaseTimings: true` is used.
    public let phaseTimings: DFlashBenchmarkPhaseTimings?
    /// Generated token ids, excluding the prompt. This is populated by the
    /// benchmark helpers so CLI diagnostics can compare exact outputs.
    public let generatedTokenIds: [Int]

    /// Generated tokens / generationSeconds.
    public var tokensPerSecond: Double {
        guard generationSeconds > 0 else { return 0 }
        return Double(generatedTokens) / generationSeconds
    }

    public init(
        generatedTokens: Int,
        prefillSeconds: Double,
        generationSeconds: Double,
        acceptLengths: [Int]? = nil,
        phaseTimings: DFlashBenchmarkPhaseTimings? = nil,
        generatedTokenIds: [Int] = []
    ) {
        self.generatedTokens = generatedTokens
        self.prefillSeconds = prefillSeconds
        self.generationSeconds = generationSeconds
        self.acceptLengths = acceptLengths
        self.phaseTimings = phaseTimings
        self.generatedTokenIds = generatedTokenIds
    }
}

/// Batched DFlash benchmark measurement for one in-process model instance.
///
/// `tokensPerSecond` is aggregate throughput across all rows. Divide by
/// `batchSize` for the average per-request rate.
public struct DFlashBatchedBenchmarkResult: Sendable {
    public let batchSize: Int
    public let generatedTokensPerRow: [Int]
    public let prefillSeconds: Double
    public let generationSeconds: Double
    public let acceptLengths: [Int]?
    public let effectiveBlockSizes: [Int]
    public let generatedTokenIds: [[Int]]

    public var totalGeneratedTokens: Int {
        generatedTokensPerRow.reduce(0, +)
    }

    public var tokensPerSecond: Double {
        guard generationSeconds > 0 else { return 0 }
        return Double(totalGeneratedTokens) / generationSeconds
    }

    public var averageTokensPerSecondPerRequest: Double {
        guard batchSize > 0 else { return 0 }
        return tokensPerSecond / Double(batchSize)
    }

    public init(
        batchSize: Int,
        generatedTokensPerRow: [Int],
        prefillSeconds: Double,
        generationSeconds: Double,
        acceptLengths: [Int]? = nil,
        effectiveBlockSizes: [Int] = [],
        generatedTokenIds: [[Int]] = []
    ) {
        self.batchSize = batchSize
        self.generatedTokensPerRow = generatedTokensPerRow
        self.prefillSeconds = prefillSeconds
        self.generationSeconds = generationSeconds
        self.acceptLengths = acceptLengths
        self.effectiveBlockSizes = effectiveBlockSizes
        self.generatedTokenIds = generatedTokenIds
    }
}

/// Keep vectorized batched Gemma DFlash verify on the shape family that has
/// parity with sequential verify and performs well under shared-instance
/// concurrency. The batched path caps subgroups to K=4, including solo tail
/// rounds, because it intentionally disables single-row verify fusions to stay
/// canonical with shared-instance serving.
public func dFlashBatchedEffectiveBlockSize(
    requestedBlockSize: Int,
    activeBatchSize: Int,
    vectorVerifyRows: Int = 16,
    totalLiveRequestCount: Int? = nil,
    singleRowSoloVerifyTokens: Int = 4,
    singleRowConcurrentVerifyTokens: Int = 4
) -> Int {
    let liveRequests = totalLiveRequestCount ?? activeBatchSize
    if liveRequests > 1 {
        return Swift.max(2, Swift.min(requestedBlockSize, singleRowConcurrentVerifyTokens))
    }
    guard activeBatchSize > 1 else {
        return Swift.max(2, Swift.min(requestedBlockSize, singleRowSoloVerifyTokens))
    }
    let batchLimited = Swift.max(2, vectorVerifyRows / Swift.max(activeBatchSize, 1))
    return Swift.max(2, Swift.min(requestedBlockSize, batchLimited))
}

let dFlashMergeSingleRowPrefillGroups: Bool = {
    switch ProcessInfo.processInfo.environment["MLX_DFLASH_MERGE_PREFILL_GROUPS"]?.lowercased() {
    case "1", "true", "yes", "on":
        true
    default:
        false
    }
}()

/// Aggregate wall-clock phase timings for DFlash greedy rounds.
///
/// These timings are diagnostic. `draftLaunchSeconds` measures graph
/// construction/launch, not full drafter execution, because the current
/// DFlash loop intentionally overlaps drafter evaluation with target verify
/// work. `verifyAndWaitSeconds` includes the explicit wait on target outputs
/// and pending draft tokens.
public struct DFlashBenchmarkPhaseTimings: Sendable {
    public let rounds: Int
    public let cacheSnapshotSeconds: Double
    public let draftLaunchSeconds: Double
    public let draftCacheTrimSeconds: Double
    public let verifyAndWaitSeconds: Double
    /// Nested inside `verifyAndWaitSeconds` when target diagnostic timings
    /// are requested and the target supports `DFlashTargetDiagnosticForwardProvider`.
    public let targetTrunkSeconds: Double
    public let targetHiddenConcatSeconds: Double
    public let targetLMHeadSeconds: Double
    public let targetSoftcapArgmaxSeconds: Double
    public let targetTrunkEmbeddingSeconds: Double
    public let targetTrunkPLESeconds: Double
    public let targetTrunkMaskSeconds: Double
    public let targetTrunkAttentionSeconds: Double
    public let targetTrunkDenseMLPSeconds: Double
    public let targetTrunkRouterSeconds: Double
    public let targetTrunkExpertsSeconds: Double
    public let targetTrunkPLEGateSeconds: Double
    public let targetTrunkFinalNormSeconds: Double
    public let acceptWalkSeconds: Double
    public let cacheRollbackSeconds: Double
    public let roundSeconds: Double

    public var accountedSeconds: Double {
        cacheSnapshotSeconds
            + draftLaunchSeconds
            + draftCacheTrimSeconds
            + verifyAndWaitSeconds
            + acceptWalkSeconds
            + cacheRollbackSeconds
    }
}

// Public only so the (public) runDFlashGreedyRound signature can name it; all
// members stay internal -- MLXFastModel passes nil and never constructs one.
public final class DFlashPhaseAccumulator {
    let collectTargetSubphaseTimings: Bool
    var rounds = 0
    var cacheSnapshotSeconds = 0.0
    var draftLaunchSeconds = 0.0
    var draftCacheTrimSeconds = 0.0
    var verifyAndWaitSeconds = 0.0
    var targetTrunkSeconds = 0.0
    var targetHiddenConcatSeconds = 0.0
    var targetLMHeadSeconds = 0.0
    var targetSoftcapArgmaxSeconds = 0.0
    var targetTrunkEmbeddingSeconds = 0.0
    var targetTrunkPLESeconds = 0.0
    var targetTrunkMaskSeconds = 0.0
    var targetTrunkAttentionSeconds = 0.0
    var targetTrunkDenseMLPSeconds = 0.0
    var targetTrunkRouterSeconds = 0.0
    var targetTrunkExpertsSeconds = 0.0
    var targetTrunkPLEGateSeconds = 0.0
    var targetTrunkFinalNormSeconds = 0.0
    var acceptWalkSeconds = 0.0
    var cacheRollbackSeconds = 0.0
    var roundSeconds = 0.0

    init(collectTargetSubphaseTimings: Bool = false) {
        self.collectTargetSubphaseTimings = collectTargetSubphaseTimings
    }

    func snapshot() -> DFlashBenchmarkPhaseTimings {
        DFlashBenchmarkPhaseTimings(
            rounds: rounds,
            cacheSnapshotSeconds: cacheSnapshotSeconds,
            draftLaunchSeconds: draftLaunchSeconds,
            draftCacheTrimSeconds: draftCacheTrimSeconds,
            verifyAndWaitSeconds: verifyAndWaitSeconds,
            targetTrunkSeconds: targetTrunkSeconds,
            targetHiddenConcatSeconds: targetHiddenConcatSeconds,
            targetLMHeadSeconds: targetLMHeadSeconds,
            targetSoftcapArgmaxSeconds: targetSoftcapArgmaxSeconds,
            targetTrunkEmbeddingSeconds: targetTrunkEmbeddingSeconds,
            targetTrunkPLESeconds: targetTrunkPLESeconds,
            targetTrunkMaskSeconds: targetTrunkMaskSeconds,
            targetTrunkAttentionSeconds: targetTrunkAttentionSeconds,
            targetTrunkDenseMLPSeconds: targetTrunkDenseMLPSeconds,
            targetTrunkRouterSeconds: targetTrunkRouterSeconds,
            targetTrunkExpertsSeconds: targetTrunkExpertsSeconds,
            targetTrunkPLEGateSeconds: targetTrunkPLEGateSeconds,
            targetTrunkFinalNormSeconds: targetTrunkFinalNormSeconds,
            acceptWalkSeconds: acceptWalkSeconds,
            cacheRollbackSeconds: cacheRollbackSeconds,
            roundSeconds: roundSeconds
        )
    }
}

/// Return the generated-token prefix length to keep when generation should stop.
public typealias DFlashStopPredicate = (_ generatedTokenIds: [Int]) -> Int?

/// Run target-only greedy generation over `promptTokens` for `maxTokens`
/// steps. This is the denominator for DFlash speedup measurements.
public func measureDFlashBaselineThroughput(
    target: any DFlashTargetModel,
    promptTokens: MLXArray,
    maxTokens: Int,
    parameters: GenerateParameters = GenerateParameters(temperature: 0),
    stopAfterGeneratedTokenCount: DFlashStopPredicate? = nil
) -> DFlashBenchmarkResult {
    guard maxTokens > 0 else {
        return DFlashBenchmarkResult(
            generatedTokens: 0,
            prefillSeconds: 0,
            generationSeconds: 0
        )
    }

    var generationParameters = parameters
    generationParameters.maxTokens = maxTokens
    generationParameters.temperature = 0

    var prompt = promptTokens
    if prompt.ndim == 1 {
        prompt = prompt[.newAxis, .ellipsis]
    }

    let cache = target.newCache(parameters: generationParameters)

    let prefillStart = Date()
    var logits = target.callAsFunction(prompt, cache: cache)
    var token = logits[0..., -1, 0...].argMax(axis: -1)
    eval(token)
    var generatedIds = [Int(token.item(Int32.self))]
    let prefillElapsed = Date().timeIntervalSince(prefillStart)
    if let stopCount = stopAfterGeneratedTokenCount?(generatedIds) {
        let kept = Array(generatedIds.prefix(clampedStopCount(stopCount, generatedIds.count)))
        return DFlashBenchmarkResult(
            generatedTokens: kept.count,
            prefillSeconds: prefillElapsed,
            generationSeconds: 0,
            generatedTokenIds: kept
        )
    }

    let generationStart = Date()
    var generated = 1
    for _ in 1 ..< maxTokens {
        logits = target.callAsFunction(token[.newAxis, .ellipsis], cache: cache)
        token = logits[0..., -1, 0...].argMax(axis: -1)
        eval(token)
        generatedIds.append(Int(token.item(Int32.self)))
        generated += 1
        if let stopCount = stopAfterGeneratedTokenCount?(generatedIds) {
            generatedIds = Array(generatedIds.prefix(clampedStopCount(stopCount, generatedIds.count)))
            generated = generatedIds.count
            break
        }
    }
    let generationElapsed = Date().timeIntervalSince(generationStart)

    return DFlashBenchmarkResult(
        generatedTokens: generated,
        prefillSeconds: prefillElapsed,
        generationSeconds: generationElapsed,
        generatedTokenIds: generatedIds
    )
}

/// Run target-only greedy generation for B requests in one model instance.
///
/// This is the denominator for in-process batched DFlash benchmarks. Prompts
/// are left-padded to a common length and use `BatchKVCache`/
/// `BatchRotatingKVCache` so the cache layout matches the batched DFlash path.
public func measureBatchedDFlashBaselineThroughput(
    target: any DFlashTargetModel,
    promptTokens: [[Int32]],
    maxTokens: Int,
    parameters: GenerateParameters = GenerateParameters(temperature: 0)
) throws -> DFlashBatchedBenchmarkResult {
    let batchSize = promptTokens.count
    guard batchSize > 0 else {
        return DFlashBatchedBenchmarkResult(
            batchSize: 0,
            generatedTokensPerRow: [],
            prefillSeconds: 0,
            generationSeconds: 0
        )
    }
    guard maxTokens > 0 else {
        return DFlashBatchedBenchmarkResult(
            batchSize: batchSize,
            generatedTokensPerRow: Array(repeating: 0, count: batchSize),
            prefillSeconds: 0,
            generationSeconds: 0,
            generatedTokenIds: Array(repeating: [], count: batchSize)
        )
    }

    var generationParameters = parameters
    generationParameters.maxTokens = maxTokens
    generationParameters.temperature = 0

    let prefillStart = Date()
    var groups: [DFlashBatchedBaselineGroup] = []
    groups.reserveCapacity(batchSize)
    var generatedIds = Array(repeating: [Int](), count: batchSize)
    var generated = Array(repeating: 0, count: batchSize)

    let promptGroups = makeDFlashPromptGroups(promptTokens)
    for promptGroup in promptGroups {
        let groupBatchSize = promptGroup.rowIndices.count
        let leftPadding = Array(repeating: 0, count: groupBatchSize)
        let prompt = MLXArray(
            promptGroup.tokens.flatMap { $0 },
            [groupBatchSize, promptGroup.length])
        let groupCache = try makeDFlashBatchedTargetCache(
            target: target,
            parameters: generationParameters,
            leftPadding: leftPadding)
        let logits = target.callAsFunction(prompt, cache: groupCache)
        let token = logits[0..., -1, 0...].argMax(axis: -1)
        eval(token)

        let first = token.asArray(Int32.self).map { Int($0) }
        for (localRow, originalRow) in promptGroup.rowIndices.enumerated() {
            generatedIds[originalRow] = [first[localRow]]
            generated[originalRow] = 1
        }
        groups.append(DFlashBatchedBaselineGroup(
            activeIndices: promptGroup.rowIndices,
            token: token,
            targetCache: groupCache))
    }
    let prefillElapsed = Date().timeIntervalSince(prefillStart)

    let generationStart = Date()
    while groups.contains(where: { group in
        group.activeIndices.contains { generated[$0] < maxTokens }
    }) {
        for groupIndex in groups.indices {
            let activeIndices = groups[groupIndex].activeIndices
            guard activeIndices.contains(where: { generated[$0] < maxTokens }) else {
                continue
            }

            let groupBatchSize = activeIndices.count
            let logits = target.callAsFunction(
                groups[groupIndex].token.reshaped([groupBatchSize, 1]),
                cache: groups[groupIndex].targetCache)
            let token = logits[0..., -1, 0...].argMax(axis: -1)
            eval(token)

            let ids = token.asArray(Int32.self).map { Int($0) }
            for (localRow, originalRow) in activeIndices.enumerated()
            where generated[originalRow] < maxTokens {
                generatedIds[originalRow].append(ids[localRow])
                generated[originalRow] += 1
            }
            groups[groupIndex].token = token
        }
    }
    let generationElapsed = Date().timeIntervalSince(generationStart)

    return DFlashBatchedBenchmarkResult(
        batchSize: batchSize,
        generatedTokensPerRow: generated,
        prefillSeconds: prefillElapsed,
        generationSeconds: generationElapsed,
        generatedTokenIds: generatedIds
    )
}

/// Run DFlash greedy generation over `promptTokens` for `maxTokens` steps.
///
/// The drafter must be compatible with the target. Binding is performed by
/// `DFlashTokenIterator` and is idempotent for an already-bound matching target.
public func measureDFlashThroughput(
    target: any DFlashTargetModel,
    drafter: DFlashDraftModel,
    promptTokens: MLXArray,
    maxTokens: Int,
    blockSize: Int? = nil,
    parameters: GenerateParameters = GenerateParameters(temperature: 0),
    collectPhaseTimings: Bool = false,
    collectVerifySubphaseTimings: Bool = false,
    stopAfterGeneratedTokenCount: DFlashStopPredicate? = nil
) throws -> DFlashBenchmarkResult {
    guard maxTokens > 0 else {
        return DFlashBenchmarkResult(
            generatedTokens: 0,
            prefillSeconds: 0,
            generationSeconds: 0
        )
    }

    var generationParameters = parameters
    generationParameters.maxTokens = maxTokens
    generationParameters.temperature = 0

    let resolvedBlockSize = blockSize ?? drafter.config.recommendedBlockSize
    guard resolvedBlockSize >= 2 else {
        throw DFlashError.invalidBlockSize(resolvedBlockSize)
    }

    try drafter.bind(target: target)

    var prompt = promptTokens
    if prompt.ndim == 1 {
        prompt = prompt[.newAxis, .ellipsis]
    }

    var targetCache = target.newCache(parameters: generationParameters)
    let draftCache = try drafter.makeCache()
    guard canTrimPromptCache(draftCache) else {
        throw DFlashError.untrimmableCache
    }

    let prefillStart = Date()
    let prefillOut = try target.forwardGreedyTokensForDFlash(
        prompt,
        cache: targetCache,
        targetLayerIds: drafter.config.targetLayerIds
    )
    let firstBonusArray = prefillOut.tokens[0..., -1]
    eval(firstBonusArray, prefillOut.targetHidden)
    let prefillElapsed = Date().timeIntervalSince(prefillStart)

    var bonus = Int(firstBonusArray.item(Int32.self))
    var targetHidden = prefillOut.targetHidden
    var generatedIds = [bonus]
    if let stopCount = stopAfterGeneratedTokenCount?(generatedIds) {
        let kept = Array(generatedIds.prefix(clampedStopCount(stopCount, generatedIds.count)))
        return DFlashBenchmarkResult(
            generatedTokens: kept.count,
            prefillSeconds: prefillElapsed,
            generationSeconds: 0,
            generatedTokenIds: kept
        )
    }

    let generationStart = Date()
    var generated = 1
    var accepts: [Int] = []
    let phases = collectPhaseTimings || collectVerifySubphaseTimings
        ? DFlashPhaseAccumulator(collectTargetSubphaseTimings: collectVerifySubphaseTimings)
        : nil

    while generated < maxTokens {
        let remaining = maxTokens - generated
        let roundBlockSize = Swift.min(resolvedBlockSize, remaining + 1)
        if roundBlockSize < 2 { break }

        let round = try runDFlashGreedyRound(
            target: target,
            drafter: drafter,
            targetCache: &targetCache,
            draftCache: draftCache,
            bonus: bonus,
            targetHidden: targetHidden,
            promptTokenCount: prompt.dim(1),
            generatedTokenCount: generated,
            blockSize: roundBlockSize,
            maxEmitCount: remaining,
            phaseAccumulator: phases
        )
        accepts.append(round.accepted)

        let emitted = round.tokens.count
        generated += emitted
        generatedIds.append(contentsOf: round.tokens)
        if let stopCount = stopAfterGeneratedTokenCount?(generatedIds) {
            generatedIds = Array(generatedIds.prefix(clampedStopCount(stopCount, generatedIds.count)))
            generated = generatedIds.count
            break
        }
        if emitted == 0 { break }

        bonus = round.bonus
        targetHidden = round.targetHidden
    }
    let generationElapsed = Date().timeIntervalSince(generationStart)

    return DFlashBenchmarkResult(
        generatedTokens: generated,
        prefillSeconds: prefillElapsed,
        generationSeconds: generationElapsed,
        acceptLengths: accepts,
        phaseTimings: phases?.snapshot(),
        generatedTokenIds: generatedIds
    )
}

/// Run DFlash greedy generation for B requests in one model instance.
///
/// This initial batched implementation targets the common benchmark/serving
/// case where rows stay lockstep enough to have a uniform acceptance length
/// each round. Non-uniform acceptance needs per-row target-cache rollback and
/// is intentionally rejected here instead of silently corrupting cache state.
public func measureBatchedDFlashThroughput(
    target: any DFlashTargetModel,
    drafter: DFlashDraftModel,
    promptTokens: [[Int32]],
    maxTokens: Int,
    blockSize: Int? = nil,
    parameters: GenerateParameters = GenerateParameters(temperature: 0)
) throws -> DFlashBatchedBenchmarkResult {
    let batchSize = promptTokens.count
    guard batchSize > 0 else {
        return DFlashBatchedBenchmarkResult(
            batchSize: 0,
            generatedTokensPerRow: [],
            prefillSeconds: 0,
            generationSeconds: 0
        )
    }
    guard maxTokens > 0 else {
        return DFlashBatchedBenchmarkResult(
            batchSize: batchSize,
            generatedTokensPerRow: Array(repeating: 0, count: batchSize),
            prefillSeconds: 0,
            generationSeconds: 0,
            generatedTokenIds: Array(repeating: [], count: batchSize)
        )
    }

    var generationParameters = parameters
    generationParameters.maxTokens = maxTokens
    generationParameters.temperature = 0

    let resolvedBlockSize = blockSize ?? drafter.config.recommendedBlockSize
    guard resolvedBlockSize >= 2 else {
        throw DFlashError.invalidBlockSize(resolvedBlockSize)
    }

    try drafter.bind(target: target)

    let prefillStart = Date()
    var groups: [DFlashBatchedGenerationGroup] = []
    groups.reserveCapacity(batchSize)
    var generatedIds = Array(repeating: [Int](), count: batchSize)
    var generated = Array(repeating: 0, count: batchSize)

    let promptGroups = makeDFlashPromptGroups(promptTokens)
    for promptGroup in promptGroups {
        let groupBatchSize = promptGroup.rowIndices.count
        let leftPadding = Array(repeating: 0, count: groupBatchSize)
        let prompt = MLXArray(
            promptGroup.tokens.flatMap { $0 },
            [groupBatchSize, promptGroup.length])
        let targetCache = try makeDFlashBatchedTargetCache(
            target: target,
            parameters: generationParameters,
            leftPadding: leftPadding)
        let draftCache = try drafter.makeBatchedCache(leftPadding: leftPadding)
        let prefillOut = try DFlashTargetRuntimeOptions.withSmallRowVerifyFusionsDisabled {
            try target.forwardGreedyTokensForDFlash(
                prompt,
                cache: targetCache,
                targetLayerIds: drafter.config.targetLayerIds
            )
        }
        let firstBonusArray = prefillOut.tokens[0..., -1]
        eval(firstBonusArray, prefillOut.targetHidden)
        let bonus = firstBonusArray.asArray(Int32.self).map { Int($0) }
        for (localRow, originalRow) in promptGroup.rowIndices.enumerated() {
            generatedIds[originalRow] = [bonus[localRow]]
            generated[originalRow] = 1
        }
        groups.append(DFlashBatchedGenerationGroup(
            activeIndices: promptGroup.rowIndices,
            bonus: bonus,
            targetHidden: prefillOut.targetHidden,
            targetCache: targetCache,
            draftCache: draftCache))
    }
    if dFlashMergeSingleRowPrefillGroups {
        groups = try mergeSingleRowDFlashGenerationGroups(groups, drafter: drafter)
    }
    let prefillElapsed = Date().timeIntervalSince(prefillStart)

    let generationStart = Date()
    var accepts: [Int] = []
    var effectiveBlockSizes: [Int] = []
    let promptLengths = promptTokens.map(\.count)

    while !groups.isEmpty {
        var nextGroups: [DFlashBatchedGenerationGroup] = []
        nextGroups.reserveCapacity(groups.count)
        let liveRequestCount = groups.reduce(0) { $0 + $1.activeIndices.count }

        for var group in groups {
            let activeB = group.activeIndices.count
            guard activeB > 0 else { continue }

            let activeRemaining = group.activeIndices.map { maxTokens - generated[$0] }
            guard let minRemaining = activeRemaining.min(), minRemaining > 0 else {
                continue
            }
            let remaining = minRemaining
            let effectiveBlockSize = dFlashBatchedEffectiveBlockSize(
                requestedBlockSize: resolvedBlockSize,
                activeBatchSize: activeB,
                totalLiveRequestCount: liveRequestCount)
            let roundBlockSize = Swift.min(effectiveBlockSize, remaining + 1)
            if roundBlockSize < 2 { continue }
            effectiveBlockSizes.append(roundBlockSize)
            let proposedCount = roundBlockSize - 1

            let draftTokens = try drafter.draftBlock(
                bonus: group.bonus,
                targetHidden: group.targetHidden,
                cache: group.draftCache,
                blockSize: roundBlockSize
            )
            asyncEval(draftTokens)

            let committedOffsets = group.activeIndices.map {
                promptLengths[$0] + generated[$0] - 1
            }
            let extraDraftContext = commonDFlashDraftCacheOvershoot(
                group.draftCache,
                committedOffsets: committedOffsets)
            if extraDraftContext > 0 {
                let trimmed = trimDFlashBatchedCache(
                    group.draftCache,
                    numTokens: extraDraftContext)
                if trimmed != extraDraftContext {
                    throw DFlashError.untrimmableCache
                }
            }

            let bonusColumn = MLXArray(group.bonus.map { Int32($0) }, [activeB, 1])
            let verifyInput = concatenated([bonusColumn, draftTokens], axis: 1)
            let rollbackProvider = target as? any DFlashTargetCacheRollbackProvider
            // See makeDefaultDFlashCacheRollbackState: the planned write width is
            // what lets the snapshot decision catch a sliding-window cache that
            // is about to wrap mid-round.
            let targetRollbackState =
                rollbackProvider?.makeDFlashCacheRollbackState(cache: group.targetCache)
                ?? target.makeDefaultDFlashCacheRollbackState(
                    cache: group.targetCache, plannedWriteCount: verifyInput.dim(1))
            let verifyOut = try DFlashTargetRuntimeOptions.withSmallRowVerifyFusionsDisabled {
                try target.forwardGreedyTokensForDFlash(
                    verifyInput,
                    cache: group.targetCache,
                    targetLayerIds: drafter.config.targetLayerIds
                )
            }
            let targetTokens = verifyOut.tokens

            let acceptedArray: MLXArray
            if proposedCount == 0 {
                acceptedArray = MLXArray.zeros([activeB], dtype: .int32)
            } else {
                let targetPrefix = targetTokens[0..., 0 ..< proposedCount]
                let matches = (draftTokens .== targetPrefix).asType(.int32)
                let prefixMatches = matches.cumprod(axis: 1)
                acceptedArray = prefixMatches.sum(axis: 1)
            }
            eval(targetTokens, draftTokens, acceptedArray)

            let acceptedPerRow = acceptedArray.asArray(Int32.self).map { Int($0) }
            var acceptedForRollback = Array(repeating: 0, count: activeB)
            var liveRowsByAccepted: [Int: [Int]] = [:]

            for row in 0 ..< activeB {
                let originalRow = group.activeIndices[row]
                let walkedTokenCount = acceptedPerRow[row] + 1
                let emitCount = Swift.min(maxTokens - generated[originalRow], walkedTokenCount)
                let accepted = emitCount < walkedTokenCount
                    ? Swift.max(0, emitCount - 1)
                    : acceptedPerRow[row]
                acceptedForRollback[row] = accepted

                let rowTokens = targetTokens[row, 0 ..< emitCount]
                    .asArray(Int32.self)
                    .map { Int($0) }
                generatedIds[originalRow].append(contentsOf: rowTokens)
                generated[originalRow] += rowTokens.count
                if let last = rowTokens.last {
                    group.bonus[row] = last
                }
                if generated[originalRow] < maxTokens {
                    liveRowsByAccepted[accepted, default: []].append(row)
                }
            }
            accepts.append(contentsOf: acceptedForRollback)

            let splitRequired = liveRowsByAccepted.count > 1
            for acceptedCount in liveRowsByAccepted.keys.sorted() {
                guard let keepLocal = liveRowsByAccepted[acceptedCount],
                    !keepLocal.isEmpty
                else { continue }

                let rejectedCount = roundBlockSize - acceptedCount - 1
                let keepIdx = MLXArray(keepLocal.map { Int32($0) }, [keepLocal.count])

                var subgroupTargetCache =
                    splitRequired
                    ? copyDFlashCache(group.targetCache)
                    : group.targetCache
                let subgroupDraftCache =
                    splitRequired
                    ? copyDFlashCache(group.draftCache)
                    : group.draftCache
                let subgroupRollbackState = filteredDFlashRollbackState(
                    targetRollbackState,
                    keepLocal: keepLocal,
                    sourceBatchSize: activeB)

                if keepLocal.count != activeB {
                    filterDFlashBatchedCache(subgroupTargetCache, keepIdx)
                    filterDFlashBatchedCache(subgroupDraftCache, keepIdx)
                }

                let subgroupVerifyInput =
                    keepLocal.count == activeB
                    ? verifyInput
                    : MLX.take(verifyInput, keepIdx, axis: 0)
                let subgroupVerifiedHidden =
                    keepLocal.count == activeB
                    ? verifyOut.targetHidden
                    : MLX.take(verifyOut.targetHidden, keepIdx, axis: 0)

                let subgroupTargetHidden: MLXArray
                if let rollbackProvider {
                    subgroupTargetHidden = try rollbackProvider.rollbackDFlashCache(
                        &subgroupTargetCache,
                        state: subgroupRollbackState,
                        verifyInput: subgroupVerifyInput,
                        acceptedTokenCount: acceptedCount,
                        rejectedTokenCount: rejectedCount,
                        targetLayerIds: drafter.config.targetLayerIds,
                        verifiedTargetHidden: subgroupVerifiedHidden
                    )
                } else {
                    subgroupTargetHidden = try target.rollbackDFlashCacheUsingDefault(
                        &subgroupTargetCache,
                        state: subgroupRollbackState,
                        verifyInput: subgroupVerifyInput,
                        acceptedTokenCount: acceptedCount,
                        rejectedTokenCount: rejectedCount,
                        targetLayerIds: drafter.config.targetLayerIds,
                        verifiedTargetHidden: subgroupVerifiedHidden
                    )
                }

                nextGroups.append(DFlashBatchedGenerationGroup(
                    activeIndices: keepLocal.map { group.activeIndices[$0] },
                    bonus: keepLocal.map { group.bonus[$0] },
                    targetHidden: subgroupTargetHidden,
                    targetCache: subgroupTargetCache,
                    draftCache: subgroupDraftCache))
            }
        }

        groups = nextGroups
    }
    let generationElapsed = Date().timeIntervalSince(generationStart)

    return DFlashBatchedBenchmarkResult(
        batchSize: batchSize,
        generatedTokensPerRow: generated,
        prefillSeconds: prefillElapsed,
        generationSeconds: generationElapsed,
        acceptLengths: accepts,
        effectiveBlockSizes: effectiveBlockSizes,
        generatedTokenIds: generatedIds
    )
}

@inline(__always)
private func clampedStopCount(_ stopCount: Int, _ upperBound: Int) -> Int {
    Swift.max(1, Swift.min(stopCount, upperBound))
}

private struct DFlashBatchedGenerationGroup {
    var activeIndices: [Int]
    var bonus: [Int]
    var targetHidden: MLXArray
    var targetCache: [KVCache]
    var draftCache: [KVCache]
}

private struct DFlashBatchedBaselineGroup {
    let activeIndices: [Int]
    var token: MLXArray
    var targetCache: [KVCache]
}

private func mergeSingleRowDFlashGenerationGroups(
    _ groups: [DFlashBatchedGenerationGroup],
    drafter: DFlashDraftModel
) throws -> [DFlashBatchedGenerationGroup] {
    guard groups.count > 1, groups.allSatisfy({ $0.activeIndices.count == 1 }) else {
        return groups
    }

    let hiddenRows = groups.map(\.targetHidden)
    let (targetHidden, leftPadding) = makeDFlashPaddedTargetHidden(hiddenRows)
    let targetCache = try mergeDFlashSingleRowCaches(groups.map(\.targetCache))
    let draftCache = try drafter.makeBatchedCache(leftPadding: leftPadding)

    return [
        DFlashBatchedGenerationGroup(
            activeIndices: groups.map { $0.activeIndices[0] },
            bonus: groups.map { $0.bonus[0] },
            targetHidden: targetHidden,
            targetCache: targetCache,
            draftCache: draftCache)
    ]
}

struct DFlashPromptGroup {
    let length: Int
    var rowIndices: [Int]
    var tokens: [[Int32]]
}

func makeDFlashPromptGroups(_ promptTokens: [[Int32]]) -> [DFlashPromptGroup] {
    var groups: [DFlashPromptGroup] = []
    groups.reserveCapacity(promptTokens.count)

    for (row, tokens) in promptTokens.enumerated() {
        let length = tokens.count
        if let groupIndex = groups.firstIndex(where: { $0.tokens.first == tokens }) {
            groups[groupIndex].rowIndices.append(row)
            groups[groupIndex].tokens.append(tokens)
        } else {
            groups.append(DFlashPromptGroup(
                length: length,
                rowIndices: [row],
                tokens: [tokens]))
        }
    }

    return groups
}

private func makeDFlashBatchedPrompt(_ promptTokens: [[Int32]]) -> (
    prompt: MLXArray, leftPadding: [Int]
) {
    let batchSize = promptTokens.count
    let maxLength = promptTokens.map(\.count).max() ?? 0
    let leftPadding = promptTokens.map { maxLength - $0.count }
    let flat = zip(promptTokens, leftPadding).flatMap { row, padding in
        Array(repeating: Int32(0), count: padding) + row
    }
    return (MLXArray(flat, [batchSize, maxLength]), leftPadding)
}

func makeDFlashBatchedTargetCache(
    target: any DFlashTargetModel,
    parameters: GenerateParameters,
    leftPadding: [Int]
) throws -> [KVCache] {
    let prototype = target.newCache(parameters: parameters)
    return try prototype.map { cache in
        if let maxSize = cache.maxSize {
            return BatchRotatingKVCache(
                maxSize: maxSize,
                leftPadding: leftPadding) as KVCache
        }
        if cache is KVCacheSimple {
            return BatchKVCache(leftPadding: leftPadding) as KVCache
        }
        throw DFlashError.unsupportedTarget(
            "batched DFlash cache does not support \(type(of: cache))")
    }
}

func makeDFlashPaddedTargetHidden(_ rows: [MLXArray]) -> (
    hidden: MLXArray, leftPadding: [Int]
) {
    guard let maxLength = rows.map({ $0.dim(1) }).max(), maxLength > 0 else {
        return (MLXArray.zeros([0, 0, 0]), [])
    }

    let leftPadding = rows.map { maxLength - $0.dim(1) }
    let paddedRows = zip(rows, leftPadding).map { row, padding in
        guard padding > 0 else {
            return row
        }
        let prefix = MLXArray.zeros(
            [row.dim(0), padding, row.dim(2)],
            dtype: row.dtype)
        return concatenated([prefix, row], axis: 1)
    }
    return (concatenated(paddedRows, axis: 0), leftPadding)
}

func mergeDFlashSingleRowCaches(_ rowCaches: [[KVCache]]) throws -> [KVCache] {
    guard let first = rowCaches.first else {
        return []
    }
    let layerCount = first.count
    guard rowCaches.allSatisfy({ $0.count == layerCount }) else {
        throw DFlashError.invalidBatchArguments("all cache rows must have the same layer count")
    }

    return try (0 ..< layerCount).map { layerIndex in
        let merged = try makeDFlashSingleRowBatchedCache(rowCaches[0][layerIndex])
        for rowIndex in 1 ..< rowCaches.count {
            let other = try makeDFlashSingleRowBatchedCache(rowCaches[rowIndex][layerIndex])
            merged.extendBatched(other)
        }
        return merged as KVCache
    }
}

private func makeDFlashSingleRowBatchedCache(_ cache: KVCache) throws -> any BatchedCache {
    if let simple = cache as? KVCacheSimple {
        return BatchKVCache.merge([simple])
    }

    if let rotating = cache as? RotatingKVCache {
        guard let maxSize = rotating.maxSize else {
            throw DFlashError.unsupportedTarget("RotatingKVCache is missing maxSize")
        }
        let batched = BatchRotatingKVCache(maxSize: maxSize, leftPadding: [0])
        let state = rotating.state
        if state.count == 2 {
            let offset = Int(rotating.metaState[3]) ?? state[0].dim(2)
            batched.state = [
                state[0],
                state[1],
                MLXArray([Int32(offset)]),
                MLXArray([Int32(0)]),
            ]
        } else {
            batched.state = [
                MLXArray([Int32(0)]),
                MLXArray([Int32(0)]),
            ]
        }
        return batched
    }

    if let batched = cache as? any BatchedCache {
        return batched
    }

    throw DFlashError.unsupportedTarget(
        "cannot merge DFlash cache layer \(type(of: cache)) into a batched cache")
}

private func rollbackDFlashBatchedTargetCache(
    _ cache: [KVCache],
    accepted: [Int],
    blockSize: Int
) throws {
    guard let maxAccepted = accepted.max() else { return }
    let trim = Swift.max(0, blockSize - maxAccepted - 1)
    if trim > 0 {
        let trimmed = trimDFlashBatchedCache(cache, numTokens: trim)
        guard trimmed == trim else {
            throw DFlashError.untrimmableCache
        }
    }

    guard maxAccepted > 0 else { return }
    let acceptedArray = MLXArray(accepted.map { Int32($0) })
    for layerCache in cache {
        if let batched = layerCache as? BatchKVCache {
            let keepLengths =
                acceptedArray.asType(.int32)
                    + Int32(batched.offset - maxAccepted)
            batched.zeroTailPerRow(keepLengths: keepLengths)
        } else if let batched = layerCache as? BatchRotatingKVCache {
            let keepLengths =
                acceptedArray.asType(.int32)
                    + Int32(batched.offset - maxAccepted)
            batched.zeroTailPerRow(keepLengths: keepLengths)
        }
    }
}

func commonDFlashDraftCacheOvershoot(
    _ cache: [KVCache],
    committedOffsets: [Int]
) -> Int {
    guard let first = cache.first, !committedOffsets.isEmpty else { return 0 }
    if let positioned = first as? BatchPositionedKVCache {
        let offsets = positioned.batchOffset.asArray(Int32.self).map { Int($0) }
        let extras = zip(offsets, committedOffsets).map { offset, committed in
            Swift.max(0, offset - committed)
        }
        return extras.min() ?? 0
    }
    let committedOffset = committedOffsets.min() ?? 0
    return Swift.max(0, first.offset - committedOffset)
}

func copyDFlashCache(_ cache: [KVCache]) -> [KVCache] {
    cache.map { $0.copy() }
}

func filteredDFlashRollbackState(
    _ state: (any DFlashTargetRollbackState)?,
    keepLocal: [Int],
    sourceBatchSize: Int
) -> (any DFlashTargetRollbackState)? {
    guard keepLocal.count != sourceBatchSize else {
        return state
    }
    guard let copiedState = state as? DFlashCopiedTargetRollbackState else {
        return state == nil ? nil : state
    }

    let keepIdx = MLXArray(keepLocal.map { Int32($0) }, [keepLocal.count])
    let filtered = copyDFlashCache(copiedState.cache)
    filterDFlashBatchedCache(filtered, keepIdx)
    return DFlashCopiedTargetRollbackState(cache: filtered)
}

func filterDFlashBatchedCache(_ cache: [KVCache], _ keepIdx: MLXArray) {
    for layerCache in cache {
        if let batched = layerCache as? BatchedCache {
            batched.filterBatched(batchIndices: keepIdx)
        }
    }
}

@discardableResult
func trimDFlashBatchedCache(_ cache: [KVCache], numTokens: Int) -> Int {
    guard numTokens > 0, !cache.isEmpty else { return 0 }
    var trimmed = Int.max
    for layerCache in cache {
        trimmed = Swift.min(trimmed, layerCache.trim(numTokens))
    }
    return trimmed == Int.max ? 0 : trimmed
}
