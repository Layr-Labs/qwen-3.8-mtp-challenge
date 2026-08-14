// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Single-batch greedy DFlash token iterator.
///
/// This is the first executable DFlash generation path. It intentionally
/// supports `temperature == 0` only, matching the currently ported greedy
/// accept/reject loop.
public struct DFlashTokenIterator: TokenIteratorProtocol {
    private let target: any DFlashTargetModel
    private let drafter: DFlashDraftModel
    private var targetCache: [KVCache]
    private var draftCache: [KVCache]
    private let blockSize: Int
    private let promptTokenCount: Int

    private var bonus: Int
    private var targetHidden: MLXArray

    private var pendingTokens: [Int] = []
    private var pendingIndex = 0

    public var tokenCount = 0
    public let maxTokens: Int?
    public var promptPrefillTime: TimeInterval = 0.0

    public init(
        input: LMInput,
        target: any DFlashTargetModel,
        drafter: DFlashDraftModel,
        targetCache: [KVCache]? = nil,
        draftCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        blockSize: Int? = nil
    ) throws {
        let resolvedBlockSize = blockSize ?? drafter.config.recommendedBlockSize
        guard resolvedBlockSize >= 2 else {
            throw DFlashError.invalidBlockSize(resolvedBlockSize)
        }
        guard parameters.temperature == 0 else {
            throw DFlashError.unsupportedSamplingTemperature(parameters.temperature)
        }

        try drafter.bind(target: target)

        var promptTokens = input.text.tokens
        if promptTokens.ndim == 1 {
            promptTokens = promptTokens[.newAxis, .ellipsis]
        }

        let resolvedTargetCache = targetCache ?? target.newCache(parameters: parameters)
        let resolvedDraftCache: [KVCache]
        if let draftCache {
            resolvedDraftCache = draftCache
        } else {
            resolvedDraftCache = try drafter.makeCache()
        }
        guard canTrimPromptCache(resolvedDraftCache) else {
            throw DFlashError.untrimmableCache
        }

        self.target = target
        self.drafter = drafter
        self.targetCache = resolvedTargetCache
        self.draftCache = resolvedDraftCache
        self.blockSize = resolvedBlockSize
        self.maxTokens = parameters.maxTokens
        self.promptTokenCount = promptTokens.dim(1)

        let prefillStart = Date()
        let prefillOut = try target.forwardGreedyTokensForDFlash(
            promptTokens,
            cache: resolvedTargetCache,
            targetLayerIds: drafter.config.targetLayerIds
        )
        let firstBonusArray = prefillOut.tokens[0..., -1]
        eval(firstBonusArray, prefillOut.targetHidden)

        self.bonus = Int(firstBonusArray.item(Int32.self))
        self.targetHidden = prefillOut.targetHidden
        self.promptPrefillTime = -prefillStart.timeIntervalSinceNow
        self.pendingTokens.append(self.bonus)
    }

    public mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            pendingIndex += 1
            tokenCount += 1
            return token
        }

        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0

        let remaining = maxTokens.map { $0 - tokenCount } ?? blockSize
        guard remaining > 0 else { return nil }

        let roundBlockSize = Swift.min(blockSize, remaining + 1)
        guard roundBlockSize >= 2 else { return nil }

        do {
            let round = try runDFlashGreedyRound(
                target: target,
                drafter: drafter,
                targetCache: &targetCache,
                draftCache: draftCache,
                bonus: bonus,
                targetHidden: targetHidden,
                promptTokenCount: promptTokenCount,
                generatedTokenCount: tokenCount,
                blockSize: roundBlockSize,
                maxEmitCount: remaining
            )
            bonus = round.bonus
            targetHidden = round.targetHidden
            pendingTokens.append(contentsOf: round.tokens)
        } catch {
            // IteratorProtocol cannot throw. Fail closed rather than
            // continuing with inconsistent target/draft caches.
            pendingTokens.removeAll(keepingCapacity: true)
            return nil
        }

        guard pendingIndex < pendingTokens.count else { return nil }
        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }
}
