// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// One streamed token from a DFlash batched decode step.
///
/// `finishReason` is `"length"` when the row reaches its max token count.
/// DFlash currently supports greedy generation, so no sampling metadata is
/// attached here.
public struct DFlashBatchedTokenResponse: Sendable {
    public let uid: Int
    public let token: Int
    public let finishReason: String?
    public let allTokens: [Int]?

    public init(
        uid: Int,
        token: Int,
        finishReason: String? = nil,
        allTokens: [Int]? = nil
    ) {
        self.uid = uid
        self.token = token
        self.finishReason = finishReason
        self.allTokens = allTokens
    }
}

/// Incremental DFlash decode for one loaded model instance and many requests.
///
/// This is the serving-shaped counterpart to `measureBatchedDFlashThroughput`.
/// Callers insert request prompts, then repeatedly call `next()` to emit one
/// token for each live request. Rows with identical prompt tokens share a DFlash
/// batch; rows that diverge in accept length are split into safe subgroups.
public final class DFlashBatchedTokenGenerator: @unchecked Sendable {
    public let target: any DFlashTargetModel
    public let drafter: DFlashDraftModel
    public let blockSize: Int
    public let parameters: GenerateParameters

    private var groups: [DFlashBatchedLiveGroup] = []
    private var rows: [Int: DFlashBatchedLiveRow] = [:]
    private var nextGeneratedUID = 0

    public init(
        target: any DFlashTargetModel,
        drafter: DFlashDraftModel,
        blockSize: Int? = nil,
        parameters: GenerateParameters = GenerateParameters(temperature: 0)
    ) throws {
        let resolvedBlockSize = blockSize ?? drafter.config.recommendedBlockSize
        guard resolvedBlockSize >= 2 else {
            throw DFlashError.invalidBlockSize(resolvedBlockSize)
        }
        guard parameters.temperature == 0 else {
            throw DFlashError.unsupportedSamplingTemperature(parameters.temperature)
        }

        try drafter.bind(target: target)

        var generationParameters = parameters
        generationParameters.temperature = 0

        self.target = target
        self.drafter = drafter
        self.blockSize = resolvedBlockSize
        self.parameters = generationParameters
    }

    public var isEmpty: Bool {
        rows.isEmpty
    }

    public var activeRequestCount: Int {
        rows.count
    }

    /// Drop a live row, for example after an EOS/stop sequence is observed by
    /// the serving layer while DFlash still has accepted tokens queued.
    @discardableResult
    public func cancel(uid: Int) -> Bool {
        guard rows[uid] != nil else { return false }
        removeFinishedRows([uid])
        return true
    }

    /// Insert prompts and allocate generated UIDs in input order.
    @discardableResult
    public func insert(
        prompts: [[Int32]],
        maxTokens: [Int]? = nil
    ) throws -> [Int] {
        let uids = (0 ..< prompts.count).map { _ in
            defer { nextGeneratedUID += 1 }
            return nextGeneratedUID
        }
        try insert(prompts: prompts, uids: uids, maxTokens: maxTokens)
        return uids
    }

    /// Insert prompts with caller-owned UIDs.
    public func insert(
        prompts: [[Int32]],
        uids: [Int],
        maxTokens: [Int]? = nil
    ) throws {
        guard prompts.count == uids.count else {
            throw DFlashError.invalidBatchArguments(
                "prompts.count (\(prompts.count)) must equal uids.count (\(uids.count))")
        }
        if let maxTokens, maxTokens.count != prompts.count {
            throw DFlashError.invalidBatchArguments(
                "maxTokens.count (\(maxTokens.count)) must equal prompts.count (\(prompts.count))")
        }
        guard !prompts.contains(where: \.isEmpty) else {
            throw DFlashError.invalidBatchArguments("prompts must not be empty")
        }

        let perRowMaxTokens = maxTokens
            ?? Array(repeating: parameters.maxTokens ?? 256, count: prompts.count)
        guard perRowMaxTokens.allSatisfy({ $0 > 0 }) else {
            throw DFlashError.invalidBatchArguments("maxTokens entries must be positive")
        }
        for uid in uids where rows[uid] != nil {
            throw DFlashError.invalidBatchArguments("duplicate live uid \(uid)")
        }

        let promptGroups = makeDFlashPromptGroups(prompts)
        for promptGroup in promptGroups {
            let groupBatchSize = promptGroup.rowIndices.count
            let leftPadding = Array(repeating: 0, count: groupBatchSize)
            let prompt = MLXArray(
                promptGroup.tokens.flatMap { $0 },
                [groupBatchSize, promptGroup.length])
            let targetCache = try makeDFlashBatchedTargetCache(
                target: target,
                parameters: parameters,
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

            var groupUIDs: [Int] = []
            groupUIDs.reserveCapacity(groupBatchSize)
            for (localRow, sourceRow) in promptGroup.rowIndices.enumerated() {
                let uid = uids[sourceRow]
                groupUIDs.append(uid)
                rows[uid] = DFlashBatchedLiveRow(
                    promptLength: prompts[sourceRow].count,
                    maxTokens: perRowMaxTokens[sourceRow],
                    producedTokens: 1,
                    emittedTokens: 0,
                    pendingTokens: [bonus[localRow]],
                    allTokens: [bonus[localRow]]
                )
            }

            groups.append(DFlashBatchedLiveGroup(
                uids: groupUIDs,
                bonus: bonus,
                targetHidden: prefillOut.targetHidden,
                targetCache: targetCache,
                draftCache: draftCache))
        }

        try mergeInitialSingleRowGroupsIfPossible()
    }

    /// Emit one token for every live row that can make progress this step.
    public func next() throws -> [DFlashBatchedTokenResponse] {
        guard !rows.isEmpty else { return [] }

        var progressed = true
        while progressed {
            progressed = false
            var nextGroups: [DFlashBatchedLiveGroup] = []
            nextGroups.reserveCapacity(groups.count)

            for var group in groups {
                guard !group.uids.isEmpty else { continue }
                let needsRound = group.uids.allSatisfy { uid in
                    guard let row = rows[uid] else { return false }
                    return row.pendingTokens.isEmpty && row.producedTokens < row.maxTokens
                }
                if needsRound {
                    let roundGroups = try runRound(for: &group)
                    nextGroups.append(contentsOf: roundGroups)
                    progressed = true
                } else {
                    nextGroups.append(group)
                }
            }

            groups = nextGroups
        }

        var responses: [DFlashBatchedTokenResponse] = []
        responses.reserveCapacity(rows.count)
        var finishedUIDs = Set<Int>()

        for uid in rows.keys.sorted() {
            guard var row = rows[uid], !row.pendingTokens.isEmpty else {
                continue
            }
            let token = row.pendingTokens.removeFirst()
            row.emittedTokens += 1

            let finishReason: String?
            let allTokens: [Int]?
            if row.emittedTokens >= row.maxTokens {
                finishReason = "length"
                allTokens = row.allTokens
                finishedUIDs.insert(uid)
            } else {
                finishReason = nil
                allTokens = nil
            }
            rows[uid] = row
            responses.append(DFlashBatchedTokenResponse(
                uid: uid,
                token: token,
                finishReason: finishReason,
                allTokens: allTokens))
        }

        if !finishedUIDs.isEmpty {
            removeFinishedRows(finishedUIDs)
        }

        return responses
    }

    private func runRound(
        for group: inout DFlashBatchedLiveGroup
    ) throws -> [DFlashBatchedLiveGroup] {
        let activeB = group.uids.count
        guard activeB > 0 else { return [] }

        let remaining = group.uids.compactMap { uid -> Int? in
            guard let row = rows[uid] else { return nil }
            return row.maxTokens - row.producedTokens
        }
        guard let minRemaining = remaining.min(), minRemaining > 0 else {
            return []
        }

        let effectiveBlockSize = dFlashBatchedEffectiveBlockSize(
            requestedBlockSize: blockSize,
            activeBatchSize: activeB,
            totalLiveRequestCount: rows.count)
        let roundBlockSize = Swift.min(effectiveBlockSize, minRemaining + 1)
        guard roundBlockSize >= 2 else { return [group] }
        let proposedCount = roundBlockSize - 1

        let draftTokens = try drafter.draftBlock(
            bonus: group.bonus,
            targetHidden: group.targetHidden,
            cache: group.draftCache,
            blockSize: roundBlockSize
        )
        asyncEval(draftTokens)

        let committedOffsets = group.uids.map { uid in
            let row = rows[uid]!
            return row.promptLength + row.producedTokens - 1
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
        // verifyInput is [bonus, drafts...]: its column count is exactly the
        // number of rows this round writes, which the snapshot decision needs to
        // see a sliding-window cache about to wrap (see
        // makeDefaultDFlashCacheRollbackState).
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

        for rowIndex in 0 ..< activeB {
            let uid = group.uids[rowIndex]
            guard var row = rows[uid] else { continue }

            let walkedTokenCount = acceptedPerRow[rowIndex] + 1
            let emitCount = Swift.min(row.maxTokens - row.producedTokens, walkedTokenCount)
            let accepted = emitCount < walkedTokenCount
                ? Swift.max(0, emitCount - 1)
                : acceptedPerRow[rowIndex]
            acceptedForRollback[rowIndex] = accepted

            let rowTokens = targetTokens[rowIndex, 0 ..< emitCount]
                .asArray(Int32.self)
                .map { Int($0) }
            row.pendingTokens.append(contentsOf: rowTokens)
            row.allTokens.append(contentsOf: rowTokens)
            row.producedTokens += rowTokens.count
            if let last = rowTokens.last {
                group.bonus[rowIndex] = last
            }
            rows[uid] = row

            if row.producedTokens < row.maxTokens {
                liveRowsByAccepted[accepted, default: []].append(rowIndex)
            }
        }

        let splitRequired = liveRowsByAccepted.count > 1
        var nextGroups: [DFlashBatchedLiveGroup] = []
        nextGroups.reserveCapacity(liveRowsByAccepted.count)

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

            nextGroups.append(DFlashBatchedLiveGroup(
                uids: keepLocal.map { group.uids[$0] },
                bonus: keepLocal.map { group.bonus[$0] },
                targetHidden: subgroupTargetHidden,
                targetCache: subgroupTargetCache,
                draftCache: subgroupDraftCache))
        }

        return nextGroups
    }

    private func mergeInitialSingleRowGroupsIfPossible() throws {
        guard dFlashMergeSingleRowPrefillGroups,
            groups.count > 1,
            groups.allSatisfy({ $0.uids.count == 1 })
        else {
            return
        }
        for group in groups {
            guard let uid = group.uids.first, let row = rows[uid],
                row.producedTokens == 1,
                row.emittedTokens == 0,
                row.pendingTokens.count == 1
            else {
                return
            }
        }

        let (targetHidden, leftPadding) = makeDFlashPaddedTargetHidden(
            groups.map(\.targetHidden))
        let targetCache = try mergeDFlashSingleRowCaches(groups.map(\.targetCache))
        let draftCache = try drafter.makeBatchedCache(leftPadding: leftPadding)

        groups = [
            DFlashBatchedLiveGroup(
                uids: groups.map { $0.uids[0] },
                bonus: groups.map { $0.bonus[0] },
                targetHidden: targetHidden,
                targetCache: targetCache,
                draftCache: draftCache)
        ]
    }

    private func removeFinishedRows(_ finishedUIDs: Set<Int>) {
        for uid in finishedUIDs {
            rows.removeValue(forKey: uid)
        }

        var filteredGroups: [DFlashBatchedLiveGroup] = []
        filteredGroups.reserveCapacity(groups.count)
        for var group in groups {
            let keepLocal = group.uids.indices.filter {
                !finishedUIDs.contains(group.uids[$0])
            }
            guard !keepLocal.isEmpty else { continue }

            if keepLocal.count != group.uids.count {
                let keepIdx = MLXArray(keepLocal.map { Int32($0) }, [keepLocal.count])
                filterDFlashBatchedCache(group.targetCache, keepIdx)
                filterDFlashBatchedCache(group.draftCache, keepIdx)
                group.targetHidden = MLX.take(group.targetHidden, keepIdx, axis: 0)
                group.bonus = keepLocal.map { group.bonus[$0] }
                group.uids = keepLocal.map { group.uids[$0] }
            }
            filteredGroups.append(group)
        }
        groups = filteredGroups
    }
}

private struct DFlashBatchedLiveGroup {
    var uids: [Int]
    var bonus: [Int]
    var targetHidden: MLXArray
    var targetCache: [KVCache]
    var draftCache: [KVCache]
}

private struct DFlashBatchedLiveRow {
    let promptLength: Int
    let maxTokens: Int
    var producedTokens: Int
    var emittedTokens: Int
    var pendingTokens: [Int]
    var allTokens: [Int]
}
