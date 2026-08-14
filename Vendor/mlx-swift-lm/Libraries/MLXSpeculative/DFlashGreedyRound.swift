// Copyright © 2026 Apple Inc.

import CryptoKit
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Operator diagnostic for the sliding-window ring seam. Off unless asked for:
/// it writes cache offsets, which are structural, not token content, but there is
/// no reason for a scored run to emit them.
///
/// The `MLX_` prefix is load-bearing: `sanitizedRuntimeWorkerEnvironment` copies
/// only an allowlist into the sandboxed worker, and `MLXFAST_*` is deliberately
/// NOT on it, so a harness-named variable never reaches this code. Same
/// convention as `MLX_DFLASH_CPU_ACCEPT_WALK` below.
private let dFlashTraceCacheSeam: Bool =
    ProcessInfo.processInfo.environment["MLX_DFLASH_TRACE_CACHE_SEAM"] == "1"

private let dFlashCPUAcceptWalk: Bool = {
    switch ProcessInfo.processInfo.environment["MLX_DFLASH_CPU_ACCEPT_WALK"]?.lowercased() {
    case "0", "false", "no", "off":
        return false
    default:
        return true
    }
}()

/// Criterion E work-binding readouts for one verified block.
///
/// These exist so a trusted parent can bind emitted tokens to target compute
/// that actually ran, at EVERY declared row including rejected ones. Token
/// output alone cannot price a verifier that is degraded everywhere but agrees
/// at the confident steps; `top2Logits` forces the lm_head per row and
/// `hiddenDigest` forces the trunk.
///
/// COMPARISON SEMANTICS -- read before using these on the parent side:
///   * `top2Tokens` / `top2Logits` are NUMERIC and must be compared to the
///     reference with a tolerance. They are the load-bearing work binder.
///   * `hiddenDigest` is an EXACT SHA-256 over the row's float32 bytes. Two
///     different builds (a candidate's optimized kernels vs the pinned
///     reference) do NOT produce bit-identical hidden states -- that is the
///     same accumulation-order frame divergence that makes exact-token
///     matching unachievable on this model. So the digest is usable for
///     SELF-consistency only (reference-vs-reference determinism, or a
///     candidate replaying its own round), never for candidate-vs-reference
///     equality.
public struct DFlashRoundWorkBinding: @unchecked Sendable {
    /// Rows pushed through the target this round (verify width).
    public let declaredRows: Int
    /// Per-row SHA-256 of the captured target hidden state (see caveat above).
    public let hiddenDigest: [String]
    /// Per-row top-2 token ids, highest logit first.
    public let top2Tokens: [[Int]]
    /// Per-row top-2 logit values, aligned with `top2Tokens`.
    public let top2Logits: [[Double]]
    /// The drafter's proposals for this round, `declaredRows - 1` of them, in
    /// the order they occupied the verify input: the verify block is
    /// `[bonus, draftTokens[0], ..., draftTokens[declaredRows - 2]]`.
    ///
    /// Reported so the trusted parent can PRICE THE REJECTED TAIL. Without it
    /// the parent knows only the emitted tokens, so it can reconstruct the
    /// accepted prefix of the verify input and nothing after it -- and the rows
    /// after it are exactly the rows an eliding verifier can fabricate for free,
    /// because no emitted token constrains them. With it the reference can
    /// replay the candidate's ACTUAL verify block, row for row, including the
    /// rows rollback discarded.
    ///
    /// Note what makes this binding hard to defeat: for a REJECTED row the
    /// candidate has no independent source for the row's own argmax. It knows
    /// every emitted token (the drafter proposed most of them), but a rejected
    /// row's output was never emitted, so reporting it requires having actually
    /// run that row's lm_head.
    public let draftTokens: [Int]
}

public struct DFlashGreedyRoundResult {
    public let accepted: Int
    public let tokens: [Int]
    public let bonus: Int
    public let targetHidden: MLXArray
    /// Populated only when the caller requested `workBinding: true`.
    public let workBinding: DFlashRoundWorkBinding?

    internal init(
        accepted: Int,
        tokens: [Int],
        bonus: Int,
        targetHidden: MLXArray,
        workBinding: DFlashRoundWorkBinding? = nil
    ) {
        self.accepted = accepted
        self.tokens = tokens
        self.bonus = bonus
        self.targetHidden = targetHidden
        self.workBinding = workBinding
    }
}

public func runDFlashGreedyRound(
    target: any DFlashTargetModel,
    drafter: DFlashDraftModel,
    targetCache: inout [KVCache],
    draftCache: [KVCache],
    bonus: Int,
    targetHidden: MLXArray,
    promptTokenCount: Int,
    generatedTokenCount: Int,
    blockSize: Int,
    maxEmitCount: Int,
    phaseAccumulator: DFlashPhaseAccumulator? = nil,
    workBinding: Bool = false
) throws -> DFlashGreedyRoundResult {
    guard blockSize >= 2 else {
        throw DFlashError.invalidBlockSize(blockSize)
    }
    let roundStart = dflashTimingStart(phaseAccumulator)

    let snapshotStart = dflashTimingStart(phaseAccumulator)
    let rollbackProvider = target as? any DFlashTargetCacheRollbackProvider
    // Pass the width this round is about to write so the snapshot decision can
    // see a sliding-window cache that is ABOUT to wrap. Without it the round
    // that crosses the ring boundary gets no snapshot and then cannot trim,
    // which is the untrimmableCache failure at a 512-token seed.
    let targetRollbackState =
        rollbackProvider?.makeDFlashCacheRollbackState(cache: targetCache)
        ?? target.makeDefaultDFlashCacheRollbackState(
            cache: targetCache, plannedWriteCount: blockSize)
    dflashRecord(snapshotStart, into: phaseAccumulator) {
        $0.cacheSnapshotSeconds += $1
    }

    let draftStart = dflashTimingStart(phaseAccumulator)
    let draftTokens = try drafter.draftBlock(
        bonus: bonus,
        targetHidden: targetHidden,
        cache: draftCache,
        blockSize: blockSize
    )

    asyncEval(draftTokens)
    dflashRecord(draftStart, into: phaseAccumulator) {
        $0.draftLaunchSeconds += $1
    }

    let draftTrimStart = dflashTimingStart(phaseAccumulator)
    let committedDraftOffset = Swift.max(0, promptTokenCount + generatedTokenCount - 1)
    if let draftOffset = draftCache.first?.offset {
        let extraDraftContext = draftOffset - committedDraftOffset
        if dFlashTraceCacheSeam {
            FileHandle.standardError.write(
                Data(
                    ("dflash-trace: draft_offset=\(draftOffset) "
                        + "committed=\(committedDraftOffset) "
                        + "extra=\(extraDraftContext) "
                        + "draft_trimmable=\(canTrimPromptCache(draftCache)) "
                        + "draft_max=\(draftCache.first?.maxSize.map(String.init) ?? "nil") "
                        + "target_offset=\(targetCache.first?.offset ?? -1) "
                        + "target_trimmable=\(canTrimPromptCache(targetCache)) "
                        + "prompt=\(promptTokenCount) "
                        + "generated=\(generatedTokenCount) "
                        + "block=\(blockSize)\n").utf8
                )
            )
        }
        if extraDraftContext > 0 {
            let trimmed = trimPromptCache(draftCache, numTokens: extraDraftContext)
            if trimmed != extraDraftContext {
                throw DFlashError.untrimmableCache
            }
        }
    }
    dflashRecord(draftTrimStart, into: phaseAccumulator) {
        $0.draftCacheTrimSeconds += $1
    }

    let verifyStart = dflashTimingStart(phaseAccumulator)
    let bonusColumn = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]
    let verifyInput = concatenated([bonusColumn, draftTokens], axis: 1)
    // Criterion E work binding needs the full logits (for per-row top-2), which
    // the greedy fast path deliberately never materializes. Take the full
    // forward in that mode and derive the same argmax tokens from it, so every
    // downstream step (accept walk, rollback, ledger) stays on ONE code path.
    // Requiring the top-2 VALUES is the point: it forces the per-row lm_head
    // that an eliding submission would skip.
    var workBindingLogits: MLXArray?
    let verifyOut: DFlashGreedyTargetForward =
        try DFlashTargetRuntimeOptions.withSmallRowVerifyFusionsEnabled {
            if workBinding {
                let full = try target.forwardForDFlash(
                    verifyInput,
                    cache: targetCache,
                    targetLayerIds: drafter.config.targetLayerIds
                )
                workBindingLogits = full.logits
                return DFlashGreedyTargetForward(
                    tokens: full.logits.argMax(axis: -1),
                    targetHidden: full.targetHidden
                )
            }
            if phaseAccumulator?.collectTargetSubphaseTimings == true,
                let diagnosticTarget = target as? any DFlashTargetDiagnosticForwardProvider
            {
                return try diagnosticTarget.forwardGreedyTokensForDFlash(
                    verifyInput,
                    cache: targetCache,
                    targetLayerIds: drafter.config.targetLayerIds,
                    collectVerifyTimings: true
                )
            }
            return try target.forwardGreedyTokensForDFlash(
                verifyInput,
                cache: targetCache,
                targetLayerIds: drafter.config.targetLayerIds
            )
        }
    if let timings = verifyOut.verifyTimings, let phaseAccumulator {
        phaseAccumulator.targetTrunkSeconds += timings.trunkSeconds
        phaseAccumulator.targetHiddenConcatSeconds += timings.hiddenConcatSeconds
        phaseAccumulator.targetLMHeadSeconds += timings.lmHeadSeconds
        phaseAccumulator.targetSoftcapArgmaxSeconds += timings.softcapArgmaxSeconds
        phaseAccumulator.targetTrunkEmbeddingSeconds += timings.trunkEmbeddingSeconds
        phaseAccumulator.targetTrunkPLESeconds += timings.trunkPLESeconds
        phaseAccumulator.targetTrunkMaskSeconds += timings.trunkMaskSeconds
        phaseAccumulator.targetTrunkAttentionSeconds += timings.trunkAttentionSeconds
        phaseAccumulator.targetTrunkDenseMLPSeconds += timings.trunkDenseMLPSeconds
        phaseAccumulator.targetTrunkRouterSeconds += timings.trunkRouterSeconds
        phaseAccumulator.targetTrunkExpertsSeconds += timings.trunkExpertsSeconds
        phaseAccumulator.targetTrunkPLEGateSeconds += timings.trunkPLEGateSeconds
        phaseAccumulator.targetTrunkFinalNormSeconds += timings.trunkFinalNormSeconds
    }
    let targetTokens = verifyOut.tokens
    let verifiedTokenCount = targetTokens.dim(1)
    let draftTokenIds = draftTokens.squeezed(axis: 0)
    let targetTokenIds = targetTokens.squeezed(axis: 0)
    let proposedCount = Swift.max(0, blockSize - 1)
    let comparableCount = Swift.min(proposedCount, verifiedTokenCount)
    let acceptStart: Date?
    let walkedAccepted: Int
    let emitted: [Int]
    let accepted: Int
    if dFlashCPUAcceptWalk {
        let targetReadCount = Swift.min(
            verifiedTokenCount,
            Swift.max(comparableCount, Swift.min(maxEmitCount, comparableCount + 1)))
        let targetIds = targetReadCount > 0
            ? targetTokenIds[0 ..< targetReadCount].asArray(Int32.self)
            : []
        let draftIds = comparableCount > 0
            ? draftTokenIds[0 ..< comparableCount].asArray(Int32.self)
            : []
        dflashRecord(verifyStart, into: phaseAccumulator) {
            $0.verifyAndWaitSeconds += $1
        }

        acceptStart = dflashTimingStart(phaseAccumulator)
        var prefix = 0
        while prefix < comparableCount, draftIds[prefix] == targetIds[prefix] {
            prefix += 1
        }
        walkedAccepted = prefix
        let walkedTokenCount = walkedAccepted + 1
        let emittedCount = Swift.min(maxEmitCount, walkedTokenCount, verifiedTokenCount)
        emitted = targetIds.prefix(emittedCount).map { Int($0) }
        accepted = emittedCount < walkedTokenCount
            ? Swift.max(0, emittedCount - 1)
            : walkedAccepted
    } else {
        let acceptedArray: MLXArray?
        if comparableCount == 0 {
            acceptedArray = nil
        } else {
            let targetPrefix = targetTokenIds[0 ..< comparableCount]
            let draftPrefix = draftTokenIds[0 ..< comparableCount]
            let matches = (draftPrefix .== targetPrefix).asType(.int32)
            let prefixMatches = matches.cumprod(axis: 0)
            acceptedArray = prefixMatches.sum()
        }
        if let acceptedArray {
            eval(targetTokens, draftTokens, acceptedArray)
        } else {
            eval(targetTokens, draftTokens)
        }
        dflashRecord(verifyStart, into: phaseAccumulator) {
            $0.verifyAndWaitSeconds += $1
        }

        acceptStart = dflashTimingStart(phaseAccumulator)
        walkedAccepted = acceptedArray.map { Int($0.item(Int32.self)) } ?? 0
        let walkedTokenCount = walkedAccepted + 1
        let emittedCount = Swift.min(maxEmitCount, walkedTokenCount, verifiedTokenCount)
        emitted = targetTokenIds[0 ..< emittedCount]
            .asArray(Int32.self)
            .map { Int($0) }
        accepted = emittedCount < walkedTokenCount
            ? Swift.max(0, emittedCount - 1)
            : walkedAccepted
    }
    dflashRecord(acceptStart, into: phaseAccumulator) {
        $0.acceptWalkSeconds += $1
    }

    // Collect the work-binding readouts BEFORE rollback: the verify-time hidden
    // states and logits describe the rows as executed this round, including the
    // rejected tail that rollback is about to discard.
    var roundWorkBinding: DFlashRoundWorkBinding?
    if workBinding, let logits = workBindingLogits {
        // Read the drafts here rather than reusing the accept walk's slice: the
        // walk only ever materializes `comparableCount` of them and only on the
        // CPU path, whereas the parent needs ALL `declaredRows - 1` regardless of
        // which walk ran. `draftTokens` is already evaluated by this point (the
        // accept walk waited on it), so this is a host read, not extra compute.
        let reportedDrafts = proposedCount > 0
            ? draftTokenIds[0 ..< proposedCount].asArray(Int32.self).map { Int($0) }
            : []
        roundWorkBinding = dflashCollectWorkBinding(
            logits: logits,
            targetHidden: verifyOut.targetHidden,
            declaredRows: verifiedTokenCount,
            draftTokens: reportedDrafts
        )
    }

    let trim = Swift.max(0, verifiedTokenCount - accepted - 1)
    let nextTargetHidden: MLXArray
    let rollbackStart = dflashTimingStart(phaseAccumulator)
    if let rollbackProvider {
        nextTargetHidden = try rollbackProvider.rollbackDFlashCache(
            &targetCache,
            state: targetRollbackState,
            verifyInput: verifyInput,
            acceptedTokenCount: accepted,
            rejectedTokenCount: trim,
            targetLayerIds: drafter.config.targetLayerIds,
            verifiedTargetHidden: verifyOut.targetHidden
        )
    } else {
        nextTargetHidden = try target.rollbackDFlashCacheUsingDefault(
            &targetCache,
            state: targetRollbackState,
            verifyInput: verifyInput,
            acceptedTokenCount: accepted,
            rejectedTokenCount: trim,
            targetLayerIds: drafter.config.targetLayerIds,
            verifiedTargetHidden: verifyOut.targetHidden
        )
    }
    dflashRecord(rollbackStart, into: phaseAccumulator) {
        $0.cacheRollbackSeconds += $1
    }
    if let phaseAccumulator {
        phaseAccumulator.rounds += 1
    }
    dflashRecord(roundStart, into: phaseAccumulator) {
        $0.roundSeconds += $1
    }

    return DFlashGreedyRoundResult(
        accepted: accepted,
        tokens: emitted,
        bonus: emitted.last ?? bonus,
        targetHidden: nextTargetHidden,
        workBinding: roundWorkBinding
    )
}

/// Derive the Criterion E per-row readouts from a verified block.
///
/// Top-2 uses the same `argPartition` extraction the DFlash parity check
/// already relies on. The hidden digest is SHA-256 over the row's float32
/// little-endian bytes -- deterministic for a given build, but see the
/// comparison caveat on `DFlashRoundWorkBinding`.
private func dflashCollectWorkBinding(
    logits: MLXArray,
    targetHidden: MLXArray,
    declaredRows: Int,
    draftTokens: [Int]
) -> DFlashRoundWorkBinding {
    var digests = [String]()
    var top2Tokens = [[Int]]()
    var top2Logits = [[Double]]()
    digests.reserveCapacity(declaredRows)
    top2Tokens.reserveCapacity(declaredRows)
    top2Logits.reserveCapacity(declaredRows)

    let hiddenRowCount = targetHidden.ndim >= 2 ? targetHidden.dim(-2) : 0
    for row in 0 ..< declaredRows {
        let logitRow = logits[0, row, 0...]
        let limit = Swift.max(1, Swift.min(2, logitRow.dim(-1)))
        let indices = argPartition(-logitRow, kth: limit - 1, axis: -1)[0 ..< limit]
        let scores = logitRow[indices]
        eval(indices, scores)
        let ids = indices.asArray(Int32.self).map { Int($0) }
        let values = scores.asArray(Float.self).map { Double($0) }
        let ordered = zip(ids, values).sorted { $0.1 > $1.1 }
        top2Tokens.append(ordered.map(\.0))
        top2Logits.append(ordered.map(\.1))

        if row < hiddenRowCount {
            let hiddenRow = targetHidden[0, row, 0...].asType(.float32)
            eval(hiddenRow)
            digests.append(dflashFloatRowDigest(hiddenRow.asArray(Float.self)))
        } else {
            digests.append("")
        }
    }

    return DFlashRoundWorkBinding(
        declaredRows: declaredRows,
        hiddenDigest: digests,
        top2Tokens: top2Tokens,
        top2Logits: top2Logits,
        draftTokens: draftTokens
    )
}

/// SHA-256 over the little-endian float32 bit patterns of `row`.
private func dflashFloatRowDigest(_ row: [Float]) -> String {
    var bytes = [UInt8]()
    bytes.reserveCapacity(row.count * 4)
    for value in row {
        // Normalize the two NaN encodings and -0.0 so a benign signed-zero
        // difference cannot change the digest.
        let normalized = value == 0 ? 0 : value
        let pattern = normalized.isNaN ? Float.nan.bitPattern : normalized.bitPattern
        bytes.append(UInt8(truncatingIfNeeded: pattern))
        bytes.append(UInt8(truncatingIfNeeded: pattern >> 8))
        bytes.append(UInt8(truncatingIfNeeded: pattern >> 16))
        bytes.append(UInt8(truncatingIfNeeded: pattern >> 24))
    }
    return SHA256.hash(data: Data(bytes))
        .map { String(format: "%02x", $0) }
        .joined()
}

@inline(__always)
private func dflashTimingStart(_ accumulator: DFlashPhaseAccumulator?) -> Date? {
    accumulator == nil ? nil : Date()
}

@inline(__always)
private func dflashRecord(
    _ start: Date?,
    into accumulator: DFlashPhaseAccumulator?,
    _ update: (DFlashPhaseAccumulator, Double) -> Void
) {
    guard let start, let accumulator else { return }
    update(accumulator, Date().timeIntervalSince(start))
}
