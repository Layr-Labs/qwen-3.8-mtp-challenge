import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXSpeculative

/// One block-decode round as the trusted parent sees it.
///
/// `tokens` is the target-confirmed block: the accepted draft prefix followed by
/// one target token (so `1...maxBlockSize` entries). `targetCacheOffset` is the
/// physical KV position after the round, which the worker cross-checks against
/// its own logical ledger before answering.
public struct LagunaDFlashBlockResult {
    public let tokens: [Int]
    public let targetCacheOffset: Int
    public let acceptedDraftCount: Int
    public let rejectedDraftCount: Int
    public let declaredRows: Int
    public let perRowHiddenDigest: [String]
    public let perRowTop2Tokens: [[Int]]
    public let perRowTop2Logits: [[Double]]
    /// The drafter's `declaredRows - 1` proposals, in verify-input order, so the
    /// parent can reconstruct the round's actual verify block
    /// (`[bonus] + draftTokens`) and have the reference price the rejected tail.
    public let draftTokens: [Int]
}

/// Re-entrant DFlash block-decode session for the `laguna-xs-2.1-dflash-v1`
/// track's runtime worker.
///
/// The vendored DFlash entry points (`DFlashTokenIterator`,
/// `generateDFlashTokens`) are one-shot: they own the whole generation loop and
/// return when it finishes. The ranked protocol is the opposite shape -- the
/// trusted parent drives one `dflash_decode_block` request at a time, chooses
/// each round's width, and owns the token budget and the timer. This session is
/// the adapter: it holds the target/draft caches, the current bonus token and
/// captured target hidden states across requests, and exposes exactly two
/// operations (`begin`, `generateBlock`).
///
/// Everything numeric is delegated to `runDFlashGreedyRound`, which is the same
/// round the validated `mlx-bench dflash` path runs, so the ranked worker and
/// the local bench cannot drift apart.
public final class LagunaDFlashBlockSession {
    private let target: any DFlashTargetModel
    private let drafter: DFlashDraftModel
    private var targetCache: [KVCache]
    private let draftCache: [KVCache]

    private var bonus: Int?
    private var targetHidden: MLXArray?

    /// Seed prompt length; the physical KV offset must stay `seedTokenCount +
    /// decodedTokenCount` for the whole session.
    public private(set) var seedTokenCount = 0
    public private(set) var decodedTokenCount = 0
    public private(set) var rollbackRoundCount = 0
    public private(set) var acceptedDraftTotal = 0
    public private(set) var rejectedDraftTotal = 0
    public private(set) var roundCount = 0

    /// Every round emits exactly one target-produced tail token (the accepted
    /// draft prefix plus one). The L3 ledger the box wrapper checks is
    /// `accepted + rejected + target_tail == declared_rows`, so the tail total
    /// is the round count by construction.
    public var targetTailTotal: Int { roundCount }

    /// Reference-side accessors, used ONLY by the reference worker built from
    /// the pinned baseline tree when it serves `dflash_reference_rows`. The
    /// candidate never reaches this kind.
    public var referenceTarget: any DFlashTargetModel { target }
    public var referenceTargetLayerIds: [Int] { drafter.config.targetLayerIds }

    /// When true every round also produces the Criterion E work-binding
    /// readouts. This costs a full-logits verify forward instead of the greedy
    /// fast path, which is intentional: the ranked contract requires the per-row
    /// lm_head to have actually run.
    private let collectWorkBinding: Bool

    public init(
        target: any DFlashTargetModel,
        drafter: DFlashDraftModel,
        collectWorkBinding: Bool = true
    ) throws {
        self.target = target
        self.drafter = drafter
        self.collectWorkBinding = collectWorkBinding
        self.targetCache = target.newCache(parameters: nil)
        self.draftCache = try drafter.makeCache()
        // The round asks the draft cache to trim only when the drafter's
        // position counter has run PAST the committed offset. Correctly aligned
        // block decode never needs that trim -- and must not, because the
        // drafter's rotating window saturates on any seed at or above 511 rows
        // and a saturated ring cannot rewind. This guard therefore only catches
        // a cache kind that could not trim even at offset 0.
        guard canTrimPromptCache(self.draftCache) else {
            throw MLXFastError.invalidInput(
                "DFlash draft cache is not trimmable; block decode cannot "
                    + "keep the drafter aligned with the target"
            )
        }
    }

    /// Untimed, input-independent warm of the block-decode graph shapes.
    ///
    /// Called after the trusted allocator clear and before the first timed
    /// round so the scored window does not pay a one-time first-touch spike.
    /// Runs on throwaway cache state -- never on the scored seed.
    public func warmWorkingSetAfterAllocatorReset() throws {
        let warmupSession = try LagunaDFlashBlockSession(
            target: target,
            drafter: drafter,
            collectWorkBinding: collectWorkBinding
        )
        try warmupSession.warmAllBlockWidths()
    }

    /// The warm itself, run on THIS session's own throwaway cache state -- so the
    /// caller must have built a session it is willing to discard.
    ///
    /// Both warm points (the worker's pre-hello warm and the post-allocator-reset
    /// re-warm) go through here. They used to warm different things, which is
    /// exactly the drift this exists to prevent.
    public func warmAllBlockWidths() throws {
        // Deliberately PAST the sliding-window ring. This used to be deliberately
        // short, to avoid a warmup tripping the wrap seam; with the seam fixed on
        // both the target side (snapshot rollback) and the drafter side (Amendment
        // 7) that rationale is inverted. A short warmup leaves the SATURATED ring
        // shapes -- the wrapped attention mask, the context-truncating drafter
        // path, the temporal-order rebuild -- to compile inside the timed window,
        // and the ranked seed is 512, so the scored run hits all of them on its
        // first round. The seed is the window plus one widest block so every one
        // of those shapes is compiled here instead.
        let seed = try begin(
            seedTokens: Array(
                repeating: 2,
                count: MLXFastConstants.experimentalDFlashWarmupSeedTokens
            )
        )
        // Warm EVERY legal width, not just the widest. The paired score divides a
        // width-K run by a width-1 run, so a shape compiled inside one side's
        // timed window and not the other's is a bias in the ratio itself, not just
        // noise. Widest first and width 1 last: a width-1 round advances the
        // target without feeding the drafter, so it must never precede a block.
        var committed = seed
        for width in stride(
            from: MLXFastConstants.experimentalDFlashMaxBlockSize, through: 2, by: -1
        ) {
            let round = try generateBlock(
                previousCommittedToken: committed,
                maxBlockSize: width
            )
            committed = round.tokens.last ?? committed
        }
        _ = try generateBlock(
            previousCommittedToken: committed,
            maxBlockSize: 1
        )
    }

    /// Seed prefill. Returns the post-prefill argmax ("bonus") token, which is
    /// the first emitted token of the run and the `previousCommittedToken` the
    /// first block request must echo back.
    public func begin(seedTokens: [Int]) throws -> Int {
        guard bonus == nil else {
            throw MLXFastError.invalidInput(
                "DFlash block session was already begun"
            )
        }
        guard !seedTokens.isEmpty else {
            throw MLXFastError.invalidInput(
                "DFlash block session requires a non-empty seed"
            )
        }
        guard seedTokens.allSatisfy({ $0 >= 0 && $0 < MLXFastConstants.vocabSize })
        else {
            throw MLXFastError.invalidInput(
                "DFlash seed contains an out-of-vocabulary token"
            )
        }

        let promptTokens = MLXArray(seedTokens.map { Int32($0) })[.newAxis, .ellipsis]
        let prefill = try target.forwardGreedyTokensForDFlash(
            promptTokens,
            cache: targetCache,
            targetLayerIds: drafter.config.targetLayerIds
        )
        let bonusArray = prefill.tokens[0..., -1]
        eval(bonusArray, prefill.targetHidden)
        let seedToken = Int(bonusArray.item(Int32.self))
        guard seedToken >= 0, seedToken < MLXFastConstants.vocabSize else {
            throw MLXFastError.invalidInput(
                "DFlash seed prefill produced an out-of-vocabulary token"
            )
        }

        bonus = seedToken
        targetHidden = prefill.targetHidden
        seedTokenCount = seedTokens.count
        decodedTokenCount = 0
        return seedToken
    }

    /// One target-verified block. `previousCommittedToken` must be the last
    /// token this session emitted -- the parent echoing it back is what proves
    /// the two sides agree on the committed prefix.
    public func generateBlock(
        previousCommittedToken: Int,
        maxBlockSize: Int
    ) throws -> LagunaDFlashBlockResult {
        guard let currentBonus = bonus, let currentHidden = targetHidden else {
            throw MLXFastError.invalidInput(
                "DFlash block requested before begin"
            )
        }
        guard previousCommittedToken == currentBonus else {
            throw MLXFastError.invalidInput(
                "DFlash block request echoed token \(previousCommittedToken) "
                    + "but the session's committed token is \(currentBonus)"
            )
        }
        guard maxBlockSize >= 1,
              maxBlockSize <= MLXFastConstants.experimentalDFlashMaxBlockSize
        else {
            throw MLXFastError.invalidInput(
                "DFlash block size \(maxBlockSize) is outside 1..."
                    + "\(MLXFastConstants.experimentalDFlashMaxBlockSize)"
            )
        }

        // K=1 is the SERIAL CONTROL the paired score divides by. It runs the
        // same worker, the same protocol and the same target forward as a real
        // block round -- just one row and zero drafts -- so the denominator
        // measures this implementation at width 1 rather than some other
        // code path. `runDFlashGreedyRound` needs at least one draft, so the
        // single-row case is handled here.
        if maxBlockSize == 1 {
            return try generateSerialRow(previousCommittedToken: currentBonus)
        }

        // The drafter re-supplies the target hidden states as cross-attention
        // context every round and derives its RoPE positions from
        // `cache.offset + contextLength`. So the context width handed to a round
        // must be exactly the number of positions the target advanced since the
        // drafter last wrote: seed width on the first block, then the previous
        // round's committed row count. A width-1 serial row in the middle of a
        // session advances the target without feeding the drafter, which would
        // leave every later draft RoPE'd at the wrong absolute position --
        // silently, since the target still verifies every emitted token. Refuse
        // instead of drafting from a desynchronized prefix.
        let drafterOffset = draftCache.first?.offset ?? 0
        let drafterContextGap = (seedTokenCount + decodedTokenCount) - drafterOffset
        guard currentHidden.dim(1) == drafterContextGap else {
            throw MLXFastError.invalidInput(
                "DFlash drafter context is \(currentHidden.dim(1)) rows wide but "
                    + "the target advanced \(drafterContextGap) positions since "
                    + "the drafter last wrote; block decode cannot keep the "
                    + "drafter aligned"
            )
        }

        let result = try runDFlashGreedyRound(
            target: target,
            drafter: drafter,
            targetCache: &targetCache,
            draftCache: draftCache,
            bonus: currentBonus,
            targetHidden: currentHidden,
            promptTokenCount: seedTokenCount,
            // `generatedTokenCount` counts EMITTED tokens, which includes the
            // seed prefill's bonus token; the round derives the drafter's
            // committed position as `prompt + generated - 1` and that has to
            // equal the target's KV offset (`seed + decoded`). `decodedTokenCount`
            // counts KV ROWS, one fewer, because the bonus token's row is written
            // by the round that consumes it. Passing the row count directly asked
            // the round to trim one row off the drafter's rotating cache, which a
            // saturated ring refuses -- so every block round threw
            // `untrimmableCache` once the seed reached the drafter's 511-slot
            // window. See docs/dflash-track-correctness-contract.md Amendment 7.
            generatedTokenCount: decodedTokenCount + 1,
            blockSize: maxBlockSize,
            maxEmitCount: maxBlockSize,
            workBinding: collectWorkBinding
        )

        guard !result.tokens.isEmpty, result.tokens.count <= maxBlockSize else {
            throw MLXFastError.invalidInput(
                "DFlash round emitted \(result.tokens.count) tokens for a "
                    + "block of \(maxBlockSize)"
            )
        }
        guard result.tokens.allSatisfy({
            $0 >= 0 && $0 < MLXFastConstants.vocabSize
        }) else {
            throw MLXFastError.invalidInput(
                "DFlash round emitted an out-of-vocabulary token"
            )
        }

        bonus = result.bonus
        targetHidden = result.targetHidden
        decodedTokenCount += result.tokens.count

        // Ledger: physical KV position must equal the logical one. A submission
        // that leaves rejected rows resident (skipping rollback to save time)
        // trips here before its tokens are ever scored.
        let observedOffset = targetCache.first?.offset ?? -1
        let expectedOffset = seedTokenCount + decodedTokenCount
        guard observedOffset == expectedOffset else {
            throw MLXFastError.invalidInput(
                "DFlash target cache offset \(observedOffset) diverged from the "
                    + "logical position \(expectedOffset)"
            )
        }

        let rejected = Swift.max(0, (maxBlockSize - 1) - result.accepted)
        if rejected > 0 {
            rollbackRoundCount += 1
        }
        acceptedDraftTotal += result.accepted
        rejectedDraftTotal += rejected
        roundCount += 1

        let binding = result.workBinding
        return LagunaDFlashBlockResult(
            tokens: result.tokens,
            targetCacheOffset: observedOffset,
            acceptedDraftCount: result.accepted,
            rejectedDraftCount: rejected,
            declaredRows: binding?.declaredRows ?? 0,
            perRowHiddenDigest: binding?.hiddenDigest ?? [],
            perRowTop2Tokens: binding?.top2Tokens ?? [],
            perRowTop2Logits: binding?.top2Logits ?? [],
            draftTokens: binding?.draftTokens ?? []
        )
    }

    /// One width-1 target row: the serial control. Zero drafts, one declared
    /// row, nothing to roll back.
    ///
    /// The full-logits forward is deliberate even though only the argmax is
    /// needed: the control has to produce the same per-row top-2 readouts the
    /// block path produces, or the parent could not compare the two sides
    /// against the same reference.
    private func generateSerialRow(
        previousCommittedToken: Int
    ) throws -> LagunaDFlashBlockResult {
        let input = MLXArray([Int32(previousCommittedToken)])[.newAxis, .ellipsis]
        let out = try target.forwardForDFlash(
            input,
            cache: targetCache,
            targetLayerIds: drafter.config.targetLayerIds
        )
        let logitRow = out.logits[0, -1, 0...]
        let (top2Tokens, top2Logits) = LagunaDFlashReference.topTwo(of: logitRow)
        eval(out.targetHidden)
        guard let next = top2Tokens.first,
              next >= 0,
              next < MLXFastConstants.vocabSize
        else {
            throw MLXFastError.invalidInput(
                "DFlash serial row produced no in-vocabulary token"
            )
        }

        bonus = next
        targetHidden = out.targetHidden
        decodedTokenCount += 1
        roundCount += 1

        let observedOffset = targetCache.first?.offset ?? -1
        let expectedOffset = seedTokenCount + decodedTokenCount
        guard observedOffset == expectedOffset else {
            throw MLXFastError.invalidInput(
                "DFlash serial row target cache offset \(observedOffset) "
                    + "diverged from the logical position \(expectedOffset)"
            )
        }

        return LagunaDFlashBlockResult(
            tokens: [next],
            targetCacheOffset: observedOffset,
            acceptedDraftCount: 0,
            rejectedDraftCount: 0,
            declaredRows: 1,
            // Amendment 1: hidden digests are a SELF-consistency instrument
            // only, never compared across builds, so the control does not pay
            // to materialize one.
            perRowHiddenDigest: [""],
            perRowTop2Tokens: [top2Tokens],
            perRowTop2Logits: [top2Logits],
            // One declared row, zero drafts: `declaredRows - 1 == 0`, so the
            // serial control satisfies the parent's length bind trivially and
            // has no rejected tail to price.
            draftTokens: []
        )
    }
}
