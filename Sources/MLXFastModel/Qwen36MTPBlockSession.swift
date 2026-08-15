import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon

// Qwen 3.6 27B native-MTP speculative decode — the worker-side hot path for the
// `qwen3.8-27b-mtp-v1` track.
//
// PROVENANCE. The accept/verify/rollback loop below is a migration of the MTP
// session's exploratory driver (`Sources/Qwen36MTPDriver/main.swift`), which is
// itself a faithful Swift port of MTPLX's `generate_mtpa`
// (MTPLX/mtplx/generation.py L10176-10420). That driver was validated 12/12
// exact-greedy to 512 tokens against the serial trajectory on M5, across all EOS
// branches, and corroborated against MTPLX. Nothing about the ALGORITHM changed
// in the migration; what changed is where it runs (the sandboxed runtime worker
// instead of a standalone target), how the head arrives (a separately pinned
// tree merged at load instead of a merged checkpoint) and that every round now
// declares an auditable row ledger to the trusted parent.
//
// Per round:
//   1. emit the pending primary; clamp the depth to the remaining token budget
//   2. draft `cycleDepth` tokens from the head (ONE fresh head cache per round,
//      shared across the sub-steps; each sub-step chains the head's own
//      post-`mtp.norm` hidden — MTPLX `mtp_cache_policy` default "persistent")
//   3. snapshot the non-trimmable (GDN/recurrent) state, then verify
//      `[primary] + drafts` in ONE batched target forward
//   4. accept the longest common prefix: row i of the verify output is the
//      target's greedy continuation of verify input i, i.e. the truth for draft i
//   5. full acceptance -> keep the verify state; the next primary is the argmax
//      of the bonus row D. Otherwise -> roll the WHOLE verify window back (trim
//      all 1+D positions from the trimmable caches AND restore the recurrent
//      snapshot) and re-forward the committed block `[primary] + acceptedDrafts`;
//      its last row is the next primary and its last hidden feeds the next draft.
//
// WHY NOT THE VENDORED DFLASH ROLLBACK. `RecurrentRollbackCache` rolls the GDN
// state forward by replaying an innovation tape the GDN forward is supposed to
// hand it via `recordTape()`. Nothing in the vendored code ever calls
// `recordTape`, so the tape is always nil and the cache silently degenerates to a
// pre-verify snapshot restore while the KV caches are trimmed to
// prefix+1+accepted — the 48 recurrent layers and the 16 attention layers desync
// on every partial acceptance. MTPLX's snapshot + rollback + re-forward needs no
// tape, which is why it is the baseline here. Grafting the tape into the Qwen35
// GDN forward is a documented LATER perf upgrade, deliberately not attempted.

/// One round's worth of committed tokens plus the row ledger the trusted parent
/// audits. Field names mirror the DFlash round result so the parent-side ledger
/// arithmetic and the box wrapper's Criterion E L3 checks are the same shape on
/// both speculative tracks.
public struct Qwen36MTPRoundResult {
    /// `[primary] + acceptedDrafts` — the tokens this round commits.
    public let tokens: [Int]
    /// `cycleDepth + 1`: one row per draft the head proposed, plus the single
    /// target tail row whose argmax becomes the next round's primary.
    public let declaredRows: Int
    /// The head's `cycleDepth` proposals, in verify-input order, so the parent
    /// can reconstruct this round's actual verify block (`[primary] + drafts`)
    /// and have the pinned reference price the rejected tail.
    public let draftTokens: [Int]
    public let acceptedDraftCount: Int
    public let rejectedDraftCount: Int
    /// `declaredRows` rows of top-2 readouts. Rows `0 ..< cycleDepth` are the
    /// verify rows that scored the drafts; the last row is the tail row.
    public let perRowTop2Tokens: [[Int]]
    public let perRowTop2Logits: [[Double]]
    /// Trimmable-cache offset after the round: `seedTokenCount + committedTotal`.
    public let targetCacheOffset: Int
    /// True when a stop token was committed this round; the parent stops asking.
    public let reachedStopToken: Bool
}

/// Errors the session raises. Every one of these is a broken invariant, not a
/// recoverable condition: the worker poisons its session on any of them.
public enum Qwen36MTPSessionError: Error, CustomStringConvertible {
    case headNotAttached
    case cacheOffsetInvariant(expected: Int, actual: Int, round: Int)
    case notBegun
    case alreadyBegun
    case invalidDepth(Int)
    case emptySeed

    public var description: String {
        switch self {
        case .headNotAttached:
            return "the Qwen 3.6 MTP head is not attached to the loaded backbone"
        case .cacheOffsetInvariant(let expected, let actual, let round):
            return "MTP cache offset invariant broken at round \(round): "
                + "trimmable offset \(actual) != seed+emitted \(expected)"
        case .notBegun:
            return "MTP round requested before the seed prefill"
        case .alreadyBegun:
            return "MTP seed prefill requested twice"
        case .invalidDepth(let depth):
            return "MTP draft depth \(depth) is out of range"
        case .emptySeed:
            return "MTP seed prefill requires a non-empty seed"
        }
    }
}

/// Native-MTP speculative decode session over one loaded Qwen 3.6 backbone with
/// its pinned MTP head attached.
///
/// Depth 1 is the SERIAL CONTROL and is served by this same class, this same
/// worker and this same forward: one draft, one verify, the accept walk. It is
/// deliberately not a second code path — the retired Gemma track ran its serial
/// side through a different verb, which put any divergence between the two paths
/// straight into the score.
public final class Qwen36MTPBlockSession {
    private let model: any Qwen36MTPTarget
    private let stopTokens: Set<Int>
    /// MTPLX default `base_hidden_variant == mtp_hidden_variant == "post_norm"`.
    private let postNorm: Bool

    private var cache: [any KVCache] = []
    /// Logits of the row that produces the next primary. One row, `[1, 1, V]`.
    private var pendingLogitsRow: MLXArray?
    /// Exact target top-1 ID for `pendingLogitsRow` when that row came from the
    /// already-materialized speculative top-2 readout. Serial/adaptive K=0 keeps
    /// using `argmaxLast` so the control path remains unchanged.
    private var pendingPrimaryID: Int?
    /// The (post-norm) trunk hidden that seeds the next draft round.
    private var pendingHidden: MLXArray?

    public private(set) var seedTokenCount = 0
    public private(set) var committedTokenCount = 0
    public private(set) var roundCount = 0
    public private(set) var acceptedDraftTotal = 0
    public private(set) var rejectedDraftTotal = 0
    public private(set) var rollbackRoundCount = 0
    public private(set) var began = false
    public private(set) var reachedStopToken = false

    public init(
        model: any Qwen36MTPTarget,
        stopTokens: Set<Int>,
        postNorm: Bool = true
    ) throws {
        guard model.hasMTPHead else { throw Qwen36MTPSessionError.headNotAttached }
        self.model = model
        self.stopTokens = stopTokens
        self.postNorm = postNorm
    }

    // MARK: - warm

    /// Input-independent shape warm, run OUTSIDE every scored window.
    ///
    /// Warms the two forward shapes a round dispatches — the batched verify at
    /// every legal width `1 ... maxDepth + 1`, and the head's single-token draft
    /// step — on throwaway cache state. Nothing here sees a seed.
    public func warmAllDepths(maxDepth: Int) throws {
        // Warms every legal verify width from 1 (the serial control's
        // single-token forward) up to maxDepth + 1, plus the head's draft step.
        // The head warm runs even for a serial-only session: the head is resident
        // on both sides, so warming it on both keeps the load shape identical.
        guard maxDepth >= 1, maxDepth <= Qwen36MTPLimits.maxDepth else {
            throw Qwen36MTPSessionError.invalidDepth(maxDepth)
        }
        let warmCache = model.newCache(parameters: nil)
        let seed = Array(repeating: 0, count: 8)
        let (logits, hidden) = model.callWithHidden(
            input: LMInput.Text(tokens: MLXArray(seed).reshaped([1, seed.count])),
            cache: warmCache, nConfirmed: 0)
        var row = hiddenRow(hidden, hidden.dim(1) - 1)
        eval(warmCache.flatMap { $0.state })
        eval(logits, row)

        let headCache = model.makeMTPCache()
        for _ in 0 ..< maxDepth {
            let (draftLogits, draftHidden) = model.mtpForwardWithHidden(
                hidden: row,
                nextTokenIds: MLXArray([0]).reshaped([1, 1]),
                cache: headCache)
            row = draftHidden[0..., (draftHidden.dim(1) - 1) ..< draftHidden.dim(1), 0...]
            eval(draftLogits, row)
        }
        for width in 1 ... (maxDepth + 1) {
            let block = Array(repeating: 0, count: width)
            let tokens = LMInput.Text(
                tokens: MLXArray(block).reshaped([1, width]))
            // Width 2 warms whichever geometry the scored K=1 round will use, so
            // its kernels compile outside every timed window. Under the fused
            // flag that also warms the reject-path rebuild, which is a shape the
            // split path never dispatches.
            let (verifyLogits, _) = width == 2 && qwenMTPFusedAcceptVerifyEnabled
                ? model.callWithHiddenStashingPrimaryBoundary(
                    input: tokens, cache: warmCache)
                : model.callWithHidden(
                    input: tokens, cache: warmCache,
                    nConfirmed: width == 2 ? 1 : 0)
            let (top2IDs, top2Values) = Self.linearTopTwoRows(verifyLogits)
            eval(verifyLogits, top2IDs, top2Values)
            if width == 2 && qwenMTPFusedAcceptVerifyEnabled {
                _ = model.recomputeFusedPrimaryBoundary(cache: warmCache)
            }
            eval(warmCache.flatMap { $0.state })
        }
    }

    // MARK: - begin

    /// Bulk-forward the seed and return the argmax of its last row — the first
    /// primary. The primary's own KV row is deliberately NOT written yet: the
    /// round-top invariant is "every emitted token is in the cache and the
    /// pending primary is not", and the verify forward writes it.
    ///
    /// SEED-TAIL VOCABULARY PROJECTION (guarded by
    /// `MLXFAST_QWEN_MTP_SEED_TAIL_PROJECTION`, default on). This method is the
    /// only place in the session that bulk-forwards a long block and then keeps
    /// ONLY its tail: two lines below, 511 of the 512 `[1, V]` logit rows the
    /// baseline just computed are dropped on the floor, and nothing downstream
    /// -- not the accept walk, not the row ledger, not the caches -- can ever
    /// read them. `callWithLastTokenHidden` runs the identical backbone forward
    /// and narrows only the final-norm + LM-head epilogue to the surviving row.
    /// The seed prefill is INSIDE the timed window by contract
    /// (`QwenRuntime.qwenMTPTimedDecode` starts its clock immediately before the
    /// begin request "so the seed cost cannot be hidden outside the window"), so
    /// this is charged work, not setup.
    ///
    /// The two branches converge immediately: both hand back a `[1, S, V]` /
    /// `[1, S, H]` pair whose LAST row is what the session keeps, with `S == 1`
    /// on the narrowed path. Every line after the call is therefore shared and
    /// index-generic, which is what makes the flag a true ablation rather than
    /// two divergent prefills.
    @discardableResult
    public func begin(seedTokens: [Int]) throws -> Int {
        guard !began else { throw Qwen36MTPSessionError.alreadyBegun }
        guard !seedTokens.isEmpty else { throw Qwen36MTPSessionError.emptySeed }
        cache = model.newCache(parameters: nil)
        let seedInput = LMInput.Text(
            tokens: MLXArray(seedTokens).reshaped([1, seedTokens.count]))
        let (logits, hidden) =
            qwenMTPSeedTailProjectionEnabled
            ? model.callWithLastTokenHidden(
                input: seedInput, cache: cache, nConfirmed: 0)
            : model.callWithHidden(
                input: seedInput, cache: cache, nConfirmed: 0)
        pendingLogitsRow = logits[0..., (logits.dim(1) - 1) ..< logits.dim(1), 0...]
        pendingHidden = hiddenRow(hidden, hidden.dim(1) - 1)
        eval(cache.flatMap { $0.state })
        eval(pendingLogitsRow!, pendingHidden!)
        seedTokenCount = seedTokens.count
        committedTokenCount = 0
        began = true
        return argmaxLast(pendingLogitsRow!)
    }

    // MARK: - draft schedule (EDITABLE POLICY)

    /// How many tokens to draft this round, given the parent's offer.
    ///
    /// THE SHIPPED DEFAULT IS A CONSTANT 2, and it is a starting line rather
    /// than a recommendation: 2 is the depth this track was pinned at while
    /// depth was an operator parameter, so an unmodified tree reproduces the
    /// measured reference behaviour exactly. A submission owns this function.
    ///
    /// Contract, enforced by a precondition at the call site and re-enforced by
    /// the TRUSTED parent against `qwenMTPMaxDraftDepth`: return a value in
    /// `0 ... min(offeredDepth, Qwen36MTPLimits.maxDepth)`. Returning 0 is an
    /// adaptive skip and costs exactly what a serial step costs.
    ///
    /// `round` is this session's own 1-based round counter -- not a position in
    /// the scored window, which the worker is never told. Acceptance history is
    /// available through `acceptedDraftTotal` / `rejectedDraftTotal` /
    /// `rollbackRoundCount`.
    // OPERATOR K-TEST VARIANT, k = 1. Draft ONE token per round at whatever
    // width the parent offers. This is the only thing that changes: the verify
    // block is still `[primary] + drafts`, acceptance is still the longest
    // common prefix over the target's own argmaxes, and the snapshot / rollback
    // / re-forward repair is untouched. The emitted stream is therefore the
    // same greedy target chain at any offer, which is what keeps every width
    // bit-exact.
    //
    // Legal by the 2026-08-14 contract for the reason the doc comment above
    // states: the return value need only land in
    // `0 ... min(offeredDepth, Qwen36MTPLimits.maxDepth)`, and the trusted
    // parent derives every ledger quantity from the drafts actually proposed.
    public var draftPolicy: (_ offeredDepth: Int, _ round: Int) -> Int = {
        offeredDepth, _ in
        Swift.min(offeredDepth, 1)
    }

    /// The shipped schedule's width. See `draftPolicy`.
    public static let defaultDraftDepth = 2

    // MARK: - one round

    /// Draft up to `depth` tokens, verify `[primary] + drafts` in one batched
    /// target forward, accept the longest common prefix, and repair the caches.
    ///
    /// `depth` IS AN OFFER, NOT AN ORDER (contract change 2026-08-14). The
    /// trusted parent offers a per-round ceiling and this session decides how
    /// many tokens it actually drafts -- 0 through `Qwen36MTPLimits.maxDepth`,
    /// per round, adaptively if it likes. The parent bounds the ACTUAL count
    /// against the trusted maximum and derives every ledger quantity from it,
    /// so a narrower round, a wider round and a round that drafts nothing are
    /// all legal and all correctly accounted.
    ///
    /// The worker is still deliberately never told how much of the decode
    /// window remains, so it cannot special-case the tail; the parent clamps
    /// the scored prefix itself.
    ///
    /// THE POLICY BELOW IS THE FIRST THING A SUBMISSION SHOULD CHANGE. It is
    /// the shipped reference schedule (`draftPolicy`), and it is deliberately
    /// dumb -- a constant 2, the depth this track measured before depth became
    /// competitive. Every acceptance-aware idea starts here: draft deeper where
    /// the head has been right, draft nothing where it has been wrong, size the
    /// round from the last round's accept run.
    public func generateRound(depth: Int) throws -> Qwen36MTPRoundResult {
        guard began, let logitsRow = pendingLogitsRow, let hidden = pendingHidden
        else { throw Qwen36MTPSessionError.notBegun }
        guard depth >= Qwen36MTPLimits.serialControlDepth,
              depth <= Qwen36MTPLimits.maxDepth
        else {
            throw Qwen36MTPSessionError.invalidDepth(depth)
        }
        roundCount += 1

        // Round-top invariant, kept as a THROW rather than a comment: every
        // emitted token is in the trimmable caches and the pending primary is
        // not. A rollback that trimmed the wrong amount shows up here, one round
        // after the mistake, instead of as a silent late divergence.
        let base = trimmableOffset()
        let expected = seedTokenCount + committedTokenCount
        guard base == expected else {
            throw Qwen36MTPSessionError.cacheOffsetInvariant(
                expected: expected, actual: base, round: roundCount)
        }

        // THE DRAFT SCHEDULE. `depth` is what the parent offered; `draftCount`
        // is what this round proposes, and from here down it is the only width
        // that matters -- the draft loop, the declared row count, the per-row
        // readouts and the rollback all key off it, so a policy change needs no
        // other edit to stay ledger-correct.
        let draftCount = draftPolicy(depth, roundCount)
        precondition(
            draftCount >= 0 && draftCount <= depth
                && draftCount <= Qwen36MTPLimits.maxDepth,
            "draftPolicy returned \(draftCount) for an offer of \(depth); a "
                + "round may propose 0 ... min(offer, maxDepth) drafts")

        // A speculative predecessor already materialized this row's exact top-1
        // as part of the trusted top-2 ledger. Consume it only on the ranked K=1
        // path; serial/adaptive K=0 and wider experiments retain their own argMax.
        let reusablePrimary = draftCount == 1 ? pendingPrimaryID : nil
        pendingPrimaryID = nil
        let primary = reusablePrimary ?? argmaxLast(logitsRow)
        var committed = [primary]
        committedTokenCount += 1

        // A stop token as the primary ends the run BEFORE any drafting: there is
        // nothing after it to predict, and drafting past it would charge the
        // measurement for work no decoder performs. The round still declares its
        // single target tail row (the row that produced this primary's successor
        // candidate is the one already spent), so the ledger stays closed.
        if stopTokens.contains(primary) {
            reachedStopToken = true
            let (tailTokens, tailLogits) = Self.topTwo(of: lastRow(logitsRow))
            pendingLogitsRow = nil
            pendingHidden = nil
            return Qwen36MTPRoundResult(
                tokens: committed,
                declaredRows: 1,
                draftTokens: [],
                acceptedDraftCount: 0,
                rejectedDraftCount: 0,
                perRowTop2Tokens: [tailTokens],
                perRowTop2Logits: [tailLogits],
                targetCacheOffset: seedTokenCount + committedTokenCount,
                reachedStopToken: true
            )
        }

        // NO DRAFTS THIS ROUND. Two ways to get here and they are not the same
        // thing. Depth 0 is THE TRUE SERIAL CONTROL -- the parent offered
        // nothing, the denominator this track divides by. A zero from
        // `draftPolicy` is an ADAPTIVE SKIP: the parent offered a width and this
        // round declined it. Both execute the identical one-token forward and
        // both declare the identical single tail row, which is the point --
        // an adaptive skip costs exactly what serial decode costs.
        //
        // One token per target forward: no
        // draft, no head cache, no head forward, no verify window and therefore
        // no rollback. The head stays ATTACHED and resident -- the paired
        // contract charges its residency to both sides, so the denominator must
        // carry the same memory and the same load shape -- but nothing on this
        // path reads it. That is the difference between "MTP off" and "MTP depth
        // 1", and it is the whole reason this branch exists.
        //
        // The single row this forward produces IS the round's target tail row:
        // its argmax becomes the next primary, exactly as the bonus row does on
        // the speculative path. So the ledger closes with declaredRows = 1,
        // accepted = rejected = 0, tail = 1 -- and `rows_per_round(0) = 1` in the
        // box wrapper agrees without any special case there.
        if depth == Qwen36MTPLimits.serialControlDepth || draftCount == 0 {
            let (serialLogits, serialHidden) = model.callWithHidden(
                input: LMInput.Text(
                    tokens: MLXArray([primary]).reshaped([1, 1])),
                cache: cache, nConfirmed: 0)
            pendingLogitsRow = serialLogits[
                0..., (serialLogits.dim(1) - 1) ..< serialLogits.dim(1), 0...]
            // Still produced, still post-norm: keeping the hidden chain identical
            // means switching depth is the ONLY difference between the two sides.
            pendingHidden = hiddenRow(serialHidden, serialHidden.dim(1) - 1)
            eval(cache.flatMap { $0.state })
            if let row = pendingLogitsRow, let h = pendingHidden { eval(row, h) }
            let (tailTokens, tailLogits) = Self.topTwo(of: lastRow(serialLogits))
            return Qwen36MTPRoundResult(
                tokens: committed,
                declaredRows: 1,
                draftTokens: [],
                acceptedDraftCount: 0,
                rejectedDraftCount: 0,
                perRowTop2Tokens: [tailTokens],
                perRowTop2Logits: [tailLogits],
                targetCacheOffset: seedTokenCount + committedTokenCount,
                reachedStopToken: false
            )
        }

        // 1. DRAFT. One fresh head cache per round, shared across the sub-steps;
        //    each sub-step chains the head's OWN post-`mtp.norm` hidden, never
        //    the trunk hidden again — re-feeding the trunk hidden would draft
        //    every level from the same state.
        let headCache = model.makeMTPCache()
        var drafts: [Int] = []
        var draftHidden = hidden
        var nextToken = primary
        for _ in 0 ..< draftCount {
            let (draftLogits, chained) = model.mtpForwardWithHidden(
                hidden: draftHidden,
                nextTokenIds: MLXArray([nextToken]).reshaped([1, 1]),
                cache: headCache)
            let proposal = argmaxLast(draftLogits)
            drafts.append(proposal)
            draftHidden = chained[0..., (chained.dim(1) - 1) ..< chained.dim(1), 0...]
            nextToken = proposal
        }

        // 2. Keep the generic pre-verify snapshot as a fallback, but use the
        //    vendored post-primary rollback checkpoint for the hot K=1 path. A
        //    rejected single draft can then retain the primary's target work and
        //    discard only the draft token instead of re-forwarding the primary.
        let fastK1 = draftCount == 1
        let fusedK1 = fastK1 && qwenMTPFusedAcceptVerifyEnabled
        let snapshot = Self.snapshotRecurrent(cache)
        let verifyInput = committed + drafts
        // FUSED ACCEPT PATH (default). `nConfirmed: 1` makes the gated-delta
        // stack run the two verify rows as two chunks so it can write the
        // post-primary checkpoint eagerly -- on ALL 48 recurrent layers, every
        // round, including the ~2/3 that fully accept and never read it. The
        // fused call runs them as one chunk (the geometry every submission
        // before the checkpoint used) and instead records the ingredients for
        // rebuilding that boundary with a single recurrence step, paid for only
        // on the rounds that reject. Both calls run the same rows through the
        // same epilogue -- final norm and the vocabulary projection over EVERY
        // row in one matmul -- and the acceptance walk below is untouched.
        // (Narrowing the head to row 0 and projecting the bonus row separately
        // afterwards turns one ~715 MB weight stream into two on the accept
        // path; that is a regression, and is deliberately not done here.)
        let verifyTokens = LMInput.Text(
            tokens: MLXArray(verifyInput).reshaped([1, verifyInput.count]))
        let (verifyLogits, verifyHidden) = fusedK1
            ? model.callWithHiddenStashingPrimaryBoundary(
                input: verifyTokens, cache: cache)
            : model.callWithHidden(
                input: verifyTokens, cache: cache, nConfirmed: fastK1 ? 1 : 0)
        let (top2IDs, top2Values) = Self.linearTopTwoRows(verifyLogits)
        eval(top2IDs, top2Values)
        let flatTop2IDs = top2IDs.asArray(Int32.self).map { Int($0) }
        let flatTop2Values = top2Values.asArray(Float.self).map { Double($0) }
        // The reducer's first ID uses the target argmax ordering: larger logit,
        // then lower token ID on an exact tie. Reuse it for acceptance instead of
        // launching a second vocabulary-wide argMax over the same rows.
        let verifyArgmax = stride(from: 0, to: flatTop2IDs.count, by: 2).map {
            flatTop2IDs[$0]
        }

        // 3. Longest-common-prefix acceptance over rows 0 ..< draftCount. Row i
        //    is the target's greedy continuation of verify input i, i.e. the
        //    truth for draft i. Row `draftCount` is the BONUS row and is only
        //    used on full acceptance.
        var acceptedCount = 0
        for index in 0 ..< drafts.count {
            guard verifyArgmax[index] == drafts[index] else { break }
            acceptedCount += 1
            if stopTokens.contains(drafts[index]) { break }
        }

        var perRowTop2Tokens: [[Int]] = []
        var perRowTop2Logits: [[Double]] = []
        perRowTop2Tokens.reserveCapacity(draftCount + 1)
        perRowTop2Logits.reserveCapacity(draftCount + 1)
        for index in 0 ..< draftCount {
            let base = index * 2
            perRowTop2Tokens.append(Array(flatTop2IDs[base ..< (base + 2)]))
            perRowTop2Logits.append(Array(flatTop2Values[base ..< (base + 2)]))
        }

        if acceptedCount == drafts.count {
            // FULL ACCEPTANCE: the verify state IS the committed state. No
            // rollback, no repair forward; the bonus row carries the next primary
            // and the last hidden row seeds the next draft.
            Self.clearRecurrentRollback(cache)
            committed.append(contentsOf: drafts)
            committedTokenCount += drafts.count
            let bonus = verifyLogits[
                0..., drafts.count ..< (drafts.count + 1), 0...]
            pendingLogitsRow = bonus
            pendingHidden = hiddenRow(verifyHidden, verifyHidden.dim(1) - 1)
            let base = drafts.count * 2
            pendingPrimaryID = flatTop2IDs[base]
            perRowTop2Tokens.append(Array(flatTop2IDs[base ..< (base + 2)]))
            perRowTop2Logits.append(Array(flatTop2Values[base ..< (base + 2)]))
        } else {
            rollbackRoundCount += 1
            committed.append(contentsOf: drafts.prefix(acceptedCount))
            committedTokenCount += acceptedCount

            // K=1 rejection: the target already computed the primary's exact
            // logits and hidden row. Restore the recurrent checkpoint written
            // immediately after that primary, trim just the rejected draft from
            // attention caches, and carry row 0 forward. The trusted tail row is
            // the same post-primary distribution, so reuse its already-recorded
            // top-2 evidence rather than running the target again.
            let committedOffset = base + committed.count
            if fusedK1 && Self.restoreAfterFusedSingleDraftReject(
                model, cache, to: committedOffset)
            {
                pendingLogitsRow = verifyLogits[
                    0..., acceptedCount ..< (acceptedCount + 1), 0...]
                pendingHidden = hiddenRow(verifyHidden, acceptedCount)
                pendingPrimaryID = flatTop2IDs[acceptedCount * 2]
                perRowTop2Tokens.append(perRowTop2Tokens[acceptedCount])
                perRowTop2Logits.append(perRowTop2Logits[acceptedCount])
            } else if fastK1 && Self.restoreAfterSingleDraftReject(
                cache, to: committedOffset)
            {
                pendingLogitsRow = verifyLogits[
                    0..., acceptedCount ..< (acceptedCount + 1), 0...]
                pendingHidden = hiddenRow(verifyHidden, acceptedCount)
                pendingPrimaryID = flatTop2IDs[acceptedCount * 2]
                perRowTop2Tokens.append(perRowTop2Tokens[acceptedCount])
                perRowTop2Logits.append(perRowTop2Logits[acceptedCount])
            } else {
                // Generic K>1 / defensive fallback: undo the whole verify window
                // and re-forward the committed block.
                Self.rollbackAfterVerify(
                    cache, snapshot, verifiedTokens: verifyInput.count, to: base)
                let (repairLogits, repairHidden) = model.callWithHidden(
                    input: LMInput.Text(
                        tokens: MLXArray(committed).reshaped([1, committed.count])),
                    cache: cache, nConfirmed: 0)
                pendingLogitsRow = repairLogits[
                    0..., (repairLogits.dim(1) - 1) ..< repairLogits.dim(1), 0...]
                pendingHidden = hiddenRow(repairHidden, repairHidden.dim(1) - 1)
                pendingPrimaryID = nil
                let (ids, values) = Self.topTwo(of: lastRow(repairLogits))
                perRowTop2Tokens.append(ids)
                perRowTop2Logits.append(values)
            }
        }

        acceptedDraftTotal += acceptedCount
        rejectedDraftTotal += drafts.count - acceptedCount
        eval(cache.flatMap { $0.state })
        if let row = pendingLogitsRow, let h = pendingHidden { eval(row, h) }

        // Truncate after the first committed stop token, keeping the stop token
        // itself — the same rule the serial reference applies.
        if let stopIndex = committed.firstIndex(where: { stopTokens.contains($0) }) {
            let dropped = committed.count - (stopIndex + 1)
            committed = Array(committed.prefix(stopIndex + 1))
            committedTokenCount -= dropped
            reachedStopToken = true
            pendingPrimaryID = nil
        }

        return Qwen36MTPRoundResult(
            tokens: committed,
            declaredRows: draftCount + 1,
            draftTokens: drafts,
            acceptedDraftCount: acceptedCount,
            rejectedDraftCount: drafts.count - acceptedCount,
            perRowTop2Tokens: perRowTop2Tokens,
            perRowTop2Logits: perRowTop2Logits,
            targetCacheOffset: seedTokenCount + committedTokenCount,
            reachedStopToken: reachedStopToken
        )
    }

    // MARK: - cache snapshot / rollback (MTPLX cache_state.py)

    /// `snapshot_untrimmable_cache`: capture the recurrent (GDN) layers' state.
    ///
    /// EVERY LEAF IS A FRESH SLICE EXPRESSION (`[.ellipsis]`), NOT A BARE
    /// REFERENCE, AND THAT IS LOAD-BEARING. `MLXArray` is a reference type and
    /// subscript-assignment mutates it IN PLACE, so a bare-reference snapshot is
    /// only safe as long as the GDN forward happens to REBIND its cache slots
    /// rather than setitem-mutate them. Today's `Qwen35GatedDeltaNet` does rebind,
    /// but nothing pins it to — an optimization that switched to in-place writes
    /// would silently rewrite the snapshot from under the rollback and produce
    /// late, rare divergence with no failing assertion anywhere. A slice
    /// expression references the array's value at capture time, so neither writer
    /// can reach it. No GPU work happens here; this is MTPLX's `_lazy_state_view`
    /// (cache_state.py:3442-3454, `value[...]`), and the same idiom the fork's own
    /// `ArraysCache.copy()` uses.
    ///
    /// See `Qwen36MTPRollbackContractTests` for the synthetic-cache regression
    /// that fails against a bare-reference snapshot.
    public static func snapshotRecurrent(_ cache: [any KVCache]) -> [Int: [MLXArray?]] {
        var snapshot: [Int: [MLXArray?]] = [:]
        for (index, entry) in cache.enumerated() {
            guard let arrays = entry as? ArraysCache else { continue }
            snapshot[index] = [arrays[0]?[.ellipsis], arrays[1]?[.ellipsis]]
        }
        return snapshot
    }

    /// `rollback_after_verify`: trim every verified position from the trimmable
    /// (KV) caches and restore the recurrent snapshot.
    ///
    /// `trim()` is NEVER used to roll a recurrent cache back: on `ArraysCache` it
    /// only decrements `offset` and leaves the SSM/conv state exactly where the
    /// verify forward left it. The state has to be restored from the snapshot,
    /// which is why the snapshot exists.
    public static func rollbackAfterVerify(
        _ cache: [any KVCache],
        _ snapshot: [Int: [MLXArray?]],
        verifiedTokens: Int,
        to base: Int
    ) {
        for (index, entry) in cache.enumerated() {
            if let arrays = entry as? ArraysCache {
                if let saved = snapshot[index] {
                    arrays[0] = saved[0]
                    arrays[1] = saved[1]
                }
                // The vendored depth-1 rollback snapshot, if the GDN forward ever
                // wrote one, describes a frame this rollback just discarded. So
                // does a fused verify's lazy boundary stash.
                arrays.rollbackState = nil
                arrays.fusedPrimaryBoundary = nil
                continue
            }
            if entry.isTrimmable, entry.offset > base {
                _ = entry.trim(Swift.min(verifiedTokens, entry.offset - base))
            }
        }
    }

    /// Restore the checkpoint produced by a two-token verify with
    /// `nConfirmed == 1`. The checkpoint is the recurrent state immediately
    /// after the primary; each attention cache is exactly one rejected draft
    /// token ahead of that same committed offset.
    ///
    /// Preflight every layer before mutating any of them. Returning `false`
    /// leaves the cache untouched so the caller can use the generic snapshot and
    /// repair path safely.
    private static func restoreAfterSingleDraftReject(
        _ cache: [any KVCache],
        to committedOffset: Int
    ) -> Bool {
        for entry in cache {
            if let arrays = entry as? ArraysCache {
                guard arrays.rollbackState != nil else { return false }
            } else if entry.isTrimmable {
                guard entry.offset == committedOffset + 1 else { return false }
            } else {
                return false
            }
        }

        for entry in cache {
            if let arrays = entry as? ArraysCache,
               let saved = arrays.rollbackState
            {
                arrays[0] = saved.0
                arrays[1] = saved.1
                arrays.rollbackState = nil
            } else if entry.isTrimmable {
                _ = entry.trim(entry.offset - committedOffset)
            }
        }
        return true
    }

    /// FUSED-VERIFY counterpart of `restoreAfterSingleDraftReject`.
    ///
    /// The fused verify wrote no checkpoint, so the post-primary recurrent state
    /// is rebuilt from the stash each gated-delta layer recorded: the
    /// convolution half is a slice already taken at forward time, and the SSM
    /// half is one `T == 1` recurrence step over the row-0 tensors that forward
    /// had already computed. The attention side is unchanged -- exactly one
    /// rejected position to trim, same as the checkpoint path.
    ///
    /// Fail-closed in the same order as the path it replaces: the attention
    /// preflight runs first, then the model's own per-layer preflight, and only
    /// then is anything mutated. A `false` from either leaves the cache
    /// untouched for the generic snapshot-and-repair fallback.
    private static func restoreAfterFusedSingleDraftReject(
        _ model: any Qwen36MTPTarget,
        _ cache: [any KVCache],
        to committedOffset: Int
    ) -> Bool {
        for entry in cache {
            if entry is ArraysCache { continue }
            guard entry.isTrimmable, entry.offset == committedOffset + 1 else {
                return false
            }
        }
        guard model.recomputeFusedPrimaryBoundary(cache: cache) else {
            return false
        }
        for entry in cache where !(entry is ArraysCache) {
            if entry.isTrimmable, entry.offset > committedOffset {
                _ = entry.trim(entry.offset - committedOffset)
            }
        }
        return true
    }

    private static func clearRecurrentRollback(_ cache: [any KVCache]) {
        for entry in cache {
            guard let arrays = entry as? ArraysCache else { continue }
            arrays.rollbackState = nil
            // The accepted frame moved past the boundary this described; a
            // surviving stash could otherwise be restored on a later round as
            // though it belonged to that round's verify.
            arrays.fusedPrimaryBoundary = nil
        }
    }

    /// Offset of the first trimmable (global-attention) cache — the sequence
    /// position. Returns -1 when the stack carries no trimmable cache at all,
    /// which the round-top invariant then reports as a broken offset rather than
    /// silently accepting.
    public static func trimmableOffset(_ cache: [any KVCache]) -> Int {
        for entry in cache where !(entry is ArraysCache) { return entry.offset }
        return -1
    }

    private func trimmableOffset() -> Int { Self.trimmableOffset(cache) }

    // MARK: - readouts

    /// Shared exact ordering for the two-stage candidate-only top-2 reduction.
    private static let linearTopTwoHeader = """
        struct qwen_top2_state {
            float first_value;
            float second_value;
            uint first_id;
            uint second_id;
            uint count;
        };

        inline qwen_top2_state qwen_top2_empty() {
            qwen_top2_state state;
            state.first_value = 0.0f;
            state.second_value = 0.0f;
            state.first_id = 0;
            state.second_id = 0;
            state.count = 0;
            return state;
        }

        inline bool qwen_top2_better(
            float candidate_value,
            uint candidate_id,
            float current_value,
            uint current_id
        ) {
            bool candidate_nan = isnan(candidate_value);
            bool current_nan = isnan(current_value);
            if (candidate_nan != current_nan) {
                return !candidate_nan;
            }
            if (candidate_value > current_value) {
                return true;
            }
            if (candidate_value < current_value) {
                return false;
            }
            return candidate_id < current_id;
        }

        inline void qwen_top2_insert(
            thread qwen_top2_state &state,
            float value,
            uint id
        ) {
            if (state.count > 0 && state.first_id == id) {
                return;
            }
            if (state.count > 1 && state.second_id == id) {
                return;
            }
            if (state.count == 0
                || qwen_top2_better(
                    value, id, state.first_value, state.first_id)) {
                if (state.count > 0) {
                    state.second_value = state.first_value;
                    state.second_id = state.first_id;
                }
                state.first_value = value;
                state.first_id = id;
                state.count = min(state.count + 1, 2u);
                return;
            }
            if (state.count == 1
                || qwen_top2_better(
                    value, id, state.second_value, state.second_id)) {
                state.second_value = value;
                state.second_id = id;
                state.count = 2;
            }
        }
    """

    /// Stage one: 32 threadgroups per row each reduce a disjoint vocabulary
    /// stripe. This exposes enough work to occupy the GPU instead of making two
    /// threadgroups serially scan almost a thousand logits per lane.
    private static let linearTopTwoPartialKernel = MLXFast.metalKernel(
        name: "qwen_mtp_linear_top2_partial",
        inputNames: ["logits"],
        outputNames: ["partial_ids", "partial_values"],
        source: """
            uint lane = thread_position_in_threadgroup.x;
            uint group_index = threadgroup_position_in_grid.x;
            uint row = group_index / 32;
            uint group = group_index % 32;
            uint vocab = uint(logits_shape[2]);
            qwen_top2_state local = qwen_top2_empty();

            for (uint index = group * 256 + lane;
                 index < vocab;
                 index += 32 * 256) {
                ulong offset = ulong(row) * ulong(logits_strides[1])
                    + ulong(index) * ulong(logits_strides[2]);
                qwen_top2_insert(local, float(logits[offset]), index);
            }

            threadgroup qwen_top2_state scratch[256];
            scratch[lane] = local;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = 128; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    qwen_top2_state merged = scratch[lane];
                    qwen_top2_state other = scratch[lane + stride];
                    if (other.count > 0) {
                        qwen_top2_insert(merged, other.first_value, other.first_id);
                    }
                    if (other.count > 1) {
                        qwen_top2_insert(merged, other.second_value, other.second_id);
                    }
                    scratch[lane] = merged;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (lane == 0) {
                uint base = (row * 32 + group) * 2;
                partial_ids[base] = int(scratch[0].first_id);
                partial_ids[base + 1] = int(scratch[0].second_id);
                partial_values[base] = scratch[0].first_value;
                partial_values[base + 1] = scratch[0].second_value;
            }
        """,
        header: linearTopTwoHeader,
        ensureRowContiguous: false
    )

    /// Stage two: one small threadgroup per row merges the 32 partial pairs.
    private static let linearTopTwoFinalizeKernel = MLXFast.metalKernel(
        name: "qwen_mtp_linear_top2_finalize",
        inputNames: ["partial_ids", "partial_values"],
        outputNames: ["top_ids", "top_values"],
        source: """
            uint lane = thread_position_in_threadgroup.x;
            uint row = threadgroup_position_in_grid.x;
            uint base = (row * 32 + lane) * 2;
            qwen_top2_state local = qwen_top2_empty();
            qwen_top2_insert(local, partial_values[base], uint(partial_ids[base]));
            qwen_top2_insert(
                local, partial_values[base + 1], uint(partial_ids[base + 1]));

            threadgroup qwen_top2_state scratch[32];
            scratch[lane] = local;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = 16; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    qwen_top2_state merged = scratch[lane];
                    qwen_top2_state other = scratch[lane + stride];
                    qwen_top2_insert(merged, other.first_value, other.first_id);
                    qwen_top2_insert(merged, other.second_value, other.second_id);
                    scratch[lane] = merged;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (lane == 0) {
                uint output_base = row * 2;
                top_ids[output_base] = int(scratch[0].first_id);
                top_ids[output_base + 1] = int(scratch[0].second_id);
                top_values[output_base] = scratch[0].first_value;
                top_values[output_base + 1] = scratch[0].second_value;
            }
        """,
        header: linearTopTwoHeader,
        ensureRowContiguous: false
    )

    private static func linearTopTwoRows(_ logits: MLXArray) -> (MLXArray, MLXArray) {
        precondition(logits.ndim == 3 && logits.dim(0) == 1)
        let rows = logits.dim(1)
        let partials = linearTopTwoPartialKernel(
            [logits],
            grid: (rows * 32 * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[rows, 32, 2], [rows, 32, 2]],
            outputDTypes: [.int32, .float32]
        )
        let outputs = linearTopTwoFinalizeKernel(
            partials,
            grid: (rows * 32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[rows, 2], [rows, 2]],
            outputDTypes: [.int32, .float32]
        )
        return (outputs[0], outputs[1])
    }

    /// Top-2 token ids and logit VALUES of a single logit row.
    ///
    /// Kept local rather than reaching into the DFlash track's reference helper:
    /// the Laguna/DFlash surface is scheduled for excision when the dedicated
    /// Qwen repository is created, and the fidelity evidence must not depend on
    /// it. The `argPartition` idiom is the same one that surface uses.
    public static func topTwo(of logitRow: MLXArray) -> ([Int], [Double]) {
        let limit = Swift.max(1, Swift.min(2, logitRow.dim(-1)))
        let indices = argPartition(-logitRow, kth: limit - 1, axis: -1)[0 ..< limit]
        let scores = logitRow[indices]
        eval(indices, scores)
        let ids = indices.asArray(Int32.self).map { Int($0) }
        let values = scores.asArray(Float.self).map { Double($0) }
        let ordered = zip(ids, values).sorted { $0.1 > $1.1 }
        return (ordered.map(\.0), ordered.map(\.1))
    }

    /// One hidden row `[1, 1, H]` in MTPLX's default `post_norm` variant.
    ///
    /// `callWithHidden` returns the PRE-norm hidden by design, so the backbone's
    /// final `model.norm` is applied here via `applyFinalNorm`. Getting this wrong
    /// does NOT break exactness — the target still decides every emitted token —
    /// it collapses ACCEPTANCE. Any validation of this path has to read the accept
    /// rate, not just the match verdict.
    private func hiddenRow(_ hidden: MLXArray, _ index: Int) -> MLXArray {
        let row = hidden[0..., index ..< (index + 1), 0...]
        return postNorm ? model.applyFinalNorm(row) : row
    }

    private func lastRow(_ logits: MLXArray) -> MLXArray {
        logits[0, logits.dim(1) - 1]
    }

    private func argmaxLast(_ logits: MLXArray) -> Int {
        let row = logits[0..., (logits.dim(1) - 1) ..< logits.dim(1), 0...]
        return argMax(row, axis: -1).item(Int.self)
    }

}

/// Compiled bounds for the native-MTP track. Deliberately not env-overridable.
public enum Qwen36MTPLimits {
    /// Single source of truth is `MLXFastConstants.qwenMTPMaxDepth`: the trusted
    /// parent bounds the same quantity and links no model code.
    public static let maxDepth = MLXFastConstants.qwenMTPMaxDepth

    /// Depth 0: MTP off, one token per target forward. See
    /// `MLXFastConstants.qwenMTPSerialControlDepth` for why this is 0 and not 1.
    public static let serialControlDepth =
        MLXFastConstants.qwenMTPSerialControlDepth
}
