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
    /// Next round's primary token, read out of the previous round's single
    /// batched eval (the row argmax the old code re-fetched with a fresh
    /// `.item()` sync at every round top). Same tensor, same `argMax` op —
    /// identical value, one less blocking boundary per round.
    private var pendingPrimary: Int?
    /// Top-2 (ids, logit values) of the row that produced `pendingPrimary` —
    /// the tail-row evidence a stop-token round must declare. Recorded from
    /// the same batched readout that produced the primary.
    private var pendingTop2: ([Int], [Double])?
    /// The (post-norm) trunk hidden that seeds the next draft round. Kept
    /// LAZY: its only consumer is the next round's GPU graph.
    private var pendingHidden: MLXArray?

    // MARK: committed head history (MTPLX `mtp_history_policy="committed"`)
    //
    // The shipped session created a FRESH, EMPTY head cache inside every round,
    // so the head drafted from ~one position of context. MTPLX's production
    // default instead keeps ONE persistent head KV cache: the prompt is
    // streamed into it once, and every committed token's fused row is appended,
    // so the head attends over the whole committed prefix when it drafts
    // (measured there: accept 0.903 with history vs 0.262 without). Everything
    // below only feeds the head, and the head only PROPOSES — a worse or
    // better draft changes the accept rate, never an emitted token — so this
    // entire mechanism is outside the exactness surface by construction.
    //
    // Layout invariant: head position p holds fused(embed(token_{p+1}),
    // trunk_hidden_p) — hidden at a position pairs with the NEXT token.
    //
    // Priming is LAZY (first drafting round), so a serial-control session
    // (offers always 0) never builds the cache and stays bit-identical to the
    // previous behaviour. History upkeep is FOLDED into the next draft
    // forward as extra leading rows — the head weights are read once per
    // drafting round either way.
    private var headHistoryCache: [any KVCache]?
    /// Committed fused rows not yet appended: (post-norm trunk hidden at t,
    /// token at t+1). Flushed as leading rows of the next draft forward.
    private var headHistoryBacklogHidden: [MLXArray] = []
    private var headHistoryBacklogTokens: [Int] = []
    /// Seed rows retained for lazy priming; released at the first flush.
    private var seedHiddenForPriming: MLXArray?
    private var seedTokensForPriming: [Int] = []

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
        // Streak-gated schedule: draft 1 by default, widen to 2 while the
        // head is on a full-acceptance streak. With per-boundary checkpoints
        // a partial accept at depth 2 is as cheap as a K=1 reject, so the
        // only marginal cost of the second draft is one extra head step —
        // and the streak gate withdraws even that on cold prompts.
        draftPolicy = { [weak self] offeredDepth, _ in
            guard let self else { return Swift.min(offeredDepth, 1) }
            return Swift.min(
                offeredDepth,
                Swift.min(1 + self.fullAcceptStreak, Self.streakDepthCap))
        }
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

        // Committed-history head shapes: the first drafting round primes the
        // retained seed suffix plus its live row through the head, and an
        // accept-fold round runs a 2-row head forward. Compile both (plus the
        // single-row lm_head slice they sample) on throwaway state here,
        // outside every scored window.
        let hDim = row.dim(-1)
        let historyWarmCache = model.makeMTPCache()
        let primeWidth = Self.headPrimeHistoryLimit + 1
        let primeHidden = MLXArray.zeros([1, primeWidth, hDim], dtype: row.dtype)
        let primeTokens = MLXArray(
            Array(repeating: Int32(0), count: primeWidth)).reshaped([1, primeWidth])
        let primed = model.mtpHeadHiddenForward(
            hidden: primeHidden, nextTokenIds: primeTokens,
            cache: historyWarmCache)
        eval(model.applyDraftLMHead(
            primed[0..., (primed.dim(1) - 1) ..< primed.dim(1), 0...]))
        let foldHidden = MLXArray.zeros([1, 2, hDim], dtype: row.dtype)
        let foldTokens = MLXArray([Int32(0), Int32(0)]).reshaped([1, 2])
        let folded = model.mtpHeadHiddenForward(
            hidden: foldHidden, nextTokenIds: foldTokens,
            cache: historyWarmCache)
        eval(model.applyDraftLMHead(
            folded[0..., (folded.dim(1) - 1) ..< folded.dim(1), 0...]))
        eval(historyWarmCache.flatMap { $0.state })
        for width in 1 ... (maxDepth + 1) {
            let block = Array(repeating: 0, count: width)
            // Every drafting width verifies with nConfirmed: 1 (per-boundary
            // checkpoints); warm the same shapes the scored rounds dispatch.
            let (verifyLogits, _) = model.callWithHidden(
                input: LMInput.Text(tokens: MLXArray(block).reshaped([1, width])),
                cache: warmCache, nConfirmed: width >= 2 ? 1 : 0)
            // Compile the two top-2 reduction kernels outside the scored window
            // at every row count a round can dispatch.
            let (warmTop2IDs, warmTop2Values) = Self.linearTopTwoRows(verifyLogits)
            eval(verifyLogits, warmTop2IDs, warmTop2Values)
            eval(warmCache.flatMap { $0.state })
        }
    }

    // MARK: - begin

    /// Bulk-forward the seed and return the argmax of its last row — the first
    /// primary. The primary's own KV row is deliberately NOT written yet: the
    /// round-top invariant is "every emitted token is in the cache and the
    /// pending primary is not", and the verify forward writes it.
    @discardableResult
    public func begin(seedTokens: [Int]) throws -> Int {
        guard !began else { throw Qwen36MTPSessionError.alreadyBegun }
        guard !seedTokens.isEmpty else { throw Qwen36MTPSessionError.emptySeed }
        cache = model.newCache(parameters: nil)
        let (seedLogits, hidden) = model.callWithHidden(
            input: LMInput.Text(
                tokens: MLXArray(seedTokens).reshaped([1, seedTokens.count])),
            cache: cache, nConfirmed: 0)
        // Seed vocabulary trim: `seedLogits` projects lm_head over all 512
        // seed rows but only the last row is ever used. It is deliberately
        // NEVER evaluated — a dead lazy graph costs nothing — and the one row
        // we need is projected directly from the post-norm hidden below.
        // RMSNorm is row-local, so norm(row)+lmHead == the sliced full
        // projection bit-for-bit (ranked receipt b5130678: +0.09%).
        _ = seedLogits
        pendingHidden = hiddenRow(hidden, hidden.dim(1) - 1)
        let lastLogits = model.applyLMHead(pendingHidden!)
        // Retain the full pre-norm seed hidden for lazy head-history priming.
        // ~5 MB at 512x5120 bf16; released at the first drafting round. The
        // eval below materialises it so no seed graph is kept alive.
        seedHiddenForPriming = hidden
        seedTokensForPriming = seedTokens
        // One batched readout: the first primary and its tail-row top-2
        // evidence come out of the same eval as the cache roots.
        let (tailIDs, tailValues) = Self.linearTopTwoRows(lastLogits)
        eval(cache.flatMap { $0.state } + [tailIDs, tailValues,
                                           pendingHidden!, hidden])
        let readTail = (
            tailIDs.asArray(Int32.self).map { Int($0) },
            tailValues.asArray(Float.self).map { Double($0) }
        )
        // Top-2 first ID == row argmax (same ordering); no separate argMax.
        pendingPrimary = readTail.0[0]
        pendingTop2 = readTail
        seedTokenCount = seedTokens.count
        committedTokenCount = 0
        began = true
        return pendingPrimary!
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

    /// Consecutive fully-accepted DRAFTING rounds. The streak-gated schedule
    /// widens by one draft per full acceptance (1 -> 2 -> 3, capped), and
    /// collapses back to 1 after any miss, so a prompt the head reads poorly
    /// degrades to plain K=1 instead of paying wasted head steps per round.
    private var fullAcceptStreak = 0
    /// Cap on the streak ladder. The verify row is nearly free (weight-bound
    /// forward), so the cap prices the marginal HEAD step against its
    /// acceptance odds; 3 keeps the wasted-work tail short on mixed prose.
    private static let streakDepthCap = 4
    /// Recent seed transitions retained when the proposal-only head cache is
    /// first primed. The full 512-token seed remains in the target cache.
    private static let headPrimeHistoryLimit = 384

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
        guard began, let primaryPending = pendingPrimary,
              let tailPending = pendingTop2, let hidden = pendingHidden
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

        let primary = primaryPending
        var committed = [primary]
        committedTokenCount += 1

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

        // A stop token as the primary ends the run BEFORE any drafting: there is
        // nothing after it to predict, and drafting past it would charge the
        // measurement for work no decoder performs. The round still declares its
        // single target tail row (the row that produced this primary's successor
        // candidate is the one already spent), so the ledger stays closed.
        if stopTokens.contains(primary) {
            reachedStopToken = true
            // The tail row to declare is the row that produced this primary —
            // its top-2 was read out of the previous round's batched eval.
            let (tailTokens, tailLogits) = tailPending
            pendingPrimary = nil
            pendingTop2 = nil
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
            // Keep the committed-history ledger complete across non-drafting
            // rounds: this round's transition is (old pending hidden, primary).
            // Pure array retention — no GPU work, so the serial control's
            // compute stream is untouched. A pure-serial session never flushes
            // this backlog (the head cache is never created).
            headHistoryBacklogHidden.append(hidden)
            headHistoryBacklogTokens.append(primary)
            let (serialLogits, serialHidden) = model.callWithHidden(
                input: LMInput.Text(
                    tokens: MLXArray([primary]).reshaped([1, 1])),
                cache: cache, nConfirmed: 0)
            // Still produced, still post-norm: keeping the hidden chain identical
            // means switching depth is the ONLY difference between the two sides.
            pendingHidden = hiddenRow(serialHidden, serialHidden.dim(1) - 1)
            // Single batched readout: next primary, tail top-2, cache roots —
            // one blocking eval instead of the previous 3-4 boundaries.
            let serialLastRow = serialLogits[
                0..., (serialLogits.dim(1) - 1) ..< serialLogits.dim(1), 0...]
            let (tailIDs, tailValues) = Self.linearTopTwoRows(serialLastRow)
            eval(cache.flatMap { $0.state } + [tailIDs, tailValues])
            let readTail = (
                tailIDs.asArray(Int32.self).map { Int($0) },
                tailValues.asArray(Float.self).map { Double($0) }
            )
            // Top-2 first ID == row argmax (same ordering); no separate argMax.
            pendingPrimary = readTail.0[0]
            pendingTop2 = readTail
            let (tailTokens, tailLogits) = readTail
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

        // 1. DRAFT — against the PERSISTENT committed-history head cache.
        //    First flush the history the head has not seen yet (lazy seed
        //    priming on the first drafting round, then any committed rows
        //    queued since the last draft), with the current round's
        //    (pendingHidden, primary) transition as the final row, in ONE head
        //    forward. Only the last row's logits are projected through the
        //    lm_head. Deeper sub-steps chain the head's OWN post-`mtp.norm`
        //    hidden exactly as before.
        let headCache: [any KVCache]
        var flushHidden: [MLXArray] = []
        var flushTokens: [Int] = []
        if let existing = headHistoryCache {
            headCache = existing
        } else {
            let fresh = model.makeMTPCache()
            headHistoryCache = fresh
            headCache = fresh
            if let seedHidden = seedHiddenForPriming,
               seedTokensForPriming.count > 1
            {
                // MTPLX priming layout: seed hidden rows 0..L-2 pair with seed
                // tokens 1..L-1 (hidden at t predicts alongside token t+1).
                let primeCount = seedTokensForPriming.count - 1
                let retainedCount = Swift.min(
                    primeCount, Self.headPrimeHistoryLimit)
                let retainedStart = primeCount - retainedCount
                flushHidden.append(
                    model.applyFinalNorm(
                        seedHidden[0..., retainedStart ..< primeCount, 0...]))
                flushTokens.append(
                    contentsOf: seedTokensForPriming[(retainedStart + 1)...])
            }
            seedHiddenForPriming = nil
            seedTokensForPriming = []
        }
        if !headHistoryBacklogHidden.isEmpty {
            flushHidden.append(contentsOf: headHistoryBacklogHidden)
            flushTokens.append(contentsOf: headHistoryBacklogTokens)
            headHistoryBacklogHidden.removeAll(keepingCapacity: true)
            headHistoryBacklogTokens.removeAll(keepingCapacity: true)
        }
        flushHidden.append(hidden)
        flushTokens.append(primary)

        let draftBase = headCache.first?.offset ?? 0
        // Every flushed position is committed history plus the (pendingHidden,
        // primary) row — primary commits unconditionally — so all of them stay
        // valid whatever the verify decides. Deeper drafted positions are
        // speculative and are trimmed after the round (MTPLX
        // `_rollback_mtp_cache(cycle_offset + 1)`).
        let validHistoryOffset = draftBase + flushTokens.count
        let draftInputHidden =
            flushHidden.count == 1 ? hidden : concatenated(flushHidden, axis: 1)
        let draftInputTokens = MLXArray(flushTokens.map(Int32.init))
            .reshaped([1, flushTokens.count])

        // Draft ids stay ON DEVICE and chain straight into the verify input —
        // no host readback between the head forward and the verify forward
        // (MTPLX batched_decode: the draft id is an mx.array stacked into the
        // verify block; the ledger reads the values from the round's single
        // batched eval afterwards). `asyncEval` submits the head chain so the
        // GPU works while the host builds the 64-layer verify graph.
        var draftIdArrays: [MLXArray] = []
        var headHidden = model.mtpHeadHiddenForward(
            hidden: draftInputHidden, nextTokenIds: draftInputTokens,
            cache: headCache)
        var draftHidden = headHidden[
            0..., (headHidden.dim(1) - 1) ..< headHidden.dim(1), 0...]
        var draftId = argMax(model.applyDraftLMHead(draftHidden), axis: -1)
            .asType(.int32)
        draftIdArrays.append(draftId)
        for _ in 1 ..< draftCount {
            headHidden = model.mtpHeadHiddenForward(
                hidden: draftHidden, nextTokenIds: draftId, cache: headCache)
            draftHidden = headHidden[
                0..., (headHidden.dim(1) - 1) ..< headHidden.dim(1), 0...]
            draftId = argMax(model.applyDraftLMHead(draftHidden), axis: -1)
                .asType(.int32)
            draftIdArrays.append(draftId)
        }
        asyncEval(draftIdArrays[draftIdArrays.count - 1])

        // 2. Keep the generic pre-verify snapshot as a fallback, but use the
        //    vendored post-primary rollback checkpoint for the hot K=1 path. A
        //    rejected single draft can then retain the primary's target work and
        //    discard only the draft token instead of re-forwarding the primary.
        let snapshot = Self.snapshotRecurrent(cache)
        let verifyTokens = concatenated(
            [MLXArray([Int32(primary)]).reshaped([1, 1])] + draftIdArrays,
            axis: 1)
        // nConfirmed: 1 at every drafting width — the fused GDN verify writes
        // a per-boundary checkpoint for EVERY row, so a partial accept at any
        // depth restores its boundary without a repair forward.
        let (verifyLogits, verifyHidden) = model.callWithHidden(
            input: LMInput.Text(tokens: verifyTokens),
            cache: cache, nConfirmed: 1)

        // THE ROUND'S SINGLE BLOCKING EVAL. Everything the host needs to read
        // this round — the per-row argmaxes (accept walk AND both candidates
        // for the next primary), the draft ids, the top-2 evidence of every
        // row including the bonus row, and the cache roots — is materialised
        // in ONE eval. The `.item()`/`.asArray` calls below then copy from
        // materialised buffers without waiting on the GPU. (MTPLX production
        // budget: 1 sync/cycle, batched_decode.py:504-525.)
        let (top2IDs, top2Values) = Self.linearTopTwoRows(verifyLogits)
        var bundle: [MLXArray] = [top2IDs, top2Values]
        bundle.append(contentsOf: draftIdArrays)
        eval(cache.flatMap { $0.state } + bundle)

        let drafts = draftIdArrays.map { Int($0.item(Int32.self)) }
        let flatTop2IDs = top2IDs.asArray(Int32.self).map { Int($0) }
        let flatTop2Values = top2Values.asArray(Float.self).map { Double($0) }
        // The top-2 reducer's first ID per row IS the row argmax under the
        // same ordering `argMax` uses (larger logit wins, lower id wins an
        // exact tie), so the separate vocabulary-wide argMax launch is
        // redundant (credit GPT-5.6 Sol, promoted b71bb35, 1.37645).
        let verifyArgmax = stride(
            from: 0, to: flatTop2IDs.count, by: 2).map { flatTop2IDs[$0] }

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
            pendingPrimary = verifyArgmax[drafts.count]
            pendingHidden = hiddenRow(verifyHidden, verifyHidden.dim(1) - 1)
            let base = drafts.count * 2
            let ids = Array(flatTop2IDs[base ..< (base + 2)])
            let values = Array(flatTop2Values[base ..< (base + 2)])
            pendingTop2 = (ids, values)
            perRowTop2Tokens.append(ids)
            perRowTop2Logits.append(values)
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
            if Self.restoreAfterPrefixReject(
                cache, acceptedCount: acceptedCount, draftCount: draftCount,
                to: committedOffset)
            {
                pendingPrimary = verifyArgmax[acceptedCount]
                pendingHidden = hiddenRow(verifyHidden, acceptedCount)
                pendingTop2 = (
                    perRowTop2Tokens[acceptedCount],
                    perRowTop2Logits[acceptedCount]
                )
                perRowTop2Tokens.append(perRowTop2Tokens[acceptedCount])
                perRowTop2Logits.append(perRowTop2Logits[acceptedCount])
            } else {
                // Generic K>1 / defensive fallback: undo the whole verify window
                // and re-forward the committed block. This rare path pays a
                // second blocking eval for its own readout.
                Self.rollbackAfterVerify(
                    cache, snapshot, verifiedTokens: draftCount + 1, to: base)
                let (repairLogits, repairHidden) = model.callWithHidden(
                    input: LMInput.Text(
                        tokens: MLXArray(committed).reshaped([1, committed.count])),
                    cache: cache, nConfirmed: 0)
                pendingHidden = hiddenRow(repairHidden, repairHidden.dim(1) - 1)
                let repairLastRow = repairLogits[
                    0..., (repairLogits.dim(1) - 1) ..< repairLogits.dim(1),
                    0...]
                let (tailIDs, tailValues) = Self.linearTopTwoRows(repairLastRow)
                eval(cache.flatMap { $0.state } + [tailIDs, tailValues])
                let ids = tailIDs.asArray(Int32.self).map { Int($0) }
                let values = tailValues.asArray(Float.self).map { Double($0) }
                // Top-2 first ID == row argmax; no separate argMax launch.
                pendingPrimary = ids[0]
                pendingTop2 = (ids, values)
                perRowTop2Tokens.append(ids)
                perRowTop2Logits.append(values)
            }
        }

        // Head-history upkeep. Trim the speculative deeper-draft rows back to
        // the valid prefix, then queue the ACCEPTED transitions for the next
        // drafting round's flush: row i of the verify output is the trunk
        // hidden at draft i's position, so (hiddenRow(i), drafts[i]) is the
        // committed pair. The rejecting round queues nothing — the next
        // round's own (pendingHidden, primary) row covers that transition.
        Self.trimTrimmable(headCache, to: validHistoryOffset)
        for index in 0 ..< acceptedCount {
            headHistoryBacklogHidden.append(hiddenRow(verifyHidden, index))
            headHistoryBacklogTokens.append(drafts[index])
        }
        fullAcceptStreak =
            acceptedCount == drafts.count ? fullAcceptStreak + 1 : 0

        acceptedDraftTotal += acceptedCount
        rejectedDraftTotal += drafts.count - acceptedCount
        // No trailing eval: the cache roots and every host-read value were
        // materialised by the round's single bundle eval above (the rare
        // generic-repair path ran its own second eval). `pendingHidden` stays
        // lazy — its only consumer is the next round's GPU graph.

        // Truncate after the first committed stop token, keeping the stop token
        // itself — the same rule the serial reference applies.
        if let stopIndex = committed.firstIndex(where: { stopTokens.contains($0) }) {
            let dropped = committed.count - (stopIndex + 1)
            committed = Array(committed.prefix(stopIndex + 1))
            committedTokenCount -= dropped
            reachedStopToken = true
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
                // The vendored rollback checkpoints, if the GDN forward ever
                // wrote them, describe a frame this rollback just discarded.
                arrays.rollbackState = nil
                arrays.rollbackCheckpoints = []
                continue
            }
            if entry.isTrimmable, entry.offset > base {
                _ = entry.trim(Swift.min(verifiedTokens, entry.offset - base))
            }
        }
    }

    /// Restore the per-boundary checkpoint written by a fused width-S verify
    /// with `nConfirmed == 1`. Checkpoint t is the recurrent state after
    /// verify row t, so a partial acceptance of `acceptedCount` drafts
    /// restores checkpoint `acceptedCount` and trims exactly the rejected
    /// rows from the attention caches — no repair forward at any depth.
    ///
    /// Preflight every layer before mutating any of them. Returning `false`
    /// leaves the cache untouched so the caller can use the generic snapshot
    /// and repair path safely.
    private static func restoreAfterPrefixReject(
        _ cache: [any KVCache],
        acceptedCount: Int,
        draftCount: Int,
        to committedOffset: Int
    ) -> Bool {
        let rejected = draftCount - acceptedCount
        guard rejected > 0 else { return false }
        for entry in cache {
            if let arrays = entry as? ArraysCache {
                guard arrays.rollbackCheckpoints.count > acceptedCount
                else { return false }
            } else if entry.isTrimmable {
                guard entry.offset == committedOffset + rejected
                else { return false }
            } else {
                return false
            }
        }

        for entry in cache {
            if let arrays = entry as? ArraysCache {
                let saved = arrays.rollbackCheckpoints[acceptedCount]
                arrays[0] = saved.0
                arrays[1] = saved.1
                arrays.rollbackState = nil
                arrays.rollbackCheckpoints = []
            } else if entry.isTrimmable {
                _ = entry.trim(entry.offset - committedOffset)
            }
        }
        return true
    }

    private static func clearRecurrentRollback(_ cache: [any KVCache]) {
        for entry in cache {
            if let arrays = entry as? ArraysCache {
                arrays.rollbackState = nil
                arrays.rollbackCheckpoints = []
            }
        }
    }

    /// Trim every trimmable cache in the stack back to `offset`. Used on the
    /// persistent head-history cache to discard speculative deeper-draft rows
    /// after a round (the head stack is all `KVCacheSimple`).
    private static func trimTrimmable(_ cache: [any KVCache], to offset: Int) {
        for entry in cache where entry.isTrimmable {
            let extra = entry.offset - offset
            if extra > 0 { _ = entry.trim(extra) }
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

    /// Top-2 token ids and logit VALUES of a single logit row.
    ///
    /// Kept local rather than reaching into the DFlash track's reference helper:
    /// the Laguna/DFlash surface is scheduled for excision when the dedicated
    /// Qwen repository is created, and the fidelity evidence must not depend on
    /// it. The `argPartition` idiom is the same one that surface uses.
    // MARK: hierarchical linear top-2 (ported from the promoted e5051ba
    // frontier, ranked 1.35254 — credit scarletbright). Replaces the
    // vocabulary-wide argPartition+gather per verify row with a two-stage
    // exact reduction: 32 threadgroups per row reduce disjoint vocabulary
    // stripes, one small threadgroup merges the partials. Ordering contract
    // is identical to `argMax` and to `topTwoRead`: value-descending, then
    // id-ascending on exact ties, NaN sorted last.

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

    /// Exact top-2 (ids, values) for every row of a `[1, rows, V]` logits
    /// array, as `[rows, 2]` int32 / float32 device arrays.
    static func linearTopTwoRows(_ logits: MLXArray) -> (MLXArray, MLXArray) {
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

    public static func topTwo(of logitRow: MLXArray) -> ([Int], [Double]) {
        let pair = topTwoLazy(logitRow)
        eval(pair.0, pair.1)
        return topTwoRead(pair)
    }

    /// Lazy half of `topTwo`: the (indices, scores) arrays, not yet evaluated,
    /// so many rows can share one batched eval.
    static func topTwoLazy(_ logitRow: MLXArray) -> (MLXArray, MLXArray) {
        let limit = Swift.max(1, Swift.min(2, logitRow.dim(-1)))
        let indices = argPartition(-logitRow, kth: limit - 1, axis: -1)[0 ..< limit]
        let scores = logitRow[indices]
        return (indices, scores)
    }

    /// Host half of `topTwo`: reads MATERIALISED (indices, scores) arrays.
    ///
    /// Tie-break pinned to value-descending THEN id-ascending: `argPartition`
    /// gives no order among equals and Swift's `sorted` is not stable, so on
    /// an exact logit tie a value-only sort could disagree with `argMax`'s
    /// lowest-index-wins rule the reference replay follows.
    static func topTwoRead(_ pair: (MLXArray, MLXArray)) -> ([Int], [Double]) {
        let ids = pair.0.asArray(Int32.self).map { Int($0) }
        let values = pair.1.asArray(Float.self).map { Double($0) }
        let ordered = zip(ids, values).sorted {
            $0.1 != $1.1 ? $0.1 > $1.1 : $0.0 < $1.0
        }
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

    private func argmaxAll(_ logits: MLXArray) -> [Int] {
        argMax(logits, axis: -1)[0].asArray(Int.self)
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
