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

    // MARK: - adaptive draft schedule state

    /// Most recent drafting round's width and acceptance, feeding
    /// `adaptiveDraftCount`. Only rounds that actually drafted update these:
    /// serial rounds (offer 0) and adaptive skips (policy returns 0) are not
    /// drafting rounds and leave the history untouched.
    private var lastRoundDrafted = 0
    private var lastRoundAccepted = 0

    /// Rolling (drafted, accepted) pairs for the most recent drafting rounds,
    /// capped at `acceptanceWindow`. The policy reads the windowed acceptance
    /// rate to decide whether speculation is paying at all. Only drafting
    /// rounds (draftCount > 0) enter the window.
    private var acceptanceHistory: [(drafted: Int, accepted: Int)] = []
    /// Adaptive skips in a row. While shut down, the policy probes the head
    /// again every `probeInterval` skip rounds so a head that recovered is
    /// not missed for the rest of the decode.
    private var consecutiveSkips = 0
    private static let acceptanceWindow = 8
    /// Below this windowed acceptance rate a verify row does not pay for
    /// itself, so the schedule falls back to serial (adaptive skip = 0).
    private static let acceptanceFloor = 0.45
    private static let probeInterval = 8

    public init(
        model: any Qwen36MTPTarget,
        stopTokens: Set<Int>,
        postNorm: Bool = true
    ) throws {
        guard model.hasMTPHead else { throw Qwen36MTPSessionError.headNotAttached }
        self.model = model
        self.stopTokens = stopTokens
        self.postNorm = postNorm
        self.draftPolicy = { [weak self] offeredDepth, round in
            self?.adaptiveDraftCount(offeredDepth: offeredDepth, round: round) ?? 0
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
        for width in 1 ... (maxDepth + 1) {
            let block = Array(repeating: 0, count: width)
            let (verifyLogits, _) = model.callWithHidden(
                input: LMInput.Text(tokens: MLXArray(block).reshaped([1, width])),
                cache: warmCache, nConfirmed: 0)
            eval(verifyLogits)
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
        let (logits, hidden) = model.callWithHidden(
            input: LMInput.Text(
                tokens: MLXArray(seedTokens).reshaped([1, seedTokens.count])),
            cache: cache, nConfirmed: 0)
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
    /// THE POLICY BELOW IS THE FIRST THING A SUBMISSION SHOULD CHANGE. The
    /// shipped reference schedule (`draftPolicy`) was a constant 2, the depth
    /// this track was pinned at while depth was an operator parameter; an
    /// unmodified tree reproduces the measured reference behaviour exactly.
    ///
    /// THIS SUBMISSION SHIPS AN ADAPTIVE SCHEDULE. The operator's sealed
    /// measurement matrix is the ground truth this is built on: a constant
    /// depth of 1 was the best measured width (+5.4% predicted raw median),
    /// a constant 2 is a measured ~0.6% regression against serial on the 3.8
    /// tower (the 0.994 calibration), and a constant 3 was far worse
    /// (−22.3%). So this schedule drafts ONE token per round normally,
    /// attempts a second only on a full-acceptance streak, and falls back to
    /// drafting nothing while the rolling acceptance rate is too low for a
    /// verify row to pay for itself. That fallback is what protects the 0.90
    /// floor: against a head the pool does not reward, the schedule degrades
    /// to serial (~1.0) instead of paying for rejected verify rows.
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
    public var draftPolicy: (_ offeredDepth: Int, _ round: Int) -> Int = {
        offeredDepth, round in
        Swift.min(offeredDepth, 0)
    }

    /// The adaptive schedule this submission ships. See `draftPolicy`.
    ///
    /// Returns 0 ... min(offer, maxDepth); the call-site precondition bounds
    /// the actual return, so the math here is only as careful as the
    /// threshold and the caps.
    private func adaptiveDraftCount(offeredDepth: Int, round: Int) -> Int {
        let maxAllowed = Swift.min(offeredDepth, Qwen36MTPLimits.maxDepth)
        guard maxAllowed >= 1 else { return 0 }

        // First round (or a fresh probe after a shutdown): no history to
        // trust. Draft one token.
        guard !acceptanceHistory.isEmpty else { return Swift.min(maxAllowed, 1) }

        // Shutdown probe: while shut down the window never refreshes, so
        // every `probeInterval` skip rounds the head gets one more chance;
        // a recovered head is never missed for the rest of the decode.
        if consecutiveSkips >= Self.probeInterval {
            acceptanceHistory.removeAll()
            lastRoundDrafted = 0
            lastRoundAccepted = 0
            consecutiveSkips = 0
            return Swift.min(maxAllowed, 1)
        }

        // While the windowed acceptance rate is below the floor, drafting does
        // not pay: every verify row costs more than the expected accepted
        // token. Draft nothing -- an adaptive skip costs exactly serial.
        if acceptanceHistory.count >= Self.acceptanceWindow {
            let drafted = acceptanceHistory.reduce(0) { $0 + $1.drafted }
            let accepted = acceptanceHistory.reduce(0) { $0 + $1.accepted }
            if drafted > 0,
               Double(accepted) / Double(drafted) < Self.acceptanceFloor
            {
                return 0
            }
        }

        // Full acceptance last round is a streak: one extra draft is the most
        // the calibration ever rewarded (constant 2 measured ~0.994 raw).
        if lastRoundDrafted >= 1, lastRoundAccepted == lastRoundDrafted {
            return Swift.min(maxAllowed, 2)
        }

        return Swift.min(maxAllowed, 1)
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

        let primary = argmaxLast(logitsRow)
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
        // The adaptive schedule records a skip only when the policy chose it
        // (an offered 0 is the control leg and leaves the schedule history
        // alone); `adaptiveDraftCount`'s probe counter keys off it.
        //
        // The single row this forward produces IS the round's target tail row:
        // its argmax becomes the next primary, exactly as the bonus row does on
        // the speculative path. So the ledger closes with declaredRows = 1,
        // accepted = rejected = 0, tail = 1 -- and `rows_per_round(0) = 1` in the
        // box wrapper agrees without any special case there.
        if depth == Qwen36MTPLimits.serialControlDepth || draftCount == 0 {
            if depth != Qwen36MTPLimits.serialControlDepth {
                consecutiveSkips += 1
            }
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

        // 2. Snapshot the recurrent (non-trimmable) state, then verify in ONE
        //    batched forward. `nConfirmed` stays 0 deliberately: a non-zero value
        //    installs the vendored depth-1-only rollback AND changes the GDN
        //    chunk geometry, both of which fight the snapshot/rollback below.
        let snapshot = Self.snapshotRecurrent(cache)
        let verifyInput = committed + drafts
        let (verifyLogits, verifyHidden) = model.callWithHidden(
            input: LMInput.Text(
                tokens: MLXArray(verifyInput).reshaped([1, verifyInput.count])),
            cache: cache, nConfirmed: 0)
        let verifyArgmax = argmaxAll(verifyLogits)

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
            let (ids, values) = Self.topTwo(of: verifyLogits[0, index])
            perRowTop2Tokens.append(ids)
            perRowTop2Logits.append(values)
        }

        if acceptedCount == drafts.count {
            // FULL ACCEPTANCE: the verify state IS the committed state. No
            // rollback, no repair forward; the bonus row carries the next primary
            // and the last hidden row seeds the next draft.
            committed.append(contentsOf: drafts)
            committedTokenCount += drafts.count
            let bonus = verifyLogits[
                0..., drafts.count ..< (drafts.count + 1), 0...]
            pendingLogitsRow = bonus
            pendingHidden = hiddenRow(verifyHidden, verifyHidden.dim(1) - 1)
            let (ids, values) = Self.topTwo(of: verifyLogits[0, drafts.count])
            perRowTop2Tokens.append(ids)
            perRowTop2Logits.append(values)
        } else {
            // PARTIAL: undo the WHOLE verify window — restore the recurrent
            // snapshot and trim all `1 + draftCount` positions from the trimmable
            // caches — then re-forward only the committed block. The correction
            // token (the target's own row-`acceptedCount` argmax) is NEVER
            // emitted here; it arrives as the next round's primary, out of the
            // repair forward, which is what keeps the emitted stream identical to
            // the serial trajectory.
            rollbackRoundCount += 1
            committed.append(contentsOf: drafts.prefix(acceptedCount))
            committedTokenCount += acceptedCount
            Self.rollbackAfterVerify(
                cache, snapshot, verifiedTokens: verifyInput.count, to: base)
            let (repairLogits, repairHidden) = model.callWithHidden(
                input: LMInput.Text(
                    tokens: MLXArray(committed).reshaped([1, committed.count])),
                cache: cache, nConfirmed: 0)
            pendingLogitsRow = repairLogits[
                0..., (repairLogits.dim(1) - 1) ..< repairLogits.dim(1), 0...]
            pendingHidden = hiddenRow(repairHidden, repairHidden.dim(1) - 1)
            let (ids, values) = Self.topTwo(of: lastRow(repairLogits))
            perRowTop2Tokens.append(ids)
            perRowTop2Logits.append(values)
        }

        acceptedDraftTotal += acceptedCount
        rejectedDraftTotal += drafts.count - acceptedCount
        lastRoundDrafted = draftCount
        lastRoundAccepted = acceptedCount
        consecutiveSkips = 0
        acceptanceHistory.append((drafted: draftCount, accepted: acceptedCount))
        if acceptanceHistory.count > Self.acceptanceWindow {
            acceptanceHistory.removeFirst()
        }
        eval(cache.flatMap { $0.state })
        if let row = pendingLogitsRow, let h = pendingHidden { eval(row, h) }

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
                // The vendored depth-1 rollback snapshot, if the GDN forward ever
                // wrote one, describes a frame this rollback just discarded.
                arrays.rollbackState = nil
                continue
            }
            if entry.isTrimmable, entry.offset > base {
                _ = entry.trim(Swift.min(verifiedTokens, entry.offset - base))
            }
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
