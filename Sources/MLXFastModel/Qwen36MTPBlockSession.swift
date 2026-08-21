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

    public init(
        model: any Qwen36MTPTarget,
        stopTokens: Set<Int>,
        postNorm: Bool = true
    ) throws {
        guard model.hasMTPHead else { throw Qwen36MTPSessionError.headNotAttached }
        self.model = model
        self.stopTokens = stopTokens
        self.postNorm = postNorm
        // Cost-model schedule (replaces the streak ladder). Choose the depth
        // that maximizes expected committed tokens per unit round time under
        // the round's measured economics:
        //
        //   T(d) = V + d·H        one width-(d+1) verify + d head steps
        //   E[tokens](d) = 1 + Σ_{k=1..d} Π_{i<k} p_i
        //
        // where p_i is the EMA-estimated acceptance of draft position i GIVEN
        // the prefix before it was accepted, and h = H/V is the head step's
        // cost relative to the weight-stream-bound verify forward (near-flat
        // in width up to the qmv limit). Greedy marginal rule: extend to
        // position k+1 exactly while
        //
        //   Π_{i<=k+1} p_i  >  h · (1 + S_k) / (1 + k·h)
        //
        // which is f(k+1) > f(k) rearranged. On hot prose (p→0.9) this runs
        // straight to the offer; on cold prompts it collapses to 1, and to a
        // free adaptive skip (0) only when even the first draft's odds are
        // below h. The streak ladder's behavior is the degenerate one-EMA
        // version of this; the per-position EMAs let depth 5-8 pay where the
        // ladder's cap of 4 left committed tokens on the table.
        draftPolicy = { [weak self] offeredDepth, _ in
            guard let self else { return Swift.min(offeredDepth, 1) }
            return self.costModelDepth(offeredDepth: offeredDepth)
        }
    }

    // MARK: - warm

    /// Keep the ranked M5-Max model allocations in Metal's residency set
    /// after the input-independent warm. MLX attaches a residency set to every
    /// command queue, but its capacity is zero until a wired limit is applied;
    /// without this one-time resize the driver must re-establish residency for
    /// the whole tower on later command buffers.
    ///
    /// Capacity is deliberately the live post-warm footprint plus only a small
    /// page-rounding allowance. After cached warm temporaries are cleared,
    /// persistent weights fit in the one resize while later scratch fails the
    /// fit test and stays on the commit-free unwired path. The ticket is never
    /// ended because shrinking the limit would evict the resident weights.
    private static let wiredZHDefaultFraction = 1.0
    private static let wiredZHDefaultSlackMB = 64
    private static let wiredTicketLock = NSLock()
    nonisolated(unsafe) private static var wiredTicketRetainer: WiredMemoryTicket?

    private final class QwenMTPWiredLimitBox: @unchecked Sendable {
        var value: Int = 0
    }

    private static func wireResidentWeightsIfEnabled() {
        let environment = ProcessInfo.processInfo.environment
        guard environment["DARKBLOOM_QWEN_MTP_WIRED_ZH"] != "0" else { return }
        guard ProcessInfo.processInfo.physicalMemory >= (UInt64(96) << 30)
        else { return }

        wiredTicketLock.lock()
        defer { wiredTicketLock.unlock() }
        guard wiredTicketRetainer == nil else { return }

        // Shape-warm locals have left scope before this method is called.
        // Remove their cached storage so the active count describes the live
        // backbone, head, and persistent runtime tensors rather than scratch.
        Memory.clearCache()
        let active = Memory.activeMemory
        guard active > 0 else { return }

        let fraction = environment["DARKBLOOM_QWEN_MTP_WIRED_ZH_FRACTION"]
            .flatMap(Double.init) ?? wiredZHDefaultFraction
        let slackMB = environment["DARKBLOOM_QWEN_MTP_WIRED_ZH_SLACK_MB"]
            .flatMap(Int.init) ?? wiredZHDefaultSlackMB
        var target = Int(Double(active) * min(max(fraction, 0.0), 1.0))
        target += max(0, slackMB) << 20

        // The MLX backend rejects a wired limit above the recommended working
        // set. Keep a 256 MiB margin for system bookkeeping and fail closed on
        // nonsensical geometry.
        if let recommended = GPU.maxRecommendedWorkingSetBytes() {
            target = min(target, max(0, recommended - (256 << 20)))
        }
        guard target > 0 else { return }

        let ticket = WiredMemoryTicket(
            size: target,
            policy: MLXLMCommon.WiredSumPolicy(cap: target),
            manager: .shared,
            kind: .active
        )
        let appliedBox = QwenMTPWiredLimitBox()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            appliedBox.value = await ticket.start()
            semaphore.signal()
        }
        let outcome = semaphore.wait(timeout: .now() + .seconds(30))
        wiredTicketRetainer = ticket

        let applied = outcome == .success ? appliedBox.value : -1
        let recommended = GPU.maxRecommendedWorkingSetBytes() ?? -1
        var line = "mlxfast: qwen-mtp wired-zh request=\(target)"
        line += " applied=\(applied) active=\(active)"
        line += " slack_mb=\(max(0, slackMB)) fraction=\(fraction)"
        line += " maxrec=\(recommended)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    /// Input-independent shape warm, run OUTSIDE every scored window.
    ///
    /// Warms the two forward shapes a round dispatches — the batched verify at
    /// every legal width `1 ... maxDepth + 1`, and the head's single-token draft
    /// step — on throwaway cache state. Nothing here sees a seed.
    public func warmAllDepths(maxDepth: Int) throws {
        // Keep the large shape-warm object graph in a separate call frame so
        // every throwaway cache and tensor is released before residency sizing.
        try warmAllDepthShapes(maxDepth: maxDepth)
        Self.wireResidentWeightsIfEnabled()
    }

    private func warmAllDepthShapes(maxDepth: Int) throws {
        // Warms every legal verify width from 1 (the serial control's
        // single-token forward) up to maxDepth + 1, plus the head's draft step.
        // The head warm runs even for a serial-only session: the head is resident
        // on both sides, so warming it on both keeps the load shape identical.
        guard maxDepth >= 1, maxDepth <= Qwen36MTPLimits.maxDepth else {
            throw Qwen36MTPSessionError.invalidDepth(maxDepth)
        }
        let warmCache = model.newCache(parameters: nil)
        // Decode kernels are not fully described by query width: the wide
        // attention path also selects against the live KV length. Warming the
        // legal widths behind an 8-token prefix left the long-prefix variants
        // to materialise inside a later scored round (the ranked prompt-5
        // receipt showed a repeatable 0.368 s one-off stall). Seed the
        // throwaway cache at the track's real 512-token prefix so every width
        // below compiles in the same long-context dispatch family as decode.
        // Token values are deliberately irrelevant here: this cache is never
        // observed by generation; only its 512-row shape selects the family.
        let seed = Array(repeating: 0, count: 512)
        let (logits, hidden) = model.callWithHidden(
            input: LMInput.Text(tokens: MLXArray(seed).reshaped([1, seed.count])),
            cache: warmCache, nConfirmed: 0)
        // As in `begin`, the full-seed lm_head projection is dead. Evaluating
        // it would warm work the scored path never performs and needlessly
        // stream the vocabulary matrix over all 512 rows.
        _ = logits
        var row = hiddenRow(hidden, hidden.dim(1) - 1)
        eval(warmCache.flatMap { $0.state })
        eval(row)

        let headCache = model.makeMTPCache()
        for _ in 0 ..< maxDepth {
            let (draftLogits, draftHidden) = model.mtpForwardWithHidden(
                hidden: row,
                nextTokenIds: MLXArray([0]).reshaped([1, 1]),
                cache: headCache)
            row = draftHidden[0..., (draftHidden.dim(1) - 1) ..< draftHidden.dim(1), 0...]
            eval(draftLogits, row)
        }

        // Committed-history head shapes: compile both the K/V-only leading-row
        // path and final full row for the full seed and a 2-row accept fold.
        let hDim = row.dim(-1)
        let historyWarmCache = model.makeMTPCache()
        // E65 rung 1 tested building this block through the live first-round
        // expression instead — applyFinalNorm over a [1, L-1, h] strided slice
        // of the retained pre-norm seed hidden, concatenated with a [1, 1, h]
        // row — on the theory that the unwarmed norm-over-slice and float
        // concat were the +23.8/+28.0/+29.7 ms of host graph build the census
        // localised in scored round 1. It measured 22.4 ms, inside the base
        // range, so those two ops are NOT the cost. Reverted; do not retry
        // without new evidence naming a different statement.
        let primeHidden = MLXArray.zeros([1, 512, hDim], dtype: row.dtype)
        let primeTokens = MLXArray(
            Array(repeating: Int32(0), count: 512)).reshaped([1, 512])
        let primed = model.mtpHeadLastHiddenWithKVOnlyHistory(
            hidden: primeHidden, nextTokenIds: primeTokens,
            cache: historyWarmCache)
            ?? model.mtpHeadHiddenForward(
                hidden: primeHidden, nextTokenIds: primeTokens,
                cache: historyWarmCache)
        // Warm the complete proposal-side expression used by a live draft.
        // The compact vocabulary changes the reduction shape and adds an
        // on-device ID map, so warming logits alone leaves both kernels to
        // cold-JIT inside the first scored round.
        //
        // LOAD-BEARING: this must warm `draftTokenID` -- the SAME expression
        // the scored rounds now dispatch -- not the old
        // `mapDraftTokenIds(argMax(applyDraftLMHead(...)))` chain. 7b33621's
        // note records that the first compact-vocabulary attempt was
        // parity-clean and faster in steady state on all 8 prompts and STILL
        // LOST, because its warm evaluated compact logits while the live graph
        // differed: first MTP block 0.941 s vs 0.402 s, the JIT paid inside
        // the scored window. A new selection kernel resets that hazard exactly.
        let primedDraftID = model.draftTokenID(
            primed[0..., (primed.dim(1) - 1) ..< primed.dim(1), 0...])
        eval(primedDraftID)
        // VERIFY-CONCAT JIT WARM. Scored rounds assemble verifyTokens as
        // concatenated([host primary] + device draftIds) over int32 [1, 1]
        // arrays. The width loop below feeds callWithHidden a single host
        // [1, width] tensor, so it never compiles that multi-input concat.
        // MLX JIT-specializes copy/concat by dtype and input count
        // (ml-explore/mlx metal JIT; first launch pays Metal library
        // compile — see Kernel Management / JIT Compilation). Those
        // copyint32int32 kernels otherwise land inside scored round 1.
        // Values are zeros / already-eval'd draft IDs and the result is
        // discarded: shape + dtype + host/device mix select the kernels.
        // Warm every legal extra-count 0...maxDepth so an adaptive
        // draftPolicy that returns 0..8 does not hit a cold width later.
        //
        // PROVENANCE, and why this block keeps disappearing. Authored by
        // fkiene and PROMOTED at 1cb1f43a7246d57af8b96dad468583364779aa73,
        // scoring 3.24417896624589 against the 3.24326223889754 base
        // (+0.0283 %). The very next promotion (ofou, ef42e0432727, now
        // upstream/main) branched from a commit PREDATING fkiene and submitted;
        // because `yukon submit` REPLACES whole files rather than merging,
        // `git diff 1cb1f43a upstream/main` on this file is 0 insertions and
        // 19 deletions -- exactly these lines, deleted by an author who never
        // opened the file. It is therefore absent from the live tip AND from
        // every tree descended from that base, including ours. Restored here
        // with its receipt so the next whole-file overlay has to argue with
        // the number instead of silently dropping it again.
        //
        // Placement is load-bearing: this sits in `warmAllDepthShapes`, i.e.
        // in the warm-up path OUTSIDE the timed window, so the JIT cost it
        // moves is paid before measurement starts. The comment 12 lines above
        // records the same hazard biting a previous candidate that warmed the
        // wrong expression: first MTP block 0.941 s vs 0.402 s.
        for extra in 0 ... maxDepth {
            var parts = [MLXArray([Int32(0)]).reshaped([1, 1])]
            for _ in 0 ..< extra {
                parts.append(primedDraftID)
            }
            eval(concatenated(parts, axis: 1))
        }
        let foldHidden = MLXArray.zeros([1, 2, hDim], dtype: row.dtype)
        let foldTokens = MLXArray([Int32(0), Int32(0)]).reshaped([1, 2])
        let folded = model.mtpHeadLastHiddenWithKVOnlyHistory(
            hidden: foldHidden, nextTokenIds: foldTokens,
            cache: historyWarmCache)
            ?? model.mtpHeadHiddenForward(
                hidden: foldHidden, nextTokenIds: foldTokens,
                cache: historyWarmCache)
        eval(model.draftTokenID(
            folded[0..., (folded.dim(1) - 1) ..< folded.dim(1), 0...]))
        eval(historyWarmCache.flatMap { $0.state })
        for width in 1 ... (maxDepth + 1) {
            let block = Array(repeating: 0, count: width)
            // Every drafting width verifies with nConfirmed: 1. Width two uses
            // the eager boundary checkpoint; wider blocks retain a replay
            // tape. Warm the same shapes the scored rounds dispatch.
            let (verifyLogits, _) = model.callWithHidden(
                input: LMInput.Text(tokens: MLXArray(block).reshaped([1, width])),
                cache: warmCache, nConfirmed: width >= 2 ? 1 : 0)
            // Compile the two top-2 reduction kernels outside the scored window
            // at every row count a round can dispatch.
            let (warmTop2IDs, warmTop2Values) = Self.linearTopTwoRows(verifyLogits)
            eval(verifyLogits, warmTop2IDs, warmTop2Values)
            eval(warmCache.flatMap { $0.state })
            if width >= 3 {
                // Warm arbitrary-prefix replay T=2...8. Restore all but the
                // final verify row and trim that same row from attention so the
                // throwaway cache remains aligned for the next width.
                precondition(model.replayRecurrentPrefix(
                    cache: warmCache, committedRows: width - 1))
                for entry in warmCache where !(entry is ArraysCache) {
                    if entry.isTrimmable { _ = entry.trim(1) }
                }
                eval(warmCache.flatMap { $0.state })
            } else {
                Self.clearRecurrentRollback(warmCache)
            }
        }

        // A K>=2 round can reject its very first draft, which replays T=1.
        // Width 2 stays on the validated eager K1 path, so compile this last
        // missing replay shape with one extra throwaway width-3 verify.
        let oneRowReplayCache = model.newCache(parameters: nil)
        let (oneRowReplayLogits, _) = model.callWithHidden(
            input: LMInput.Text(tokens: MLXArray([0, 0, 0]).reshaped([1, 3])),
            cache: oneRowReplayCache, nConfirmed: 1)
        eval(oneRowReplayLogits)
        eval(oneRowReplayCache.flatMap { $0.state })
        precondition(model.replayRecurrentPrefix(
            cache: oneRowReplayCache, committedRows: 1))
        eval(oneRowReplayCache.flatMap { $0.state })

        // SEED-PREFILL SHAPE WARM (M=512 backbone). Keep this as the final
        // warm so the promoted allocator/pipeline end state is preserved.
        // The phase trace measured
        // `begin` at ~0.9 s of eval wall for a 512-token seed — mostly
        // first-touch pipeline compilation and allocator growth for the
        // M=512 shapes, charged inside the timed window because this warm
        // path previously exercised only M=8 and the decode widths. One
        // input-independent 512-zero forward on a throwaway cache moves that
        // first-touch out here, into the untimed warm, replaying `begin`'s
        // exact op sequence: full-seed forward whose full logits are a dead
        // lazy graph (never evaluated, exactly as `begin` leaves them), the
        // final-norm over the priming rows, and the single-row lm_head
        // readout. Zero tokens in, nothing read out — pure shape warm, the
        // same contract as every warm above.
        let seedWarmCache = model.newCache(parameters: nil)
        let seedWarmTokens = Array(repeating: 0, count: 512)
        let (seedWarmLogits, seedWarmHidden) = model.callWithHidden(
            input: LMInput.Text(
                tokens: MLXArray(seedWarmTokens).reshaped([1, 512])),
            cache: seedWarmCache, nConfirmed: 0)
        _ = seedWarmLogits
        let seedWarmRow = hiddenRow(seedWarmHidden, seedWarmHidden.dim(1) - 1)
        let seedWarmNorm = model.applyFinalNorm(
            seedWarmHidden[0..., 0 ..< 511, 0...])
        let (seedWarmIDs, seedWarmValues) =
            Self.linearTopTwoRows(model.applyLMHead(seedWarmRow))
        eval(seedWarmCache.flatMap { $0.state }
            + [seedWarmIDs, seedWarmValues, seedWarmNorm])

        // TARGET-SIDE later-window SDPA compile. Distinct from the rejected
        // #674 proposal-head HOST KV walk (3.23670). After the 512-row seed
        // the 16 FA caches sit at kL=512; scored decode walks that prefix to
        // kL~1024. The width ladder above only compiled kL≈512+width. HOST-
        // extend throwaway FA K/V to kL>=1024 (dummy concat, no 64-layer
        // forward) and dispatch the three fused-vector shapes the ranked
        // path actually fires: qL=1 (serial / chunk-B of width 6) plus the
        // exactness-chunk pair qL=5 / qL=4. Live `begin()` caches untouched.
        Self.warmTargetLaterWindowSDPA(seedWarmCache)
    }

    /// Untimed. Throwaway FA caches only. Token-neutral: dummy K/V never
    /// enter a scored forward. Compiles later-window `MLXFast` SDPA so the
    /// first decode step past the 512-row seed does not first-touch those
    /// pipeline variants inside the scored window.
    private static func warmTargetLaterWindowSDPA(_ cache: [KVCache]) {
        var extended: [MLXArray] = []
        var firstKV: (MLXArray, MLXArray)?
        var faCount = 0
        for entry in cache {
            guard entry is KVCacheSimple else { continue }
            let st = entry.state
            guard st.count == 2 else { continue }
            let k = st[0]
            let v = st[1]
            guard k.ndim == 4, v.ndim == 4, k.dim(2) > 0, v.dim(2) == k.dim(2)
            else { continue }
            faCount += 1
            let pad = max(0, 1024 - k.dim(2))
            let extK: MLXArray
            let extV: MLXArray
            if pad > 0 {
                let kPad = MLXArray.zeros(
                    [k.dim(0), k.dim(1), pad, k.dim(3)], dtype: k.dtype)
                let vPad = MLXArray.zeros(
                    [v.dim(0), v.dim(1), pad, v.dim(3)], dtype: v.dtype)
                extK = concatenated([k, kPad], axis: 2)
                extV = concatenated([v, vPad], axis: 2)
            } else {
                extK = k
                extV = v
            }
            extended.append(contentsOf: [extK, extV])
            if firstKV == nil { firstKV = (extK, extV) }
        }
        // Pinned Qwen 3.8 tower: 16 FA + 48 GDN. Wrong geometry → no-op.
        guard faCount == 16, let (extK, extV) = firstKV, extK.dim(2) >= 1024
        else { return }
        eval(extended)
        // 4 KV heads × 6 GQA = 24 Q heads; head_dim from the live FA tensor
        // (config pins 256). Scale matches Qwen35Attention.
        let qHeads = extK.dim(1) * 6
        let headDim = extK.dim(3)
        let scale = 1 / Float(headDim).squareRoot()
        var outs: [MLXArray] = []
        for qL in [1, 5, 4] {
            let q = MLXArray.zeros(
                [extK.dim(0), qHeads, qL, headDim], dtype: extK.dtype)
            outs.append(
                MLXFast.scaledDotProductAttention(
                    queries: q,
                    keys: extK,
                    values: extV,
                    scale: scale,
                    mask: .causal
                )
            )
        }
        eval(outs)
        // Scored decode walks N past 1024 (512 seed + 512 decode).
        // `sdpa_vector_2pass` on this arch bumps blocks 64→128 when N>1024.
        // The kL=1024 warm above compiles the 64-block family. Compile the
        // 128-block family at kL=1025 for the same qL={1,5,4} only.
        if extK.dim(2) == 1024 {
            let kPad1 = MLXArray.zeros(
                [extK.dim(0), extK.dim(1), 1, extK.dim(3)], dtype: extK.dtype)
            let vPad1 = MLXArray.zeros(
                [extV.dim(0), extV.dim(1), 1, extV.dim(3)], dtype: extV.dtype)
            let k1025 = concatenated([extK, kPad1], axis: 2)
            let v1025 = concatenated([extV, vPad1], axis: 2)
            var outs1025: [MLXArray] = []
            for qL in [1, 5, 4] {
                let q = MLXArray.zeros(
                    [k1025.dim(0), qHeads, qL, headDim], dtype: k1025.dtype)
                outs1025.append(
                    MLXFast.scaledDotProductAttention(
                        queries: q,
                        keys: k1025,
                        values: v1025,
                        scale: scale,
                        mask: .causal
                    )
                )
            }
            eval(outs1025)
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
        let tBegin0 = Self.traceRounds ? DispatchTime.now().uptimeNanoseconds : 0
        let cpuBegin0 = Self.traceRounds ? Self.threadCPUNanoseconds() : 0
        cache = model.newCache(parameters: nil)
        let (seedLogits, hidden) = model.callWithHidden(
            input: LMInput.Text(
                tokens: MLXArray(seedTokens).reshaped([1, seedTokens.count])),
            cache: cache, nConfirmed: 0)
        let tBeginBuilt = Self.traceRounds ? DispatchTime.now().uptimeNanoseconds : 0
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
        if Self.traceRounds {
            let tBeginDone = DispatchTime.now().uptimeNanoseconds
            let cpuBeginDone = Self.threadCPUNanoseconds()
            Self.traceWrite("mtp-trace: begin seed=\(seedTokens.count) "
                + "build_us=\((tBeginBuilt - tBegin0) / 1000) "
                + "eval_wall_us=\((tBeginDone - tBeginBuilt) / 1000) "
                + "wall_us=\((tBeginDone - tBegin0) / 1000) "
                + "cpu_us=\((cpuBeginDone - cpuBegin0) / 1000) "
                + "head_submit=\(Self.headSubmitPolicy)\n")
        }
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

    /// Consecutive fully-accepted DRAFTING rounds. Kept as a public-ish
    /// telemetry counter; the cost-model schedule below reads the per-position
    /// EMAs, not this.
    private var fullAcceptStreak = 0

    /// The last row of a head-chain hidden block. Every step after the first
    /// feeds ONE row in and gets ONE row back, and `lastHiddenWithKVOnlyHistory`
    /// already returns only the final row — so the trailing-row slice those
    /// call sites took was an identity slice on all but the flush step, costing
    /// a host graph node and a device op per PROPOSED token for nothing. The
    /// guard is on the shape, not on the call site, so a multi-row block still
    /// takes the real slice.
    @inline(__always)
    private static func lastHiddenRow(_ block: MLXArray) -> MLXArray {
        let rows = block.dim(1)
        guard rows > 1 else { return block }
        return block[0..., (rows - 1) ..< rows, 0...]
    }

    /// Local phase-trace gate, read once. `MLX_` prefix on purpose: the
    /// trusted harness strips `MLXFAST_*` from the sandboxed worker's env
    /// but allows the `MLX_` prefix through.
    ///
    /// Requires `MLXFAST_NO_SANDBOX=1` on the wrapper. The runtime worker
    /// sandbox in `writeRuntimeWorkerSandboxProfile` denies every write
    /// except `/dev/null`, including TMPDIR, so `traceSink` cannot open its
    /// file and silently degrades to the stderr fallback below. The
    /// `mtp-timed` parent then discards that stderr, because it calls
    /// `runtimeWorkerOptions` without `forwardsWorkerStderr`. Setting the
    /// trace variables alone therefore yields a clean run and no trace at
    /// all. `MLXFAST_NO_SANDBOX` is refused when
    /// `MLXFAST_OFFICIAL_BENCHMARK_RUN=1`, so this stays local-only.
    private static let traceRounds =
        ProcessInfo.processInfo.environment["MLX_QWEN_MTP_TRACE"] == "1"
    /// Attribution probe only. `verify_build_us` measures the window in which
    /// the host builds the verify graph WHILE the asynchronously submitted head
    /// chain runs on the GPU, so a head-chain stall is indistinguishable from
    /// host build cost there. Draining the chain before the window moves that
    /// GPU time into `draft_build_us`. Never enable on a timed candidate: it
    /// destroys the head/verify overlap the round is designed around.
    ///
    /// DRAINING THE CHAIN DOES NOT LEAVE `verify_build_us` AS PURE HOST GRAPH
    /// CONSTRUCTION, and an earlier version of this comment said it did. The
    /// window still contains the decode asyncEval ladder fired from
    /// `Qwen35TextModelInner.callAsFunction`, so the host blocks there on the
    /// MLX async-submission throttle while the GPU runs. E86 measured the split
    /// by removing the ladder (`MLX_QWEN_MTP_LADDER=off`), which is the only
    /// configuration that leaves the window free of GPU submissions. Declared
    /// head, 512 decode tokens, --sync-head, M4 Pro, median us per round:
    ///
    ///     ladder off:      verify_build   2,294   eval_wall  149,866
    ///     ladder shipped:  verify_build  72,330   eval_wall   77,092
    ///
    /// Host encode of the whole 64-layer verify graph is 2,294 us. Under the
    /// shipped ladder `verify_build_us` reads 72,330 us, so that counter is
    /// ~97 % GPU wait and only ~3 % host build. The two counters partition one
    /// GPU cost at the rung positions: their sum is flat at ~149.3 ms for every
    /// non-empty rung set. Read `verify_build_us + eval_wall_us` as the verify
    /// pipeline cost, and never read `verify_build_us` alone as host op-count
    /// evidence.
    ///
    /// In production, where the chain is not drained, the window also absorbs
    /// the head-chain GPU time: the same shipped rung set reads 83,202 us of
    /// `verify_build_us` and 4,766 us of `d_submit2_us`, against 72,330 us and
    /// 15,985 us under --sync-head. The ~11 ms difference is head GPU execute
    /// moving out of `d_submit2_us` and into the verify window, which is the
    /// overlap this flag exists to undo.
    private static let traceSyncHeadChain =
        ProcessInfo.processInfo.environment["MLX_QWEN_MTP_TRACE_SYNC_HEAD"] == "1"

    /// E90 rung 1: head-chain submission granularity.
    ///
    ///     ship     one `asyncEval` after the loop, as shipped
    ///     everyN   submit each Nth chain step as soon as it is built
    ///     mid      one extra submission at step ceil(d/2)
    ///
    /// SCHEDULING ONLY. Each arm evaluates the same graph, in the same order,
    /// with the same inputs, so every arm must emit the identical token stream
    /// and the identical round sequence. A mismatch is a defect in the arm.
    private static let headSubmitPolicy =
        ProcessInfo.processInfo.environment["MLX_QWEN_MTP_HEAD_SUBMIT"] ?? "ship"
    private static let headSubmitEvery: Int = {
        guard headSubmitPolicy.hasPrefix("every") else { return 0 }
        guard let n = Int(headSubmitPolicy.dropFirst("every".count)), n >= 1 else {
            preconditionFailure(
                "MLX_QWEN_MTP_HEAD_SUBMIT=\(headSubmitPolicy) is not everyN")
        }
        return n
    }()
    private static let headSubmitMid: Bool = {
        precondition(
            headSubmitPolicy == "ship" || headSubmitPolicy == "mid"
                || headSubmitEvery > 0,
            "MLX_QWEN_MTP_HEAD_SUBMIT=\(headSubmitPolicy) is not ship, mid or everyN")
        return headSubmitPolicy == "mid"
    }()

    @inline(__always)
    private static func headChainSubmits(step: Int, draftCount: Int) -> Bool {
        if headSubmitEvery > 0 { return step % headSubmitEvery == 0 }
        if headSubmitMid { return step == (draftCount + 1) / 2 }
        return false
    }
    /// Opened O_APPEND so the reference, verify and timed workers can each
    /// write the same file without a later process truncating an earlier
    /// one's rounds. Falls back to stderr when no path is configured, which
    /// the `mtp-timed` parent discards: `runtimeWorkerOptions` is called
    /// there without `forwardsWorkerStderr`, so it defaults to false and the
    /// drain installs a swallowing emitter.
    private static let traceSink: FileHandle = {
        guard let path = ProcessInfo.processInfo
            .environment["MLX_QWEN_MTP_TRACE_PATH"], !path.isEmpty
        else { return FileHandle.standardError }
        let fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        guard fd >= 0 else { return FileHandle.standardError }
        return FileHandle(fileDescriptor: fd, closeOnDealloc: false)
    }()

    private static func traceWrite(_ line: String) {
        traceSink.write(Data(line.utf8))
    }

    /// CPU nanoseconds this thread has consumed. `CLOCK_THREAD_CPUTIME_ID`
    /// advances only while the thread runs, so pairing it with the wall clock
    /// separates a descheduled host from a slow one.
    @inline(__always)
    private static func threadCPUNanoseconds() -> UInt64 {
        var value = timespec()
        guard clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0 else { return 0 }
        return UInt64(value.tv_sec) * 1_000_000_000 + UInt64(value.tv_nsec)
    }

    /// Exact-value row dump for the LOCAL width-wall gate: hexfloat (`%a`)
    /// per declared top-2 value so the serial leg's rows and a wide round's
    /// rows can be compared BIT-FOR-BIT by position — the comparison the
    /// local argmax-only reference check does not do and the ranked ledger
    /// replay does. Same env gate as the phase trace; never on at rank.
    private static func traceRow(pos: Int, ids: [Int], values: [Double]) {
        guard traceRounds else { return }
        let hex = values.map { String(format: "%a", $0) }.joined(separator: ",")
        traceWrite("mtp-row: pos=\(pos) ids=\(ids[0]),\(ids[1]) v=\(hex)\n")
    }

    // MARK: - cost-model depth schedule

    /// Per-position acceptance EMAs: `positionAcceptEMA[i]` estimates
    /// P(draft i accepted | drafts 0..<i accepted). Seeded with an optimistic,
    /// gently decaying prior so the first rounds draft rather than stall; the
    /// EMA half-life (~9 observed rounds at 0.15) adapts well inside a
    /// 512-token window while surviving one unlucky reject.
    /// PRIORS: optimistic-decaying, by measurement. The real-prose
    /// production conditionals (0.92/0.70/0.50, MTPLX) were tried and taxed
    /// the ramp two extra rounds on easy prose (22.15 vs 21.5 local) — and
    /// the published MEDIAN is set by the easy-mid prompts, so a ramp tax
    /// lands exactly where it hurts. The EMAs converge to the prompt's
    /// truth within ~10 rounds regardless; what protects the hard prompts
    /// is the 0.95 optimism CAP below (the p5 over-draft bug was the
    /// uncapped transfer, not the prior).
    private var positionAcceptEMA: [Double] = (0 ..< Qwen36MTPLimits.maxDepth)
        .map { 0.85 * pow(0.98, Double($0)) }
    private static let acceptEMAAlpha = 0.15

    /// h = (one head draft step) / (one batched verify forward), the only
    /// constant the marginal rule needs. Derivation from the campaign's
    /// measured budgets: the verify forward is weight-stream bound on the
    /// ~14.1 GiB 4-bit backbone and near-flat in width; a head step streams
    /// the head layer plus the full lm_head readout (~0.65 GiB 4-bit) and
    /// carries the chained-launch overhead of the committed-history path.
    /// h HISTORY, because it was mispriced twice. 0.12 (arm 1) and 0.09
    /// (arm 2) both divided total window time by rounds WITHOUT subtracting
    /// the ~0.9 s seed prologue charged inside the local window — a prologue
    /// artifact that made depth look nearly free. Steady-state regression on
    /// the phase-traced receipts (draft_build ≈ 2.4 ms/step CPU, eval_wall
    /// 79→89→106 ms for widths 7→8→9) puts the TRUE marginal cost of an
    /// extra draft at ~10-16 ms against a ~24-40 ms round base: h ≈ 0.6 on
    /// the bf16-head (pinned) stack. Underpricing h over-drafts d=6-8 on
    /// hard hidden prompts — invisible on degenerate local prose at accept
    /// ≈ 1.0, and worth up to -20% on a per-pair tail. Re-fit from
    /// forced-depth arms after every head-variant change.
    ///
    /// FOURTH FIT — and the resolution of the 0.20-vs-0.43 dispute. The
    /// capped-regime phase trace measured ~10.75 ms marginal per draft on a
    /// ~27 ms base (0.20) in the fully-accepted case. MTPLX ships a
    /// break-even of ~0.43 — but their reject pays a REPAIR FORWARD, while
    /// this stack's per-row GDN checkpoints make a prefix reject nearly
    /// free (restoreAfterPrefixReject, no repair at any depth). Their
    /// constant prices a cost this stack deleted; 0.40 measured -4.5% on
    /// the easy-prose receipt (held d2-3 where d4 pays). A fit near 0.20 is
    /// the honest READING of that trace FOR THIS ROLLBACK MECHANISM; the
    /// shipped level is 0.18, which is the end-to-end optimum bracketed on
    /// both sides by ranked receipts (0.14 -> 2.766, 0.15 -> 2.667,
    /// 0.32 -> 2.84585). The wasted-work term a reject does keep (the
    /// drafted head steps past the break) is already inside the marginal
    /// the rule prices.
    private static let headStepCostRatio = 0.18

    /// E68: the depth price as a per-position vector.
    ///
    /// `headStepCostRatio` prices every extra draft the same, so the shipped
    /// cost model is `T(d) = V + d * h * V` with the verify forward `V` flat
    /// in width. The measured verify curve is not flat in width: the QMV
    /// dispatch table changes group shape at several widths, so the step into
    /// one width can cost a multiple of the step into its neighbour.
    ///
    /// Every arm holds the total at `maxDepth * headStepCostRatio`, so an arm
    /// changes the SHAPE of the price and never its level. The level is
    /// already measured: `h = 0.32` scored 2.84585, a clean -3%, because it
    /// shortened every draft. This pool rewards depth, so E68 asks only
    /// whether the price is distributed correctly across positions.
    internal struct DepthPrice {
        /// `marginal[d]` prices the step into verify width `d + 2`.
        let marginal: [Double]
        /// `cumulative[d]` is the running cost BEFORE step `d` is taken, so
        /// `cumulative[0]` is 1.0: the verify forward on its own.
        let cumulative: [Double]
    }

    /// The one-boundary tier factor E56 fitted, retained so `pb5` and `pb7`
    /// reproduce that experiment's published arithmetic exactly.
    internal static let boundaryTierFactor = 2.0301

    /// The shipped flat price. `cumulative` repeats the tip's closed form
    /// instead of accumulating: `1.0 + 0.18 + 0.18 + 0.18` and
    /// `1.0 + 3.0 * 0.18` differ by one ulp, and a control arm that is not
    /// bit-identical to the tip is not a control.
    internal static func makeUniformDepthPrice() -> DepthPrice {
        DepthPrice(
            marginal: [Double](repeating: headStepCostRatio,
                               count: Qwen36MTPLimits.maxDepth),
            cumulative: (0 ... Qwen36MTPLimits.maxDepth).map {
                1.0 + Double($0) * headStepCostRatio
            })
    }

    /// One priced boundary, holding the total. `width` is the verify width
    /// the priced step ENTERS, so it selects index `width - 2`.
    internal static func makeBoundaryDepthPrice(
        enteringVerifyWidth width: Int
    ) -> DepthPrice {
        let count = Qwen36MTPLimits.maxDepth
        let within = Double(count) * headStepCostRatio
            / (Double(count - 1) + boundaryTierFactor)
        var marginal = [Double](repeating: within, count: count)
        marginal[width - 2] = within * boundaryTierFactor
        return DepthPrice(marginal: marginal,
                          cumulative: prefixCosts(marginal))
    }

    /// `headStepCostRatio + (C(d + 2) - C(d + 1)) / V` from the E68 rung-1
    /// session, before rescaling. `C(M)` is the whole-table isolated QMV cost
    /// at verify width `M`, median of the three shipped legs of rung-1 job
    /// `21ac5458`, and `V = 0.060300` s is the verify-forward normaliser that
    /// session named. `makeMeasuredDepthPrice` rescales to the shipped total,
    /// so only the SHAPE of these numbers reaches the scheduler; the level
    /// stays at `maxDepth * headStepCostRatio`. An empty array is a trap
    /// rather than a silent fallback to `ship`.
    ///
    /// E68 rung 3 measured this shape end to end: candidate MTP seconds per
    /// token 0.031457267 -> 0.030356223, **-3.500 % against a 0.143 % null**,
    /// nine mirrored-palindrome legs, real 40 C gate on every leg, one
    /// byte-identical 513-token stream `da92be8a0dc02229` on all nine.
    ///
    /// The shape is fitted to the CURRENT kernel dispatch table, whose step
    /// into verify width 6 costs 27.308 ms against 13.405 ms for the step
    /// into width 5. A change to that table invalidates the fit, not only its
    /// magnitude. Refit from a fresh rung-1 curve whenever the QMV group
    /// shapes move.
    internal static let measuredRawDepthPrice: [Double] = [
        0.26300121724709807,
        0.29195567495854047,
        0.34642143034825884,
        0.40231023217247086,
        0.63287276451077956,
        0.43601634825870655,
        0.35457813598673293,
        0.42510483416251998,
    ]

    internal static func makeMeasuredDepthPrice() -> DepthPrice {
        precondition(
            measuredRawDepthPrice.count == Qwen36MTPLimits.maxDepth,
            "E68 pbfit: measuredRawDepthPrice is not filled from rung 1")
        let total = Double(Qwen36MTPLimits.maxDepth) * headStepCostRatio
        let scale = total / measuredRawDepthPrice.reduce(0.0, +)
        let marginal = measuredRawDepthPrice.map { $0 * scale }
        return DepthPrice(marginal: marginal,
                          cumulative: prefixCosts(marginal))
    }

    internal static func prefixCosts(_ marginal: [Double]) -> [Double] {
        var out = [1.0]
        var running = 1.0
        for value in marginal {
            running += value
            out.append(running)
        }
        return out
    }

    internal enum DepthPriceArm: String {
        case ship, pb5, pb7, pbfit
    }

    /// THE ONE LINE AN ARM SESSION PATCHES. `QwenMTPDepthPriceTests` pins the
    /// shipped value so a leg session cannot leave another arm behind.
    ///
    /// The shipped default is `ship` (uniform). `pbfit` wins by -3.5 % on this
    /// host's kernel dispatch table and loses that win entirely on the crown
    /// table (E75 rung B/D: +0.33 % on crown, a +3.8 pp interaction). The
    /// shape is fitted to one dispatch table, so it is a research arm, not a
    /// shipped constant. Refit and re-price on the live table before shipping
    /// any non-uniform shape.
    internal static let depthPriceArm: DepthPriceArm = .ship

    /// Built once. A computed property here would allocate two arrays on
    /// every round, inside the timed path.
    internal static let depthPrice: DepthPrice = {
        switch depthPriceArm {
        case .ship: return makeUniformDepthPrice()
        case .pb5: return makeBoundaryDepthPrice(enteringVerifyWidth: 5)
        case .pb7: return makeBoundaryDepthPrice(enteringVerifyWidth: 7)
        case .pbfit: return makeMeasuredDepthPrice()
        }
    }()

    /// HARD DEPTH CAP 4 — WIDTHS ABOVE 5 ARE STRUCTURALLY CLOSED on this
    /// stack, by bitwise measurement (hexfloat row gate, two attempts):
    /// verify widths 6-9 drift from the serial trajectory in top-2 VALUES
    /// (ids hold) even with (a) <= 5-row query chunking and (b) per-row
    /// prefix-sliced sdpa at exactly the serial kL — identical mismatch
    /// pattern both times, so the attention was never the (only) source;
    /// the gated-delta scan's internal chunk geometry changes above S=5
    /// (the invariant-#7 note warned about exactly this). Worse, the
    /// drifted K/V rows the wide forwards write CONTAMINATE every later
    /// round — a single wide round poisons the whole window under the
    /// ranked exact-value replay, while staying invisible to the local
    /// argmax-only check. Width 5 measured 5/5 bit-exact, which is why
    /// every promoted receipt at cap 4 survived rank. Do not raise this
    /// without a bit-exact >width-5 GDN scan AND a fresh hexfloat row gate.
    ///
    /// RESOLUTION of the wall's mechanism, and the door through it: the GDN
    /// scan kernel is sequential in T with T-independent per-row arithmetic
    /// (one register-resident fp32 state walked t = 0..<T), so the scan was
    /// never the drift source. Quantized projections at M in 6..9 still ride
    /// the per-row-exact QMV dispatch (host qmv batch limit 10+ on this
    /// generation for these shapes). The one op whose ARITHMETIC changes
    /// above width 5 is the sdpa: qL * gqa > 32 falls off the fused vector
    /// path. `attentionWithCacheUpdate` therefore splits a 6..9-row causal
    /// decode attention into two <= 5-row sdpa calls whose bottom-right-
    /// aligned windows are byte-identical to the promoted <= 5 rounds' —
    /// after which a deep round is ONE ordinary model call. Measured on the
    /// hexfloat row gate: widths 6..8 bit-exact per position against the
    /// serial trajectory. Segmenting the whole FORWARD instead (two model
    /// calls, 5+k) was measured bit-exact too but pays a second full weight
    /// pass (~25 ms) and loses on net; the chunk lives at the sdpa only.
    private static let sdpaWidthWallDepthCap = 5

    /// Depth cap for streak-qualified deep rounds. 8 is the trusted
    /// per-round maximum; rows_per_round = depth + 1 stays ledger-legal.
    /// Gated on a full-accept streak so the deep rounds only fire where the
    /// head has been perfect, mirroring the streak ladder that qualified
    /// cap 4; any reject resets the streak.
    private static let segmentedVerifyDepthCap = 7
    /// 2, not 3 — the FOURTH restore of this literal, and it has still never
    /// lost on its merits.
    ///
    /// Ranked history: newjordan 2.91995 (PROMOTED, then reverted by a later
    /// archive that happened to carry 3); hadakang 2.92976 against the 2.92622
    /// frontier of its day (beat its contemporary crown, lost the race); and my
    /// own `4650c96e` scored 2.93524 against the 2.93429 base it was built on
    /// (+0.03%) and again lost only because the crown moved to 2.94662 while it
    /// validated. Three independent runs, three times ahead of its own base.
    ///
    /// A fourth, independent line of evidence, from a negative result of mine.
    /// I raised `headStepCostRatio` 0.18 -> 0.32 on the directly measured
    /// marginal (`fc62d1aa`): it scored 2.84585, a clean -3% with the baseline
    /// leg FLAT (0.038092 -> 0.038070, so not a draw artifact). It shortened
    /// every draft — 4.35/4.89/5.78/5.33/5.04 -> 3.36/4.01/4.53/4.03/4.76 — and
    /// candidate decode time ROSE 0.95%. **This pool rewards depth**: the
    /// marginal draft is worth more than its verify row costs. With 0.15 (2.667)
    /// and 0.14 (2.766) failing below, h is now bracketed on both sides and 0.18
    /// is a true local optimum — so the way to buy depth is NOT the price.
    ///
    /// It is the cap, and that is what makes it safe. `h` moves the marginal
    /// rule on EVERY round including the hard prompts (0.32 dragged prompt 6
    /// from 0.17 drafts to 0.06). This gate is conditioned on OBSERVED perfect
    /// acceptance and any reject resets `fullAcceptStreak` to 0, so it cannot
    /// touch a cold or hard prompt at all — it only shortens the
    /// re-qualification ramp on stretches the head is already proving. Gate 1 is
    /// measured dead (2.833, -7.1%); gate 0 only tied (2.9200).
    private static let segmentedStreakGate = 2

    /// The greedy marginal-depth rule described at the policy's assignment.
    private func costModelDepth(offeredDepth: Int) -> Int {
        // The width wall binds the SINGLE-CALL verify; a qualifying
        // full-accept streak opens the segmented cap (the round then feeds
        // the target <= 5-row segments, never a wider launch). Any reject
        // resets the streak, so a cold or struggling prompt never sees a
        // deep round.
        // FLAT CAP 7, NO GATE. Two ranked receipts of the organizer frontier,
        // each isolating one half of this: capping at 7 gave `919318e1` (always
        // the x4 slot) 0.011967 s/tok at draft length 4.32 — still the fastest
        // reading for that prompt from any tree on this board — while removing
        // the streak gate's width-wall FLOOR gave `00142a44` 0.011070 at draft
        // length 5.10, a -2.6% move to within 0.4% of its own board minimum.
        //
        // The floor, not the ceiling, is what was truncating `00142a44`: it
        // drops the cap to `sdpaWidthWallDepthCap` on every round after a
        // reject, cutting rounds the marginal rule would have taken deeper. So
        // the ceiling stays at 7 and the floor goes, which is both receipts in
        // one schedule.
        //
        // Safety does not depend on the gate: `reach` is a product of the
        // per-position acceptance EMAs and collapses on a cold stretch by
        // itself. Widths 6..8 are bit-exact per position against the serial
        // trajectory through the sdpa exactness chunk, so 7 is policy.
        //
        // A floor of 6 under the ceiling of 7 was tried and REVERTED. The
        // argument for it read the x4 slot's 4.38 / 0.012191 s/tok off
        // `89cbdc02` as a depth error, but `89cbdc02` was a slow TREE: on the
        // four submissions that ran this exact eight-prompt schedule, plutarch
        // (92% non-drafting) is flat to +-0.1% while every drafting prompt
        // spreads 1.2-2.5%. The x4 cost was drafting-path cost, not depth.
        // Re-measured on a fast tree, this flat cap took x4 to 0.011979 and
        // the median to 3.30955573 against 3.30221310 for the floor.
        //
        // Imported from promoted submission c6af1e24 (organizer 88578f92,
        // official 3.30955573); it supersedes ead84bba (official 3.30221310).
        let widthCap = Self.segmentedVerifyDepthCap
        let cap = Swift.min(
            Swift.min(offeredDepth, Qwen36MTPLimits.maxDepth),
            widthCap)
        // Snapshot BEFORE the walk and before any drafting: by the time the
        // round's trace line is emitted, the EMAs, the streak and `pendingTop2`
        // have all been advanced by this round's own outcome, so reading them
        // there would describe the next round's inputs, not this one's.
        if Self.traceRounds { snapshotScheduleSignal(widthCap: widthCap) }
        guard cap > 0 else { return 0 }
        let price = Self.depthPrice
        var reach = 1.0
        var expected = 0.0
        var depth = 0
        while depth < cap {
            var p = positionAcceptEMA[depth]
            if depth == 0, let tail = pendingTop2, tail.1.count >= 2 {
                let margin = tail.1[0] - tail.1[1]
                let conf = 1.0 / (1.0 + exp(-margin / 2.0))
                p = Swift.min(p, conf)
            } else if depth == 1, let tail = pendingTop2, tail.1.count >= 2 {
                let margin = tail.1[0] - tail.1[1]
                let conf2 = 1.0 / (1.0 + exp(-margin / 3.0))
                p = Swift.min(p, conf2)
            }
            reach *= p
            let threshold = price.marginal[depth] * (1.0 + expected) /
                price.cumulative[depth]
            if Self.traceRounds {
                scheduleTrace += String(
                    format: "%d:%.6f/%.6f/%.6f;", depth, p, reach, threshold)
            }
            guard reach > threshold else { break }
            expected += reach
            depth += 1
        }
        return depth
    }

    /// Trace-gated record of the schedule's inputs and its extension walk.
    /// Written only when the phase trace is on, so the scored schedule runs
    /// byte-identical arithmetic without it.
    private var scheduleTrace = ""

    /// Every scalar the schedule may legally read BEFORE it proposes anything:
    /// the pending primary's target top-2 margin, the per-position EMAs, the
    /// full-accept streak and the width cap in force. Recorded so an offline
    /// fit can ask which of these separates a round that accepts its whole
    /// chain from one that accepts nothing, without spending a second run.
    private func snapshotScheduleSignal(widthCap: Int) {
        let margin: Double
        if let tail = pendingTop2, tail.1.count >= 2 {
            margin = tail.1[0] - tail.1[1]
        } else {
            margin = Double.nan
        }
        let emas = positionAcceptEMA
            .map { String(format: "%.6f", $0) }.joined(separator: ",")
        scheduleTrace = "arm=" + Self.depthPriceArm.rawValue + " " + String(
            format: "m=%.6f streak=%d cap=%d ema=",
            margin, fullAcceptStreak, widthCap) + emas + " sched="
    }

    /// Fold one round's acceptance outcome into the per-position EMAs.
    /// Positions before the accepted count observed a success; the position
    /// AT the accepted count observed a failure only if the walk actually
    /// rejected there (not when it ended early on a committed stop token);
    /// deeper positions were never reached and observe nothing.
    private func recordAcceptOutcome(acceptedCount: Int, drafts: [Int]) {
        let alpha = Self.acceptEMAAlpha
        for index in 0 ..< acceptedCount where index < positionAcceptEMA.count {
            positionAcceptEMA[index] += alpha * (1.0 - positionAcceptEMA[index])
        }
        let stoppedEarly = acceptedCount > 0 && acceptedCount <= drafts.count
            && stopTokens.contains(drafts[acceptedCount - 1])
        if acceptedCount < drafts.count, !stoppedEarly,
           acceptedCount < positionAcceptEMA.count
        {
            positionAcceptEMA[acceptedCount] +=
                alpha * (0.0 - positionAcceptEMA[acceptedCount])
        } else if acceptedCount == drafts.count, !drafts.isEmpty,
                  acceptedCount < positionAcceptEMA.count
        {
            // Optimism transfer: a FULLY accepted round is evidence about the
            // position just past the round's depth too — the chain was hot and
            // only the schedule ended it. Without this the first unreached
            // position keeps its cold prior and the product-of-EMAs reach can
            // never clear the deep threshold inside a short window; this is
            // the streak ladder's widening step, recast as evidence. Capped
            // at 0.95: transferred optimism is inference, not observation,
            // and deep positions never merit a certainty estimate
            // without treating that inference as a real observation.
            if positionAcceptEMA[acceptedCount] < 0.95 {
                positionAcceptEMA[acceptedCount] +=
                    alpha * (0.95 - positionAcceptEMA[acceptedCount])
            }
        }
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
        guard began, let primaryPending = pendingPrimary,
              pendingTop2 != nil, let hidden = pendingHidden
        else { throw Qwen36MTPSessionError.notBegun }
        guard depth >= Qwen36MTPLimits.serialControlDepth,
              depth <= Qwen36MTPLimits.maxDepth
        else {
            throw Qwen36MTPSessionError.invalidDepth(depth)
        }
        roundCount += 1
        // Local-only phase trace (MLXFAST_QWEN_MTP_TRACE=1): three boundaries
        // split a round into head-chain graph build, verify graph build, and
        // the single blocking eval's GPU wall. Never on in a ranked run.
        let tRound0 = Self.traceRounds ? DispatchTime.now().uptimeNanoseconds : 0
        let cpuRound0 = Self.traceRounds ? Self.threadCPUNanoseconds() : 0
        var tDraftBuilt: UInt64 = 0
        var tSnapshotDone: UInt64 = 0
        var tVerifyBuilt: UInt64 = 0
        var tEvalDone: UInt64 = 0
        var tReadDone: UInt64 = 0
        var tCommitDone: UInt64 = 0

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

        // A STOP TOKEN IS COMMITTED LIKE ANY OTHER TOKEN, and this round keeps
        // drafting past it. The parent owns the decode window: its loop runs to
        // the configured total and it checks every emitted index against the
        // serial trajectory (`QwenRuntimeMTPDriver.swift` :121, :216-226), which
        // the shipped 1024-token golden continues for 722 tokens past its first
        // `248044`. Ending the round here instead nilled the pendings and killed
        // the session for good -- the next round threw `.notBegun` -- which
        // capped both legs of every local window at 301 tokens.

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
            Self.traceRow(
                pos: seedTokenCount + committedTokenCount,
                ids: tailTokens, values: tailLogits)
            return Qwen36MTPRoundResult(
                tokens: committed,
                declaredRows: 1,
                draftTokens: [],
                acceptedDraftCount: 0,
                rejectedDraftCount: 0,
                perRowTop2Tokens: [tailTokens],
                perRowTop2Logits: [tailLogits],
                targetCacheOffset: seedTokenCount + committedTokenCount
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
        let tDraft0 = Self.traceRounds
            ? DispatchTime.now().uptimeNanoseconds : 0
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
                flushHidden.append(
                    model.applyFinalNorm(seedHidden[0..., 0 ..< primeCount, 0...]))
                flushTokens.append(contentsOf: seedTokensForPriming[1...])
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
        // (Per-step asyncEval was tried here and measured NEUTRAL — the
        // ~2.4 ms/step is host graph BUILD, not GPU work to overlap; see
        // idea.md V6 journal. Single submission after the loop, as before.)
        let tFlushBuilt = Self.traceRounds
            ? DispatchTime.now().uptimeNanoseconds : 0
        var draftIdArrays: [MLXArray] = []
        var headHidden = model.mtpHeadLastHiddenWithKVOnlyHistory(
            hidden: draftInputHidden, nextTokenIds: draftInputTokens,
            cache: headCache)
            ?? model.mtpHeadHiddenForward(
                hidden: draftInputHidden, nextTokenIds: draftInputTokens,
                cache: headCache)
        var draftHidden = Self.lastHiddenRow(headHidden)
        var draftId = model.draftTokenID(draftHidden)
        draftIdArrays.append(draftId)
        // Early submission of the FIRST head step: its graph exists ~2.4 ms
        // before the rest of the chain is built, and unlike the per-step
        // variant (measured neutral — nothing but build time between steps)
        // the first step carries the history flush, which IS real GPU work
        // the device can start while the host builds steps 2..d.
        let tHead1Built = Self.traceRounds
            ? DispatchTime.now().uptimeNanoseconds : 0
        asyncEval(draftId)
        let tSubmit1 = Self.traceRounds
            ? DispatchTime.now().uptimeNanoseconds : 0
        for step in 1 ..< draftCount {
            headHidden = model.mtpHeadHiddenForward(
                hidden: draftHidden, nextTokenIds: draftId, cache: headCache)
            draftHidden = Self.lastHiddenRow(headHidden)
            draftId = model.draftTokenID(draftHidden)
            draftIdArrays.append(draftId)
            if Self.headChainSubmits(step: step, draftCount: draftCount) {
                asyncEval(draftId)
            }
        }
        let tChainBuilt = Self.traceRounds
            ? DispatchTime.now().uptimeNanoseconds : 0
        asyncEval(draftIdArrays[draftIdArrays.count - 1])
        if Self.traceSyncHeadChain {
            eval(draftIdArrays[draftIdArrays.count - 1])
        }
        if Self.traceRounds { tDraftBuilt = DispatchTime.now().uptimeNanoseconds }

        // 2. Keep the generic pre-verify snapshot as a fallback, but use the
        //    vendored post-primary rollback checkpoint for the hot K=1 path. A
        //    rejected single draft can then retain the primary's target work and
        //    discard only the draft token instead of re-forwarding the primary.
        let snapshot = Self.snapshotRecurrent(cache)
        if Self.traceRounds { tSnapshotDone = DispatchTime.now().uptimeNanoseconds }
        let verifyTokens = concatenated(
            [MLXArray([Int32(primary)]).reshaped([1, 1])] + draftIdArrays,
            axis: 1)
        // nConfirmed: 1 at every drafting width. K=1 writes its promoted eager
        // primary checkpoint; K>=2 keeps exact recurrence inputs so a partial
        // accept can replay only its committed prefix without a repair forward.
        //
        // Widths 6..9 ride the SAME single call: every quantized projection
        // at M in 6..9 still routes through the per-row-exact QMV dispatch
        // (the host's qmv batch limit is 10+ on this generation for these
        // shapes), the GDN scan kernel is sequential in T with T-independent
        // per-row arithmetic, and the one op that DID change arithmetic
        // above width 5 — the fused sdpa vector path's qL bound — is handled
        // by the exactness chunk inside `attentionWithCacheUpdate` (two
        // <= 5-row sdpa calls, byte-identical windows). One tape, one
        // rollback story, one readout, no second weight pass.
        // Publish the post-norm block this verify forward already computes so
        // accepted head-history rows do not each repeat the same row-local
        // RMSNorm through applyFinalNorm. Conformers that return nil retain the
        // old path through the guarded hiddenRow overload below.
        let (verifyLogits, verifyHidden, verifyNormed) =
            model.callWithHiddenAndNormed(
                input: LMInput.Text(tokens: verifyTokens),
                cache: cache, nConfirmed: 1)
        if Self.traceRounds { tVerifyBuilt = DispatchTime.now().uptimeNanoseconds }

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
        if Self.traceRounds { tEvalDone = DispatchTime.now().uptimeNanoseconds }

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

        if Self.traceRounds { tReadDone = DispatchTime.now().uptimeNanoseconds }

        if acceptedCount == drafts.count {
            // FULL ACCEPTANCE: the verify state IS the committed state. No
            // rollback, no repair forward; the bonus row carries the next primary
            // and the last hidden row seeds the next draft.
            Self.clearRecurrentRollback(cache)
            committed.append(contentsOf: drafts)
            committedTokenCount += drafts.count
            pendingPrimary = verifyArgmax[drafts.count]
            pendingHidden = hiddenRow(
                verifyHidden, verifyNormed, verifyHidden.dim(1) - 1)
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
                model, cache,
                acceptedCount: acceptedCount, draftCount: draftCount,
                to: committedOffset)
            {
                pendingPrimary = verifyArgmax[acceptedCount]
                pendingHidden = hiddenRow(
                    verifyHidden, verifyNormed, acceptedCount)
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

        if Self.traceRounds { tCommitDone = DispatchTime.now().uptimeNanoseconds }

        // Head-history upkeep. Trim the speculative deeper-draft rows back to
        // the valid prefix, then queue the ACCEPTED transitions for the next
        // drafting round's flush: row i of the verify output is the trunk
        // hidden at draft i's position, so (hiddenRow(i), drafts[i]) is the
        // committed pair. The rejecting round queues nothing — the next
        // round's own (pendingHidden, primary) row covers that transition.
        Self.trimTrimmable(headCache, to: validHistoryOffset)
        if acceptedCount > 0 {
            // Keep accepted post-norm rows as one contiguous block. The backlog
            // already supports multi-row blocks (seed priming uses one), while
            // the token list remains flat and preserves the same row order.
            if let block = normedRows(
                verifyHidden, verifyNormed, 0 ..< acceptedCount)
            {
                headHistoryBacklogHidden.append(block)
            } else {
                // Preserve the exact pre-existing per-row normalization path
                // whenever no matching published block is available.
                for index in 0 ..< acceptedCount {
                    headHistoryBacklogHidden.append(
                        hiddenRow(verifyHidden, index))
                }
            }
            headHistoryBacklogTokens.append(
                contentsOf: drafts.prefix(acceptedCount))
        }
        fullAcceptStreak =
            acceptedCount == drafts.count ? fullAcceptStreak + 1 : 0
        recordAcceptOutcome(acceptedCount: acceptedCount, drafts: drafts)
        if Self.traceRounds {
            // Row i's distribution follows (primary + drafts[0..<i]); only
            // rows on the accepted trajectory align with the serial leg.
            let rowBase = expected + 1
            for index in 0 ... acceptedCount where index < perRowTop2Tokens.count {
                Self.traceRow(
                    pos: rowBase + index,
                    ids: perRowTop2Tokens[index],
                    values: perRowTop2Logits[index])
            }
        }

        acceptedDraftTotal += acceptedCount
        rejectedDraftTotal += drafts.count - acceptedCount
        if Self.traceRounds {
            // Five-way split of the round. `eval_wall` is the only segment the
            // GPU owns; everything after it is host time that the device could
            // in principle be overlapping, so the tail segments are the budget
            // for any further pipelining work.
            let tTailDone = DispatchTime.now().uptimeNanoseconds
            let line = "mtp-trace: round=\(roundCount) d=\(draftCount) "
                + "acc=\(acceptedCount) "
                + "draft_build_us=\((tDraftBuilt - tRound0) / 1000) "
                // Complete split of draft_build, so a first-round cold cost
                // names the statement that pays it instead of the section.
                + "d_pre_us=\((tDraft0 - tRound0) / 1000) "
                + "d_flush_us=\((tFlushBuilt - tDraft0) / 1000) "
                + "d_head1_us=\((tHead1Built - tFlushBuilt) / 1000) "
                + "d_submit1_us=\((tSubmit1 - tHead1Built) / 1000) "
                + "d_chain_us=\((tChainBuilt - tSubmit1) / 1000) "
                + "d_submit2_us=\((tDraftBuilt - tChainBuilt) / 1000) "
                + "verify_build_us=\((tVerifyBuilt - tDraftBuilt) / 1000) "
                + "eval_wall_us=\((tEvalDone - tVerifyBuilt) / 1000) "
                + "readout_us=\((tReadDone - tEvalDone) / 1000) "
                + "commit_us=\((tCommitDone - tReadDone) / 1000) "
                + "upkeep_us=\((tTailDone - tCommitDone) / 1000) "
                + "round_us=\((tTailDone - tRound0) / 1000) "
                // Thread CPU nanoseconds this round consumed, beside the wall
                // clock. A round whose wall time rises while this stays flat
                // lost the CPU to something else; a round where both rise ran
                // the same host work at a lower clock. E89 measures the same
                // field on a second host under the same name and units.
                + "host_thread_cpu_ns=\(Self.threadCPUNanoseconds() &- cpuRound0) "
                + scheduleTrace + "\n"
            Self.traceWrite(line)
            // Absolute anchors on the mach uptime clock, so an offline reader
            // can intersect the round's inter-anchor windows with the GPU
            // execution intervals E90GPUIntervals records on the same axis.
            // `verify_build_us` above keeps its historical meaning (it still
            // spans the recurrent snapshot), and the split appears here.
            Self.traceWrite(
                "mtp-anchor: round=\(roundCount) d=\(draftCount) "
                    + "acc=\(acceptedCount) "
                    // One trace file collects every worker a leg spawns, and
                    // the GPU interval ledger is per process, so the reader
                    // needs the pid to join the two without mixing workers.
                    + "pid=\(ProcessInfo.processInfo.processIdentifier) "
                    + "t_round0=\(tRound0) t_draft0=\(tDraft0) "
                    + "t_flush_built=\(tFlushBuilt) t_head1_built=\(tHead1Built) "
                    + "t_submit1=\(tSubmit1) t_chain_built=\(tChainBuilt) "
                    + "t_draft_built=\(tDraftBuilt) "
                    + "t_snapshot_done=\(tSnapshotDone) "
                    + "t_verify_built=\(tVerifyBuilt) t_eval_done=\(tEvalDone) "
                    + "t_read_done=\(tReadDone) t_commit_done=\(tCommitDone) "
                    + "t_tail_done=\(tTailDone)\n")
        }
        // No trailing eval: every host-read value was materialised by the
        // round bundle above. A successful wide-prefix replay intentionally
        // installs lazy recurrent roots; only the next GPU graph consumes
        // them. The rare generic-repair path ran its own second eval.
        // `pendingHidden` is likewise device-only until the next round.

        return Qwen36MTPRoundResult(
            tokens: committed,
            declaredRows: draftCount + 1,
            draftTokens: drafts,
            acceptedDraftCount: acceptedCount,
            rejectedDraftCount: drafts.count - acceptedCount,
            perRowTop2Tokens: perRowTop2Tokens,
            perRowTop2Logits: perRowTop2Logits,
            targetCacheOffset: seedTokenCount + committedTokenCount
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
                arrays.prefixReplayTape = nil
                continue
            }
            if entry.isTrimmable, entry.offset > base {
                _ = entry.trim(Swift.min(verifiedTokens, entry.offset - base))
            }
        }
    }

    /// Restore the committed boundary from a width-S verify with
    /// `nConfirmed == 1`. K=1 consumes its eager checkpoint. K>=2 replays
    /// `acceptedCount + 1` target rows from the exact pre-verify recurrent
    /// state. Both paths then trim exactly the rejected attention rows.
    ///
    /// Preflight every layer before mutating any of them. Returning `false`
    /// leaves the cache untouched so the caller can use the generic snapshot
    /// and repair path safely.
    private static func restoreAfterPrefixReject(
        _ model: any Qwen36MTPTarget,
        _ cache: [any KVCache],
        acceptedCount: Int,
        draftCount: Int,
        to committedOffset: Int
    ) -> Bool {
        let rejected = draftCount - acceptedCount
        guard rejected > 0 else { return false }

        // Preserve the officially validated eager K=1 path byte-for-byte.
        // Wider verifies retain a compact recurrence tape instead of eagerly
        // materialising one fp32 state at every possible boundary.
        if draftCount > 1 {
            for entry in cache where !(entry is ArraysCache) {
                guard entry.isTrimmable,
                      entry.offset == committedOffset + rejected
                else { return false }
            }
            guard model.replayRecurrentPrefix(
                cache: cache, committedRows: acceptedCount + 1)
            else { return false }
            for entry in cache where !(entry is ArraysCache) {
                if entry.isTrimmable, entry.offset > committedOffset {
                    _ = entry.trim(entry.offset - committedOffset)
                }
            }
            return true
        }

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
                arrays.prefixReplayTape = nil
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
                arrays.prefixReplayTape = nil
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

    /// Slice rows from a matching post-norm block when the verify forward
    /// published one. RMSNorm reduces only over the final axis, so these are
    /// the same values as normalizing each row again.
    private func normedRows(
        _ hidden: MLXArray, _ normed: MLXArray?, _ range: Range<Int>
    ) -> MLXArray? {
        guard postNorm, let normed,
              hidden.ndim == 3, normed.ndim == 3,
              normed.dim(0) == hidden.dim(0),
              normed.dim(1) == hidden.dim(1),
              normed.dim(2) == hidden.dim(2),
              range.lowerBound >= 0,
              range.upperBound <= normed.dim(1),
              !range.isEmpty
        else { return nil }
        return normed[0..., range, 0...]
    }

    /// Any shape surprise falls back to the pre-existing per-row path.
    private func hiddenRow(
        _ hidden: MLXArray, _ normed: MLXArray?, _ index: Int
    ) -> MLXArray {
        normedRows(hidden, normed, index ..< (index + 1))
            ?? hiddenRow(hidden, index)
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
