import Foundation
import MLXFastCore

// Trusted-parent driver for the DFlash block-decode track. Links no MLX: it
// spawns the sandboxed worker, owns the timer and the block schedule, and feeds
// every emitted row to the Criterion E validator.

/// On-disk reference material for one DFlash prompt.
///
/// This is the artifact the pinned-baseline reference verifier (contract layer
/// L1) produces: for each decode position, the reference's K=1 sequential
/// argmax plus -- where a declared-frame replay was performed -- the reference's
/// argmax at that block width, and the reference top-2 readouts used for work
/// binding. Keeping it a file keeps the trusted parent free of MLX.
public struct DFlashReferenceGolden: Codable {
    public struct Row: Codable {
        public let sequentialArgmax: Int
        public let declaredFrameArgmax: [String: Int]?
        public let top2Tokens: [Int]?
        public let top2Logits: [Double]?
        /// The reference's own top-1 logit at this row, and the token this row
        /// predicted in the chain the golden was generated against together with
        /// that token's reference logit (Amendment 16). Optional so a golden
        /// written before the field existed still decodes; the validator falls
        /// back to the top-2 form for such a row.
        ///
        /// Note what the recorded token means HERE: it is the GOLDEN's chain, so
        /// a run whose candidate diverged must not read this logit as a statement
        /// about the candidate's token. The validator enforces that by comparing
        /// the id before using the value, which is why the id is stored at all.
        public let top1Logit: Double?
        public let emittedToken: Int?
        public let emittedTokenLogit: Double?

        enum CodingKeys: String, CodingKey {
            case sequentialArgmax = "sequential_argmax"
            case declaredFrameArgmax = "declared_frame_argmax"
            case top2Tokens = "top2_tokens"
            case top2Logits = "top2_logits"
            case top1Logit = "top1_logit"
            case emittedToken = "emitted_token"
            case emittedTokenLogit = "emitted_token_logit"
        }
    }

    public let seedTokens: [Int]
    public let referenceSeedToken: Int
    public let rows: [Row]
    /// Set by the reference generator once its self-consistency replay passed
    /// (contract layer L1 requirement R5). A run must refuse to score against a
    /// golden that never proved reference determinism.
    public let referenceSelfConsistent: Bool?
    /// The emitted chain these rows were replayed against, when the reference
    /// generated it itself. Carried so a plan can be reconstructed from the
    /// golden alone; it leaks nothing the rows do not already contain, since
    /// `rows[i].sequentialArgmax` is the same chain for a generated golden.
    public let emittedTokens: [Int]?

    enum CodingKeys: String, CodingKey {
        case seedTokens = "seed_tokens"
        case referenceSeedToken = "reference_seed_token"
        case rows
        case referenceSelfConsistent = "reference_self_consistent"
        case emittedTokens = "emitted_tokens"
    }

    public init(
        seedTokens: [Int],
        referenceSeedToken: Int,
        rows: [Row],
        referenceSelfConsistent: Bool?,
        emittedTokens: [Int]? = nil
    ) {
        self.seedTokens = seedTokens
        self.referenceSeedToken = referenceSeedToken
        self.rows = rows
        self.referenceSelfConsistent = referenceSelfConsistent
        self.emittedTokens = emittedTokens
    }
}

/// Reference oracle backed by a pinned-baseline-generated golden file.
public struct DFlashGoldenReferenceOracle: DFlashReferenceOracle {
    private let rows: [DFlashReferenceGolden.Row]

    public init(golden: DFlashReferenceGolden) {
        self.rows = golden.rows
    }

    public func referenceRows(
        emittedPrefix: [Int],
        startOffset: Int,
        count: Int,
        declaredBlockWidth: Int
    ) throws -> [DFlashReferenceRow] {
        guard startOffset >= 0, count >= 0 else {
            throw MLXFastError.invalidInput(
                "DFlash reference request has a negative range"
            )
        }
        guard startOffset + count <= rows.count else {
            throw MLXFastError.invalidInput(
                "DFlash reference golden covers \(rows.count) rows; run reached "
                    + "\(startOffset + count)"
            )
        }
        return (0 ..< count).map { index in
            let row = rows[startOffset + index]
            return DFlashReferenceRow(
                sequentialArgmax: row.sequentialArgmax,
                declaredFrameArgmax: row.declaredFrameArgmax?[
                    String(declaredBlockWidth)
                ],
                top2Tokens: row.top2Tokens ?? [],
                top2Logits: row.top2Logits ?? [],
                top1Logit: row.top1Logit,
                emittedToken: row.emittedToken,
                emittedTokenLogit: row.emittedTokenLogit
            )
        }
    }
}

/// Reference oracle backed by a LIVE pinned-baseline reference worker, replaying
/// the CANDIDATE's OWN emitted chain (contract layer L1).
///
/// This is what a stored golden cannot be. A golden's row `i` was teacher-forced
/// on whatever chain generated the file, so as soon as the candidate emits a
/// different -- and admissible -- token at some position, every later golden row
/// answers a question about a prefix the candidate never had. The contract has
/// always said the parent journals and the pinned reference replays afterwards on
/// the candidate's own prefix; this is that replay.
///
/// Requests must arrive in emission order. The reference session holds one
/// continuously-advanced width-1 cache (see `LagunaDFlashReference`), so an
/// in-order walk is a pure continuation; an out-of-order request would force it
/// to rebuild the frame from scratch. `validateJournalAgainstReference` replays
/// the journal in order, which satisfies that.
struct DFlashLiveReferenceOracle: DFlashReferenceOracle {
    let client: RuntimeWorkerClient
    /// Length of the seed the candidate bulk-prefilled.
    let seedTokenCount: Int
    /// `seed + [seed argmax] + every token the candidate emitted`. Row `j` of a
    /// request at emitted offset `o` is fed `context[seedTokenCount + o + j]`.
    let context: [Int]

    func referenceRows(
        emittedPrefix: [Int],
        startOffset: Int,
        count: Int,
        declaredBlockWidth: Int
    ) throws -> [DFlashReferenceRow] {
        try referenceBatch(
            emittedPrefix: emittedPrefix,
            startOffset: startOffset,
            count: count,
            declaredBlockWidth: declaredBlockWidth,
            // No journalled drafts supplied, so no verify block is replayed.
            declaredRows: count,
            draftTokens: []
        ).rows
    }

    func referenceBatch(
        emittedPrefix: [Int],
        startOffset: Int,
        count: Int,
        declaredBlockWidth: Int,
        declaredRows: Int,
        draftTokens: [Int]
    ) throws -> DFlashReferenceBatch {
        guard count > 0 else { return DFlashReferenceBatch(rows: []) }
        guard startOffset >= 0, emittedPrefix.count == startOffset else {
            throw MLXFastError.invalidInput(
                "DFlash live reference replay is out of order: prefix of "
                    + "\(emittedPrefix.count) at offset \(startOffset)"
            )
        }
        // The whole point of the live path: the rows must be produced against the
        // candidate's own emitted prefix, so prove that is what is being replayed
        // rather than assume it.
        let chainStart = seedTokenCount + 1
        guard chainStart + startOffset + count <= context.count,
              context[chainStart ..< (chainStart + startOffset)]
                  .elementsEqual(emittedPrefix)
        else {
            throw MLXFastError.invalidInput(
                "DFlash live reference context does not carry the candidate's "
                    + "own emitted prefix at offset \(startOffset)"
            )
        }

        let absoluteOffset = seedTokenCount + startOffset
        // The reference needs an input token for every row of a frame, so a frame
        // can never be wider than the tokens the candidate actually emitted. This
        // is the same bound the golden generator applies.
        let availableRows = context.count - absoluteOffset
        let widestFrame = Swift.max(
            count,
            Swift.min(declaredBlockWidth, availableRows)
        )
        // The candidate's OWN verify block, reconstructed here rather than taken
        // from the worker: row 0 is the parent's committed token at this position,
        // and only the drafts after it are worker-asserted. The emitted-context
        // frames above cannot reach the rejected tail -- they are fed emitted
        // tokens, and the emitted chain stops describing the verify input at the
        // first rejection -- and they are additionally clamped by `availableRows`,
        // so the final round's tail could not be replayed at all. This one carries
        // its own inputs, so neither limit applies.
        var verifyBlockTokens: [Int]?
        if declaredRows > 1,
           draftTokens.count == declaredRows - 1,
           absoluteOffset < context.count
        {
            verifyBlockTokens = [context[absoluteOffset]] + draftTokens
        }
        let response = try client.dflashReferenceRows(
            prefixTokens: context,
            seedTokenCount: seedTokenCount,
            startOffset: absoluteOffset,
            rowCount: count,
            widestFrame: widestFrame,
            verifyBlockTokens: verifyBlockTokens
        )
        guard response.ok,
              let k1 = response.referenceK1Argmax,
              let top2Tokens = response.referenceTop2Tokens,
              let top2Logits = response.referenceTop2Logits,
              let frameWidths = response.referenceFrameWidths,
              let frameArgmax = response.referenceFrameArgmax,
              k1.count == count,
              top2Tokens.count == count,
              top2Logits.count == count,
              frameWidths.count == frameArgmax.count,
              // The emitted-token readouts cover a prefix of the batch, but the
              // two arrays must describe the same rows as each other.
              (response.referenceEmittedTokens?.count ?? 0)
                  == (response.referenceEmittedTokenLogits?.count ?? 0)
        else {
            throw MLXFastError.invalidInput(
                "DFlash live reference returned an incomplete row batch at "
                    + "offset \(startOffset): "
                    + (response.error ?? "no reason reported")
            )
        }
        // Asked for and not answered is an operator fault, never a silent
        // degradation: this is the only readout that reaches a rejected row, so a
        // scored run must not quietly fall back to leaving the tail unpriced.
        var verifyTop2Tokens: [[Int]]?
        var verifyTop2Logits: [[Double]]?
        if let verifyBlockTokens {
            guard let tokens = response.referenceVerifyTop2Tokens,
                  let logits = response.referenceVerifyTop2Logits,
                  tokens.count == verifyBlockTokens.count,
                  logits.count == verifyBlockTokens.count
            else {
                throw MLXFastError.invalidInput(
                    "DFlash live reference did not return the "
                        + "\(verifyBlockTokens.count)-row verify-block replay it "
                        + "was asked for at offset \(startOffset)"
                )
            }
            verifyTop2Tokens = tokens
            verifyTop2Logits = logits
        }
        var frames = [Int: [Int]]()
        for (width, argmax) in zip(frameWidths, frameArgmax) {
            frames[width] = argmax
        }
        let declaredFrame = frames[declaredBlockWidth]
        let top1Logits = response.referenceTop1Logits
        let emittedTokens = response.referenceEmittedTokens ?? []
        let emittedTokenLogits = response.referenceEmittedTokenLogits ?? []
        let rows = (0 ..< count).map { index in
            DFlashReferenceRow(
                sequentialArgmax: k1[index],
                declaredFrameArgmax: declaredFrame.flatMap {
                    index < $0.count ? $0[index] : nil
                },
                top2Tokens: top2Tokens[index],
                top2Logits: top2Logits[index],
                top1Logit: top1Logits.flatMap {
                    index < $0.count ? $0[index] : nil
                },
                emittedToken: index < emittedTokens.count
                    ? emittedTokens[index]
                    : nil,
                emittedTokenLogit: index < emittedTokenLogits.count
                    ? emittedTokenLogits[index]
                    : nil
            )
        }
        return DFlashReferenceBatch(
            rows: rows,
            verifyBlockTop2Tokens: verifyTop2Tokens,
            verifyBlockTop2Logits: verifyTop2Logits
        )
    }
}

extension QwenRuntime {
    /// Run a validated, parent-timed DFlash block-decode measurement.
    ///
    /// Timing is parent-owned end to end: the clock starts after the seed
    /// prefill response and stops when the configured token total is committed,
    /// and the denominator is that configured total -- never a worker-reported
    /// count.
    ///
    /// Validation happens in two phases, and the split is a contract requirement
    /// rather than a timing optimization. Inline, per round, the parent checks
    /// everything that is pure arithmetic on the worker's own report. Then, with
    /// the candidate torn down, a pinned reference worker replays the journal on
    /// the candidate's OWN emitted prefix and scores every token. Reference rows
    /// read from a pre-generated file cannot do this: they are anchored to the
    /// chain that generated the file, so the first legitimate divergence makes
    /// every later row a comparison against a prefix the candidate never had.
    ///
    /// `referenceWorkerOptions` selects the reference binary; `nil` reuses the
    /// candidate's worker options, which is what a local diagnostic run wants. A
    /// ranked run passes the pinned baseline worker together with
    /// `options.referenceWeightsPath`.
    public static func experimentalDFlashBenchmark(
        options: ExperimentalDFlashOptions,
        workerOptions: RuntimeWorkerOptions,
        scheduleSeed: UInt64,
        tolerance: DFlashWorkBindingTolerance = DFlashWorkBindingTolerance(),
        referenceWorkerOptions: RuntimeWorkerOptions? = nil
    ) throws -> ExperimentalDFlashReport {
        let goldenData = try Data(
            contentsOf: URL(fileURLWithPath: options.goldenPath)
        )
        let golden = try JSONDecoder().decode(
            DFlashReferenceGolden.self,
            from: goldenData
        )
        guard golden.referenceSelfConsistent != false else {
            throw DFlashContractViolation(
                kind: .workBindingMissing,
                detail: "reference golden reports a failed self-consistency "
                    + "replay; this is an operator fault, not a submission fault"
            )
        }
        guard options.totalTokenCount > 0,
              options.totalTokenCount <= golden.rows.count
        else {
            throw MLXFastError.invalidInput(
                "DFlash golden carries \(golden.rows.count) reference rows but "
                    + "\(options.totalTokenCount) tokens were requested"
            )
        }

        // The golden stays the FALLBACK oracle and keeps two jobs on the scored
        // path: it supplies the seed-token expectation and it bounds `--tokens`.
        // The admissibility verdicts come from the live replay below.
        let oracle = DFlashGoldenReferenceOracle(golden: golden)
        let validator = LagunaDFlashBlockValidator(
            oracle: oracle,
            seedTokenCount: golden.seedTokens.count,
            totalTokenCount: options.totalTokenCount,
            tolerance: tolerance
        )
        // A maxBlockSize of 1 is the SERIAL CONTROL: pin the schedule to width 1
        // so every round is a one-row target step. Same worker, same protocol,
        // same forward -- only the width differs, which is what makes the paired
        // ratio a like-for-like comparison.
        //
        // `minBlockSize` 2 for every non-control run is LOAD-BEARING, not a
        // stylistic floor. A width-1 round advances the target without feeding the
        // drafter, whose cross-attention context must be exactly as wide as the
        // positions the target advanced since it last wrote. One such row still
        // leaves a gap of 1 and stays legal; two in a row does not, and the
        // session refuses the following block rather than draft from a prefix the
        // drafter never saw. If L6's randomized schedule is ever widened to
        // include width 1 alongside wider blocks, `LagunaDFlashBlockSession` has
        // to accumulate the skipped hidden rows first.
        var schedule = DFlashBlockSchedule(
            seed: scheduleSeed,
            maxBlockSize: options.maxBlockSize,
            minBlockSize: options.maxBlockSize == 1 ? 1 : 2
        )

        let client = try RuntimeWorkerClient(
            options: workerOptions,
            weightsPath: options.targetWeightsPath,
            dflashDrafterPath: options.drafterPath
        )
        // Closed EXPLICITLY before the reference worker starts, not just on the
        // way out: the text tower is ~21.6 GB, so two live residencies would put
        // the box under memory pressure the measurement never accounted for.
        // `close()` is idempotent, so this defer only covers the failure paths.
        defer { client.close() }

        // Untimed phase start, BEFORE the clock: the allocator clear and the
        // working-set re-touch that follows it never see the seed, so timing them
        // measures nothing about the submission -- it only charges the warm to the
        // score. It used to ride inside the begin request, which is why the warm
        // had to stay small; widening it to cover the saturated ring and every
        // block width then cost 24-27% of absolute decode time.
        let warm = try client.warmDFlashDecode()
        guard warm.ok else {
            throw MLXFastError.invalidInput(
                "DFlash worker failed the untimed warm: "
                    + (warm.error ?? "no reason reported")
            )
        }

        // Seed prefill IS charged to the decode measurement as a whole (the
        // retired MTP contract charged it the same way): the clock starts
        // immediately before the request so the seed cost cannot be hidden.
        let started = Date()
        let begin = try client.beginDFlashDecode(seedTokens: golden.seedTokens)
        guard begin.ok, let seedToken = begin.seedToken else {
            throw MLXFastError.invalidInput(
                "DFlash worker failed the seed prefill: "
                    + (begin.error ?? "no seed token returned")
            )
        }
        guard seedToken == golden.referenceSeedToken else {
            throw DFlashContractViolation(
                kind: .tokenNotAdmissible,
                step: 0,
                detail: "seed prefill token disagreed with the reference"
            )
        }

        var previousCommittedToken = seedToken
        var acceptedTotal = 0
        var rejectedTotal = 0

        while validator.committedTokens.count < options.totalTokenCount {
            let blockSize = schedule.nextBlockSize()
            let roundStart = Date()
            let response = try client.dflashDecodeBlock(
                previousCommittedToken: previousCommittedToken,
                maxBlockSize: blockSize
            )
            let latency = Date().timeIntervalSince(roundStart)
            guard response.ok, let tokens = response.tokens else {
                throw MLXFastError.invalidInput(
                    "DFlash block request failed: "
                        + (response.error ?? "no tokens returned")
                )
            }

            let round = DFlashObservedRound(
                requestedBlockSize: blockSize,
                tokens: tokens,
                declaredRows: response.declaredRows ?? 0,
                perRowTop2Tokens: response.perRowTop2Tokens ?? [],
                perRowTop2Logits: response.perRowTop2Logits ?? [],
                draftTokens: response.draftTokens ?? [],
                acceptedDraftCount: response.acceptedDraftCount ?? 0,
                rejectedDraftCount: response.rejectedDraftCount ?? 0,
                targetCacheOffset: response.targetCacheOffset ?? -1,
                latencySeconds: latency
            )
            // The parent asks for a full block every time; if the round would
            // overrun the scored window, validate only the prefix that fits.
            let remaining = options.totalTokenCount
                - validator.committedTokens.count
            if tokens.count > remaining {
                let trimmed = DFlashObservedRound(
                    requestedBlockSize: blockSize,
                    tokens: Array(tokens.prefix(remaining)),
                    declaredRows: round.declaredRows,
                    perRowTop2Tokens: round.perRowTop2Tokens,
                    perRowTop2Logits: round.perRowTop2Logits,
                    draftTokens: round.draftTokens,
                    acceptedDraftCount: round.acceptedDraftCount,
                    rejectedDraftCount: round.rejectedDraftCount,
                    // The worker's offset counts every token it committed; the
                    // parent only scores the prefix, so re-base the expectation.
                    targetCacheOffset: golden.seedTokens.count
                        + validator.committedTokens.count + remaining,
                    latencySeconds: latency
                )
                try validator.acceptStructural(round: trimmed)
            } else {
                try validator.acceptStructural(round: round)
            }
            acceptedTotal += round.acceptedDraftCount
            rejectedTotal += round.rejectedDraftCount
            previousCommittedToken = tokens.last ?? previousCommittedToken
        }
        let decodeSeconds = Date().timeIntervalSince(started)
        try validator.requireComplete()

        // --- post-run reference replay (contract layer L1) --------------------
        //
        // The clock has stopped and the candidate is gone. Only now is a
        // reference verdict meaningful, because only now does the parent know the
        // candidate's own emitted chain to teacher-force the reference on.
        client.close()
        try replayDFlashJournalAgainstLiveReference(
            validator: validator,
            golden: golden,
            candidateSeedToken: seedToken,
            options: options,
            workerOptions: referenceWorkerOptions ?? workerOptions
        )
        try validator.requireReferenceValidated()

        return ExperimentalDFlashReport(
            totalTokenCount: options.totalTokenCount,
            decodeSeconds: decodeSeconds,
            roundCount: validator.roundLatencies.count,
            acceptedDraftTotal: acceptedTotal,
            rejectedDraftTotal: rejectedTotal,
            declaredRowTotal: validator.declaredRowTotal,
            residualDivergenceCount: validator.residualDivergenceCount,
            admissibleExactCount: validator.admissibleExactCount,
            admissibleDeclaredFrameCount: validator.admissibleDeclaredFrameCount,
            admissibleNearTieCount: validator.admissibleNearTieCount,
            maxOverMedianRoundLatency: validator.maxOverMedianRoundLatency(),
            allTokensAdmissible: true,
            seedTokenCount: golden.seedTokens.count,
            // Parent-derived, never worker-reported: the ledger the wrapper
            // checks must not be something the measured party can assert.
            targetCacheOffsetFinal: golden.seedTokens.count
                + validator.committedTokens.count,
            referenceCheckedRowTotal: validator.referenceCheckedRowTotal,
            rejectedRowsReferenceChecked:
                validator.rejectedRowsReferenceChecked,
            verifyBlockReplayedRoundCount:
                validator.verifyBlockReplayedRoundCount,
            rejectedTailComparisonCount: validator.rejectedTailComparisonCount,
            maxRejectedTailLogitDelta: validator.maxRejectedTailLogitDelta,
            // One target-produced tail token per round, by construction.
            targetTailTotal: validator.roundLatencies.count,
            maxBlockRequestSeconds: validator.maxBlockRequestSeconds,
            p50BlockRequestSeconds: validator.p50BlockRequestSeconds,
            blockSize: options.maxBlockSize,
            usesTrainedDrafter: options.maxBlockSize > 1,
            workBindingComparisonCount: validator.workBindingComparisonCount,
            maxTop2LogitDelta: validator.maxWorkBindingLogitDelta,
            meanTop2LogitDelta: validator.meanWorkBindingLogitDelta,
            p50Top2LogitDelta: validator.p50WorkBindingLogitDelta,
            p99Top2LogitDelta: validator.p99WorkBindingLogitDelta,
            maxTop2LogitRelativeDelta: validator.maxWorkBindingLogitRelativeDelta,
            p99Top2LogitRelativeDelta: validator.p99WorkBindingLogitRelativeDelta,
            workBindingLogitDeltas: validator.workBindingLogitDeltas,
            workBindingComparisonWidths: validator.workBindingComparisonWidths,
            workBindingToleranceAbsolute: tolerance.absolute,
            workBindingToleranceRelative: tolerance.relative
        )
    }

    /// Open the pinned reference worker and score the journal against it.
    ///
    /// Ordering is load-bearing on two axes. Memory: the candidate must already
    /// be closed, so this is only ever called after `client.close()` -- the two
    /// workers each hold the ~21.6 GB text tower. Frame: the reference is
    /// prefilled over the SAME seed the candidate bulk-prefilled, which leaves
    /// its continuous width-1 cache positioned at the end of the seed, so the
    /// first row request is a plain continuation and every later one stays one.
    private static func replayDFlashJournalAgainstLiveReference(
        validator: LagunaDFlashBlockValidator,
        golden: DFlashReferenceGolden,
        candidateSeedToken: Int,
        options: ExperimentalDFlashOptions,
        workerOptions: RuntimeWorkerOptions
    ) throws {
        let referenceClient = try RuntimeWorkerClient(
            options: workerOptions,
            weightsPath: options.referenceWeightsPath
                ?? options.targetWeightsPath,
            dflashDrafterPath: options.referenceDrafterPath
                ?? options.drafterPath
        )
        defer { referenceClient.close() }

        let prefill = try referenceClient.dflashReferencePrefill(
            seedTokens: golden.seedTokens
        )
        guard prefill.ok, let referenceSeedToken = prefill.seedToken else {
            throw MLXFastError.invalidInput(
                "DFlash reference replay failed its seed prefill: "
                    + (prefill.error ?? "no seed token returned")
            )
        }
        // The candidate was already checked against the golden's recorded seed
        // token. If the LIVE reference disagrees with that record, the golden and
        // the reference build are not the same reference, so the replay would
        // score against a chain neither party produced. That is an operator
        // fault -- a mismatched reference tree or weights -- not a submission
        // fault, and it must fail loudly instead of being scored.
        guard referenceSeedToken == golden.referenceSeedToken else {
            throw DFlashContractViolation(
                kind: .workBindingMissing,
                step: 0,
                detail: "the live reference's seed token disagrees with the "
                    + "golden's record; the reference build or weights do not "
                    + "match the golden (operator fault)"
            )
        }

        let oracle = DFlashLiveReferenceOracle(
            client: referenceClient,
            seedTokenCount: golden.seedTokens.count,
            context: golden.seedTokens + [candidateSeedToken]
                + validator.committedTokens
        )
        try validator.validateJournalAgainstReference(oracle: oracle)
    }
}

// MARK: - L1 reference golden generation

/// The emitted plan a reference pass replays.
///
/// Produced by whoever ran the candidate (locally: the bench script; on the box:
/// the measurement wrapper). It carries only what the reference needs to rebuild
/// the same positions: the seed prompt, the tokens the candidate emitted, and
/// the block width the candidate declared per round.
public struct DFlashEmittedPlan: Codable {
    public struct Round: Codable {
        public let blockSize: Int
        public let count: Int

        enum CodingKeys: String, CodingKey {
            case blockSize = "block_size"
            case count
        }
    }

    public let seedTokens: [Int]
    public let emitted: [Int]
    public let rounds: [Round]?

    enum CodingKeys: String, CodingKey {
        case seedTokens = "seed_tokens"
        case emitted
        case rounds
    }
}

/// Reference-driven chain generation for `dflash-reference --generate`.
///
/// Without this, a golden can only ever be as long as an `emitted` array someone
/// typed by hand, which makes the long-context and wrap-seam cases untestable.
/// Here the REFERENCE produces the chain itself: sequential width-1 argmax over
/// its own growing context, one token at a time, in a single process with the
/// model resident once.
///
/// `seedExtensionSteps` is the seam lever. The trusted binary links no
/// tokenizer, so a long seed cannot be typed as text; instead the seed becomes
/// the supplied prompt plus that many reference-generated tokens. That is real
/// model-shaped context, and it lets the seed length be dialled to any position
/// relative to Laguna's 512-slot sliding-window ring -- including the ~505..511
/// band where a block round STARTS trimmable and ENDS wrapped.
public struct DFlashReferenceChainOptions {
    public let seedExtensionSteps: Int
    public let generateTokenCount: Int
    public let roundBlockSize: Int
    public let scheduleSeed: UInt64
    public let planOutputPath: String?

    public init(
        seedExtensionSteps: Int = 0,
        generateTokenCount: Int,
        roundBlockSize: Int = 1,
        scheduleSeed: UInt64 = 0,
        planOutputPath: String? = nil
    ) {
        self.seedExtensionSteps = seedExtensionSteps
        self.generateTokenCount = generateTokenCount
        self.roundBlockSize = roundBlockSize
        self.scheduleSeed = scheduleSeed
        self.planOutputPath = planOutputPath
    }
}

/// Outcome of a reference pass, including the self-consistency verdict.
public struct DFlashReferenceGoldenResult {
    public let rowCount: Int
    public let referenceSeedToken: Int
    public let selfConsistent: Bool
    public let selfConsistencyRowCount: Int
    public let selfConsistencyDetail: String
    /// Rows whose `sequentialArgmax` disagreed with the emitted token at the
    /// same index, for a chain the reference generated itself. Must be zero: a
    /// nonzero count means the reference contradicts its own chain, so the
    /// admissible sets this golden defines are not the ones a sequential
    /// decoder lands in. Folded into `selfConsistent`.
    public let chainRowContradictionCount: Int
    /// Seed length actually used, i.e. after any `--seed-generate` extension.
    public let seedTokenCount: Int
    /// Declared block widths recorded per row, ascending.
    public let recordedFrameWidths: [Int]
    /// Where the reconstructed emitted plan was written, if it was.
    public let planOutputPath: String?

    public init(
        rowCount: Int,
        referenceSeedToken: Int,
        selfConsistent: Bool,
        selfConsistencyRowCount: Int,
        selfConsistencyDetail: String,
        chainRowContradictionCount: Int = 0,
        seedTokenCount: Int = 0,
        recordedFrameWidths: [Int] = [],
        planOutputPath: String? = nil
    ) {
        self.rowCount = rowCount
        self.referenceSeedToken = referenceSeedToken
        self.selfConsistent = selfConsistent
        self.selfConsistencyRowCount = selfConsistencyRowCount
        self.selfConsistencyDetail = selfConsistencyDetail
        self.chainRowContradictionCount = chainRowContradictionCount
        self.seedTokenCount = seedTokenCount
        self.recordedFrameWidths = recordedFrameWidths
        self.planOutputPath = planOutputPath
    }
}

extension QwenRuntime {
    /// Generate the DFlash reference golden (contract layer L1).
    ///
    /// The worker spawned here MUST be the one built from the pinned baseline
    /// tree, loading organizer-transformed weights -- never the candidate's. The
    /// caller enforces that by pointing `workerOptions.executablePath` and
    /// `targetWeightsPath` at the pinned tree; this function additionally runs
    /// strictly on its own, after the timed phase, with the candidate gone.
    ///
    /// R5 self-consistency has two halves, and BOTH are mandatory.
    ///
    /// First, one round is replayed and required to be bit-identical. The replay
    /// deliberately targets the first round, whose start offset is behind the
    /// reference's live walk, so it takes the rebuild-from-scratch path: the
    /// replay therefore also proves the fallback construction agrees with the
    /// continuous walk, which is the property that makes a stateful reference
    /// admissible at all.
    ///
    /// Second, when the reference generated the chain itself, every row's
    /// width-1 argmax must equal the token at that index of the chain. This half
    /// used to be missing, and its absence is what let goldens ship with
    /// `reference_self_consistent: true` while `emitted_tokens[i]` disagreed
    /// with `rows[i].sequential_argmax` -- the reference contradicting itself,
    /// and with it the honest K=1 serial control being rejected on rows the
    /// reference's own chain had produced.
    ///
    /// Either failure is an OPERATOR fault rather than a submission fault.
    public static func experimentalDFlashReferenceGolden(
        plan: DFlashEmittedPlan,
        chain: DFlashReferenceChainOptions? = nil,
        targetWeightsPath: String,
        drafterPath: String,
        outputPath: String,
        workerOptions: RuntimeWorkerOptions
    ) throws -> DFlashReferenceGoldenResult {
        guard !plan.seedTokens.isEmpty else {
            throw MLXFastError.invalidInput(
                "DFlash reference plan has an empty seed"
            )
        }
        if let chain {
            guard chain.generateTokenCount > 0 else {
                throw MLXFastError.invalidInput(
                    "DFlash reference chain generation needs a positive token "
                        + "count"
                )
            }
            guard chain.seedExtensionSteps >= 0, chain.roundBlockSize >= 1,
                  chain.roundBlockSize
                      <= MLXFastConstants.experimentalDFlashMaxBlockSize
            else {
                throw MLXFastError.invalidInput(
                    "DFlash reference chain generation has an out-of-range "
                        + "seed extension or block width"
                )
            }
        } else {
            guard !plan.emitted.isEmpty else {
                throw MLXFastError.invalidInput(
                    "DFlash reference plan has no emitted tokens to verify"
                )
            }
        }

        let client = try RuntimeWorkerClient(
            options: workerOptions,
            weightsPath: targetWeightsPath,
            dflashDrafterPath: drafterPath
        )
        defer { client.close() }

        // --- optional seed extension (one process, model resident once) ------
        var seedTokens = plan.seedTokens
        if let chain, chain.seedExtensionSteps > 0 {
            // Extension tokens are produced the way a decoder produces them:
            // one bulk forward over the SUPPLIED prompt, then one single-token
            // forward per position after it. They then become part of the seed
            // the candidate bulk-prefills, so this span is model-shaped context
            // for dialling seed length -- every scored row sits after it.
            let extensionPrefill = try client.dflashReferencePrefill(
                seedTokens: plan.seedTokens
            )
            guard extensionPrefill.ok,
                  let firstExtensionToken = extensionPrefill.seedToken
            else {
                throw MLXFastError.invalidInput(
                    "DFlash reference seed extension failed its prefill: "
                        + (extensionPrefill.error ?? "no seed token returned")
                )
            }
            var extended = [firstExtensionToken]
            extended += try walkReferenceChain(
                client: client,
                context: plan.seedTokens + extended,
                seedTokenCount: plan.seedTokens.count,
                steps: chain.seedExtensionSteps - 1,
                label: "seed-generate"
            )
            seedTokens += extended
        }

        // The seed token is the post-prefill argmax over the WHOLE seed: ONE
        // bulk forward, argmax of the last row. That is precisely what
        // `LagunaDFlashBlockSession.begin` computes, so a correct candidate
        // reproduces it exactly -- where bulk-forwarding all but the last seed
        // token and then single-stepping it is a different computation that can
        // disagree at a near tie and fail an honest run at its first check.
        // Asking the reference for it avoids trusting any candidate-supplied
        // value, and it primes the reference's continuous width-1 frame at the
        // end of the seed so the first row request continues the walk.
        let seedPrefill = try client.dflashReferencePrefill(
            seedTokens: seedTokens
        )
        guard seedPrefill.ok, let seedArgmax = seedPrefill.seedToken else {
            throw MLXFastError.invalidInput(
                "DFlash reference could not establish the seed token: "
                    + (seedPrefill.error ?? "no seed token returned")
            )
        }

        // --- the emitted chain: supplied, or generated by the reference ------
        let emitted: [Int]
        if let chain {
            emitted = try walkReferenceChain(
                client: client,
                context: seedTokens + [seedArgmax],
                seedTokenCount: seedTokens.count,
                steps: chain.generateTokenCount,
                label: "generate"
            )
        } else {
            emitted = plan.emitted
        }

        // Full context the emitted rows were produced against.
        let context = seedTokens + [seedArgmax] + emitted
        // Row i is fed context[seedLen + i] and predicts context[seedLen+i+1],
        // i.e. emitted[i].
        let rowInputBase = seedTokens.count

        // Rounds give the declared block widths. Absent, treat every row as its
        // own width-1 round so the golden is still usable for a serial control.
        let rounds: [DFlashEmittedPlan.Round]
        if let chain {
            // Lay the generated rounds out on the SAME parent schedule the run
            // will use. The parent owns block widths, so with full acceptance
            // the golden's frame boundaries land exactly where the run's do; a
            // partially-accepting round shifts them, which is what the capped
            // residual bucket is for.
            rounds = scheduledRounds(
                tokenCount: emitted.count,
                maxBlockSize: chain.roundBlockSize,
                scheduleSeed: chain.scheduleSeed
            )
        } else {
            rounds = plan.rounds
                ?? emitted.map { _ in
                    DFlashEmittedPlan.Round(blockSize: 1, count: 1)
                }
        }
        let plannedRowTotal = rounds.reduce(0) { $0 + $1.count }
        guard plannedRowTotal == emitted.count else {
            throw MLXFastError.invalidInput(
                "DFlash reference plan rounds cover \(plannedRowTotal) rows but "
                    + "\(emitted.count) tokens were emitted"
            )
        }

        if let path = chain?.planOutputPath {
            let reconstructed = DFlashEmittedPlan(
                seedTokens: seedTokens,
                emitted: emitted,
                rounds: rounds
            )
            let planEncoder = JSONEncoder()
            planEncoder.outputFormatting = [
                .prettyPrinted, .sortedKeys, .withoutEscapingSlashes,
            ]
            try planEncoder.encode(reconstructed)
                .write(to: URL(fileURLWithPath: path))
        }

        var goldenRows = [DFlashReferenceGolden.Row]()
        goldenRows.reserveCapacity(emitted.count)
        var replayRequest: (offset: Int, count: Int, width: Int)?
        var replayExpected: RuntimeWorkerResponse?
        var recordedWidths = Set<Int>([1])

        var emittedOffset = 0
        for round in rounds {
            guard round.count > 0, round.blockSize >= round.count else {
                throw MLXFastError.invalidInput(
                    "DFlash reference plan has a round emitting \(round.count) "
                        + "tokens at declared width \(round.blockSize)"
                )
            }
            let startOffset = rowInputBase + emittedOffset

            // Record a GENUINE block frame for every width the parent could
            // have declared here, not just one.
            //
            // The parent picks each round's width from a randomized schedule, and
            // a round that rejects drafts emits fewer tokens than its declared
            // width, so the width a row is scored at is not known when the
            // golden is built. Replaying widths `count ... blockSize` covers
            // every width the schedule can request. Widening the frame is sound
            // for the scored rows: attention is causally masked, so rows
            // `0 ..< count` cannot see the tail rows this widening filled in
            // with later emitted tokens rather than the candidate's (unknown)
            // rejected drafts. Kernel tiling can still perturb the last bits of
            // a wider frame, which is what the top-2 tolerance absorbs.
            //
            // ONE request covers all of them. Each frame is a `copy()` of the
            // reference's continuous cache taken at this round boundary, so the
            // widths are siblings of the width-1 walk rather than a separate
            // bulk re-prefill -- and asking for them one request at a time would
            // rewind the walk and force exactly that re-prefill back into
            // existence, at O(n^2).
            let availableRows = context.count - startOffset
            let widestFrame = Swift.max(
                round.count,
                Swift.min(round.blockSize, availableRows)
            )
            let response = try client.dflashReferenceRows(
                prefixTokens: context,
                seedTokenCount: seedTokens.count,
                startOffset: startOffset,
                rowCount: round.count,
                widestFrame: widestFrame
            )
            guard response.ok,
                  let k1 = response.referenceK1Argmax,
                  let top2Tokens = response.referenceTop2Tokens,
                  let top2Logits = response.referenceTop2Logits,
                  let frameWidths = response.referenceFrameWidths,
                  let frameArgmax = response.referenceFrameArgmax,
                  k1.count == round.count,
                  top2Tokens.count == round.count,
                  top2Logits.count == round.count,
                  frameWidths.count == frameArgmax.count,
                  frameWidths.contains(round.count),
                  frameArgmax.allSatisfy({ $0.count == round.count })
            else {
                throw MLXFastError.invalidInput(
                    "DFlash reference returned an incomplete row batch"
                )
            }
            var frames = [Int: [Int]]()
            for (width, argmax) in zip(frameWidths, frameArgmax) {
                frames[width] = argmax
                recordedWidths.insert(width)
            }

            for index in 0 ..< round.count {
                var declaredFrames = [String: Int]()
                for (width, argmax) in frames where index < argmax.count {
                    declaredFrames[String(width)] = argmax[index]
                }
                // Width 1 needs no request: a one-row frame IS the sequential
                // frame, so the serial control (max block size 1) always has a
                // declared-frame entry to be scored against.
                declaredFrames["1"] = k1[index]
                let top1Logit = response.referenceTop1Logits.flatMap {
                    index < $0.count ? $0[index] : nil
                }
                let emittedToken = response.referenceEmittedTokens.flatMap {
                    index < $0.count ? $0[index] : nil
                }
                let emittedTokenLogit = response.referenceEmittedTokenLogits
                    .flatMap { index < $0.count ? $0[index] : nil }
                goldenRows.append(
                    DFlashReferenceGolden.Row(
                        sequentialArgmax: k1[index],
                        declaredFrameArgmax: declaredFrames,
                        top2Tokens: top2Tokens[index],
                        top2Logits: top2Logits[index],
                        top1Logit: top1Logit,
                        emittedToken: emittedToken,
                        emittedTokenLogit: emittedTokenLogit
                    )
                )
            }

            // Replay the FIRST round: it is the one whose context is shortest,
            // so a determinism failure there is unambiguous.
            if replayRequest == nil {
                replayRequest = (startOffset, round.count, widestFrame)
                replayExpected = response
            }
            emittedOffset += round.count
        }

        // --- R5: self-consistency replay in the same reference build ---------
        var selfConsistent = false
        var selfConsistencyDetail = "no round available to replay"
        var selfConsistencyRowCount = 0
        if let request = replayRequest, let expected = replayExpected {
            let again = try client.dflashReferenceRows(
                prefixTokens: context,
                seedTokenCount: seedTokens.count,
                startOffset: request.offset,
                rowCount: request.count,
                widestFrame: request.width
            )
            selfConsistencyRowCount = request.count
            if !again.ok {
                selfConsistencyDetail =
                    "replay failed: " + (again.error ?? "unknown error")
            } else if again.referenceK1Argmax != expected.referenceK1Argmax {
                selfConsistencyDetail = "width-1 argmax differed between replays"
            } else if again.referenceBlockArgmax != expected.referenceBlockArgmax {
                selfConsistencyDetail = "block-frame argmax differed between replays"
            } else if again.referenceFrameWidths != expected.referenceFrameWidths
                || again.referenceFrameArgmax != expected.referenceFrameArgmax
            {
                selfConsistencyDetail =
                    "declared-frame argmax differed between replays"
            } else if again.referenceTop2Tokens != expected.referenceTop2Tokens {
                selfConsistencyDetail = "top-2 token ids differed between replays"
            } else if again.referenceTop2Logits != expected.referenceTop2Logits {
                selfConsistencyDetail = "top-2 logit values differed between replays"
            } else if again.referenceTop1Logits != expected.referenceTop1Logits
                || again.referenceEmittedTokens != expected.referenceEmittedTokens
                || again.referenceEmittedTokenLogits
                    != expected.referenceEmittedTokenLogits
            {
                selfConsistencyDetail =
                    "emitted-token logit readouts differed between replays"
            } else {
                selfConsistent = true
                selfConsistencyDetail =
                    "replayed \(request.count) row(s) bit-identically"
            }
        }

        // --- R5, second half: the rows must reproduce their own chain ---------
        //
        // Only meaningful for a chain the REFERENCE generated: a supplied
        // `emitted` array is the candidate's, and Criterion E exists precisely
        // because a candidate may legitimately land on a different admissible
        // token. But when the reference produced the chain by width-1 argmax,
        // `rows[i].sequentialArgmax` IS that chain, and any disagreement means
        // the golden's own two halves were computed in different frames.
        var chainContradictions = [Int]()
        if chain != nil {
            for index in 0 ..< Swift.min(emitted.count, goldenRows.count)
            where goldenRows[index].sequentialArgmax != emitted[index] {
                chainContradictions.append(index)
            }
            if goldenRows.count != emitted.count {
                chainContradictions.append(Swift.min(goldenRows.count, emitted.count))
            }
        }
        if !chainContradictions.isEmpty {
            selfConsistent = false
            selfConsistencyDetail =
                "replay: \(selfConsistencyDetail); but the reference chain "
                + "contradicts its own rows at \(chainContradictions.count) of "
                + "\(emitted.count) position(s), first index "
                + "\(chainContradictions[0])"
        }

        let golden = DFlashReferenceGolden(
            seedTokens: seedTokens,
            referenceSeedToken: seedArgmax,
            rows: goldenRows,
            referenceSelfConsistent: selfConsistent,
            emittedTokens: chain == nil ? nil : emitted
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(golden).write(to: URL(fileURLWithPath: outputPath))

        return DFlashReferenceGoldenResult(
            rowCount: goldenRows.count,
            referenceSeedToken: seedArgmax,
            selfConsistent: selfConsistent,
            selfConsistencyRowCount: selfConsistencyRowCount,
            selfConsistencyDetail: selfConsistencyDetail,
            chainRowContradictionCount: chainContradictions.count,
            seedTokenCount: seedTokens.count,
            recordedFrameWidths: recordedWidths.sorted(),
            planOutputPath: chain?.planOutputPath
        )
    }

    /// Generate `steps` tokens by walking the reference's width-1 frame forward.
    ///
    /// `context` is the seed (`seedTokenCount` tokens) followed by the positions
    /// already walked; step `i` feeds `context.last` and appends the argmax. The
    /// reference keeps ONE continuously-advanced cache across these requests, so
    /// every generated token comes out of a genuine one-token decode on the state
    /// its predecessor left -- the same frame the golden's row replay uses, and
    /// the same frame a serial decoder runs in.
    ///
    /// The previous generator re-prefilled the whole growing prefix on every
    /// step. That was quadratic, and worse, it generated the chain in a frame
    /// that no longer matched the one the chain was later checked in the moment
    /// the prefix passed the seed: bulk-forwarding 517 tokens is not
    /// bulk-forwarding 512 and single-stepping 5. The visible symptom was a
    /// golden whose `emitted_tokens` disagreed with its own `sequential_argmax`.
    private static func walkReferenceChain(
        client: RuntimeWorkerClient,
        context: [Int],
        seedTokenCount: Int,
        steps: Int,
        label: String
    ) throws -> [Int] {
        guard steps > 0 else { return [] }
        var context = context
        var generated = [Int]()
        generated.reserveCapacity(steps)
        for index in 0 ..< steps {
            let response = try client.dflashReferenceRows(
                prefixTokens: context,
                seedTokenCount: seedTokenCount,
                startOffset: context.count - 1,
                rowCount: 1,
                widestFrame: 1
            )
            guard response.ok,
                  let token = response.referenceK1Argmax?.first
            else {
                throw MLXFastError.invalidInput(
                    "DFlash reference chain generation failed at \(label) step "
                        + "\(index): " + (response.error ?? "no row returned")
                )
            }
            generated.append(token)
            context.append(token)
            if (index + 1) % 32 == 0 || index + 1 == steps {
                fputs(
                    "dflash-reference: \(label) \(index + 1)/\(steps) "
                        + "(context \(context.count))\n",
                    stderr
                )
            }
        }
        return generated
    }

    /// Round layout for a generated chain, replaying the parent block schedule.
    private static func scheduledRounds(
        tokenCount: Int,
        maxBlockSize: Int,
        scheduleSeed: UInt64
    ) -> [DFlashEmittedPlan.Round] {
        var schedule = DFlashBlockSchedule(
            seed: scheduleSeed,
            maxBlockSize: maxBlockSize,
            minBlockSize: maxBlockSize == 1 ? 1 : 2
        )
        var rounds = [DFlashEmittedPlan.Round]()
        var remaining = tokenCount
        while remaining > 0 {
            let width = schedule.nextBlockSize()
            let count = Swift.min(width, remaining)
            rounds.append(
                DFlashEmittedPlan.Round(blockSize: width, count: count)
            )
            remaining -= count
        }
        return rounds
    }
}
