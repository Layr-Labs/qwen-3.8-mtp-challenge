import Foundation
import MLXFastCore

// Trusted-parent driver for the Qwen 3.6 native-MTP track. Links no MLX: it
// spawns the sandboxed worker, owns the clock and the depth, journals every
// round, and then -- with the candidate torn down -- has the pinned reference
// replay the journal.
//
// Both verbs share this one path. `mtp-timed` reports the parent's own wall
// clock over its own configured token total; `mtp-verify` runs the identical
// decode untimed and additionally retains the row ledger as evidence. Running
// them through the same code is deliberate: the retired Gemma track ran its two
// sides through different verbs, which put any divergence between the code paths
// straight into the score, and the DFlash track fixed that by moving the width
// instead. Here the depth moves, and the verbs differ only in what they report.

extension QwenRuntime {
    /// Run one native-MTP decode over the golden and audit it.
    ///
    /// - Parameters:
    ///   - verb: the reporting verb name. It selects what is REPORTED, never
    ///     what is run: both verbs execute this identical decode, and only
    ///     `mtp-timed` treats the parent's wall clock as authoritative.
    ///   - retainLedger: `mtp-verify` keeps the per-row evidence; `mtp-timed`
    ///     does not (a 512-token ledger of top-2 readouts would dominate a
    ///     timing report).
    public static func qwenMTPDecode(
        verb: String,
        options: QwenMTPOptions,
        workerOptions: RuntimeWorkerOptions,
        referenceWorkerOptions: RuntimeWorkerOptions? = nil,
        tolerance: QwenMTPNearTieTolerance = QwenMTPNearTieTolerance(),
        retainLedger: Bool
    ) throws -> QwenMTPReport {
        let golden = try loadQwenMTPGolden(options.goldenPath)
        guard golden.referenceSelfConsistent != false else {
            throw QwenMTPContractViolation(
                kind: .referenceNotSelfConsistent,
                detail: "the reference rows report a failed self-consistency "
                    + "replay; this is an operator fault, not a submission fault"
            )
        }
        // ONE ROW MORE THAN THE BUDGET, and the +1 is not slack. `rows[i]`
        // describes the token emitted at index `i + 1` (index 0 is the seed
        // argmax, which has no row and is pinned by `reference_seed_token`), so
        // covering N emitted tokens needs N-1 rows. The LAST round's target tail
        // row then predicts the token at index N -- one past the window -- and
        // that row is declared, so it has to be reference-checked too. Requiring
        // N+1 rows makes a short reference a legible abort here rather than a
        // "row not reference-checked" failure 500 tokens into a gated run.
        guard options.totalTokenCount > 0,
              golden.rows.count >= options.totalTokenCount + 1
        else {
            throw MLXFastError.invalidInput(
                "the MTP reference carries \(golden.rows.count) rows; a "
                    + "\(options.totalTokenCount)-token window needs at least "
                    + "\(options.totalTokenCount + 1) (one per emitted token "
                    + "after the seed argmax, plus one for the final round's "
                    + "target tail row). Regenerate with --generate "
                    + "\(options.totalTokenCount + 1)."
            )
        }
        guard options.depth >= MLXFastConstants.qwenMTPSerialControlDepth else {
            throw MLXFastError.invalidInput(
                "--mtp-depth must be at least "
                    + "\(MLXFastConstants.qwenMTPSerialControlDepth) "
                    + "(0 is the true serial control: MTP off)")
        }

        let client = try RuntimeWorkerClient(
            options: workerOptions,
            weightsPath: options.targetWeightsPath,
            mtpHeadPath: options.mtpHeadPath
        )
        // Closed EXPLICITLY before the reference worker starts, not only on the
        // way out: two live residencies of this checkpoint would put the box
        // under memory pressure the measurement never accounted for. `close()` is
        // idempotent, so this defer only covers the failure paths.
        defer { client.close() }

        // Untimed phase start, BEFORE the clock: the allocator clear and the
        // round-shape warm never see the seed, so timing them measures nothing
        // about the submission.
        let warm = try client.warmMTPDecode()
        guard warm.ok else {
            throw MLXFastError.invalidInput(
                "the MTP worker failed the untimed warm: "
                    + (warm.error ?? "no reason reported"))
        }

        // The seed prefill IS charged to the decode measurement, the way the
        // paired contract requires: the clock starts immediately before the
        // request so the seed cost cannot be hidden outside the window.
        let started = Date()
        let begin = try client.beginMTPDecode(seedTokens: golden.seedTokens)
        // The prefill's share of the charged window, read the instant the seed
        // request returns. Observability only: `started` stays the decode
        // clock's origin and nothing is subtracted, so the scored quantity is
        // bit-identical to a build without this line (see the report field).
        let seedPrefillSeconds = Date().timeIntervalSince(started)
        guard begin.ok, let seedToken = begin.seedToken else {
            throw MLXFastError.invalidInput(
                "the MTP worker failed the seed prefill: "
                    + (begin.error ?? "no seed token returned"))
        }
        guard seedToken == golden.referenceSeedToken else {
            throw QwenMTPContractViolation(
                kind: .seedTokenMismatch,
                step: 0,
                detail: "seed prefill token \(seedToken) disagreed with the "
                    + "reference's \(golden.referenceSeedToken)"
            )
        }

        // The seed argmax IS the first emitted token: the session holds it as the
        // pending primary and commits it at the top of round 1.
        var emitted: [Int] = []
        var rounds: [QwenMTPObservedRound] = []
        var latencies: [Double] = []

        while emitted.count < options.totalTokenCount {
            // THE OFFERED CEILING, and it lives HERE rather than in the worker.
            //
            // OPERATOR-RATIFIED 2026-08-14: this number is an OFFER, not a
            // schedule. The candidate decides how many tokens it actually
            // drafts in the round (0 through `qwenMTPMaxDraftDepth`), and the
            // trusted bound the ledger is closed against is that constant, not
            // this offer -- see `requireStructurallySound`. What the parent
            // still owns is the tail: the worker is deliberately never told how
            // much of the decode window remains, because a worker that knew
            // could special-case the tail of a scored window, so the parent
            // narrows its own offer as the budget runs down. A round that
            // overruns anyway (the candidate drafted wider than the offer near
            // the tail, which is legal) is TRUNCATED below; the rows it
            // declared are still reference-checked, which is why
            // `declared_rows_total >= emitted_token_total` is an inequality.
            //
            // The serial control is exempt: depth 0 commits exactly one token
            // per round, so it can never overrun, and clamping it upward would
            // silently turn the denominator back into a speculative decoder.
            let remaining = options.totalTokenCount - emitted.count
            let requestedDepth =
                options.depth == MLXFastConstants.qwenMTPSerialControlDepth
                ? MLXFastConstants.qwenMTPSerialControlDepth
                : Swift.max(
                    1,
                    Swift.min(
                        options.depth,
                        MLXFastConstants.qwenMTPMaxDraftDepth,
                        remaining - 1))
            let roundStart = Date()
            let response = try client.mtpDecodeRound(depth: requestedDepth)
            let latency = Date().timeIntervalSince(roundStart)
            guard response.ok, let tokens = response.tokens, !tokens.isEmpty else {
                throw MLXFastError.invalidInput(
                    "the MTP round request failed: "
                        + (response.error ?? "no tokens returned"))
            }
            let round = QwenMTPObservedRound(
                emittedBaseIndex: emitted.count,
                requestedDepth: requestedDepth,
                tokens: tokens,
                declaredRows: response.declaredRows ?? 0,
                draftTokens: response.draftTokens ?? [],
                acceptedDraftCount: response.acceptedDraftCount ?? 0,
                rejectedDraftCount: response.rejectedDraftCount ?? 0,
                perRowTop2Tokens: response.perRowTop2Tokens ?? [],
                perRowTop2Logits: response.perRowTop2Logits ?? [],
                targetCacheOffset: response.targetCacheOffset ?? -1,
                latencySeconds: latency
            )
            try requireStructurallySound(round: round, depth: requestedDepth)
            // NON-DRAFTING ROUNDS ARE LEGAL (operator-ratified 2026-08-14).
            //
            // This is where a drafts-empty round used to be refused as
            // `stopTokenInsideWindow`. That refusal was a defence of the OLD
            // scoring anchor: with the score normalised so an unmodified
            // depth-2 tree read 1.0, a candidate that quietly stopped drafting
            // would have been measuring serial against a normalising reference
            // it no longer resembled. Under serial-anchored scoring the
            // incentive is gone by construction -- a candidate that never
            // drafts IS the serial control and scores exactly 1.0 -- so an
            // adaptive policy is free to skip a round it expects the head to
            // miss.
            //
            // WHAT DID NOT MOVE. A round must still commit at least one token
            // (`tokens.isEmpty` is rejected above), the loop still runs until
            // the parent's own configured total is reached, and every emitted
            // token is still required to equal the serial trajectory below --
            // so a trajectory that genuinely terminated early cannot short the
            // window silently. It shows up as a token mismatch, which is the
            // stronger signal, not as a short denominator.
            emitted.append(contentsOf: tokens)
            rounds.append(round)
            latencies.append(latency)
        }
        let decodeSeconds = Date().timeIntervalSince(started)

        // TRUNCATE TO THE PARENT'S OWN DENOMINATOR. With the clamp above a round
        // can overrun by at most one token (`remaining == 1` and the single draft
        // accepted), and the scored window is the parent's configured total, not
        // whatever the last round happened to commit. The extra token is simply
        // outside the window: the round that produced it still declared its rows
        // and those rows are still reference-checked, which is why
        // `declared_rows_total >= emitted_token_total` is an inequality.
        //
        // Every offset reported below is PARENT-DERIVED from this truncated
        // total, never read back from the worker -- the ledger the box wrapper
        // audits must not be something the measured party can assert.
        if emitted.count > options.totalTokenCount {
            emitted = Array(emitted.prefix(options.totalTokenCount))
        }
        let emittedTotal = emitted.count

        // --- exactness against the serial trajectory --------------------------
        var firstDivergence: Int?
        for index in 0 ..< emittedTotal {
            let expected = index == 0
                ? golden.referenceSeedToken
                : golden.rows[index - 1].sequentialArgmax
            if emitted[index] != expected {
                firstDivergence = index
                break
            }
        }

        // --- post-window reference replay ------------------------------------
        //
        // The clock has stopped and the candidate is gone. Only now is a
        // reference verdict meaningful, because only now does the parent know the
        // candidate's own emitted chain.
        client.close()
        let audit = try auditQwenMTPJournal(
            rounds: rounds,
            emitted: emitted,
            golden: golden,
            options: options,
            workerOptions: referenceWorkerOptions ?? workerOptions,
            tolerance: tolerance
        )

        let declaredRowTotal = rounds.reduce(0) { $0 + $1.declaredRows }
        guard audit.rows.count == declaredRowTotal else {
            throw QwenMTPContractViolation(
                kind: .rowNotReferenceChecked,
                detail: "the reference checked \(audit.rows.count) of "
                    + "\(declaredRowTotal) declared rows"
            )
        }

        var divergenceMargin: Double?
        if let index = firstDivergence {
            divergenceMargin = index == 0
                ? nil
                : golden.rows[index - 1].top2Logits.flatMap {
                    $0.count >= 2 ? $0[0] - $0[1] : nil
                }
        }

        // Whole-window summaries use the SAME lower-median rule as the
        // after-first p50, so the payload carries exactly one definition of
        // "p50" rather than two that differ on even-length windows.
        let sortedLatencies = latencies.sorted()
        return QwenMTPReport(
            verb: verb,
            depth: options.depth,
            seedTokenCount: golden.seedTokens.count,
            decodeTokenCount: options.totalTokenCount,
            emittedTokenTotal: emittedTotal,
            allTokensMatched: firstDivergence == nil,
            firstDivergenceIndex: firstDivergence,
            firstDivergenceReferenceMargin: divergenceMargin,
            roundCount: rounds.count,
            acceptedDraftTotal: rounds.reduce(0) { $0 + $1.acceptedDraftCount },
            rejectedDraftTotal: rounds.reduce(0) { $0 + $1.rejectedDraftCount },
            // One target-produced tail row per round, by construction.
            targetTailTotal: rounds.count,
            declaredRowTotal: declaredRowTotal,
            referenceCheckedRowTotal: audit.rows.count,
            rejectedRowsReferenceChecked: audit.rejectedRowsChecked,
            verifyBlockReplayedRoundCount: audit.replayedRoundCount,
            residualDivergenceCount: audit.residualDivergenceCount,
            maxRejectedTailLogitDelta: audit.maxRejectedTailLogitDelta,
            targetCacheOffsetFinal: golden.seedTokens.count + emittedTotal,
            decodeSeconds: decodeSeconds,
            roundRequestSeconds: latencies,
            maxRoundRequestSeconds: sortedLatencies.last ?? 0,
            p50RoundRequestSeconds: QwenMTPReport.lowerMedian(latencies),
            seedPrefillSeconds: seedPrefillSeconds,
            // PARENT-DERIVED, from the parent's own journal: the number of
            // drafts each round actually proposed. Every effective-depth
            // summary in the sealed report is computed from this array, so a
            // candidate cannot assert its own schedule any more than it can
            // assert its own timing.
            effectiveDraftLengths: rounds.map { $0.draftTokens.count },
            ledger: retainLedger ? audit.rows : []
        )
    }

    /// Arithmetic the parent can check without any reference, per round.
    ///
    /// THE BOUND IS THE TRUSTED MAXIMUM, NOT THE OFFER. `depth` is what the
    /// parent asked for; `round.draftTokens.count` is what the candidate
    /// actually proposed, and since 2026-08-14 those are allowed to differ in
    /// either direction. What may not differ is the ledger: every quantity
    /// below is derived from the ACTUAL draft count, so a round is sound
    /// exactly when the rows it declared, the tokens it committed and the
    /// drafts it proposed all agree with each other and sit inside
    /// `qwenMTPMaxDraftDepth`.
    private static func requireStructurallySound(
        round: QwenMTPObservedRound,
        depth: Int
    ) throws {
        let expectedRows = round.draftTokens.isEmpty
            ? 1
            : QwenMTPRowAccounting.rowsPerRound(depth: round.draftTokens.count)
        guard round.declaredRows == expectedRows,
              round.perRowTop2Tokens.count == round.declaredRows,
              round.perRowTop2Logits.count == round.declaredRows,
              round.acceptedDraftCount + round.rejectedDraftCount
                  == round.draftTokens.count,
              round.acceptedDraftCount >= 0,
              round.rejectedDraftCount >= 0,
              round.draftTokens.count
                  <= MLXFastConstants.qwenMTPMaxDraftDepth,
              // The serial control drafts nothing, ever. Depth 0 is the
              // denominator this track divides by, and a "serial" leg that
              // quietly drafted would be an accelerated denominator -- the
              // exact mislabel that produced a 0.875x "speedup" before depth 0
              // existed. This is the one place a requested depth still binds.
              depth != MLXFastConstants.qwenMTPSerialControlDepth
                  || round.draftTokens.isEmpty,
              // The committed block is the primary plus exactly the accepted
              // drafts; anything else means the worker emitted a token the accept
              // walk did not accept.
              round.tokens.count == 1 + round.acceptedDraftCount,
              round.tokens.dropFirst().elementsEqual(
                  round.draftTokens.prefix(round.acceptedDraftCount))
        else {
            throw QwenMTPContractViolation(
                kind: .rowLedgerNotClosed,
                step: round.emittedBaseIndex,
                detail: "round declared \(round.declaredRows) rows for "
                    + "\(round.draftTokens.count) drafts, committed "
                    + "\(round.tokens.count) tokens with "
                    + "\(round.acceptedDraftCount) accepted"
            )
        }
    }

    struct QwenMTPAudit {
        var rows: [QwenMTPLedgerRow] = []
        var rejectedRowsChecked = 0
        var replayedRoundCount = 0
        var residualDivergenceCount = 0
        var maxRejectedTailLogitDelta: Double = 0
    }

    /// Reference-check every declared row.
    ///
    /// TWO SOURCES, AND THE SPLIT IS FORCED. A draft row `i` is the target's
    /// greedy continuation of `[primary] + drafts[0 ..< i]`. While `i <=
    /// accepted`, that prefix is exactly the serial trajectory, so the SERIAL
    /// GOLDEN describes the row -- including the first REJECTED row, whose
    /// reference is the serial token the draft failed to match. Past that the
    /// prefix contains a rejected token and no serial golden can describe the
    /// row at all; those rows are priced by replaying the candidate's own verify
    /// block on the pinned reference. Dropping the second source would leave the
    /// rejected tail unpriced, which is precisely the accounting hole the box
    /// wrapper's EQUALITY on `reference_checked_row_total` closes.
    private static func auditQwenMTPJournal(
        rounds: [QwenMTPObservedRound],
        emitted: [Int],
        golden: QwenMTPReferenceGolden,
        options: QwenMTPOptions,
        workerOptions: RuntimeWorkerOptions,
        tolerance: QwenMTPNearTieTolerance
    ) throws -> QwenMTPAudit {
        var audit = QwenMTPAudit()
        var rowIndex = 0

        // The reference walks the candidate's OWN chain, so the frame it prices a
        // row in is the frame the candidate produced it in.
        let context = golden.seedTokens + emitted
        let seedCount = golden.seedTokens.count
        // A round needs the replay exactly when it has rows past the FIRST
        // rejection: `goldenBackedDrafts = min(accepted + 1, draftCount)`, so
        // `draftCount > goldenBackedDrafts` reduces to `rejected > 1`. A run with
        // no such round never opens a second worker at all.
        let needsReplay = rounds.contains { $0.rejectedDraftCount > 1 }

        var referenceClient: RuntimeWorkerClient?
        defer { referenceClient?.close() }
        if needsReplay {
            let client = try RuntimeWorkerClient(
                options: workerOptions,
                weightsPath: options.referenceWeightsPath
                    ?? options.targetWeightsPath,
                mtpHeadPath: options.referenceHeadPath ?? options.mtpHeadPath
            )
            let prefill = try client.mtpReferencePrefill(
                seedTokens: golden.seedTokens)
            guard prefill.ok, let referenceSeedToken = prefill.seedToken else {
                throw MLXFastError.invalidInput(
                    "the MTP reference replay failed its seed prefill: "
                        + (prefill.error ?? "no seed token returned"))
            }
            // If the LIVE reference disagrees with the golden's record, the
            // golden and the reference build are not the same reference, so the
            // replay would price rows against a chain neither party produced.
            // That is an operator fault -- a mismatched tree -- not a submission
            // fault, and it must fail loudly instead of being scored.
            guard referenceSeedToken == golden.referenceSeedToken else {
                throw QwenMTPContractViolation(
                    kind: .referenceNotSelfConsistent,
                    step: 0,
                    detail: "the live reference's seed token "
                        + "\(referenceSeedToken) disagrees with the golden's "
                        + "\(golden.referenceSeedToken); the reference build or "
                        + "weights do not match the golden (operator fault)"
                )
            }
            referenceClient = client
        }

        for (roundNumber, round) in rounds.enumerated() {
            let base = round.emittedBaseIndex
            let accepted = round.acceptedDraftCount
            let draftCount = round.draftTokens.count

            // Rows the serial golden can describe: drafts 0 ... accepted (capped
            // at the draft window) and the tail row.
            let goldenBackedDrafts = Swift.min(accepted + 1, draftCount)
            for index in 0 ..< goldenBackedDrafts {
                let referenceToken = try serialReference(
                    golden: golden, emittedIndex: base + index + 1)
                audit.rows.append(
                    makeRow(
                        rowIndex: rowIndex,
                        round: roundNumber,
                        kind: .draft,
                        draftIndex: index,
                        accepted: index < accepted,
                        token: round.draftTokens[index],
                        round: round,
                        rowSlot: index,
                        referenceToken: referenceToken.token,
                        referenceMargin: referenceToken.margin,
                        source: .serialGolden
                    ))
                rowIndex += 1
                if index >= accepted { audit.rejectedRowsChecked += 1 }
            }

            // Rows past the first rejection: no serial reference exists.
            if draftCount > goldenBackedDrafts {
                guard let client = referenceClient else {
                    throw QwenMTPContractViolation(
                        kind: .rowNotReferenceChecked,
                        step: base,
                        detail: "round \(roundNumber) has "
                            + "\(draftCount - goldenBackedDrafts) rejected-tail "
                            + "rows and no reference worker was opened to price "
                            + "them"
                    )
                }
                let block = round.verifyBlockTokens
                let response = try client.mtpReferenceRows(
                    prefixTokens: context,
                    seedTokenCount: seedCount,
                    startOffset: seedCount + base,
                    rowCount: 1,
                    verifyBlockTokens: block
                )
                guard response.ok,
                      let replayTokens = response.referenceVerifyTop2Tokens,
                      let replayLogits = response.referenceVerifyTop2Logits,
                      replayTokens.count == block.count,
                      replayLogits.count == block.count
                else {
                    throw MLXFastError.invalidInput(
                        "the MTP reference did not return the \(block.count)-row "
                            + "verify-block replay it was asked for at round "
                            + "\(roundNumber): "
                            + (response.error ?? "no reason reported"))
                }
                audit.replayedRoundCount += 1
                for index in goldenBackedDrafts ..< draftCount {
                    let referenceRow = replayTokens[index]
                    let referenceValues = replayLogits[index]
                    let margin = referenceValues.count >= 2
                        ? referenceValues[0] - referenceValues[1]
                        : Double.infinity
                    let candidateTop1 = round.perRowTop2Tokens[index].first ?? -1
                    let candidateLogit =
                        round.perRowTop2Logits[index].first ?? 0
                    audit.maxRejectedTailLogitDelta = Swift.max(
                        audit.maxRejectedTailLogitDelta,
                        abs(candidateLogit - (referenceValues.first ?? 0)))
                    if candidateTop1 != (referenceRow.first ?? -1) {
                        // A disagreement is admitted ONLY when the reference row
                        // is a near tie; otherwise the two paths genuinely
                        // disagree, which is a logic bug, not numerics.
                        guard margin < tolerance.referenceMargin else {
                            throw QwenMTPContractViolation(
                                kind: .rejectedTailDiverged,
                                step: base,
                                detail: "rejected-tail row \(index) of round "
                                    + "\(roundNumber) declared top-1 "
                                    + "\(candidateTop1) but the reference "
                                    + "replayed \(referenceRow.first ?? -1) at a "
                                    + "margin of \(margin)"
                            )
                        }
                        audit.residualDivergenceCount += 1
                    }
                    audit.rows.append(
                        makeRow(
                            rowIndex: rowIndex,
                            round: roundNumber,
                            kind: .draft,
                            draftIndex: index,
                            accepted: false,
                            token: round.draftTokens[index],
                            round: round,
                            rowSlot: index,
                            referenceToken: referenceRow.first ?? -1,
                            referenceMargin: margin,
                            source: .verifyBlockReplay
                        ))
                    rowIndex += 1
                    audit.rejectedRowsChecked += 1
                }
            }

            // The tail row: its argmax is the next round's primary, i.e. the
            // serial token at emitted index base + accepted + 1.
            let tailReference = try serialReference(
                golden: golden, emittedIndex: base + accepted + 1)
            audit.rows.append(
                makeRow(
                    rowIndex: rowIndex,
                    round: roundNumber,
                    kind: .targetTail,
                    draftIndex: nil,
                    accepted: true,
                    token: tailReference.token,
                    round: round,
                    rowSlot: round.declaredRows - 1,
                    referenceToken: tailReference.token,
                    referenceMargin: tailReference.margin,
                    source: .serialGolden
                ))
            rowIndex += 1
        }
        return audit
    }

    private static func serialReference(
        golden: QwenMTPReferenceGolden,
        emittedIndex: Int
    ) throws -> (token: Int, margin: Double) {
        if emittedIndex == 0 {
            return (golden.referenceSeedToken, .infinity)
        }
        let rowIndex = emittedIndex - 1
        guard rowIndex >= 0, rowIndex < golden.rows.count else {
            throw QwenMTPContractViolation(
                kind: .rowNotReferenceChecked,
                step: emittedIndex,
                detail: "the reference carries \(golden.rows.count) rows; the "
                    + "run reached emitted index \(emittedIndex). Regenerate the "
                    + "reference with at least one more row than the token "
                    + "budget."
            )
        }
        let row = golden.rows[rowIndex]
        let margin = row.top2Logits.flatMap {
            $0.count >= 2 ? $0[0] - $0[1] : nil
        } ?? Double.infinity
        return (row.sequentialArgmax, margin)
    }

    private static func makeRow(
        rowIndex: Int,
        round roundNumber: Int,
        kind: QwenMTPLedgerRow.Kind,
        draftIndex: Int?,
        accepted: Bool,
        token: Int,
        round: QwenMTPObservedRound,
        rowSlot: Int,
        referenceToken: Int,
        referenceMargin: Double,
        source: QwenMTPLedgerRow.ReferenceSource
    ) -> QwenMTPLedgerRow {
        QwenMTPLedgerRow(
            rowIndex: rowIndex,
            round: roundNumber,
            kind: kind,
            draftIndex: draftIndex,
            accepted: accepted,
            token: token,
            top2Tokens: rowSlot < round.perRowTop2Tokens.count
                ? round.perRowTop2Tokens[rowSlot] : [],
            top2Logits: rowSlot < round.perRowTop2Logits.count
                ? round.perRowTop2Logits[rowSlot] : [],
            referenceToken: referenceToken,
            referenceCheckedBy: source,
            referenceMargin: referenceMargin
        )
    }

    static func loadQwenMTPGolden(_ path: String) throws -> QwenMTPReferenceGolden {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try JSONDecoder().decode(QwenMTPReferenceGolden.self, from: data)
    }
}

// MARK: - reference row generation (`mtp-verify --generate`)

/// The seed prompt a reference pass starts from.
public struct QwenMTPEmittedPlan: Codable {
    public let seedTokens: [Int]
    public let emitted: [Int]?

    enum CodingKeys: String, CodingKey {
        case seedTokens = "seed_tokens"
        case emitted
    }

    public init(seedTokens: [Int], emitted: [Int]? = nil) {
        self.seedTokens = seedTokens
        self.emitted = emitted
    }
}

public struct QwenMTPReferenceResult {
    public let rowCount: Int
    public let referenceSeedToken: Int
    public let selfConsistent: Bool
    public let selfConsistencyDetail: String
    public let chainRowContradictionCount: Int
    public let seedTokenCount: Int
    public let planOutputPath: String?
}

extension QwenRuntime {
    /// Generate the track's serial reference rows.
    ///
    /// The worker spawned here MUST be the one built from the pinned baseline
    /// tree. Locally it is the candidate's own build, which is why
    /// `benchmark-qwen-mtp.sh` labels its result NOT RANKABLE and says so in
    /// prose: a candidate that generates its own reference has proven internal
    /// consistency, not fidelity.
    ///
    /// R5 self-consistency has two halves and BOTH are mandatory:
    ///   1. one row is replayed and required to come back bit-identical. The
    ///      replay targets the FIRST row, whose offset is behind the reference's
    ///      live walk, so it takes the rebuild-from-scratch path -- proving the
    ///      fallback construction agrees with the continuous walk, which is what
    ///      makes a stateful reference admissible at all;
    ///   2. every row's argmax must equal the chain token at that index. Three
    ///      DFlash goldens shipped with `reference_self_consistent: true` while
    ///      `emitted_tokens[i]` disagreed with `rows[i].sequential_argmax`, i.e.
    ///      the reference contradicting itself, so this half is not optional.
    public static func qwenMTPReferenceGolden(
        plan: QwenMTPEmittedPlan,
        generateTokenCount: Int,
        targetWeightsPath: String,
        mtpHeadPath: String,
        outputPath: String,
        planOutputPath: String?,
        workerOptions: RuntimeWorkerOptions
    ) throws -> QwenMTPReferenceResult {
        guard !plan.seedTokens.isEmpty else {
            throw MLXFastError.invalidInput(
                "the MTP reference plan has an empty seed")
        }
        guard generateTokenCount > 0 else {
            throw MLXFastError.invalidInput(
                "--generate needs a positive token count")
        }

        let client = try RuntimeWorkerClient(
            options: workerOptions,
            weightsPath: targetWeightsPath,
            mtpHeadPath: mtpHeadPath
        )
        defer { client.close() }

        let prefill = try client.mtpReferencePrefill(seedTokens: plan.seedTokens)
        guard prefill.ok, let seedArgmax = prefill.seedToken else {
            throw MLXFastError.invalidInput(
                "the MTP reference could not establish the seed token: "
                    + (prefill.error ?? "no seed token returned"))
        }

        // Walk the width-1 frame forward one token at a time. Every generated
        // token comes out of a genuine one-token decode on the state its
        // predecessor left -- the same frame the rows are later checked in, and
        // the same frame a serial decoder runs in. Re-prefilling the growing
        // prefix on every step would be quadratic AND would generate the chain in
        // a frame that stops matching the one it is checked in the moment the
        // prefix passes the seed.
        var context = plan.seedTokens + [seedArgmax]
        var rows: [QwenMTPReferenceGolden.Row] = []
        var emitted: [Int] = []
        var firstRowExpectation: (tokens: [Int], logits: [Double])?

        for step in 0 ..< generateTokenCount {
            let response = try client.mtpReferenceRows(
                prefixTokens: context,
                seedTokenCount: plan.seedTokens.count,
                startOffset: context.count - 1,
                rowCount: 1
            )
            guard response.ok,
                  let argmax = response.referenceK1Argmax?.first,
                  let top2Tokens = response.referenceTop2Tokens?.first,
                  let top2Logits = response.referenceTop2Logits?.first
            else {
                throw MLXFastError.invalidInput(
                    "the MTP reference chain failed at step \(step): "
                        + (response.error ?? "no row returned"))
            }
            rows.append(
                QwenMTPReferenceGolden.Row(
                    sequentialArgmax: argmax,
                    top2Tokens: top2Tokens,
                    top2Logits: top2Logits,
                    top1Logit: top2Logits.first
                ))
            emitted.append(argmax)
            context.append(argmax)
            if firstRowExpectation == nil {
                firstRowExpectation = (top2Tokens, top2Logits)
            }
            if (step + 1) % 32 == 0 || step + 1 == generateTokenCount {
                fputs(
                    "mtp-verify: generate \(step + 1)/\(generateTokenCount) "
                        + "(context \(context.count))\n",
                    stderr
                )
            }
        }

        // R5 first half: replay the FIRST row. Its offset is far behind the live
        // walk, so this takes the rebuild path.
        var selfConsistent = false
        var detail = "no row available to replay"
        if let expected = firstRowExpectation {
            let again = try client.mtpReferenceRows(
                prefixTokens: context,
                seedTokenCount: plan.seedTokens.count,
                startOffset: plan.seedTokens.count,
                rowCount: 1
            )
            if !again.ok {
                detail = "replay failed: " + (again.error ?? "unknown error")
            } else if again.referenceTop2Tokens?.first != expected.tokens {
                detail = "top-2 token ids differed between replays"
            } else if again.referenceTop2Logits?.first != expected.logits {
                detail = "top-2 logit values differed between replays"
            } else {
                selfConsistent = true
                detail = "replayed 1 row bit-identically"
            }
        }

        // R5 second half: the rows must reproduce their own chain.
        var contradictions = 0
        for index in 0 ..< Swift.min(emitted.count, rows.count)
        where rows[index].sequentialArgmax != emitted[index] {
            contradictions += 1
        }
        if contradictions > 0 {
            selfConsistent = false
            detail = "replay: \(detail); but the reference chain contradicts its "
                + "own rows at \(contradictions) of \(emitted.count) positions"
        }

        let golden = QwenMTPReferenceGolden(
            seedTokens: plan.seedTokens,
            referenceSeedToken: seedArgmax,
            rows: rows,
            referenceSelfConsistent: selfConsistent,
            emittedTokens: emitted
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [
            .prettyPrinted, .sortedKeys, .withoutEscapingSlashes,
        ]
        try encoder.encode(golden).write(to: URL(fileURLWithPath: outputPath))
        if let planOutputPath {
            let reconstructed = QwenMTPEmittedPlan(
                seedTokens: plan.seedTokens,
                emitted: [seedArgmax] + emitted
            )
            try encoder.encode(reconstructed)
                .write(to: URL(fileURLWithPath: planOutputPath))
        }

        return QwenMTPReferenceResult(
            rowCount: rows.count,
            referenceSeedToken: seedArgmax,
            selfConsistent: selfConsistent,
            selfConsistencyDetail: detail,
            chainRowContradictionCount: contradictions,
            seedTokenCount: plan.seedTokens.count,
            planOutputPath: planOutputPath
        )
    }
}
