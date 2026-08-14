import Foundation
import MLXFastCore
import MLXFastHarness
import Testing

// Unit-level guards for the Qwen 3.6 native-MTP verbs. NOTHING here loads a
// model, opens a worker or touches a network: every assertion reads checked-in
// text, runs pure arithmetic, or feeds a synthetic report through the same code
// a real one goes through.
//
// The three classes of bug these exist for, in the order they have actually
// happened on this repo's other speculative track:
//
//   1. A GATE THAT ASSERTS A FIELD THE PAYLOAD DOES NOT EMIT. `jq -e` then fails
//      on every run and the gate is dead before it ever gates anything. The
//      DFlash parity gate shipped that way (it required `.experimental` and
//      `.target_verification_mode`); the general form of the check is what would
//      have caught it, so it is applied here from day one.
//   2. A LEDGER THAT CLOSES BY ARITHMETIC RATHER THAN BY AUDIT. The retired MTP
//      track validated `2*pairs - rollbacks + serialRows == totalTokenCount`,
//      which a worker that verified nothing passes.
//   3. A FLAG OR FIELD RENAMED ON ONE SIDE OF A TWO-CONSUMER CONTRACT.

@Suite
struct QwenMTPPayloadSchemaTests {
    private typealias S = DFlashGateTextSupport

    private static let workflowPath =
        ".github/workflows/qwen-mtp-ranked-benchmark.yml"
    private static let cliPath = "Sources/MLXFastCLI/main.swift"

    /// The keys `emitQwenMTPPayload` actually puts in its payload — the
    /// authority for what any gate is allowed to assert on.
    static func payloadKeys() throws -> Set<String> {
        let cli = try S.text(cliPath)
        let function = try #require(
            cli.range(of: "private static func emitQwenMTPPayload("),
            "emitQwenMTPPayload is gone from the CLI"
        )
        let tail = cli[function.upperBound...]
        let literal = try #require(
            tail.range(of: "var payload: [String: Any] = ["),
            "emitQwenMTPPayload no longer builds a `payload` dictionary"
        )
        let serialize = try #require(
            tail.range(
                of: "JSONSerialization.data(",
                range: literal.upperBound ..< tail.endIndex
            ),
            "could not find the end of the MTP payload construction"
        )
        let region = String(tail[literal.upperBound ..< serialize.lowerBound])
        var keys = Set(S.captures(#""([a-z][a-z0-9_]*)":"#, in: region))
        for key in S.captures(#"payload\["([a-z][a-z0-9_]*)"\]"#, in: region) {
            keys.insert(key)
        }
        return keys
    }

    /// Anti-vacuity: the extractor must find the keys the payload demonstrably
    /// has, so a parsing regression cannot make every check below pass silently.
    @Test
    func thePayloadKeyExtractorFindsRealKeys() throws {
        let keys = try Self.payloadKeys()
        for key in [
            "track_id", "verb", "mtp_depth", "all_tokens_matched",
            "parity_all_ok", "declared_rows_total",
            "reference_checked_row_total", "uses_native_mtp_head",
            "uses_pinned_mtp_head",
        ] {
            #expect(keys.contains(key), "payload is missing \(key): \(keys)")
        }
        #expect(
            keys.count > 20,
            "MTP payload key extraction looks broken: \(keys)"
        )
    }

    /// THE CHECK THE DFLASH GATE NEEDED AND DID NOT HAVE. Every field the ranked
    /// correctness gate's jq requires must be a field the payload emits.
    @Test
    func theRankedGateAssertsOnlyFieldsTheReportEmits() throws {
        let workflow = try S.text(Self.workflowPath)
        let step = try S.stepBody(
            workflow, "Qwen-MTP correctness and parity gate (untimed)")
        let asserted = S.jqAssertedFields(in: S.executable(step))
        let emitted = try Self.payloadKeys()

        #expect(
            !asserted.required.isEmpty,
            "the gate asserts nothing; either the step moved or the jq parser broke"
        )
        let missing = asserted.required.subtracting(emitted)
        #expect(
            missing.isEmpty,
            """
            the Qwen-MTP correctness gate requires \(missing.sorted()), which \
            mtp-verify never emits. `jq -e` will fail on EVERY run and the gate \
            will be dead before it gates anything. Either drop the conjunct or \
            add the field -- do not leave them disagreeing.
            """
        )
        // The absence assertions must stay absences: a payload that grew a
        // `score` key would turn the untimed fidelity verb into a second,
        // ungated scoring path.
        let wronglyPresent = asserted.forbidden.intersection(emitted)
        #expect(
            wronglyPresent.isEmpty,
            """
            mtp-verify now emits \(wronglyPresent.sorted()), which the gate \
            requires to be ABSENT. The untimed verb must not be able to publish \
            a number.
            """
        )
        #expect(asserted.forbidden.contains("score"))
        #expect(asserted.forbidden.contains("speedup"))
    }

    /// The box wrapper is not in this repository, so its field list is mirrored
    /// here BY NAME. Every field `measure-qwen-mtp-job.sh` reads out of a phase
    /// report or an exactness probe has to exist in the payload, or a ranked run
    /// rejects a valid measurement for a reason no local test would ever show.
    @Test
    func thePayloadCarriesEveryFieldTheBoxWrapperReads() throws {
        let emitted = try Self.payloadKeys()
        // Sourced from measure-qwen-mtp-job.sh: check_row_accounting's ledger
        // projection, run_phase_once's expect_side / accept jq, the timing
        // readout, run_exactness_probe's jq, and parity_all_ok_from_reports.
        let wrapperFields = [
            "track_id",
            "all_tokens_matched",
            "official_score_produced",
            "parent_measured_seconds_per_token",
            "uses_native_mtp_head",
            "mtp_depth",
            "decode_token_count",
            "emitted_token_total",
            "declared_rows_total",
            "reference_checked_row_total",
            "accepted_draft_total",
            "rejected_draft_total",
            "target_tail_total",
            "round_count",
            "seed_token_count",
            "target_cache_offset_final",
            // check_stall_guardrail, which FAILS CLOSED without one of the two
            // routes and will not fall back to whole-window max/p50.
            "block_request_seconds",
            "first_block_seconds",
            "max_block_request_seconds_after_first",
            "p50_block_request_seconds_after_first",
        ]
        for field in wrapperFields {
            #expect(
                emitted.contains(field),
                """
                measure-qwen-mtp-job.sh reads `.\(field)` out of a phase report \
                and the payload does not emit it. That wrapper is box-owned and \
                is not editable from this branch; the payload has to satisfy it.
                """
            )
        }
    }

    /// THE SERIAL CONTROL IS DEPTH 0, AND THE LEDGER CLOSES THERE.
    ///
    /// Box 3 measured the old depth-1 "control" running 512 tokens in 302 rounds
    /// at a 0.699 accept rate — an already-accelerated baseline — which is why
    /// the paired ratio came out at 0.875x for a method the authors measured at
    /// 1.34x @ 128 against true serial. Depth 0 emits exactly one token per
    /// round, so rounds == tokens and every declared row is the round's own
    /// target tail row.
    @Test
    func theSerialControlIsDepthZeroAndItsLedgerCloses() throws {
        #expect(MLXFastConstants.qwenMTPSerialControlDepth == 0)
        // `rows_per_round(0) == 1` — the box wrapper's shell arithmetic
        // (`depth + 1`) needs no special case for the control.
        #expect(QwenMTPRowAccounting.rowsPerRound(depth: 0) == 1)

        let tokens = 512
        #expect(
            QwenMTPRowAccounting.closes(
                emittedTokenTotal: tokens,
                configuredTokenTotal: tokens,
                declaredRowTotal: tokens,
                referenceCheckedRowTotal: tokens,
                acceptedDraftTotal: 0,
                rejectedDraftTotal: 0,
                targetTailTotal: tokens,
                roundCount: tokens,
                depth: 0,
                seedTokenCount: 512,
                targetCacheOffsetFinal: 512 + tokens
            ))
        // A "serial" run that drafted anything must NOT close at depth 0: that is
        // the one-deep speculative decoder this depth exists to exclude.
        #expect(
            !QwenMTPRowAccounting.closes(
                emittedTokenTotal: tokens,
                configuredTokenTotal: tokens,
                declaredRowTotal: tokens + 40,
                referenceCheckedRowTotal: tokens + 40,
                acceptedDraftTotal: 40,
                rejectedDraftTotal: 0,
                targetTailTotal: tokens,
                roundCount: 302,
                depth: 0,
                seedTokenCount: 512,
                targetCacheOffsetFinal: 512 + tokens
            ),
            """
            a depth-0 ledger with accepted drafts still closes. Depth 0 must \
            draft nothing; a control that speculates is an accelerated \
            denominator and every score divided by it is understated.
            """
        )
    }

    /// The control depth is one constant, shared by the payload, the local
    /// runner and the box wrapper's QMTP_SERIAL_DEPTH.
    @Test
    func theControlDepthIsOneConstantAcrossEveryConsumer() throws {
        let cli = try S.text(Self.cliPath)
        #expect(
            cli.contains(
                "\"serial_control_depth\": MLXFastConstants.qwenMTPSerialControlDepth"),
            "the payload hard-codes a control depth instead of sharing the constant")
        let runner = try S.text("benchmark-qwen-mtp.sh")
        #expect(
            runner.contains("--mtp-depth 0 > \"${serial_report}\""),
            """
            the local runner's baseline leg is not at depth 0. Depth 1 still \
            drafts and accepts, so it is not a serial control.
            """
        )
        #expect(runner.contains(".is_serial_control == true"))
    }

    /// THE STALL GUARDRAIL CONTRACT. The wrapper fails CLOSED unless a timed
    /// report offers one of two routes, and the payload must satisfy at least
    /// one of them or every ranked run rejects on the report contract.
    @Test
    func theTimedPayloadSatisfiesBothStallGuardrailRoutes() throws {
        let cli = try S.text(Self.cliPath)
        // Route 1: the full per-block array (preferred -- the wrapper slices and
        // reduces it itself, so the guard's arithmetic is not asserted by the
        // measured side).
        #expect(cli.contains(#"payload["block_request_seconds"] = report.roundRequestSeconds"#))
        // Route 2: the after-first trio, for a window the array route declines.
        #expect(cli.contains(#"payload["first_block_seconds"]"#))
        #expect(cli.contains(#"payload["max_block_request_seconds_after_first"]"#))
        #expect(cli.contains(#"payload["p50_block_request_seconds_after_first"]"#))
        // Whole-window fields survive for audit; the guard no longer reads them.
        #expect(cli.contains(#"payload["max_block_request_seconds"]"#))
        #expect(cli.contains(#"payload["p50_block_request_seconds"]"#))
    }

    /// The array and the trio must agree BIT FOR BIT, because the wrapper picks
    /// a route by array length and a report offering two different answers would
    /// pass or fail depending on which branch it happened to take.
    ///
    /// The p50 rule is the wrapper's own: `sorted[floor((n - 1) / 2)]`, the LOWER
    /// median. Swift's natural `sorted[count / 2]` is the UPPER median and
    /// differs on every even-length window -- which a 512-token run very often
    /// is.
    @Test
    func theArrayAndTheTrioAgreeIncludingOnEvenLengthWindows() throws {
        // Even post-first population (5 rounds -> 4 after the first), where the
        // upper and lower medians differ.
        let evenReport = Self.timedReport(blocks: [0.26, 0.05, 0.01, 0.04, 0.02])
        #expect(evenReport.firstBlockSeconds == 0.26)
        #expect(evenReport.maxRoundRequestSecondsAfterFirst == 0.05)
        // sorted post-first = [0.01, 0.02, 0.04, 0.05]; floor((4-1)/2) = 1.
        #expect(evenReport.p50RoundRequestSecondsAfterFirst == 0.02)

        // Odd post-first population.
        let oddReport = Self.timedReport(blocks: [0.26, 0.05, 0.01, 0.04])
        #expect(oddReport.p50RoundRequestSecondsAfterFirst == 0.04)

        // The whole-window p50 uses the SAME rule, so the payload carries one
        // definition of "p50" rather than two.
        #expect(
            QwenMTPReport.lowerMedian([0.26, 0.05, 0.01, 0.04]) == 0.04,
            "the whole-window p50 rule drifted from the after-first rule")

        // Degenerate cases must not trap: the wrapper treats a non-positive p50
        // as "no usable post-first population" and skips the ratio.
        #expect(Self.timedReport(blocks: [0.26]).p50RoundRequestSecondsAfterFirst == 0)
        #expect(Self.timedReport(blocks: []).firstBlockSeconds == 0)
    }

    /// The array is a per-ROUND journal, so its length is the round count. A
    /// short array would silently shrink the population the guard reduces over.
    @Test
    func theBlockArrayLengthIsTheRoundCount() throws {
        let report = Self.timedReport(blocks: [0.26, 0.05, 0.01, 0.04, 0.02])
        #expect(report.roundRequestSeconds.count == report.roundCount)
        #expect(report.firstBlockSeconds == report.roundRequestSeconds[0])
    }

    /// The first block is EXCLUDED FROM THE RATIO, NOT FROM THE REPORT. Box 3
    /// measured a depth-1 phase at max_block 0.261 s against p50 0.0389 s --
    /// 6.71x, tripping a factor-4 guard -- and the shape was a one-time
    /// post-prefill warmup. Prove the exclusion is what rescues that run, and
    /// that a genuine stall in the post-first population is still caught.
    @Test
    func excludingTheFirstBlockRescuesAWarmupAndStillCatchesAStall() throws {
        func ratio(_ blocks: [Double]) -> Double {
            let report = Self.timedReport(blocks: blocks)
            let p50 = report.p50RoundRequestSecondsAfterFirst
            guard p50 > 0 else { return 0 }
            return report.maxRoundRequestSecondsAfterFirst / p50
        }
        // The measured box-3 shape: one 0.261 s first block over a ~0.039 s body.
        let warmup = [0.261] + Array(repeating: 0.0389, count: 300)
        #expect(
            ratio(warmup) < 4,
            """
            the measured box-3 warmup shape still trips a factor-4 guard after \
            the first block is excluded.
            """
        )
        // A real stall mid-window is still a rejection.
        var stalled = Array(repeating: 0.0389, count: 300)
        stalled[150] = 0.5
        #expect(ratio([0.261] + stalled) > 4)
    }

    /// A minimal timed report carrying a given per-round journal.
    static func timedReport(blocks: [Double]) -> QwenMTPReport {
        let sorted = blocks.sorted()
        return QwenMTPReport(
            verb: "mtp-timed",
            depth: 2,
            seedTokenCount: 512,
            decodeTokenCount: blocks.count,
            emittedTokenTotal: blocks.count,
            allTokensMatched: true,
            firstDivergenceIndex: nil,
            firstDivergenceReferenceMargin: nil,
            roundCount: blocks.count,
            acceptedDraftTotal: 0,
            rejectedDraftTotal: 0,
            targetTailTotal: blocks.count,
            declaredRowTotal: blocks.count,
            referenceCheckedRowTotal: blocks.count,
            rejectedRowsReferenceChecked: 0,
            verifyBlockReplayedRoundCount: 0,
            residualDivergenceCount: 0,
            maxRejectedTailLogitDelta: 0,
            targetCacheOffsetFinal: 512 + blocks.count,
            decodeSeconds: blocks.reduce(0, +),
            roundRequestSeconds: blocks,
            maxRoundRequestSeconds: sorted.last ?? 0,
            p50RoundRequestSeconds: QwenMTPReport.lowerMedian(blocks)
        )
    }

    /// `rows_per_round` is duplicated across a Swift constant and a shell
    /// function in a file this repository does not own. Pin the value so the two
    /// cannot drift silently.
    @Test
    func rowsPerRoundMatchesTheWrapperEquation() throws {
        // The wrapper's `rows_per_round() { echo $((depth + 1)); }`: the head
        // proposes D draft rows and the target contributes one tail row.
        for depth in 1 ... 8 {
            #expect(
                QwenMTPRowAccounting.rowsPerRound(depth: depth) == depth + 1,
                """
                rows_per_round(\(depth)) changed. measure-qwen-mtp-job.sh \
                computes depth + 1 in shell and its comment names this constant \
                as the ONE place to change; both have to move together.
                """
            )
        }
    }

    /// The L3 equations, exercised on a synthetic ledger rather than trusted to
    /// read correctly. Depth 2 over 4 rounds: 3 full accepts and 1 round that
    /// accepted one draft and rejected one.
    @Test
    func theRowAccountingEquationsCloseOnAnHonestLedger() throws {
        let rounds = 4
        let depth = 2
        let accepted = 3 * 2 + 1        // 7
        let rejected = 1                // the one rejected draft
        let tail = rounds               // one target row per round
        let declared = accepted + rejected + tail   // 12 == (depth + 1) * rounds
        let emitted = rounds + accepted             // 11
        #expect(
            QwenMTPRowAccounting.closes(
                emittedTokenTotal: emitted,
                configuredTokenTotal: emitted,
                declaredRowTotal: declared,
                referenceCheckedRowTotal: declared,
                acceptedDraftTotal: accepted,
                rejectedDraftTotal: rejected,
                targetTailTotal: tail,
                roundCount: rounds,
                depth: depth,
                seedTokenCount: 512,
                targetCacheOffsetFinal: 512 + emitted
            ))
    }

    /// The equality on `reference_checked_row_total` is the accounting hole the
    /// rejected-tail replay closes, and the wrapper's header forbids relaxing it
    /// back to `>= emitted`. Prove a ledger that leaves the tail unpriced FAILS.
    @Test
    func aLedgerThatLeavesTheRejectedTailUnpricedIsRejected() throws {
        let rounds = 4
        let depth = 2
        let accepted = 7
        let rejected = 1
        let tail = rounds
        let declared = accepted + rejected + tail
        let emitted = rounds + accepted
        #expect(
            !QwenMTPRowAccounting.closes(
                emittedTokenTotal: emitted,
                configuredTokenTotal: emitted,
                declaredRowTotal: declared,
                // Everything but the rejected row was checked. Under the OLD
                // `>= emitted` invariant this passed.
                referenceCheckedRowTotal: declared - rejected,
                acceptedDraftTotal: accepted,
                rejectedDraftTotal: rejected,
                targetTailTotal: tail,
                roundCount: rounds,
                depth: depth,
                seedTokenCount: 512,
                targetCacheOffsetFinal: 512 + emitted
            ),
            """
            a ledger whose rejected tail was never reference-checked still \
            closes. That is the exact hole measure-qwen-mtp-job.sh's EQUALITY on \
            reference_checked_row_total exists to close, and its header says the \
            fix for a rejection is to measure a tree that carries the readouts, \
            never to weaken the comparison.
            """
        )
    }

    /// A ledger whose declared rows exceed what the TRUSTED MAXIMUM could have
    /// produced is rejected: a worker cannot declare work the round shape does
    /// not allow.
    ///
    /// RE-AIMED 2026-08-14. This used to bound rows by the REQUESTED depth,
    /// which stopped being the right question when the candidate took ownership
    /// of its own per-round draft count: a round that drafts wider than the
    /// parent's tail-narrowed offer is legal now, and rejecting it would reject
    /// a legal adaptive run. What is still impossible is declaring more rows
    /// than `qwenMTPMaxDraftDepth` drafts plus a tail row could produce, which
    /// is what this asserts. The example below is 4 rounds at the trusted
    /// maximum: 4 * (8 + 1) = 36 rows is the ceiling, so 37 cannot close.
    @Test
    func aLedgerDeclaringMoreRowsThanTheTrustedMaximumAllowsIsRejected() throws {
        #expect(
            !QwenMTPRowAccounting.closes(
                emittedTokenTotal: 11,
                configuredTokenTotal: 11,
                declaredRowTotal: 37,
                referenceCheckedRowTotal: 37,
                acceptedDraftTotal: 32,
                rejectedDraftTotal: 1,
                targetTailTotal: 4,
                roundCount: 4,
                depth: 2,
                seedTokenCount: 512,
                targetCacheOffsetFinal: 523
            ))
    }

    /// `parity_all_ok` must be an AND, not a copy of the exactness verdict: a run
    /// whose ledger did not close has not proven its tokens were produced by the
    /// work it declared, whatever the tokens happened to be.
    @Test
    func parityAllOKRequiresBothExactnessAndACloseLedger() throws {
        func report(
            matched: Bool,
            referenceChecked: Int,
            declared: Int
        ) -> QwenMTPReport {
            QwenMTPReport(
                verb: "mtp-timed",
                depth: 2,
                seedTokenCount: 512,
                decodeTokenCount: 11,
                emittedTokenTotal: 11,
                allTokensMatched: matched,
                firstDivergenceIndex: matched ? nil : 3,
                firstDivergenceReferenceMargin: nil,
                roundCount: 4,
                acceptedDraftTotal: 7,
                rejectedDraftTotal: 1,
                targetTailTotal: 4,
                declaredRowTotal: declared,
                referenceCheckedRowTotal: referenceChecked,
                rejectedRowsReferenceChecked: 1,
                verifyBlockReplayedRoundCount: 0,
                residualDivergenceCount: 0,
                maxRejectedTailLogitDelta: 0,
                targetCacheOffsetFinal: 523,
                decodeSeconds: 1,
                maxRoundRequestSeconds: 0.1,
                p50RoundRequestSeconds: 0.05
            )
        }
        #expect(report(matched: true, referenceChecked: 12, declared: 12)
            .parityAllOK)
        #expect(!report(matched: false, referenceChecked: 12, declared: 12)
            .parityAllOK)
        #expect(!report(matched: true, referenceChecked: 11, declared: 12)
            .parityAllOK)
    }

    /// The two boolean spellings of "the pinned native MTP head drafted this
    /// run" must stay a single predicate. Two consumers pinned different names
    /// before the payload existed; the payload emits both, so nothing may make
    /// them diverge.
    @Test
    func theTwoHeadBooleansAreOnePredicate() throws {
        let cli = try S.text(Self.cliPath)
        #expect(
            cli.contains(
                #""uses_native_mtp_head": report.usesNativeMTPHead,"#))
        #expect(
            cli.contains(
                #""uses_pinned_mtp_head": report.usesNativeMTPHead,"#),
            """
            uses_pinned_mtp_head is no longer derived from the same expression \
            as uses_native_mtp_head. The box wrapper reads one and the ranked \
            workflow reads the other; if they can disagree, one consumer can \
            accept a run the other would reject.
            """
        )
        // ... and the predicate itself, which was REDEFINED on 2026-08-14 from
        // "the parent asked for a speculative depth" (depth > 0) to "the head
        // ACTUALLY DRAFTED" (a draft was proposed). The old form was only ever
        // a proxy, and once the candidate owns its own per-round schedule the
        // proxy can be simply false: a run offered depth 8 may legally draft
        // nothing. Asserted on the behaviour, not on the source text.
        //
        // The serial control still reports false, because depth 0 cannot draft
        // -- which is what the local runner's baseline-leg jq reads.
        #expect(!Self.headBooleanReport(depth: 0, drafted: 0).usesNativeMTPHead)
        #expect(Self.headBooleanReport(depth: 0, drafted: 0).isSerialControl)
        // A speculative offer that DID draft reports true at every depth,
        // including 1: the head really does draft there (box 3 measured a 0.699
        // accept rate over 302 rounds at 512 tokens), and calling that "not
        // using the head" is what let a one-deep speculative decoder serve as
        // the serial denominator.
        #expect(Self.headBooleanReport(depth: 1, drafted: 7).usesNativeMTPHead)
        #expect(!Self.headBooleanReport(depth: 1, drafted: 7).isSerialControl)
        #expect(Self.headBooleanReport(depth: 8, drafted: 3).usesNativeMTPHead)
        #expect(!Self.headBooleanReport(depth: 8, drafted: 3).isSerialControl)
        // THE NEW CASE, and the one the redefinition exists for: a legal
        // adaptive candidate that was offered a width and declined it every
        // round. It is not the serial control -- the offer was speculative --
        // and it did not use the head. Both statements are true at once, and
        // neither is a rejection: under serial-anchored scoring this candidate
        // simply scores 1.0.
        #expect(!Self.headBooleanReport(depth: 8, drafted: 0).usesNativeMTPHead)
        #expect(!Self.headBooleanReport(depth: 8, drafted: 0).isSerialControl)
    }

    /// A minimal report used only to exercise the head predicate. `drafted` is
    /// how many drafts the run proposed in total, which is what the predicate
    /// reads since 2026-08-14.
    static func headBooleanReport(depth: Int, drafted: Int = 0) -> QwenMTPReport {
        QwenMTPReport(
            verb: "mtp-timed",
            depth: depth,
            seedTokenCount: 512,
            decodeTokenCount: 1,
            emittedTokenTotal: 1,
            allTokensMatched: true,
            firstDivergenceIndex: nil,
            firstDivergenceReferenceMargin: nil,
            roundCount: 1,
            acceptedDraftTotal: drafted,
            rejectedDraftTotal: 0,
            targetTailTotal: 1,
            declaredRowTotal: 1,
            referenceCheckedRowTotal: 1,
            rejectedRowsReferenceChecked: 0,
            verifyBlockReplayedRoundCount: 0,
            residualDivergenceCount: 0,
            maxRejectedTailLogitDelta: 0,
            targetCacheOffsetFinal: 513,
            decodeSeconds: 1,
            maxRoundRequestSeconds: 1,
            p50RoundRequestSeconds: 1
        )
    }

    /// The depth flag has one canonical spelling and one accepted alias, and the
    /// ambiguous case is refused rather than resolved.
    @Test
    func theDepthFlagHasACanonicalSpellingAndARefusedAmbiguity() throws {
        let cli = try S.text(Self.cliPath)
        #expect(cli.contains(#""--mtp-depth""#))
        #expect(cli.contains(#""--depth""#))
        #expect(
            cli.contains("--depth is an alias of --mtp-depth, not a second knob"),
            """
            the CLI no longer refuses a --mtp-depth/--depth disagreement. Silently \
            preferring one would let a caller believe it measured a depth the run \
            did not use.
            """
        )
        // Both in-repo callers use the canonical spelling; the box wrapper's is
        // canonical by construction.
        for path in [Self.workflowPath, "benchmark-qwen-mtp.sh"] {
            let body = try S.text(path)
            #expect(
                body.contains("--mtp-depth"),
                "\(path) does not pass --mtp-depth")
            #expect(
                !body.contains("--depth "),
                """
                \(path) still passes the alias --depth. The canonical spelling \
                is --mtp-depth (the box wrapper's own CLI takes only that), and \
                a caller left on the alias will keep working until the alias is \
                removed and then fail on a ranked run.
                """
            )
        }
    }

    /// `MLXFAST_QWEN_MTP_TARGET_DIR` must never become a LOAD input.
    ///
    /// The box wrapper exports it around every verb invocation and nothing reads
    /// it, which reads like a bug and is not one: it names the RAW pinned HF
    /// snapshot, while `--weights` names the TRANSFORMED tree derived from it.
    /// Wiring the env var into the weights resolution would silently load the
    /// 16-file raw snapshot instead of the 1,847-tensor transformed tree — a
    /// different model, scored as the candidate's. It is provenance, it is
    /// logged, and this test keeps it that way.
    @Test
    func thePinnedReferenceDirectoryIsProvenanceNotAWeightsSource() throws {
        let cli = try S.text(Self.cliPath)
        let resolver = try #require(
            cli.range(of: "private static func qwenMTPWeightsPath("))
        let end = try #require(
            cli.range(
                of: "\n    }\n", range: resolver.upperBound ..< cli.endIndex))
        let body = String(cli[resolver.upperBound ..< end.lowerBound])
        #expect(
            !body.contains("MLXFAST_QWEN_MTP_TARGET_DIR"),
            """
            the MTP weights resolver now falls back to             MLXFAST_QWEN_MTP_TARGET_DIR. That variable is the RAW pinned             snapshot, not the transformed tree; using it as --weights loads a             different model and scores it as the candidate's.
            """
        )
        #expect(body.contains("MLXFAST_WEIGHTS_PATH"))
        // It IS read for provenance, so an audit can see which pinned reference
        // the run was staged from.
        #expect(cli.contains("pinned_reference="))
    }

    /// Both MTP verbs must be reachable from the CLI dispatch, because the ranked
    /// workflow and the local runner both assert their presence by grepping this
    /// exact shape.
    @Test
    func bothVerbsAreDeclaredInTheShapeTheCallersGrepFor() throws {
        let cli = try S.text(Self.cliPath)
        for verb in ["mtp-verify", "mtp-timed"] {
            #expect(
                cli.contains("case \"\(verb)\":"),
                """
                the CLI no longer declares `case "\(verb)":`. Both \
                benchmark-qwen-mtp.sh and the ranked workflow assert this literal \
                shape and abort with QWEN-MTP-PENDING-PHASE3 without it.
                """
            )
        }
    }
}

// MARK: - the head manifest fixture

@Suite
struct QwenMTPHeadManifestTests {
    private typealias S = DFlashGateTextSupport

    private static let manifestPath = "fixtures/qwen3_8_27b_mtp_head.sha256"

    struct Record {
        let sha256: String
        let bytes: Int
        let path: String
    }

    static func records() throws -> [Record] {
        try S.text(manifestPath).split(separator: "\n").compactMap { line in
            guard !line.hasPrefix("#"), !line.isEmpty else { return nil }
            let fields = line.split(separator: " ", omittingEmptySubsequences: true)
            guard fields.count == 3, let bytes = Int(fields[1]) else {
                return nil
            }
            return Record(
                sha256: String(fields[0]),
                bytes: bytes,
                path: String(fields[2])
            )
        }
    }

    /// The manifest is well-formed in the same format the target manifest uses:
    /// the workflow's `verify_cache` reads both with one `while read` loop.
    @Test
    func theHeadManifestIsWellFormed() throws {
        let records = try Self.records()
        #expect(records.count == 4, "expected 4 records, got \(records.count)")
        for record in records {
            #expect(
                record.sha256.count == 64
                    && record.sha256.allSatisfy {
                        $0.isHexDigit && !$0.isUppercase
                    },
                "\(record.path) has a malformed digest: \(record.sha256)"
            )
            #expect(record.bytes > 0, "\(record.path) has no bytes")
            #expect(
                !record.path.contains("/"),
                """
                \(record.path) is nested. `verify_cache` compares each manifest \
                entry against `basename` of a maxdepth-1 find, so a nested entry \
                would verify a file the inventory check then reports as \
                unexpected.
                """
            )
        }
        let paths = Set(records.map(\.path))
        #expect(paths.count == records.count, "duplicate manifest entries")
        // config.json and model.safetensors.index.json are required because
        // Qwen36MTPHeadAttachment.verifyHeadTree REFUSES a head directory that
        // does not carry them -- the index to assert 15 bare tensor names, the
        // config to assert the architecture family and the single MTP layer.
        // The tokenizer files the 3.6 head shipped are NOT required and are not
        // present: the head merges under the `mtp.` prefix into the target's
        // tree and carries no independent tokenizer.
        for required in [
            "config.json", "model.safetensors",
            "model.safetensors.index.json",
        ] {
            #expect(
                paths.contains(required),
                "the head manifest does not pin \(required)")
        }
        for absent in ["tokenizer.json", "tokenizer_config.json"] {
            #expect(
                !paths.contains(absent),
                "the head tree does not ship \(absent); pinning it would fail "
                    + "verify_cache's inventory half against the published tree")
        }
        // `.gitattributes` IS pinned, exactly as it is now pinned in the backbone
        // manifest (16 repo files, 16 records). The inventory half of
        // `verify_cache` rejects any file in the cache that the manifest does not
        // name, and the file is a genuine member of the pinned upstream revision
        // that a stock `snapshot_download` stages -- so omitting the record does
        // not keep a git plumbing file out of the pins, it just fails the run
        // (run 31665285024 died on the target side of exactly this). The record
        // set and the staged tree must agree file-for-file.
        #expect(paths.contains(".gitattributes"))
    }

    /// The whole-manifest shape pins must equal what the file actually carries.
    /// Per-file hashing cannot see a manifest that LOST records; it would verify
    /// fewer files and report success.
    ///
    /// FULLY RESTORED at the 2026-08-14 head publish, and REPINNED the same day
    /// when the head was republished at 26a328e0 carrying config.json and
    /// model.safetensors.index.json -- the two files verifyHeadTree refuses a
    /// head tree without. That took the published tree from 2 records /
    /// 849401866 bytes to 4 records / 849406438 bytes. Both halves of the
    /// assertion move together on purpose: the pins are checked against the
    /// workflow AND against the file, so a body edited without the pins (or the
    /// reverse) fails here rather than at `verify_cache` on the runner.
    @Test
    func theWorkflowShapePinsMatchTheCheckedInManifest() throws {
        let records = try Self.records()
        let environment = try S.jobEnvironment(
            try S.text(".github/workflows/qwen-mtp-ranked-benchmark.yml"))
        #expect(
            environment["MLXFAST_QWEN_MTP_HEAD_MANIFEST_PATH"]
                == Self.manifestPath)
        #expect(
            environment["MLXFAST_QWEN_MTP_HEAD_MANIFEST_RECORDS"] == "4",
            "the head record-count pin matches the published four-file tree"
        )
        #expect(
            environment["MLXFAST_QWEN_MTP_HEAD_MANIFEST_BYTES"] == "849406438",
            "the head byte-total pin matches the published tree"
        )
        // The published manifest is internally consistent.
        #expect(records.count == 4)
        #expect(records.reduce(0) { $0 + $1.bytes } == 849_406_438)
    }

    /// The manifest header must name the artifact it pins, at the revision the
    /// workflow pins. A manifest that pinned a different revision's bytes would
    /// verify a head nobody agreed to.
    ///
    /// FULLY RESTORED at the 2026-08-14 head publish: repository AND revision
    /// are real (EigenLabs/Qwen3.8-27B-MTP-bf16 @ 26a328e0…, the republish that
    /// added the two config files verifyHeadTree requires), the records describe
    /// the published four-file tree, and the 3.6 head's identity survives only
    /// as provenance prose in the manifest header.
    @Test
    func theHeadManifestNamesThePinnedRepositoryAndRevision() throws {
        let body = try S.text(Self.manifestPath)
        let environment = try S.jobEnvironment(
            try S.text(".github/workflows/qwen-mtp-ranked-benchmark.yml"))
        let repository = try #require(environment["MLXFAST_QWEN_MTP_HEAD_REPO"])
        let revision = try #require(
            environment["MLXFAST_QWEN_MTP_HEAD_REVISION"])
        #expect(repository == "EigenLabs/Qwen3.8-27B-MTP-bf16")
        #expect(revision == "26a328e070875b0314d652a039b6b59902690f03")
        // The manifest header names the same repository at the same published
        // revision, and keeps the LAST PINNED (3.6) head's identity as
        // provenance prose.
        #expect(body.contains(repository))
        #expect(body.contains(revision))
        #expect(body.contains("mlx-community/Qwen3.6-27B-MTP-4bit"))
        #expect(body.contains("83795d546e9d328160e593fb0bf10b2bf2fe637e"))
    }

    /// The tensor count the loader enforces and the count the payload reports are
    /// one constant.
    ///
    /// 15, MEASURED on the 3.8 head 2026-08-14, replacing the 31 inherited from
    /// the 3.6 head's index. The two are not in conflict: the 3.6 head was 4-bit
    /// group-64, so its 8 matrices appeared as weight/scales/biases triples
    /// (24) alongside 7 norms; the 3.8 head is bf16, so the same 8 matrices are
    /// 8 entries and 8 + 7 = 15. The contract fixture's mtp_head.tensor_count
    /// mirrors this.
    @Test
    func theHeadTensorCountIsOneConstant() throws {
        #expect(MLXFastConstants.qwenMTPHeadTensorCount == 15)
        let cli = try S.text("Sources/MLXFastCLI/main.swift")
        #expect(
            cli.contains("MLXFastConstants.qwenMTPHeadTensorCount"),
            """
            the payload no longer reports the shared head tensor count. A second \
            literal here would let the number the loader enforces and the number \
            the evidence reports drift apart.
            """
        )
    }
}
