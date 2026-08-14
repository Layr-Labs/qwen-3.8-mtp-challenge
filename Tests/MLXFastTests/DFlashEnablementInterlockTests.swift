import Foundation
import Testing

/// The DFlash enablement interlock answers two separate questions, and the
/// suite's job is to keep them separate:
///
///   may this job RUN?              `confirm_track_enabled`, matching
///                                  `track_id`, non-empty golden pins.
///   may it PUBLISH a ranked score? `official_scoring_enabled` AND
///                                  `reference_baseline.publication_allowed`.
///
/// Requiring the PUBLISH answer in order to RUN made the pipeline untestable
/// before go-live: the correctness gates and every fail-closed guard could only
/// be exercised by first throwing the switch they gate. A gates-only dispatch
/// (`run_benchmark=false`) produces no ranked score, so it is admitted against a
/// not-yet-enabled track; anything that can publish still requires both flags.
///
/// The risk this suite guards is the obvious one: that the split quietly became
/// a way to publish a score without the go-live flags. So it asserts the ranked
/// refusal survives, and asserts the check exists at the SCORING step too --
/// a guard 49 steps upstream of the thing it protects is a guard whose coverage
/// depends on step order.
@Suite("DFlash enablement interlock")
struct DFlashEnablementInterlockTests {
    private static let workflowPath = ".github/workflows/dflash-benchmark.yml"
    private static let fixturePath = "fixtures/laguna_xs_2_1_dflash_track.json"
    private static let enablementStep = "Enforce DFlash track enablement"
    private static let scoringStep = "Compute DFlash score and enforce floor"

    private static func workflow() throws -> String {
        try String(contentsOfFile: workflowPath, encoding: .utf8)
    }

    /// A step's `run:` body with comment-only lines removed, so no assertion
    /// here can be satisfied by the prose explaining it.
    private static func executableBody(_ workflow: String, _ step: String) throws -> String {
        let marker = "- name: \(step)\n"
        let start = try #require(
            workflow.range(of: marker),
            "step '\(step)' is missing from \(workflowPath)"
        )
        let rest = workflow[start.upperBound...]
        let end = rest.range(of: "\n      - name: ")
        let body = end.map { String(rest[..<$0.lowerBound]) } ?? String(rest)
        return body
            .split(separator: "\n", omittingEmptySubsequences: false)
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("#") }
            .joined(separator: "\n")
    }

    // MARK: - the RUN question stays mandatory for both kinds of dispatch

    @Test
    func confirmTrackEnabledIsRequiredEvenForAGatesOnlyDryRun() throws {
        let step = try Self.executableBody(try Self.workflow(), Self.enablementStep)

        // The interlock check must NOT be conditioned on run_benchmark: a dry
        // run is still a run on the operator's box.
        let check = try #require(
            step.range(of: #"if [[ "${CONFIRM_TRACK_ENABLED}" != "1" ]]; then"#),
            "the confirm_track_enabled interlock vanished"
        )
        let refusal = String(step[check.lowerBound...])
        #expect(
            refusal.contains("exit 1"),
            "the confirm_track_enabled interlock warns instead of refusing"
        )
        // The RUN gate must not have been made ranked-only by the split.
        let beforeCheck = String(step[..<check.lowerBound])
        let lastRunBenchmarkBranch = beforeCheck.range(
            of: #"if [[ "${RUN_BENCHMARK}" == "1" ]]; then"#,
            options: .backwards
        )
        if let branch = lastRunBenchmarkBranch {
            let between = String(beforeCheck[branch.upperBound...])
            #expect(
                between.contains("fi"),
                """
                the confirm_track_enabled interlock is inside an unclosed \
                RUN_BENCHMARK branch, so a gates-only dry run skips it.
                """
            )
        }
    }

    @Test
    func goldenPinsAreRequiredEvenForAGatesOnlyDryRun() throws {
        let step = try Self.executableBody(try Self.workflow(), Self.enablementStep)
        for pin in [
            "MLXFAST_DFLASH_CORRECTNESS_GOLDEN_SHA256",
            "MLXFAST_DFLASH_CORRECTNESS_GOLDEN_BYTES",
            "MLXFAST_DFLASH_BENCH_GOLDEN_SHA256",
            "MLXFAST_DFLASH_BENCH_GOLDEN_BYTES",
        ] {
            #expect(step.contains(pin), "golden pin \(pin) is no longer required")
        }
    }

    // MARK: - the PUBLISH question still refuses a ranked run

    @Test
    func aRankedRunAgainstAnUnenabledTrackIsStillRefused() throws {
        let step = try Self.executableBody(try Self.workflow(), Self.enablementStep)

        #expect(step.contains("official_scoring_enabled"))
        #expect(step.contains("publication_allowed"))

        // The refusal must be reachable when RUN_BENCHMARK=1 and must exit.
        let unenabled = try #require(
            step.range(
                of: #"if [[ "${scoring_enabled}" != "true" || "${publication_allowed}" != "true" ]]; then"#
            ),
            "the enablement comparison vanished or changed shape"
        )
        let branch = String(step[unenabled.upperBound...])
        let rankedGuard = try #require(
            branch.range(of: #"if [[ "${RUN_BENCHMARK}" == "1" ]]; then"#),
            """
            an unenabled track no longer distinguishes a ranked dispatch from a \
            dry run, so either every run is refused (the old chicken-and-egg) or \
            every run is admitted (a ranked run against an unenabled track).
            """
        )
        let rankedBranch = String(branch[rankedGuard.upperBound...])
        let exitIndex = try #require(
            rankedBranch.range(of: "exit 1"),
            "a ranked dispatch against an unenabled track no longer exits"
        )
        // The exit must come before the branch closes -- i.e. it belongs to the
        // ranked arm, not to something after it.
        let fiIndex = try #require(rankedBranch.range(of: "\n            fi"))
        #expect(
            exitIndex.lowerBound < fiIndex.lowerBound,
            "the ranked refusal's exit 1 is outside the ranked branch"
        )
    }

    /// The dry-run arm must announce itself and must not be silent: an operator
    /// reading the log has to be able to tell a validated dry run from a ranked
    /// one that quietly published nothing.
    @Test
    func theDryRunArmAnnouncesThatNoScoreIsProduced() throws {
        let step = try Self.executableBody(try Self.workflow(), Self.enablementStep)
        #expect(step.contains("::notice::"))
        #expect(
            step.contains("MLXFAST_DFLASH_DRY_RUN_UNENABLED=1"),
            """
            the dry-run arm does not record that it was admitted unenabled, so \
            the scoring step cannot refuse it independently.
            """
        )
        #expect(step.contains("GATES-ONLY"))
    }

    // MARK: - the second, order-independent check

    @Test
    func theScoringStepIndependentlyRefusesAnUnenabledTrack() throws {
        let step = try Self.executableBody(try Self.workflow(), Self.scoringStep)

        #expect(
            step.contains("official_scoring_enabled"),
            """
            \(Self.scoringStep) does not check official_scoring_enabled itself. \
            The only enablement check would then be 49 steps upstream, so its \
            coverage depends on step order.
            """
        )
        #expect(step.contains("publication_allowed"))
        #expect(
            step.contains("MLXFAST_DFLASH_DRY_RUN_UNENABLED"),
            """
            \(Self.scoringStep) does not refuse a dispatch that was admitted as \
            unenabled, so a dry run taught to reach scoring would publish.
            """
        )
        // Both refusals must exit, not warn.
        let refusals = step
            .split(separator: "\n", omittingEmptySubsequences: false)
            .filter { $0.contains("::error::") }
        #expect(
            refusals.count >= 2,
            "expected both the flag check and the dry-run check to error; found \(refusals.count)"
        )
    }

    // MARK: - behavioural: execute the real step under every input combination

    /// Everything above reads the step's text. This one RUNS it.
    ///
    /// The distinction is not academic: the redactor gap in Amendment 23 would
    /// have passed a text-matching test, because the missing arms were absent
    /// rather than misspelled. A split interlock is exactly the kind of change
    /// where the text can look right and the boolean can be inverted, so the
    /// truth table is asserted by execution.
    @Test
    func theInterlockAdmitsTheDryRunAndOnlyTheDryRun() throws {
        // (label, trackEnabled, rankedRun, confirmed, expectAdmit)
        let cases:
            [(String, Bool, Bool, Bool, Bool)] = [
                // The attack the split could have introduced.
                ("unenabled + ranked + confirmed", false, true, true, false),
                // The point of the split.
                ("unenabled + dry run + confirmed", false, false, true, true),
                // The RUN question must bind BOTH kinds of dispatch.
                ("unenabled + dry run + unconfirmed", false, false, false, false),
                ("enabled + ranked + unconfirmed", true, true, false, false),
                // A properly enabled track is unaffected.
                ("enabled + ranked + confirmed", true, true, true, true),
                ("enabled + dry run + confirmed", true, false, true, true),
            ]

        for (label, trackEnabled, rankedRun, confirmed, expectAdmit) in cases {
            let admitted = try runEnablementStep(
                trackEnabled: trackEnabled,
                rankedRun: rankedRun,
                confirmed: confirmed
            )
            #expect(
                admitted == expectAdmit,
                """
                enablement step \(admitted ? "ADMITTED" : "REFUSED") '\(label)', \
                expected \(expectAdmit ? "ADMIT" : "REFUSE"). The interlock's \
                truth table changed: a ranked dispatch against a track whose \
                official_scoring_enabled / publication_allowed are false must be \
                refused, a gates-only dispatch must be admitted, and neither may \
                run without confirm_track_enabled.
                """
            )
        }
    }

    /// Extracts the step's real `run:` body and executes it against a synthetic
    /// contract, returning whether it admitted the dispatch.
    private func runEnablementStep(
        trackEnabled: Bool,
        rankedRun: Bool,
        confirmed: Bool
    ) throws -> Bool {
        let workspace = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("dflash-interlock-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: workspace,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: workspace) }

        let contract = workspace.appendingPathComponent("contract.json")
        try """
        {"track_id":"laguna-xs-2.1-dflash-v1",
         "official_scoring_enabled":\(trackEnabled),
         "reference_baseline":{"publication_allowed":\(trackEnabled)}}
        """.write(to: contract, atomically: true, encoding: .utf8)

        let body = try Self.rawBody(try Self.workflow(), Self.enablementStep)
        let githubEnv = workspace.appendingPathComponent("github_env")
        FileManager.default.createFile(atPath: githubEnv.path, contents: nil)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["-c", body]
        process.currentDirectoryURL = workspace
        process.environment = [
            "PATH": ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin",
            "MLXFAST_DFLASH_CONTRACT_PATH": contract.path,
            "MLXFAST_DFLASH_TRACK_ID": "laguna-xs-2.1-dflash-v1",
            // Non-empty pins: this test is about the enablement booleans, and
            // goldenPinsAreRequiredEvenForAGatesOnlyDryRun covers the pins.
            "MLXFAST_DFLASH_CORRECTNESS_GOLDEN_SHA256": "aa",
            "MLXFAST_DFLASH_CORRECTNESS_GOLDEN_BYTES": "1",
            "MLXFAST_DFLASH_BENCH_GOLDEN_SHA256": "bb",
            "MLXFAST_DFLASH_BENCH_GOLDEN_BYTES": "2",
            "CONFIRM_TRACK_ENABLED": confirmed ? "1" : "0",
            "RUN_BENCHMARK": rankedRun ? "1" : "0",
            "GITHUB_ENV": githubEnv.path,
        ]
        let sink = Pipe()
        process.standardOutput = sink
        process.standardError = sink
        try process.run()
        _ = sink.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()

        let admitted = process.terminationStatus == 0
        // An admitted dry run against an unenabled track must hand the marker to
        // the scoring step; without it the second guard cannot fire.
        if admitted && !trackEnabled {
            let recorded =
                (try? String(contentsOf: githubEnv, encoding: .utf8)) ?? ""
            #expect(
                recorded.contains("MLXFAST_DFLASH_DRY_RUN_UNENABLED=1"),
                """
                the step admitted an unenabled dispatch without recording \
                MLXFAST_DFLASH_DRY_RUN_UNENABLED, so the scoring step's \
                independent refusal has nothing to key on.
                """
            )
        }
        return admitted
    }

    /// The step's `run:` body verbatim, comments included -- this one is
    /// executed, so stripping comments would change what runs.
    private static func rawBody(_ workflow: String, _ step: String) throws -> String {
        let marker = "- name: \(step)\n"
        let start = try #require(workflow.range(of: marker))
        let rest = workflow[start.upperBound...]
        let end = rest.range(of: "\n      - name: ")
        let block = end.map { String(rest[..<$0.lowerBound]) } ?? String(rest)
        // Take the `run: |` scalar and strip its YAML block indentation.
        let runMarker = try #require(
            block.range(of: "run: |\n"),
            "step '\(step)' has no literal run: block"
        )
        let scalar = String(block[runMarker.upperBound...])
        let lines = scalar.split(separator: "\n", omittingEmptySubsequences: false)
        let indent =
            lines
            .first { !$0.trimmingCharacters(in: .whitespaces).isEmpty }?
            .prefix { $0 == " " }
            .count ?? 0

        // A YAML block scalar ends at the first non-blank line indented LESS
        // than the block. Without this the extraction ran past the step into
        // the comment introducing the NEXT step, and a blind dedent turned
        // "      # DFlash-specific host contract..." into "lash-specific host
        // contract..." -- a line bash then tried to execute. That produced a
        // non-zero exit for every case and read exactly like the interlock
        // refusing everything.
        var body: [String] = []
        for line in lines {
            let blank = line.trimmingCharacters(in: .whitespaces).isEmpty
            if !blank && line.prefix { $0 == " " }.count < indent { break }
            body.append(line.count >= indent ? String(line.dropFirst(indent)) : String(line))
        }
        return body.joined(separator: "\n")
    }

    // MARK: - behavioural: the anti-lottery floor, executed

    /// A pool the participant could memorise must not be enough to RANK, but
    /// must still be enough to validate the pipeline. That asymmetry is the
    /// anti-lottery floor.
    ///
    /// Why it matters: with a single timed target a failed ranked run is free to
    /// retry against a prompt it has already seen, which turns every output-side
    /// gate into submit-until-green and lets a drafter-confidence threshold be
    /// tuned across attempts. The fixture's timed_prompt_pool_note has always
    /// said 8 DISTINCT targets.
    ///
    /// This test RUNS the step. The version it replaces read the step's text,
    /// and that is exactly why it is gone. It extracted everything from the
    /// floor's comparison to the END of the step and asserted the slice
    /// contained "exit 1" -- which the per-entry completeness check further down
    /// satisfies all by itself. Replacing the floor's own `exit 1` with a
    /// `::warning::`, so that a one-entry pool RANKS, left all four of its
    /// assertions passing and the whole 582-test suite green. Prose cannot
    /// satisfy an exit status.
    @Test
    func aRankedRunRequiresEightDistinctTimedTargets() throws {
        // (label, contract, expect the step to admit the dispatch)
        let cases: [(String, String, Bool)] = [
            ("a one-entry pool", DFlashWorkflowStep.distinctPool(1), false),
            ("seven distinct targets", DFlashWorkflowStep.distinctPool(7), false),
            // F3: pool LENGTH 8, anti-lottery value zero. Eight dispatches of
            // dflash-provision-goldens.yml against one bench_seed_r2_path
            // produce exactly this, so it is the operator-error case, not just
            // the adversarial one.
            (
                "today's single golden listed eight times",
                DFlashWorkflowStep.clonedPool(8), false
            ),
            // The same content re-uploaded under eight different keys. Distinct
            // r2_path, one distinct digest: still one prompt.
            (
                "eight keys, one digest",
                DFlashWorkflowStep.pool(
                    (0 ..< 8).map {
                        #"{"r2_path":"correctness_prompts/laguna-xs-2.1-dflash/k\#($0).json","#
                            + #""sha256":"5b9e2c4a09904391213fd2d80f4b5d1ccde39a163b88de22990ec34f3f781ad8","#
                            + #""bytes":185433}"#
                    }
                ), false
            ),
            // One key listed eight times with eight digests: not a wider pool,
            // and it biases the sampling toward whichever object that key holds.
            (
                "one key, eight digests",
                DFlashWorkflowStep.pool(
                    (0 ..< 8).map {
                        #"{"r2_path":"correctness_prompts/laguna-xs-2.1-dflash/same.json","#
                            + #""sha256":"\#(String(repeating: "\($0)", count: 64))","#
                            + #""bytes":\#(1000 + $0)}"#
                    }
                ), false
            ),
            // F4: eight distinct targets, one of them malformed. Under the old
            // sample-then-validate order this ranked on 7 dispatches in 8.
            (
                "eight distinct targets, one with no sha256",
                DFlashWorkflowStep.pool(
                    (0 ..< 8).map {
                        $0 == 3
                            ? #"{"r2_path":"correctness_prompts/laguna-xs-2.1-dflash/timed-3.json","bytes":1003}"#
                            : DFlashWorkflowStep.entry(seed: $0)
                    }
                ), false
            ),
            // The pool the floor exists to admit.
            ("eight distinct well-formed targets", DFlashWorkflowStep.distinctPool(8), true),
        ]

        for (label, contract, expectAdmit) in cases {
            let result = try DFlashWorkflowStep.runSelection(contract: contract, ranked: true)
            #expect(
                result.admitted == expectAdmit,
                """
                the timed-target selection step \(result.admitted ? "ADMITTED" : "REFUSED") \
                a RANKED dispatch against \(label), expected \
                \(expectAdmit ? "ADMIT" : "REFUSE"). The anti-lottery floor must \
                count DISTINCT targets -- unique sha256 AND unique r2_path -- \
                because pool length is satisfiable without widening the pool at \
                all, and every entry must be well formed BEFORE sampling, \
                because validating only the sampled entry gives a pool of N \
                one-in-N coverage. Output:
                \(result.output)
                """
            )
            guard !expectAdmit else { continue }
            #expect(
                result.output.contains("::error::"),
                "\(label): the refusal must annotate as a workflow error"
            )
            #expect(
                result.githubOutput.isEmpty,
                """
                \(label): the step emitted selection outputs while refusing, so \
                'Prepare hidden DFlash goldens' would receive a target the floor \
                rejected.
                """
            )
        }
    }

    /// A malformed pool must be refused on EVERY dispatch, not on one in N.
    ///
    /// The step used to validate only the SAMPLED entry, which gives a pool of
    /// N one-in-N coverage. Measured against an 8-entry pool whose entry[3]
    /// carried no sha256: 39 of 40 ranked selections were ADMITTED. The one that
    /// was not died ~25 minutes into a ranked job and read as a flake.
    ///
    /// One draw cannot distinguish "validated the pool" from "happened to draw
    /// the bad entry", so this draws repeatedly. Under sample-then-validate the
    /// probability of 24 consecutive refusals is 8^-24; under validate-then-
    /// sample it is 1.
    @Test
    func aMalformedPoolEntryIsRefusedOnEveryDispatchNotOneInEight() throws {
        let contract = DFlashWorkflowStep.pool(
            (0 ..< 8).map {
                $0 == 3
                    ? #"{"r2_path":"correctness_prompts/laguna-xs-2.1-dflash/timed-3.json","bytes":1003}"#
                    : DFlashWorkflowStep.entry(seed: $0)
            }
        )
        var admitted = 0
        for _ in 0 ..< 24 {
            let result = try DFlashWorkflowStep.runSelection(contract: contract, ranked: true)
            if result.admitted { admitted += 1 }
        }
        #expect(
            admitted == 0,
            """
            \(admitted) of 24 ranked selections were ADMITTED against an \
            8-entry pool whose entry[3] has no sha256. Per-entry completeness \
            must be checked over EVERY entry BEFORE sampling: a pool is either \
            well formed or it is not, and that is a property of the dispatch, \
            not of the draw. Validating only the sampled entry turns a fixture \
            bug into an intermittent mid-run failure that costs a ranked slot \
            and looks like flake.
            """
        )
    }

    /// The refusal has to say WHICH coordinate is short and BY HOW MUCH, or the
    /// operator's next move is a guess. "8 targets" and "8 distinct targets"
    /// look identical from a log line that only reports the length.
    @Test
    func theFloorNamesTheCoordinateThatIsShort() throws {
        let cloned = try DFlashWorkflowStep.runSelection(
            contract: DFlashWorkflowStep.clonedPool(8), ranked: true
        )
        #expect(cloned.output.contains("distinct sha256"))
        #expect(cloned.output.contains("distinct r2_path"))
        #expect(
            cloned.output.contains("short by 7"),
            """
            the refusal does not quantify the shortfall, so a pool of 8 clones \
            and a pool of 7 distinct targets read the same in the log. Output:
            \(cloned.output)
            """
        )

        let malformed = try DFlashWorkflowStep.runSelection(
            contract: DFlashWorkflowStep.pool([
                DFlashWorkflowStep.entry(seed: 0),
                #"{"r2_path":"correctness_prompts/laguna-xs-2.1-dflash/timed-1.json","bytes":1001}"#,
            ]),
            ranked: true
        )
        // A missing sha256 must be diagnosed as MALFORMED, not laundered into
        // the distinct count. `[null, "aa..."] | unique | length` is 2, so an
        // entry-less digest would otherwise read as "one fewer distinct target"
        // -- a true number attached to the wrong diagnosis, which is how a
        // malformed pool gets "fixed" by adding more entries.
        #expect(
            malformed.output.contains("malformed") && malformed.output.contains("entry 1"),
            """
            an entry with no sha256 was not reported as malformed by index. \
            Output:
            \(malformed.output)
            """
        )
    }

    /// The exemption a gates-only dispatch enjoys comes from the step's own
    /// `if:`, NOT from the `MLXFAST_RUN_BENCHMARK` conjunct inside it.
    ///
    /// The test this replaces asserted the conjunct with the rationale that
    /// without it the floor "either blocks gates-only validation or does not
    /// protect ranked runs". That rationale was false: the step-level
    /// `if: ${{ inputs.run_benchmark }}` already means a gates-only dispatch
    /// never reaches the body, so the conjunct can never be false where it is
    /// evaluated. Keeping it is defence in depth for a future in which the `if:`
    /// is widened or the body is lifted elsewhere -- but the `if:` is the thing
    /// that actually grants the exemption, so the `if:` is what gets pinned.
    ///
    /// Stated honestly: the first assertion here reads the condition rather
    /// than executing GitHub's evaluation of it. The executed form lives in
    /// `DFlashDryRunReachesItsGateTests`, whose step-set evaluator computes the
    /// real `if:` expressions and asserts that a `run_benchmark=false` dispatch
    /// does not run this step at all. What this test adds is the exact
    /// condition, so a rewrite to something that happens to evaluate true for a
    /// dry run does not slip past a step-set that only checks membership.
    @Test
    func aGatesOnlyDispatchIsExemptedByTheStepsOwnCondition() throws {
        let condition = try DFlashWorkflowStep.condition(
            try Self.workflow(), DFlashWorkflowStep.selectionStep
        )
        #expect(
            condition == "${{ inputs.run_benchmark }}",
            """
            '\(DFlashWorkflowStep.selectionStep)' no longer carries \
            `if: ${{ inputs.run_benchmark }}` (found \(condition ?? "no condition")). \
            That condition is what exempts a gates-only dry run from the \
            8-distinct-target floor; without it the dry run dies red on main's \
            deliberately sub-floor pool, one step short of the only \
            DFlash-specific gate it exists to exercise.
            """
        )

        // Defence in depth, executed: with the step-level `if:` bypassed -- as
        // it is here, and as it would be by a future edit that widens it -- the
        // env conjunct still lets a sub-floor pool through for validation.
        let dryRun = try DFlashWorkflowStep.runSelection(
            contract: DFlashWorkflowStep.distinctPool(1), ranked: false
        )
        #expect(
            dryRun.admitted,
            """
            the anti-lottery floor refused a NON-ranked invocation of the step \
            body against a one-entry pool. Requiring 8 targets to validate the \
            pipeline recreates exactly the chicken-and-egg the enablement split \
            removed. Output:
            \(dryRun.output)
            """
        )

        // ...but the >= 1 fail-closed check is NOT ranked-scoped: an empty pool
        // still fails both modes.
        for (label, contract) in [
            ("an empty pool", #"{"timed_prompt_pool":[]}"#),
            ("a missing pool", #"{"track_id":"laguna-xs-2.1-dflash-v1"}"#),
        ] {
            let result = try DFlashWorkflowStep.runSelection(contract: contract, ranked: false)
            #expect(
                !result.admitted,
                """
                the step admitted a non-ranked dispatch against \(label). The \
                >= 1 fail-closed check must survive the ranked-only scoping of \
                the floor: there is nothing to sample from. Output:
                \(result.output)
                """
            )
            // Non-zero is not enough. Delete the >= 1 check and the step still
            // exits non-zero -- but as `division by 0` from the rejection-
            // sampling range, or as a raw jq error, ~25 minutes into a job.
            // Requiring the annotation is what keeps the refusal NAMED.
            #expect(
                result.output.contains("::error::"),
                """
                \(label) was refused without a ::workflow error:: annotation, so \
                the run summary shows a bash or jq crash rather than "the pool \
                is empty". Output:
                \(result.output)
                """
            )
        }
    }

    /// Everything in this suite that reads the contract reads the HARDCODED
    /// path `fixtures/laguna_xs_2_1_dflash_track.json`. The workflow does not:
    /// both publish checks, the `>= 1` fail-closed check and the anti-lottery
    /// floor all resolve the contract through `MLXFAST_DFLASH_CONTRACT_PATH`.
    ///
    /// Nothing pinned that binding, so the inertness argument rested on an
    /// unpinned indirection. Measured: repointing that one workflow line at a
    /// second JSON with `official_scoring_enabled: true`,
    /// `publication_allowed: true` and an eight-entry pool defeats all four
    /// locks with 582/582 still green. Pin the binding, and the file this suite
    /// reads is the file the workflow uses.
    @Test
    func theWorkflowResolvesTheContractToTheFixtureThisSuiteReads() throws {
        for path in [Self.workflowPath, ".github/workflows/dflash-provision-goldens.yml"] {
            let text = try String(contentsOfFile: path, encoding: .utf8)
            let bindings = Self.contractPathBindings(text)
            #expect(
                bindings == [Self.fixturePath],
                """
                \(path) binds MLXFAST_DFLASH_CONTRACT_PATH to \(bindings), not \
                exactly [\(Self.fixturePath)]. Every DFlash lock -- the two \
                enablement flag checks, the >= 1 fail-closed check and the \
                anti-lottery floor -- reads whatever that variable names, while \
                theTrackFixtureRemainsInertOnMain reads the hardcoded fixture. \
                If the two can diverge, the inertness this suite asserts is \
                about a file the ranked job does not read.
                """
            )
        }
    }

    /// Every `MLXFAST_DFLASH_CONTRACT_PATH: <value>` YAML binding in a workflow.
    ///
    /// Matched with a colon, so the shell assignment
    /// `MLXFAST_DFLASH_CONTRACT_PATH="${MLXFAST_DFLASH_CONTRACT_PATH}"` that
    /// forwards the value into a sandboxed command is correctly ignored -- it
    /// re-exports the binding rather than choosing it.
    private static func contractPathBindings(_ workflow: String) -> [String] {
        let pattern = "(?m)^[ ]*MLXFAST_DFLASH_CONTRACT_PATH:[ ]*(\\S+)[ ]*$"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let range = NSRange(workflow.startIndex..., in: workflow)
        return regex.matches(in: workflow, range: range).compactMap {
            Range($0.range(at: 1), in: workflow).map { String(workflow[$0]) }
        }
    }

    /// The fixture is LIVE on main as of go-live (2026-07-31, Amendment 31).
    /// This was `theTrackFixtureRemainsInertOnMain`, which asserted the flags
    /// were false; the go-live commit flipped them here in the same change, as
    /// that test's own message instructed. It now pins the enabled state, so a
    /// silent REVERT to inert (which would break Yukon imports) is caught, and
    /// it pins the go-live invariant: scoring may be enabled only with the
    /// fidelity gate implemented (mirrors trackCannotBeEnabledWhileTheFidelityGateIsUnspecified).
    @Test
    func theTrackFixtureIsEnabledForGoLive() throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: Self.fixturePath))
        let object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )

        #expect(
            object["official_scoring_enabled"] as? Bool == true,
            "official_scoring_enabled is not true; the track is offline and Yukon scored dispatches will be refused"
        )
        let baseline = try #require(object["reference_baseline"] as? [String: Any])
        #expect(baseline["publication_allowed"] as? Bool == true)

        // Go-live invariant: enabling scoring REQUIRES the fidelity gate signed
        // off (status implemented). This mirrors the manifest-side build gate so
        // the two cannot drift.
        let proposed = try #require(object["proposed_scoring"] as? [String: Any])
        #expect(
            proposed["token_fidelity_gate_status"] as? String == "implemented",
            "official scoring is enabled while the token-fidelity gate is not implemented"
        )

        // The anti-lottery pool must stay at >= 8 DISTINCT targets or a ranked
        // run cannot proceed. Distinctness, not length. Still checked post-go-live.
        let pool = object["timed_prompt_pool"] as? [Any] ?? []
        let entries = pool.compactMap { $0 as? [String: Any] }
        let distinctDigests = Set(entries.compactMap { $0["sha256"] as? String }).count
        let distinctKeys = Set(entries.compactMap { $0["r2_path"] as? String }).count
        #expect(
            distinctDigests >= 8 && distinctKeys >= 8,
            """
            timed_prompt_pool has \(distinctDigests) distinct digests over \
            \(distinctKeys) distinct keys; the staged pool must keep at least 8 \
            of each or a ranked run cannot proceed even after go-live. Inertness \
            before go-live is held by official_scoring_enabled / \
            publication_allowed, both asserted false above.
            """
        )
        // Every entry must still be complete -- in the same shape the selection
        // step enforces, so the fixture and the workflow cannot disagree about
        // what "well formed" means. A dispatch-time refusal there would be a
        // fixture bug this test should have caught.
        #expect(entries.count == pool.count, "a pool entry is not a JSON object")
        for (index, entry) in entries.enumerated() {
            let path = entry["r2_path"] as? String ?? ""
            let sha = entry["sha256"] as? String ?? ""
            let bytes = entry["bytes"] as? Int ?? 0
            #expect(!path.isEmpty, "pool entry \(index) has no r2_path")
            #expect(
                sha.count == 64 && sha.allSatisfy { "0123456789abcdef".contains($0) },
                "pool entry \(index) sha256 is not a 64-character LOWERCASE hex digest"
            )
            #expect(bytes > 0, "pool entry \(index) has no positive byte count")
        }
    }
}
