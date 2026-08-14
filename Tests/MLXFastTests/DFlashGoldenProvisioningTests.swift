import Foundation
import Testing

/// `.github/workflows/dflash-provision-goldens.yml` — the operator job that
/// freezes and uploads the two hidden DFlash goldens.
///
/// It exists because the objects the ranked job downloads DO NOT EXIST in R2:
/// run 30604267251 died in "Prepare hidden DFlash goldens" with HTTP 404
/// `NoSuchKey` after six correctly-signed attempts, every other gate green. The
/// credentials are GitHub environment secrets in `benchmark-private-prompts-v2`
/// injected per run, so generation and upload have to happen inside a job bound
/// to that environment.
///
/// THE HAZARD THIS SUITE IS REALLY ABOUT. The go-live runbook's step B told the
/// operator to build these goldens with `dflash-reference --seed-generate N`,
/// which extends the seed by REFERENCE-GENERATED tokens — greedy
/// self-continuation, i.e. exactly the material
/// `docs/dflash-track-correctness-contract.md` Amendment 10 measured and
/// condemned (122 distinct tokens in 512, ZERO rows with a top-2 gap below 0.25,
/// minimum gap 1.875-2.625; Amendment 11 prices it at 1.117x against prose's
/// 0.840x). A golden frozen that way advertises a speedup that does not exist.
///
/// Amendment 18's lesson is why the degeneracy tests below are shaped the way
/// they are: L2 survived seventeen amendments and fell to the first real attack,
/// because every review asked whether it rejected honest work and none asked
/// whether it accepted dishonest work. So the degenerate profile here is built
/// from Amendment 10's REAL numbers and the guard is attacked with it first;
/// only then is it shown to accept prose. And every assertion that can be
/// executed is executed — a test that reads workflow TEXT is satisfied by prose,
/// and prose is what failed here in the first place.
@Suite("DFlash golden provisioning")
struct DFlashGoldenProvisioningTests {
    private static let workflowPath = ".github/workflows/dflash-provision-goldens.yml"
    private static let rankedWorkflowPath = ".github/workflows/dflash-benchmark.yml"
    private static let jobKey = "dflash-provision-goldens"
    private static let guardScript = ".github/scripts/check-dflash-golden-degeneracy.sh"
    private static let trustedScript =
        ".github/scripts/enforce-trusted-dflash-provision-workflow.sh"
    private static let publicProseFixture =
        "correctness_prompts/public_longcopy_gate_english_512_256.json"

    private static let interlockStep = "Enforce provisioning interlock and required inputs"
    private static let seedStep = "Fetch operator seed plans"
    private static let generateStep = "Generate both goldens from the pinned baseline"
    private static let verifyStep = "Verify both goldens"
    private static let pinStep = "Re-download the uploaded objects and pin from those bytes"

    // MARK: - 1. dispatch-only

    /// `workflow_dispatch` and nothing else. A push trigger on this job would
    /// upload a golden — and rewrite hidden ranked material — on every merge to
    /// main.
    @Test
    func theProvisioningJobIsDispatchOnly() throws {
        let workflow = try Self.workflow()
        let triggers = try Self.triggerKeys(workflow)

        #expect(
            triggers == ["workflow_dispatch"],
            """
            dflash-provision-goldens declares triggers \(triggers). It must be \
            workflow_dispatch ONLY: it holds R2 WRITE credentials and its output \
            becomes hidden ranked material, so anything that can fire it without \
            an operator deciding to is a way to overwrite a golden by merging.
            """
        )

        // And the trusted-context script refuses a non-dispatch event even if
        // the trigger list is ever widened. Executed, not read.
        let pushed = try Self.runTrustedGuard(
            ref: "refs/heads/main", event: "push", workflow: "dflash-provision-goldens.yml"
        )
        #expect(
            pushed.status != 0,
            "the trusted-context guard admitted a push event: \(pushed.output)"
        )
    }

    /// main only. The ranked job admits `submissions/*`, `baseline/*` and
    /// `yukon/baseline/*`; copying that allowlist here would let any branch a
    /// participant can create overwrite a hidden golden.
    @Test
    func onlyMainMayDispatchProvisioning() throws {
        let allowed = try Self.runTrustedGuard(
            ref: "refs/heads/main", event: "workflow_dispatch",
            workflow: "dflash-provision-goldens.yml"
        )
        #expect(allowed.status == 0, "main was refused: \(allowed.output)")

        for ref in [
            "refs/heads/submissions/example",
            "refs/heads/baseline/reference",
            "refs/heads/yukon/baseline/718528521cd7a7df341b750bc3ccb28478ff045b",
            "refs/heads/feature/anything",
            "refs/tags/v1",
        ] {
            let result = try Self.runTrustedGuard(
                ref: ref, event: "workflow_dispatch",
                workflow: "dflash-provision-goldens.yml"
            )
            #expect(
                result.status != 0,
                """
                the provisioning guard ADMITTED \(ref). This job writes to the \
                private R2 bucket; a branch namespace a participant can create \
                must never reach it.
                """
            )
        }

        // A dispatch whose GITHUB_WORKFLOW_REF names a different workflow file
        // is refused too, so the guard cannot be borrowed by another job.
        let wrongFile = try Self.runTrustedGuard(
            ref: "refs/heads/main", event: "workflow_dispatch", workflow: "ci.yml"
        )
        #expect(wrongFile.status != 0, "the guard accepted a foreign workflow ref")
    }

    /// Both scripts the workflow invokes BY PATH must be executable in the
    /// index. This is not hygiene: the workflow runs
    /// `.github/scripts/enforce-trusted-dflash-provision-workflow.sh` as a
    /// command, so a 0644 mode makes the trusted-context step die with exit 126
    /// — before the interlock, naming no cause. It shipped that way once and was
    /// caught only because `onlyMainMayDispatchProvisioning` EXECUTES the script
    /// instead of reading it; asserting on workflow text would have passed.
    @Test
    func theScriptsTheWorkflowInvokesAreExecutable() throws {
        for script in [Self.guardScript, Self.trustedScript] {
            // git's view, not the checkout's: the mode bit that travels is the
            // one recorded in the index.
            let listing = try Self.bash(
                "git ls-files --stage -- \(script)", cwd: Self.repoRoot, env: [:]
            )
            #expect(
                listing.output.hasPrefix("100755"),
                """
                \(script) is mode \(listing.output.prefix(6)) in the git index, \
                not 100755. The workflow invokes it as a command; a non-executable \
                mode fails the job with exit 126 and no diagnosis. `chmod +x` it \
                and commit the mode change.
                """
            )
            #expect(
                FileManager.default.isExecutableFile(
                    atPath: Self.repoRoot.appendingPathComponent(script).path
                ),
                "\(script) is not executable in the working tree"
            )
        }
    }

    // MARK: - 2. the environment must equal the ranked job's

    /// Both jobs read the SAME R2 secret NAMES and those resolve per
    /// environment. A different binding here uploads the goldens into a
    /// different bucket with different credentials — this job reports success
    /// and the ranked job keeps getting its 404. Equality, not a literal, so a
    /// rotation to `-v3` fails until BOTH move.
    @Test
    func provisioningBindsTheSameEnvironmentAsTheRankedJob() throws {
        let provisioning = try Self.jobEnvironment(Self.workflowPath)
        let ranked = try Self.jobEnvironment(Self.rankedWorkflowPath)

        #expect(
            provisioning == ranked,
            """
            dflash-provision-goldens binds environment '\(provisioning)' but the \
            ranked DFlash job binds '\(ranked)'. The R2 secret names are \
            identical in both and resolve per environment, so the goldens would \
            be written where the ranked job never looks — the same class of \
            mismatch that cost run 30588892999 six retried HTTP 400s with the \
            binding as the only variable.
            """
        )
        #expect(!provisioning.isEmpty)
    }

    // MARK: - 3. every fail-closed arm, executed

    /// The interlock and required-input wall. Each arm must exit non-zero AND
    /// must leave no partial step outputs behind — a half-formed selection is
    /// how a later step ends up with an empty object key.
    @Test
    func everyInterlockArmRefusesAndEmitsNoOutputs() throws {
        let base = [
            "CONFIRM_PROVISION": "1",
            "CORRECTNESS_SEED_R2_PATH": "correctness_prompts/seed-c.json",
            "CORRECTNESS_OBJECT_PATH": "correctness_prompts/golden-c.json",
            "BENCH_SEED_R2_PATH": "correctness_prompts/seed-b.json",
            "BENCH_OBJECT_PATH": "correctness_prompts/golden-b.json",
            "MLXFAST_PRIVATE_PROMPTS_R2_PRESENT": "1",
        ]

        // The happy path first, so a suite that refuses everything is visible.
        let happy = try Self.runStep(Self.interlockStep, env: base)
        #expect(happy.status == 0, "the interlock refused a complete dispatch: \(happy.output)")
        let outputs = Self.parseOutputs(happy.githubOutput)
        #expect(outputs["correctness_seed"] == base["CORRECTNESS_SEED_R2_PATH"])
        #expect(outputs["bench_object"] == base["BENCH_OBJECT_PATH"])

        let arms: [(String, [String: String], String)] = [
            (
                "interlock not ticked", ["CONFIRM_PROVISION": "0"],
                "confirm_provision_goldens=true"
            ),
            (
                "no correctness seed", ["CORRECTNESS_SEED_R2_PATH": ""],
                "runbook.md step A"
            ),
            ("no bench seed", ["BENCH_SEED_R2_PATH": ""], "runbook.md step A"),
            (
                "no correctness destination", ["CORRECTNESS_OBJECT_PATH": ""],
                "no upload destination named"
            ),
            (
                "no bench destination", ["BENCH_OBJECT_PATH": ""],
                "no upload destination named"
            ),
            (
                "both goldens to one key",
                ["BENCH_OBJECT_PATH": base["CORRECTNESS_OBJECT_PATH"]!],
                "same key"
            ),
            (
                "both goldens from one seed",
                ["BENCH_SEED_R2_PATH": base["CORRECTNESS_SEED_R2_PATH"]!],
                "same seed"
            ),
            (
                "no R2 credentials", ["MLXFAST_PRIVATE_PROMPTS_R2_PRESENT": "0"],
                "SAME GitHub environment"
            ),
        ]

        for (label, override, expected) in arms {
            let result = try Self.runStep(
                Self.interlockStep, env: base.merging(override) { _, new in new }
            )
            #expect(
                result.status != 0,
                "the interlock ADMITTED a dispatch with '\(label)': \(result.output)"
            )
            #expect(
                result.output.contains("::error::"),
                "\(label): the refusal must annotate as a workflow error"
            )
            #expect(
                result.output.contains(expected),
                "\(label): the refusal does not name the cause (wanted '\(expected)'): \(result.output)"
            )
            #expect(
                result.githubOutput.isEmpty,
                "\(label): the step emitted step outputs while refusing"
            )
        }
    }

    /// The missing-seed refusal must not merely fail — it must refuse to reach
    /// `--seed-generate`. This is the exact substitution the old runbook invited.
    @Test
    func aMissingSeedNeverFallsBackToSeedGenerate() throws {
        let result = try Self.runStep(
            Self.interlockStep,
            env: [
                "CONFIRM_PROVISION": "1",
                "CORRECTNESS_SEED_R2_PATH": "",
                "CORRECTNESS_OBJECT_PATH": "a.json",
                "BENCH_SEED_R2_PATH": "",
                "BENCH_OBJECT_PATH": "b.json",
                "MLXFAST_PRIVATE_PROMPTS_R2_PRESENT": "1",
            ]
        )
        #expect(result.status != 0)
        #expect(
            result.output.contains("--seed-generate")
                && result.output.contains("Amendment 10"),
            """
            the missing-seed refusal does not tell the operator WHY there is no \
            fallback. The runbook told them to use --seed-generate; the message \
            that replaces that instruction has to name it. Output:
            \(result.output)
            """
        )
    }

    /// The seed-shape and seed-character wall, executed with a stub downloader.
    @Test
    func theSeedFetchRefusesAnythingThatIsNotAProseSeedPlan() throws {
        let prose = try Self.publicProseTokens()

        let cases: [(String, String, String)] = [
            (
                "no seed_tokens array", #"{"emitted":[]}"#,
                "no usable .seed_tokens array"
            ),
            (
                "empty seed", #"{"seed_tokens":[],"emitted":[]}"#,
                "no usable .seed_tokens array"
            ),
            (
                "seed_tokens is a string", #"{"seed_tokens":"hello","emitted":[]}"#,
                "no usable .seed_tokens array"
            ),
            (
                "a supplied chain, not a seed",
                #"{"seed_tokens":\#(Self.jsonArray(prose)),"emitted":[1,2,3]}"#,
                "must be seed-only"
            ),
            (
                // `DFlashEmittedPlan.emitted` is a non-optional `[Int]` with
                // synthesized Codable, so an absent key is a decode failure —
                // reported here, on the seed, instead of by a Swift keyNotFound
                // several minutes into a 300-minute job.
                "emitted key absent entirely",
                #"{"seed_tokens":\#(Self.jsonArray(prose))}"#,
                "no .emitted array"
            ),
            (
                "emitted is not an array",
                #"{"seed_tokens":\#(Self.jsonArray(prose)),"emitted":0}"#,
                "no .emitted array"
            ),
            (
                // THE ONE THAT MATTERS: Amendment 10's measured degenerate seed
                // profile, 122 distinct tokens in 512.
                "greedy self-continuation seed",
                #"{"seed_tokens":\#(Self.jsonArray(Self.seedTokens(length: 512, distinct: 122))),"emitted":[]}"#,
                "seed variety"
            ),
        ]

        for (label, badSeed, expected) in cases {
            let result = try Self.runSeedFetch(
                correctnessSeed: badSeed,
                benchSeed: #"{"seed_tokens":\#(Self.jsonArray(prose)),"emitted":[]}"#
            )
            #expect(
                result.status != 0,
                "'Fetch operator seed plans' ADMITTED \(label): \(result.output)"
            )
            #expect(
                result.output.contains(expected),
                "\(label): refusal does not name the cause (wanted '\(expected)'): \(result.output)"
            )
        }

        // ...and a real prose seed on both sides is admitted, so the wall is not
        // simply "refuse everything".
        let good = try Self.runSeedFetch(
            correctnessSeed: #"{"seed_tokens":\#(Self.jsonArray(prose)),"emitted":[]}"#,
            benchSeed:
                #"{"seed_tokens":\#(Self.jsonArray(Self.seedTokens(length: 512, distinct: 317))),"emitted":[]}"#
        )
        #expect(good.status == 0, "a pair of prose seeds was refused: \(good.output)")
        #expect(good.requestedKeys.count == 2, "both seeds must be fetched")
        #expect(
            good.output.contains("dflash-degeneracy [correctness-seed]")
                && good.output.contains("dflash-degeneracy [bench-seed]"),
            "the pre-generation screen did not report both seeds' character"
        )
    }

    /// The golden-verification wall, executed against staged private files. Each
    /// arm is a property some shipped golden has actually violated.
    @Test
    func everyGoldenVerificationArmRefuses() throws {
        let good = Self.golden(
            seedLength: 512, seedDistinct: 317, rowCount: 8, profile: .prose
        )

        let arms: [(String, [String: Any], String)] = [
            (
                "self-consistency flag false",
                good.merging(["reference_self_consistent": false]) { _, new in new },
                "reference_self_consistent=false"
            ),
            (
                "self-consistency flag absent",
                good.filter { $0.key != "reference_self_consistent" },
                "reference_self_consistent=false"
            ),
            (
                // Amendment 10 Defect 1: three goldens shipped with
                // reference_self_consistent=true while emitted_tokens[i]
                // disagreed with rows[i].sequential_argmax.
                "chain contradicts its own rows",
                good.merging([
                    "emitted_tokens": Self.contradictedChain(rowCount: 8, at: 3)
                ]) { _, new in new },
                "contradicts itself"
            ),
            (
                "no emitted chain at all",
                good.filter { $0.key != "emitted_tokens" },
                "no-emitted-chain"
            ),
            (
                "fewer rows than the gate requests",
                Self.golden(seedLength: 512, seedDistinct: 317, rowCount: 3, profile: .prose),
                "rows but the gate will request"
            ),
            (
                // Amendment 10's measured degenerate profile, verbatim.
                "degenerate material",
                Self.golden(
                    seedLength: 512, seedDistinct: 122, rowCount: 8, profile: .degenerate
                ),
                "degenerate DFlash material rejected"
            ),
        ]

        // Baseline: the honest artifact is accepted, so "refuses everything" is
        // not what is being measured.
        let accepted = try Self.runVerify(correctness: good, bench: good, wantedTokens: 8)
        #expect(accepted.status == 0, "a valid golden pair was refused: \(accepted.output)")

        for (label, damaged, expected) in arms {
            let result = try Self.runVerify(
                correctness: damaged, bench: good, wantedTokens: 8
            )
            #expect(
                result.status != 0,
                "'Verify both goldens' ADMITTED \(label): \(result.output)"
            )
            #expect(
                result.output.contains(expected),
                "\(label): refusal does not name the cause (wanted '\(expected)'): \(result.output)"
            )
        }

        // The BENCH side is verified too, not just the correctness side. A guard
        // that covers one of two artifacts reports success on the one.
        let benchDamaged = try Self.runVerify(
            correctness: good,
            bench: Self.golden(
                seedLength: 512, seedDistinct: 122, rowCount: 8, profile: .degenerate
            ),
            wantedTokens: 8
        )
        #expect(
            benchDamaged.status != 0,
            "the timed golden was not screened; only the correctness golden was"
        )
    }

    // MARK: - 4. the pins come from the re-downloaded object

    /// A pin is only worth anything if it describes the bytes the ranked job
    /// will download. Executed against a stub bucket.
    ///
    /// Honest scope: with the round-trip `cmp` in place the downloaded digest and
    /// the local digest are forced EQUAL on the happy path, so no test can tell
    /// them apart by value there. What is discriminated instead is stronger and
    /// is what the requirement is actually protecting: no pin can be produced
    /// without a successful download of that exact key, and a bucket that
    /// returns different bytes is refused rather than pinned around.
    @Test
    func pinsComeFromTheReDownloadedObjectAndNotTheLocalFile() throws {
        let golden = Self.golden(
            seedLength: 512, seedDistinct: 317, rowCount: 8, profile: .prose
        )

        // (a) happy path: the pin equals the digest of the bytes the bucket
        //     served, and both keys were actually requested.
        let ok = try Self.runPinStep(
            local: golden, served: golden, serveFailure: nil
        )
        #expect(ok.status == 0, "the pin step failed on a clean round trip: \(ok.output)")
        #expect(
            ok.requestedKeys.sorted() == [
                "correctness_prompts/golden-b.json", "correctness_prompts/golden-c.json",
            ],
            """
            the pin step did not re-download both objects (\(ok.requestedKeys)). \
            A pin computed without a round trip describes what this job MEANT to \
            upload, which is exactly the state that produced run 30604267251's \
            404.
            """
        )
        let served = try Self.canonicalJSON(golden)
        let outputs = Self.parseOutputs(ok.githubOutput)
        #expect(outputs["correctness_sha256"] == Self.sha256(served))
        #expect(outputs["correctness_bytes"] == "\(served.utf8.count)")
        #expect(outputs["bench_sha256"] == Self.sha256(served))

        // (b) the bucket has the object but the bytes differ: refuse. A step
        //     that hashed the local file and skipped the comparison would pass
        //     this and hand the ranked job a pin it can never match.
        var tampered = golden
        tampered["reference_seed_token"] = 999_999
        let mismatch = try Self.runPinStep(
            local: golden, served: tampered, serveFailure: nil
        )
        #expect(
            mismatch.status != 0,
            """
            the object read back from R2 differed from what was uploaded and the \
            step pinned it anyway. Output:
            \(mismatch.output)
            """
        )
        #expect(mismatch.output.contains("differs from the bytes this job uploaded"))
        #expect(
            Self.parseOutputs(mismatch.githubOutput)["correctness_sha256"] == nil,
            "a pin was emitted for an object whose round trip failed"
        )

        // (c) the object is not in the bucket at all — the 404 that started
        //     this. The local file is perfect; the step must still refuse,
        //     which is what proves the pin depends on the download.
        let missing = try Self.runPinStep(
            local: golden, served: golden, serveFailure: "correctness_prompts/golden-c.json"
        )
        #expect(
            missing.status != 0,
            """
            the bucket returned no object and the step produced a pin anyway. \
            That is the run-30604267251 failure with a green tick on it. Output:
            \(missing.output)
            """
        )
        #expect(Self.parseOutputs(missing.githubOutput)["correctness_sha256"] == nil)

        // (d) the re-downloaded bytes are themselves re-screened, so a pin can
        //     never describe material the guard would reject.
        let degenerate = Self.golden(
            seedLength: 512, seedDistinct: 122, rowCount: 8, profile: .degenerate
        )
        let degeneratePins = try Self.runPinStep(
            local: degenerate, served: degenerate, serveFailure: nil
        )
        #expect(
            degeneratePins.status != 0,
            "the pin step pinned degenerate material that had reached the bucket"
        )

        // Structural backstop for the one thing the round-trip cmp hides: the
        // digest must be read from the fetched path.
        let body = try Self.stepBody(Self.workflowPath, Self.pinStep)
        #expect(
            body.contains(#"sha="$(shasum -a 256 < "${fetched}" | awk '{print $1}')""#),
            "the pin is no longer computed from the re-downloaded file"
        )
        #expect(
            body.contains(#"bytes="$(wc -c < "${fetched}" | tr -d ' ')""#),
            "the byte count is no longer computed from the re-downloaded file"
        )
    }

    // MARK: - 5. the degeneracy guard, attacked before it is trusted

    /// Amendment 10's MEASURED degenerate profile, rebuilt from the table, must
    /// be rejected — and rejected on every arm, by name.
    ///
    /// | golden               | seed len | distinct | rows gap<0.25 | min gap |
    /// | seam-512-golden.json | 512      | 122      | 0             | 1.875   |
    /// | seam-b-golden.json   | 600      | 122      | 0             | 2.625   |
    @Test
    func theGuardRejectsAmendmentTensMeasuredDegenerateProfile() throws {
        let profiles: [(String, [String: Any], Double)] = [
            (
                "seam-512-golden.json (ranked-window fixture)",
                Self.golden(
                    seedLength: 512, seedDistinct: 122, rowCount: 128,
                    profile: .degenerate, minimumGap: 1.875
                ),
                1.875
            ),
            (
                "seam-b-golden.json",
                Self.golden(
                    seedLength: 600, seedDistinct: 122, rowCount: 128,
                    profile: .degenerate, minimumGap: 2.625
                ),
                2.625
            ),
        ]

        for (label, profile, minimumGap) in profiles {
            let result = try Self.runGuard(profile, label: "degenerate")
            #expect(
                result.status != 0,
                """
                the degeneracy guard ACCEPTED \(label) — Amendment 10's measured \
                degenerate profile, the material on which every green result on \
                this track was wrongly obtained. Output:
                \(result.output)
                """
            )

            // All three arms must fire, and each must be visible. A guard that
            // rejects for one reason while two checks silently pass is a guard
            // whose other two checks nobody has tested.
            #expect(
                result.output.contains("seed variety"),
                "\(label): the seed-variety arm did not fire"
            )
            #expect(
                result.output.contains("top-2 gap below 0.25"),
                "\(label): the near-tie-count arm did not fire"
            )
            #expect(
                result.output.contains("minimum top-2 gap"),
                "\(label): the minimum-gap arm did not fire"
            )

            // The three statistics are reported even on a rejection.
            #expect(result.output.contains("distinct_seed_tokens=122"))
            #expect(result.output.contains("rows_with_top2_gap_lt_0.25=0"))
            #expect(result.output.contains("min_top2_gap=\(Self.trimmed(minimumGap))"))
            #expect(
                result.output.contains("Amendment 10"),
                "\(label): the refusal does not cite the measurement it is derived from"
            )
        }
    }

    /// Only now, prose. Both samples: Amendment 10's own varied fixture, and one
    /// seeded with the ACTUAL token ids of the checked-in public English prose
    /// fixture — real bytes from this repository, not a synthetic stand-in.
    @Test
    func theGuardAcceptsProse() throws {
        let amendmentProse = Self.golden(
            seedLength: 512, seedDistinct: 317, rowCount: 128, profile: .prose
        )
        let varied = try Self.runGuard(amendmentProse, label: "prose")
        #expect(
            varied.status == 0,
            """
            the degeneracy guard REJECTED Amendment 10's own varied-512 prose \
            profile (317 distinct in 512, 3 rows under a 0.25 gap, minimum gap \
            0.0000). A guard that rejects the honest material is not strict, it \
            is broken. Output:
            \(varied.output)
            """
        )
        #expect(varied.output.contains("distinct_seed_tokens=317"))
        #expect(varied.output.contains("rows_with_top2_gap_lt_0.25=3"))
        #expect(varied.output.contains("ACCEPTED"))

        let real = try Self.publicProseTokens()
        // 277 distinct in 512 (was 276 under the Laguna tokenizer). The PROSE
        // is byte-identical -- only the tokenizer changed with the Qwen 3.6
        // target repoint -- and the seed-variety ratio moved 0.5391 -> 0.5410,
        // still far above the guard script's 0.40 floor, so the floor itself
        // needs no re-derivation. If this pin ever moves because the prose
        // changed, re-derive the constant in \(Self.guardScript) instead.
        #expect(
            Set(real).count == 277 && real.count == 512,
            """
            \(Self.publicProseFixture) no longer carries the 512-token / \
            277-distinct English prose sample the seed-variety threshold was \
            derived against. Re-derive the constant in \(Self.guardScript) \
            rather than adjusting this expectation.
            """
        )
        var publicProse = amendmentProse
        publicProse["seed_tokens"] = real
        let checkedIn = try Self.runGuard(publicProse, label: "public-english-prose")
        #expect(
            checkedIn.status == 0,
            """
            the guard rejected a seed that IS the tokenization of the English \
            prose checked into this repository (\(Self.publicProseFixture), \
            cases[0].prompt_tokens: 277 distinct in 512). The 0.40 seed-variety \
            floor is above real prose and must be re-derived. Output:
            \(checkedIn.output)
            """
        )

        // The seed-only pre-screen agrees with the full screen on the same seed,
        // so the two cannot drift into disagreeing about the same material.
        let seedOnly = try Self.runGuard(
            ["seed_tokens": real, "emitted": [Int]()],
            label: "public-english-seed", seedOnly: true
        )
        #expect(seedOnly.status == 0, "the pre-screen rejected a seed the full screen accepts")
        let degenerateSeedOnly = try Self.runGuard(
            ["seed_tokens": Self.seedTokens(length: 512, distinct: 122), "emitted": [Int]()],
            label: "degenerate-seed", seedOnly: true
        )
        #expect(
            degenerateSeedOnly.status != 0,
            "the pre-screen accepted the degenerate seed the full screen rejects"
        )
    }

    /// The material the guard cannot measure is refused, not waved through.
    /// "Unmeasurable" being treated as "fine" is the shape of the original
    /// defect.
    @Test
    func theGuardRefusesMaterialItCannotMeasure() throws {
        var noReadouts = Self.golden(
            seedLength: 512, seedDistinct: 317, rowCount: 8, profile: .prose
        )
        noReadouts["rows"] = (0 ..< 8).map { ["sequential_argmax": 5000 + $0] }
        let result = try Self.runGuard(noReadouts, label: "no-top2")
        #expect(result.status != 0, "a golden with no top-2 readouts was accepted")
        #expect(result.output.contains("cannot be measured"))

        var noRows = Self.golden(
            seedLength: 512, seedDistinct: 317, rowCount: 8, profile: .prose
        )
        noRows["rows"] = [Any]()
        #expect(try Self.runGuard(noRows, label: "no-rows").status != 0)
    }

    /// The thresholds are constants in the script, not knobs a caller can
    /// loosen. A threshold a dispatch input can relax is a threshold that gets
    /// relaxed by the operator who is in a hurry.
    @Test
    func theThresholdsAreNotEnvironmentOverridable() throws {
        let degenerate = Self.golden(
            seedLength: 512, seedDistinct: 122, rowCount: 128, profile: .degenerate
        )
        for (name, value) in [
            ("MIN_DISTINCT_SEED_FRACTION", "0.0"),
            ("MIN_NEAR_TIE_ROWS", "0"),
            ("MAX_MIN_TOP2_GAP", "99"),
            ("NEAR_TIE_GAP", "99"),
        ] {
            let result = try Self.runGuard(degenerate, label: "override", extraEnv: [name: value])
            #expect(
                result.status != 0,
                """
                exporting \(name)=\(value) turned the degeneracy guard off. These \
                constants are derived from Amendment 10's measurements and must \
                be changed in \(Self.guardScript), in a commit, with a reason.
                """
            )
        }
    }

    // MARK: - 6. the generation argv, executed

    /// `--seed-generate` must not reach the reference. Executed: the real
    /// generation step runs against a `sudo` shim that records the exact argv
    /// handed to bench-exec.
    ///
    /// Asserting on the workflow's text could not do this — the file mentions
    /// `--seed-generate` five times, in the comments and error messages that
    /// explain why it is absent.
    @Test
    func theGenerationArgvNeverCarriesSeedGenerate() throws {
        let run = try Self.runGenerationWithSudoShim()
        #expect(run.status == 0, "the generation step failed against the shim: \(run.output)")
        #expect(
            run.invocations.count == 2,
            "expected one reference invocation per golden, got \(run.invocations.count)"
        )

        for argv in run.invocations {
            #expect(
                !argv.contains("--seed-generate"),
                """
                the reference was invoked with --seed-generate: \(argv). That \
                flag extends the seed with the model's own greedy continuation — \
                Amendment 10's degenerate material, which Amendment 11 prices at \
                1.117x against prose's 0.840x. This is the exact substitution the \
                old runbook step B invited.
                """
            )
            #expect(argv.contains("dflash-reference"), "argv is not a dflash-reference call: \(argv)")
            #expect(argv.contains("--generate"), "the chain is not generated: \(argv)")
            #expect(argv.contains("--emitted"), "the operator seed is not supplied: \(argv)")
        }
        // One correctness golden and one bench golden, at the contract's token
        // counts.
        let generated = run.invocations.compactMap { argv -> String? in
            guard let index = argv.firstIndex(of: "--generate"), index + 1 < argv.count else {
                return nil
            }
            return argv[index + 1]
        }
        #expect(generated == ["512", "512"], "unexpected --generate counts: \(generated)")
    }

    // MARK: - 7. it cannot publish a score, and does not touch the ranked path

    @Test
    func provisioningPublishesNoScoreAndTouchesNoRankedState() throws {
        let workflow = try Self.workflow()
        let executable = Self.executableLines(workflow)

        // Nothing that times, scores, or writes ranked state.
        for forbidden in [
            "measure-dflash-job",
            "MLXFAST_DFLASH_MEASURE_JOB",
            "MLXFAST_MEASURE_STATE_DIR",
            "MLXFAST_DFLASH_BASELINE_CALIBRATION",
            "score.json",
            "decode_speedup",
            "failure-counts",
            "upload-artifact",
        ] {
            #expect(
                !executable.contains(forbidden),
                """
                dflash-provision-goldens executes '\(forbidden)'. This job is an \
                operator tool: it must not time, score, or write anything the \
                ranked pipeline reads as state.
                """
            )
        }

        // It must not edit the ranked workflow's pins, or commit, or push.
        for forbidden in ["sed -i", "yq -i", "git commit", "git push", "tee "] {
            #expect(
                !executable.contains(forbidden),
                """
                dflash-provision-goldens contains '\(forbidden)'. Applying the \
                pins must stay a reviewed commit: a job that both produces the \
                material and declares it trusted has no review between those two \
                acts.
                """
            )
        }

        // MENTIONING the ranked workflow is required — the printed pin block has
        // to tell the operator which file to edit — so the assertion is about
        // MUTATION, per line: no executable line may both name a workflow file
        // and redirect into it. A blunt "the string must not appear" check fails
        // on the very guidance that makes the job usable, which is how a
        // meaningful assertion gets deleted instead of sharpened.
        var namesRankedWorkflow = false
        for line in executable.split(separator: "\n", omittingEmptySubsequences: false) {
            guard line.contains(".github/workflows") else { continue }
            namesRankedWorkflow = true
            // A workflow path appearing AFTER a redirect operator is a write to
            // it. Before one it is prose — including `echo "... " >&2`, where the
            // path is the message and the redirect is only stderr.
            let afterRedirect = line.range(of: ">").map { String(line[$0.upperBound...]) } ?? ""
            #expect(
                !afterRedirect.contains(".github/workflows"),
                """
                an executable line redirects into a workflow file: \(line)
                The pins are PRINTED for a human to apply; a job that rewrites \
                the ranked workflow declares its own output trusted with no \
                review in between.
                """
            )
        }
        #expect(
            namesRankedWorkflow,
            """
            no executable line names .github/workflows at all, so the printed pin \
            block no longer tells the operator which file to apply the pins to. \
            This assertion exists to keep that guidance present, not merely to \
            forbid writes.
            """
        )

        // Its bench workspace must be a different path from the ranked job's, or
        // a provisioning dispatch would delete a ranked run's workspace.
        let provisioningWS = try #require(Self.jobEnv(workflow, "MLXFAST_JOB_WS"))
        let rankedWS = try #require(
            Self.jobEnv(try String(contentsOfFile: Self.rankedWorkflowPath, encoding: .utf8),
                        "MLXFAST_JOB_WS")
        )
        #expect(
            provisioningWS != rankedWS,
            "provisioning shares the ranked job's bench workspace \(rankedWS)"
        )
        #expect(provisioningWS.hasPrefix("/Users/Shared/bench-jobs/"))

        // And the pins are PRINTED. If that step ever disappears the whole job
        // becomes an upload with no way to use its output.
        #expect(workflow.contains("- name: Print the pins to apply"))
        for pin in [
            "MLXFAST_DFLASH_CORRECTNESS_GOLDEN_SHA256",
            "MLXFAST_DFLASH_CORRECTNESS_GOLDEN_BYTES",
            "MLXFAST_DFLASH_BENCH_GOLDEN_SHA256",
            "MLXFAST_DFLASH_BENCH_GOLDEN_BYTES",
        ] {
            #expect(workflow.contains(pin), "the printed pin block omits \(pin)")
        }
    }

    /// The goldens must come from the PINNED BASELINE tree, not from a build of
    /// whatever main happens to be — the reference defines the admissible sets a
    /// candidate is judged against.
    @Test
    func generationRunsFromThePinnedBaseline() throws {
        let workflow = try Self.workflow()
        #expect(
            Self.jobEnv(workflow, "MLXFAST_DFLASH_BASELINE_WS")
                == Self.jobEnv(
                    try String(contentsOfFile: Self.rankedWorkflowPath, encoding: .utf8),
                    "MLXFAST_DFLASH_BASELINE_WS"
                ),
            "provisioning and the ranked job disagree about where the pinned DFlash baseline is"
        )
        let preflight = try Self.stepBody(Self.workflowPath, "DFlash host preflight")
        #expect(preflight.contains("MLXFAST_DFLASH_BASELINE_RESOLVED"))
        #expect(
            preflight.contains(#"test -d "${resolved}/weights""#),
            "the baseline is not checked for transformed weights before generation"
        )
    }
}

// MARK: - fixtures built from Amendment 10's measured numbers

extension DFlashGoldenProvisioningTests {
    enum Profile {
        /// Amendment 10: zero rows with a top-2 gap below 0.25.
        case degenerate
        /// Amendment 10: 3 rows below 0.25, minimum exactly 0.0000.
        case prose
    }

    /// A seed of `length` tokens containing exactly `distinct` distinct ids.
    /// Deterministic, so a failure is reproducible.
    static func seedTokens(length: Int, distinct: Int) -> [Int] {
        precondition(distinct <= length && distinct > 0)
        var tokens = Array(1000 ..< (1000 + distinct))
        var state: UInt64 = 0x2545_F491_4F6C_DD1D
        func next(_ bound: Int) -> Int {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return Int(state % UInt64(bound))
        }
        while tokens.count < length {
            tokens.append(1000 + next(distinct))
        }
        for index in stride(from: tokens.count - 1, to: 0, by: -1) {
            tokens.swapAt(index, next(index + 1))
        }
        return tokens
    }

    /// Row top-2 gaps for a profile, matching Amendment 10's measured columns.
    static func gaps(_ profile: Profile, rowCount: Int, minimumGap: Double) -> [Double] {
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        func uniform(_ low: Double, _ high: Double) -> Double {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return low + (high - low) * (Double(state % 100_000) / 100_000.0)
        }
        switch profile {
        case .degenerate:
            // Nothing under 0.25; the smallest gap is exactly the measured one.
            var values = [minimumGap]
            while values.count < rowCount {
                values.append(uniform(minimumGap + 0.05, 9.0))
            }
            return values
        case .prose:
            // 3 near-ties per 128 positions, including a true 0.0000 tie.
            var values = (0 ..< rowCount).map { _ in uniform(0.30, 9.0) }
            let nearTies = [0.0, 0.0625, 0.1875]
            for (offset, gap) in nearTies.enumerated() where rowCount > offset {
                values[(offset * 37 + 11) % rowCount] = gap
            }
            return values
        }
    }

    static func golden(
        seedLength: Int,
        seedDistinct: Int,
        rowCount: Int,
        profile: Profile,
        minimumGap: Double = 1.875
    ) -> [String: Any] {
        let gapValues = gaps(profile, rowCount: rowCount, minimumGap: minimumGap)
        var rows: [[String: Any]] = []
        rows.reserveCapacity(rowCount)
        for index in 0 ..< rowCount {
            let top1: Int = 5000 + index
            let top2: Int = 6000 + index
            let secondLogit: Double = 20.0 - gapValues[index]
            var row: [String: Any] = [:]
            row["sequential_argmax"] = top1
            row["declared_frame_argmax"] = ["1": top1]
            row["top2_tokens"] = [top1, top2]
            row["top2_logits"] = [20.0, secondLogit]
            row["top1_logit"] = 20.0
            rows.append(row)
        }
        return [
            "seed_tokens": seedTokens(length: seedLength, distinct: seedDistinct),
            "reference_seed_token": 4242,
            "reference_self_consistent": true,
            "emitted_tokens": (0 ..< rowCount).map { 5000 + $0 },
            "rows": rows,
        ]
    }

    /// A chain that disagrees with `rows[at].sequential_argmax` — Amendment 10
    /// Defect 1's shipped contradiction.
    static func contradictedChain(rowCount: Int, at index: Int) -> [Int] {
        var chain = (0 ..< rowCount).map { 5000 + $0 }
        chain[index] = 77_777
        return chain
    }

    static func publicProseTokens() throws -> [Int] {
        let data = try Data(contentsOf: URL(fileURLWithPath: publicProseFixture))
        let root = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let cases = try #require(root["cases"] as? [[String: Any]])
        return try #require(cases.first?["prompt_tokens"] as? [Int])
    }

    static func jsonArray(_ values: [Int]) -> String {
        "[" + values.map(String.init).joined(separator: ",") + "]"
    }

    static func canonicalJSON(_ object: [String: Any]) throws -> String {
        let data = try JSONSerialization.data(
            withJSONObject: object, options: [.sortedKeys]
        )
        return String(decoding: data, as: UTF8.self)
    }

    static func trimmed(_ value: Double) -> String {
        var text = String(value)
        while text.hasSuffix("0") { text.removeLast() }
        if text.hasSuffix(".") { text.removeLast() }
        return text
    }
}

// MARK: - execution harnesses

extension DFlashGoldenProvisioningTests {
    struct Run {
        var status: Int32
        var output: String
        var githubOutput: String = ""
        var requestedKeys: [String] = []
        var invocations: [[String]] = []
    }

    final class Sandbox {
        let root: URL
        init() throws {
            root = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("dflash-provision-\(UUID().uuidString)")
            try FileManager.default.createDirectory(
                at: root, withIntermediateDirectories: true
            )
        }
        deinit { try? FileManager.default.removeItem(at: root) }
        func dir(_ name: String) throws -> URL {
            let url = root.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            return url
        }
    }

    static func workflow() throws -> String {
        try String(contentsOfFile: workflowPath, encoding: .utf8)
    }

    static func bash(
        _ script: String,
        cwd: URL,
        env: [String: String]
    ) throws -> (status: Int32, output: String) {
        var environment = env
        let inherited = ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin"
        if environment["PATH"] == nil {
            environment["PATH"] = "\(inherited):/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["-c", script]
        process.currentDirectoryURL = cwd
        process.environment = environment
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(decoding: data, as: UTF8.self))
    }

    static var repoRoot: URL {
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }

    /// Runs a real step body from the workflow, in the repository root, with a
    /// fresh `GITHUB_OUTPUT`.
    static func runStep(
        _ step: String,
        env: [String: String],
        cwd: URL? = nil
    ) throws -> Run {
        let outputFile = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("gho-\(UUID().uuidString)")
        FileManager.default.createFile(atPath: outputFile.path, contents: nil)
        defer { try? FileManager.default.removeItem(at: outputFile) }

        var environment = env
        environment["GITHUB_OUTPUT"] = outputFile.path
        let run = try bash(
            try stepBody(workflowPath, step),
            cwd: cwd ?? repoRoot,
            env: environment
        )
        return Run(
            status: run.status,
            output: run.output,
            githubOutput: (try? String(contentsOf: outputFile, encoding: .utf8)) ?? ""
        )
    }

    /// A sandbox that looks enough like a checkout to run a step that calls
    /// `.github/scripts/*`: the REAL degeneracy guard, a stub R2 downloader.
    static func stagedCheckout(_ sandbox: Sandbox, objects: [String: String]) throws -> URL {
        let scripts = try sandbox.dir(".github/scripts")
        let objectDir = try sandbox.dir("objects")
        let fm = FileManager.default

        try fm.copyItem(
            at: repoRoot.appendingPathComponent(guardScript),
            to: scripts.appendingPathComponent("check-dflash-golden-degeneracy.sh")
        )
        try fm.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: scripts.appendingPathComponent(
                "check-dflash-golden-degeneracy.sh"
            ).path
        )

        for (key, body) in objects {
            try body.write(
                to: objectDir.appendingPathComponent(key.replacingOccurrences(of: "/", with: "_")),
                atomically: true,
                encoding: .utf8
            )
        }

        let stub = """
            #!/bin/bash
            set -u
            printf '%s\\n' "${1-}" >> "${STUB_REQUEST_LOG}"
            key="${1-}"
            if [[ -n "${STUB_FAIL_KEY:-}" && "${key}" == "${STUB_FAIL_KEY}" ]]; then
              echo "stub: <Error><Code>NoSuchKey</Code></Error> for '${key}'" >&2
              exit 22
            fi
            src="${STUB_OBJECT_DIR}/$(printf '%s' "${key}" | tr '/' '_')"
            if [[ -z "${key}" || ! -f "${src}" ]]; then
              echo "stub download-r2-object.sh: no object staged for key '${key}'" >&2
              exit 22
            fi
            cp "${src}" "${2}"
            """
        let stubURL = scripts.appendingPathComponent("download-r2-object.sh")
        try stub.write(to: stubURL, atomically: true, encoding: .utf8)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: stubURL.path)
        return objectDir
    }

    static func runSeedFetch(correctnessSeed: String, benchSeed: String) throws -> Run {
        let sandbox = try Sandbox()
        let correctnessKey = "correctness_prompts/seed-c.json"
        let benchKey = "correctness_prompts/seed-b.json"
        _ = try stagedCheckout(
            sandbox, objects: [correctnessKey: correctnessSeed, benchKey: benchSeed]
        )
        let privateDir = try sandbox.dir("private")
        let requestLog = sandbox.root.appendingPathComponent("requests.log")
        FileManager.default.createFile(atPath: requestLog.path, contents: nil)

        var run = try runStep(
            seedStep,
            env: [
                "MLXFAST_PRIVATE_DIR": privateDir.path,
                "CORRECTNESS_SEED_R2_PATH": correctnessKey,
                "BENCH_SEED_R2_PATH": benchKey,
                "STUB_REQUEST_LOG": requestLog.path,
                "STUB_OBJECT_DIR": sandbox.root.appendingPathComponent("objects").path,
            ],
            cwd: sandbox.root
        )
        run.requestedKeys = readLines(requestLog)
        return run
    }

    static func runVerify(
        correctness: [String: Any],
        bench: [String: Any],
        wantedTokens: Int
    ) throws -> Run {
        let sandbox = try Sandbox()
        _ = try stagedCheckout(sandbox, objects: [:])
        let privateDir = try sandbox.dir("private")
        try canonicalJSON(correctness).write(
            to: privateDir.appendingPathComponent("correctness_golden.json"),
            atomically: true, encoding: .utf8
        )
        try canonicalJSON(bench).write(
            to: privateDir.appendingPathComponent("bench_golden.json"),
            atomically: true, encoding: .utf8
        )
        return try runStep(
            verifyStep,
            env: [
                "MLXFAST_PRIVATE_DIR": privateDir.path,
                "MLXFAST_DFLASH_CORRECTNESS_TOKENS": "\(wantedTokens)",
                "MLXFAST_DFLASH_DECODE_TOKENS": "\(wantedTokens)",
            ],
            cwd: sandbox.root
        )
    }

    static func runPinStep(
        local: [String: Any],
        served: [String: Any],
        serveFailure: String?
    ) throws -> Run {
        let sandbox = try Sandbox()
        let correctnessKey = "correctness_prompts/golden-c.json"
        let benchKey = "correctness_prompts/golden-b.json"
        let servedBody = try canonicalJSON(served)
        _ = try stagedCheckout(
            sandbox, objects: [correctnessKey: servedBody, benchKey: servedBody]
        )
        let privateDir = try sandbox.dir("private")
        let localBody = try canonicalJSON(local)
        for side in ["correctness", "bench"] {
            try localBody.write(
                to: privateDir.appendingPathComponent("\(side)_golden.json"),
                atomically: true, encoding: .utf8
            )
        }
        let requestLog = sandbox.root.appendingPathComponent("requests.log")
        FileManager.default.createFile(atPath: requestLog.path, contents: nil)

        var env: [String: String] = [
            "MLXFAST_PRIVATE_DIR": privateDir.path,
            "CORRECTNESS_OBJECT_PATH": correctnessKey,
            "BENCH_OBJECT_PATH": benchKey,
            "STUB_REQUEST_LOG": requestLog.path,
            "STUB_OBJECT_DIR": sandbox.root.appendingPathComponent("objects").path,
        ]
        if let serveFailure { env["STUB_FAIL_KEY"] = serveFailure }

        var run = try runStep(pinStep, env: env, cwd: sandbox.root)
        run.requestedKeys = readLines(requestLog)
        return run
    }

    /// Runs the real generation step against a `sudo` shim that records argv and
    /// fabricates the golden the step expects to find.
    static func runGenerationWithSudoShim() throws -> Run {
        let sandbox = try Sandbox()
        let fm = FileManager.default
        let bin = try sandbox.dir("bin")
        let privateDir = try sandbox.dir("private")
        let jobWS = try sandbox.dir("jobws")
        _ = try sandbox.dir("jobws/.dflash-provision")
        let argvLog = sandbox.root.appendingPathComponent("argv.log")
        fm.createFile(atPath: argvLog.path, contents: nil)

        for side in ["correctness", "bench"] {
            try "{\"seed_tokens\":[1,2,3],\"emitted\":[]}".write(
                to: privateDir.appendingPathComponent("\(side)_seed.json"),
                atomically: true, encoding: .utf8
            )
        }

        // Records the FULL argv as ONE tab-separated line and writes whatever
        // `--output` names, so the step's own `test -s` is satisfied.
        //
        // `%q`, not `%s`: one of the arguments is the inner `bash -c` script,
        // which contains newlines. With `%s` that argument spills across lines
        // and a line-based reader sees three fabricated "invocations" made of
        // script fragments -- which is exactly what it did the first time.
        let shim = """
            #!/bin/bash
            set -u
            printf '%q\\t' "$@" >> "${SHIM_ARGV_LOG}"
            printf '\\n' >> "${SHIM_ARGV_LOG}"
            out=""
            prev=""
            for arg in "$@"; do
              if [[ "${prev}" == "--output" ]]; then out="${arg}"; fi
              prev="${arg}"
            done
            if [[ -n "${out}" ]]; then
              mkdir -p "$(dirname "${SHIM_JOB_WS}/${out}")"
              printf '{"generated":true}' > "${SHIM_JOB_WS}/${out}"
            fi
            """
        let shimURL = bin.appendingPathComponent("sudo")
        try shim.write(to: shimURL, atomically: true, encoding: .utf8)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: shimURL.path)

        let inherited = ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin"
        let run = try runStep(
            generateStep,
            env: [
                "PATH": "\(bin.path):\(inherited):/usr/bin:/bin",
                "MLXFAST_PRIVATE_DIR": privateDir.path,
                "MLXFAST_JOB_WS": jobWS.path,
                "MLXFAST_BENCH_EXEC": "/opt/bench/bench-exec.sh",
                "MLXFAST_OFFICIAL_BENCHMARK_RUN": "1",
                "MLXFAST_DFLASH_ASSISTANT_DIR": "/opt/bench-runner/cache/dflash/assistant",
                "MLXFAST_DFLASH_TARGET_DIR": "/opt/bench-runner/cache/target",
                "MLXFAST_DFLASH_BLOCK_SIZE": "3",
                "MLXFAST_DFLASH_CORRECTNESS_TOKENS": "512",
                "MLXFAST_DFLASH_DECODE_TOKENS": "512",
                "SHIM_ARGV_LOG": argvLog.path,
                "SHIM_JOB_WS": jobWS.path,
            ],
            cwd: sandbox.root
        )
        let invocations = readLines(argvLog).map { line in
            line.split(separator: "\t", omittingEmptySubsequences: true).map(String.init)
        }
        return Run(status: run.status, output: run.output, invocations: invocations)
    }

    /// Executes the REAL degeneracy script against a staged artifact.
    static func runGuard(
        _ object: [String: Any],
        label: String,
        seedOnly: Bool = false,
        extraEnv: [String: String] = [:]
    ) throws -> Run {
        let sandbox = try Sandbox()
        let path = sandbox.root.appendingPathComponent("artifact.json")
        try canonicalJSON(object).write(to: path, atomically: true, encoding: .utf8)

        var environment = extraEnv
        environment["PATH"] =
            (ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin")
            + ":/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        let flag = seedOnly ? "--seed-only " : ""
        let run = try bash(
            "\(repoRoot.appendingPathComponent(guardScript).path) \(flag)"
                + "\(path.path) \(label)",
            cwd: repoRoot,
            env: environment
        )
        return Run(status: run.status, output: run.output)
    }

    static func runTrustedGuard(
        ref: String,
        event: String,
        workflow: String
    ) throws -> Run {
        let run = try bash(
            repoRoot.appendingPathComponent(trustedScript).path,
            cwd: repoRoot,
            env: [
                "GITHUB_REPOSITORY": "Layr-Labs/mlxfast-challenge-dev",
                "GITHUB_REF": ref,
                "GITHUB_WORKFLOW_REF":
                    "Layr-Labs/mlxfast-challenge-dev/.github/workflows/\(workflow)@\(ref)",
                "GITHUB_EVENT_NAME": event,
            ]
        )
        return Run(status: run.status, output: run.output)
    }

    static func readLines(_ url: URL) -> [String] {
        ((try? String(contentsOf: url, encoding: .utf8)) ?? "")
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map(String.init)
    }

    static func sha256(_ text: String) -> String {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("sha-\(UUID().uuidString)")
        try? text.write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }
        let run = try? bash(
            "shasum -a 256 < \(url.path) | awk '{print $1}'",
            cwd: URL(fileURLWithPath: NSTemporaryDirectory()),
            env: [:]
        )
        return (run?.output ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func parseOutputs(_ text: String) -> [String: String] {
        var result: [String: String] = [:]
        for line in text.split(separator: "\n") {
            guard let split = line.firstIndex(of: "=") else { continue }
            result[String(line[..<split])] = String(line[line.index(after: split)...])
        }
        return result
    }
}

// MARK: - workflow parsing

extension DFlashGoldenProvisioningTests {
    /// The job's `environment:` value.
    static func jobEnvironment(_ path: String) throws -> String {
        let text = try String(contentsOfFile: path, encoding: .utf8)
        let line = try #require(
            text.split(separator: "\n", omittingEmptySubsequences: false).first {
                $0.hasPrefix("    environment:")
                    && !$0.trimmingCharacters(in: .whitespaces).hasPrefix("#")
            },
            "\(path) declares no job-level environment:, so it gets no private-material secrets"
        )
        return line.replacingOccurrences(of: "    environment:", with: "")
            .trimmingCharacters(in: .whitespaces)
    }

    /// A job-level `env:` value, comments excluded.
    static func jobEnv(_ workflow: String, _ key: String) -> String? {
        for line in workflow.split(separator: "\n", omittingEmptySubsequences: false)
        where line.hasPrefix("      \(key): ") {
            return String(line.dropFirst("      \(key): ".count))
                .trimmingCharacters(in: .whitespaces)
        }
        return nil
    }

    /// Top-level trigger keys under `on:`.
    static func triggerKeys(_ workflow: String) throws -> [String] {
        let lines = workflow.split(separator: "\n", omittingEmptySubsequences: false)
        guard let start = lines.firstIndex(of: "on:") else {
            Issue.record("no top-level on: block")
            return []
        }
        var keys: [String] = []
        for line in lines[(start + 1)...] {
            if line.trimmingCharacters(in: .whitespaces).isEmpty { continue }
            let indent = line.prefix { $0 == " " }.count
            if indent == 0 { break }
            guard indent == 2, let colon = line.firstIndex(of: ":") else { continue }
            let key = line[line.startIndex ..< colon].trimmingCharacters(in: .whitespaces)
            if !key.hasPrefix("#") { keys.append(key) }
        }
        return keys
    }

    /// A step's `run:` block scalar, dedented, comments preserved (it is
    /// executed, so stripping them would change what runs).
    static func stepBody(_ path: String, _ step: String) throws -> String {
        let workflow = try String(contentsOfFile: path, encoding: .utf8)
        let start = try #require(
            workflow.range(of: "- name: \(step)\n"),
            "step '\(step)' is missing from \(path)"
        )
        var block: [Substring] = []
        for line in workflow[start.upperBound...].split(
            separator: "\n", omittingEmptySubsequences: false
        ) {
            let blank = line.trimmingCharacters(in: .whitespaces).isEmpty
            if !blank, line.prefix(while: { $0 == " " }).count <= 6 { break }
            block.append(line)
        }
        let text = block.joined(separator: "\n")
        let marker = try #require(
            text.range(of: "run: |\n"), "step '\(step)' has no literal run: block"
        )
        let scalar = String(text[marker.upperBound...])
        let lines = scalar.split(separator: "\n", omittingEmptySubsequences: false)
        let indent =
            lines.first { !$0.trimmingCharacters(in: .whitespaces).isEmpty }?
            .prefix { $0 == " " }.count ?? 0
        var body: [String] = []
        for line in lines {
            let blank = line.trimmingCharacters(in: .whitespaces).isEmpty
            if !blank, line.prefix(while: { $0 == " " }).count < indent { break }
            body.append(line.count >= indent ? String(line.dropFirst(indent)) : String(line))
        }
        return body.joined(separator: "\n")
    }

    /// The workflow with comment-only lines removed, so an "is absent"
    /// assertion cannot be defeated by the prose explaining the absence.
    static func executableLines(_ workflow: String) -> String {
        workflow
            .split(separator: "\n", omittingEmptySubsequences: false)
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("#") }
            .joined(separator: "\n")
    }
}
