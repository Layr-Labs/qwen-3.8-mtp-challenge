import CryptoKit
import Foundation
import Testing

// The track-scoped constants pinned by QwenMTPTrackScopedConstantsTests.
import MLXFastCore

// NAMING LAW FOR THE QWEN 3.8 NATIVE-MTP TRACK (qwen3.8-27b-mtp-v1).
//
// The organizer's rule is that no retired Gemma/Laguna MTP surface name may
// survive anywhere in the repository, as a substring, in any new name. The
// existing guard is BenchmarkScriptTests'
// `localDFlashScriptsNameNoRetiredMTPSurface` (the local DFlash scripts do not
// NAME them); its companion `retiredMTPNamesStayRetired` (the retired FILES
// stay deleted) went with DFlashTrackTests at the 2026-08-13 repo split. Both
// were written for the DFlash track and neither knows about this one.
//
// Renaming must be TESTED, not just done, and the test has to run in BOTH
// directions:
//
//   NEGATIVE  every entry of `retiredMTPNames` must still trip the guard's own
//             substring check. A track that "passes the guard" because the
//             guard was quietly narrowed has proven nothing.
//   POSITIVE  every name this track introduces must be proven free of all of
//             them -- enumerated here by name, so a future rename is caught by
//             this file rather than by a reviewer reading a diff.
//
// THE TRAP THIS FILE EXISTS FOR. The retired names are matched as SUBSTRINGS,
// so a "qwen-" prefix does NOT make a retired name new: the natural spellings
// for this track's workflow, its probe verb and its reference-row artifacts all
// reproduce a retired string with a prefix in front of it. Only changing what
// follows "mtp-" clears the rule. `retiredNamesAreNotClearedByAQwenPrefix`
// below pins exactly that, so the reasoning survives even if everyone who
// remembers it has moved on.
//
// Nothing here loads a model, runs a benchmark or needs a network: every
// assertion reads checked-in text.
@Suite
struct QwenMTPTrackNamingTests {
    private typealias S = DFlashGateTextSupport

    // MARK: - The guard's own check

    /// The substring check `localDFlashScriptsNameNoRetiredMTPSurface` applies,
    /// factored out so both directions below run the SAME predicate. If that
    /// guard's matching rule ever changes, this mirror must change with it --
    /// which is the point: two different notions of "names a retired surface"
    /// would let a name pass one and fail the other.
    private static func retiredNames(in text: String) -> [String] {
        S.retiredMTPNames.filter { text.contains($0) }
    }

    // MARK: - The new surface

    /// Every name this track introduces, labelled. The label is what a failure
    /// message says, so it names the thing to rename rather than the string.
    ///
    /// ADD TO THIS LIST WHEN THE TRACK GROWS A SURFACE. It is the enumeration
    /// the operator asked for: the value of the positive case comes entirely
    /// from its completeness, and a name that is not listed here is a name
    /// nobody is checking.
    static let newSurfaceNames: [(label: String, name: String)] = [
        // --- the four checked-in files -----------------------------------
        ("ranked workflow filename", "qwen-mtp-ranked-benchmark.yml"),
        ("ranked workflow path", ".github/workflows/qwen-mtp-ranked-benchmark.yml"),
        ("ranked workflow name", "qwen-mtp-ranked-benchmark"),
        // The provisioning + R2-key-probe workflows. The probe file dodges the
        // retired names deliberately: putting the probe verb directly after
        // "mtp-" would reproduce a retired substring (this file is scanned WHOLE,
        // so the retired string is described, never spelled), so the verb is
        // spelled "r2-key-probe" -- probe LAST -- and the file is
        // qwen-mtp-r2-key-probe.yml.
        ("provisioning workflow filename", "qwen-mtp-provision-goldens.yml"),
        ("provisioning workflow path", ".github/workflows/qwen-mtp-provision-goldens.yml"),
        ("provisioning workflow name", "qwen-mtp-provision-goldens"),
        ("r2-key-probe workflow filename", "qwen-mtp-r2-key-probe.yml"),
        ("r2-key-probe workflow path", ".github/workflows/qwen-mtp-r2-key-probe.yml"),
        ("r2-key-probe workflow name", "qwen-mtp-r2-key-probe"),
        // The trusted-context guards the two credentialled workflows invoke BY
        // PATH. A script path is as hard to rename as a workflow filename --
        // the `run:` line names it literally and a rename that misses one end
        // fails the job with exit 127 -- so both are enumerated here, and the
        // probe guard's own verb placement matters for the same reason the
        // probe workflow's does: the verb is spelled "r2-probe", never directly
        // after "mtp-".
        (
            "provisioning trusted-context guard",
            ".github/scripts/enforce-trusted-qwen-mtp-provision-workflow.sh"
        ),
        (
            "r2-key-probe trusted-context guard",
            ".github/scripts/enforce-trusted-qwen-mtp-r2-probe-workflow.sh"
        ),
        ("local benchmark runner", "benchmark-qwen-mtp.sh"),
        ("local setup runner", "setup-qwen-mtp.sh"),
        ("track manifest filename", "benchmark.qwen-mtp.json"),
        // --- track identity ------------------------------------------------
        ("track id", "qwen3.8-27b-mtp-v1"),
        ("track name", "mlxfast-challenge-dev-qwen38-mtp"),
        ("leaderboard namespace", "qwen3.8-27b-mtp-v1"),
        ("timed decode target id", "lowsim-prose-qwen38-v1"),
        ("calibration-ready interlock", "MLXFAST_QWEN_MTP_CALIBRATION_READY"),
        ("workflow env prefix", "MLXFAST_QWEN_MTP_"),
        // --- CLI verbs -------------------------------------------------------
        // The natural spellings -- appending "probe", "benchmark" or
        // "reference" directly after "mtp-" -- are all retired entries.
        ("untimed verify verb", "mtp-verify"),
        ("timed measurement verb", "mtp-timed"),
        ("canonical depth flag", "--mtp-depth"),
        ("head tree flag", "--mtp-head"),
        // --- runtime-worker surface (phase 2) --------------------------------
        // The worker subcommand and the five protocol kinds. Wire strings are
        // surface names too: a retired substring in a request kind would survive
        // in every transcript and every sandbox profile that names it.
        ("runtime worker subcommand", "mtp-runtime-worker"),
        ("worker warm kind", "mtp_decode_warm"),
        ("worker begin kind", "mtp_decode_begin"),
        ("worker round kind", "mtp_decode_round"),
        ("worker reference prefill kind", "mtp_reference_prefill"),
        ("worker reference rows kind", "mtp_reference_rows"),
        ("head directory environment variable", "MLXFAST_QWEN_MTP_HEAD_DIR"),
        // --- Swift sources this track adds -----------------------------------
        ("worker hot-path source", "Qwen36MTPBlockSession.swift"),
        ("head attachment source", "Qwen36MTPHeadAttachment.swift"),
        ("serial reference source", "Qwen36MTPReferenceSession.swift"),
        ("worker harness source", "QwenRuntimeMTPWorker.swift"),
        ("model-surface protocol source", "Qwen36MTPTarget.swift"),
        ("model-surface protocol", "Qwen36MTPTarget"),
        ("backbone layout test suite source", "QwenMTPBackboneLayoutTests.swift"),
        ("metallib all-roots flag", "--all-build-roots"),
        ("pinned reference environment variable", "MLXFAST_QWEN_MTP_TARGET_DIR"),
        ("trusted contract source", "QwenRuntimeMTP.swift"),
        ("trusted driver source", "QwenRuntimeMTPDriver.swift"),
        ("verb test suite source", "QwenMTPVerbTests.swift"),
        ("rollback test suite source", "QwenMTPRollbackContractTests.swift"),
        ("trusted-context guard test suite source", "QwenMTPTrustedContextGuardTests.swift"),
        // --- payload fields ---------------------------------------------------
        // The evidence payload's own key names, enumerated for the same reason:
        // they are read by a box-owned wrapper and a workflow, so they are as
        // hard to rename as a filename.
        ("native head boolean", "uses_native_mtp_head"),
        ("pinned head boolean", "uses_pinned_mtp_head"),
        ("head attachment boolean", "mtp_head_attached"),
        ("depth field", "mtp_depth"),
        ("serial control depth field", "serial_control_depth"),
        ("row ledger field", "row_ledger"),
        ("parity gate field", "parity_all_ok"),
        // --- box-owned and per-track paths -----------------------------------
        ("measure wrapper filename", "measure-qwen-mtp-job.sh"),
        ("per-track state dir", "/opt/bench-runner/state/qwen3.8-27b-mtp-v1"),
        ("per-track state dir leaf", "qwen3.8-27b-mtp-v1"),
        ("per-track baseline dir", "/opt/bench-runner/baseline/qwen3.8-27b-mtp-v1/current"),
        ("per-track baseline calibration", "/opt/bench-runner/state/qwen3.8-27b-mtp-v1/baseline-calibration.json"),
        ("MTP head cache dir", "/opt/bench-runner/cache/qwen38/qwen3.8-27b-mtp-v1/mtp-head"),
        ("runner label", "m5-qwen38-27b-mtp"),
        ("bench job workspace", "/Users/Shared/bench-jobs/qwen-mtp-ranked-current"),
        // --- fixtures -------------------------------------------------------
        ("track contract fixture", "fixtures/qwen3_8_27b_mtp_track.json"),
        ("MTP head manifest fixture", "fixtures/qwen3_8_27b_mtp_head.sha256"),
        ("target manifest fixture", "fixtures/reference_qwen3_8_27b_4bit.sha256"),
        // --- pinned checkpoint identity (repinned 2026-08-14) -----------------
        // Enumerated for the same reason every other name here is: these strings
        // are spelled into a workflow, a shell script and a compiled constant,
        // so they are as hard to change as a filename -- and the naming law
        // applies to every string this track introduces, not only to the ones
        // this repository invented. The suffix rule is what makes the check
        // non-trivial: a repository name ending in a retired surface name would
        // trip the guard exactly as a filename would. Both repositories are
        // ours, so this list is now checking names WE chose -- exactly the case
        // the law exists for.
        //
        // The revision entries below are the PUBLISHED shas (2026-08-14); while
        // publication was pending they were markers, enumerated so the gate
        // could catch a placeholder reaching a download or a comparison.
        ("pinned checkpoint repository", "EigenLabs/Qwen3.8-27B-4bit"),
        ("pinned MTP head repository", "EigenLabs/Qwen3.8-27B-MTP-bf16"),
        (
            "pinned checkpoint cache dir",
            "/opt/bench-runner/cache/huggingface/hub/models--EigenLabs--Qwen3.8-27B-4bit"
        ),
        ("upstream bf16 base repository", "Qwen/Qwen3.8-27B"),
        ("pinned checkpoint revision", "eda45ab47f465d08d6558f0353a2346e2eb9d5b3"),
        ("pinned MTP head revision", "26a328e070875b0314d652a039b6b59902690f03"),
        // --- workflow-internal names ----------------------------------------
        ("hidden-golden staging dir", ".qwen-mtp-ranked-src"),
        ("transformed weights hash artifact", "qwen-mtp-weights.sha256"),
        ("parity gate report artifact", "qwen-mtp-gates-report.json"),
        ("paired results artifact", "qwen-mtp-paired-results.json"),
        ("measure verdict artifact", "qwen-mtp-measure-verdict.txt"),
        ("correctness artifact bundle", "qwen-mtp-correctness-results"),
        ("audit artifact bundle", "qwen-mtp-ranked-audit"),
        ("redacted failure bundle", "qwen-mtp-ranked-failure"),
        ("bench job id prefix", "qwen-mtp-ranked-"),
        ("score mode", "qwen-mtp-paired-decode-only"),
        // --- local runner names ----------------------------------------------
        ("local scratch root", ".mlxfast-local-qwen-mtp"),
        ("local reference rows file", "qwen-mtp-golden-rows.json"),
        ("local head cache root", "~/.cache/mlxfast/qwen3.8-27b-mtp-v1"),
        ("local head cache leaf", "mtp-head"),
        ("local score oracle label", "candidate-local-mtp-golden-rows"),
    ]

    /// The checked-in files this track adds. Scanned WHOLE -- comments
    /// included, which is stricter than `DFlashGateTextSupport.executable`
    /// permits for the DFlash workflow. That is deliberate for a brand-new
    /// surface: there is no historical comment here that has to mention a dead
    /// name, so the strongest available check costs nothing, and it keeps a
    /// retired name from creeping in as "just a comment" and later being copied
    /// into a value.
    ///
    /// THIS FILE IS IN THE LIST. It is a new file too, and a naming test that
    /// exempted itself would be free to spell a retired name in a comment and
    /// then have that comment copied into a value. Every retired string this
    /// suite needs is built at run time from `retiredMTPNames`, never spelled
    /// here, so the self-check costs nothing and closes the loop.
    static let newSurfaceFiles = [
        ".github/workflows/qwen-mtp-ranked-benchmark.yml",
        ".github/workflows/qwen-mtp-provision-goldens.yml",
        ".github/workflows/qwen-mtp-r2-key-probe.yml",
        // The two trusted-context guards. Scanned WHOLE like everything else
        // here: their headers explain themselves at length by comparison with
        // the DFlash twins, which is exactly the kind of prose that reaches for
        // a dead name.
        ".github/scripts/enforce-trusted-qwen-mtp-provision-workflow.sh",
        ".github/scripts/enforce-trusted-qwen-mtp-r2-probe-workflow.sh",
        "benchmark-qwen-mtp.sh",
        "setup-qwen-mtp.sh",
        "benchmark.qwen-mtp.json",
        "fixtures/qwen3_8_27b_mtp_head.sha256",
        "Tests/MLXFastTests/QwenMTPTrackNamingTests.swift",
        // Phase 2: the runtime and verb sources. Scanned WHOLE, comments
        // included -- these files talk ABOUT the retired surface (the DFlash
        // rollback, the retired row equation) and must do so without ever
        // spelling a retired name.
        "Sources/MLXFastModel/Qwen36MTPBlockSession.swift",
        "Sources/MLXFastModel/Qwen36MTPHeadAttachment.swift",
        "Sources/MLXFastModel/Qwen36MTPReferenceSession.swift",
        "Sources/MLXFastHarness/QwenRuntimeMTPWorker.swift",
        "Sources/MLXFastModel/Qwen36MTPTarget.swift",
        "Tests/MLXFastTests/QwenMTPBackboneLayoutTests.swift",
        "Sources/MLXFastTrustedHarness/QwenRuntimeMTP.swift",
        "Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift",
        "Tests/MLXFastTests/QwenMTPVerbTests.swift",
        "Tests/MLXFastTests/QwenMTPRollbackContractTests.swift",
        "Tests/MLXFastTests/QwenMTPTrustedContextGuardTests.swift",
    ]

    // MARK: - NEGATIVE: the guard still catches every retired name

    /// Each retired name must still be caught by the guard's own check. This is
    /// the half that keeps the positive case honest: "no retired name found" is
    /// only evidence if the finder still finds them.
    @Test
    func everyRetiredMTPNameStillTripsTheGuard() throws {
        #expect(
            !S.retiredMTPNames.isEmpty,
            "the retired-name list is empty; the guard cannot catch anything"
        )
        for retired in S.retiredMTPNames {
            // A synthetic fixture in the shape the guard actually scans: a
            // retired name embedded in surrounding text, not the bare string.
            let fixture = """
                #!/usr/bin/env bash
                # a script that reaches for a dead surface
                exec mlxfast-swift \(retired) --weights weights
                """
            let caught = Self.retiredNames(in: fixture)
            #expect(
                caught.contains(retired),
                """
                the retired MTP name '\(retired)' no longer trips the guard's \
                substring check. Every entry must stay catchable: a guard that \
                has been narrowed makes every "no retired name found" result \
                below vacuous.
                """
            )
        }
    }

    /// The retired list must not SHRINK. The Qwen track was named to dodge the
    /// guard rather than to weaken it (the plan's phase 4 records that the
    /// tripwire may not need weakening at all), so a smaller list means someone
    /// took the other route -- deleting an entry instead of renaming a surface.
    @Test
    func theRetiredMTPNameListDoesNotShrink() throws {
        #expect(
            S.retiredMTPNames.count >= 12,
            """
            DFlashGateTextSupport.retiredMTPNames now holds \
            \(S.retiredMTPNames.count) entries, fewer than the 12 that were \
            retired with the Gemma/Laguna MTP track. The Qwen-MTP track is named \
            to PASS this guard unweakened; if a new surface collides, rename the \
            surface, do not delete the entry.
            """
        )
        // Duplicates would inflate the count above without widening coverage.
        #expect(
            Set(S.retiredMTPNames).count == S.retiredMTPNames.count,
            "retiredMTPNames contains duplicate entries"
        )
    }

    /// THE SUFFIX RULE, pinned. Prefixing a retired name with this track's
    /// prefix does NOT produce a new name: the retired string survives as a
    /// substring and the guard still fires. This is the single most repeatable
    /// mistake available here, and it is why this track's workflow filename does
    /// not put "-benchmark" directly after "mtp-" and its untimed verb does not
    /// end in "-probe": prefixing either of those with "qwen-" would have left a
    /// retired string intact inside the new name.
    @Test
    func retiredNamesAreNotClearedByAQwenPrefix() throws {
        for retired in S.retiredMTPNames {
            for prefix in ["qwen-", "qwen3.6-", "qwen36-", "qwen3.8-", "qwen38-"] {
                let prefixed = prefix + retired
                #expect(
                    !Self.retiredNames(in: prefixed).isEmpty,
                    """
                    '\(prefixed)' was expected to STILL trip the guard: prefixing \
                    a retired name does not clear it, because the check is a \
                    substring match. If this ever passes, the matching rule \
                    changed and every name in newSurfaceNames must be re-derived.
                    """
                )
            }
        }
    }

    // MARK: - POSITIVE: nothing this track introduces names a retired surface

    /// Every enumerated new surface name is free of every retired name.
    @Test
    func newQwenMTPSurfaceNamesCarryNoRetiredMTPName() throws {
        for (label, name) in Self.newSurfaceNames {
            let collisions = Self.retiredNames(in: name)
            #expect(
                collisions.isEmpty,
                """
                the \(label) '\(name)' contains the retired MTP name(s) \
                \(collisions.joined(separator: ", ")). Rename the surface -- \
                note that a "qwen-" prefix does not clear a retired name, only \
                changing what FOLLOWS "mtp-" does.
                """
            )
        }
    }

    /// The four checked-in files carry no retired name anywhere in their text.
    /// This is the file-level twin of `localDFlashScriptsNameNoRetiredMTPSurface`
    /// for this track, and it catches names the enumeration above forgot.
    @Test
    func newQwenMTPFilesCarryNoRetiredMTPName() throws {
        for path in Self.newSurfaceFiles {
            let body = try S.text(path)
            let collisions = Self.retiredNames(in: body)
            #expect(
                collisions.isEmpty,
                """
                \(path) names the retired MTP surface(s) \
                \(collisions.joined(separator: ", ")). Every one of these is a \
                subcommand that does not exist, a file that was deleted, or a \
                track id nothing will score.
                """
            )
        }
    }

    // MARK: - Anti-vacuity: the enumeration describes the real files

    /// An enumeration of names is worth nothing if the files use different
    /// ones. Pin each load-bearing name to the file that must actually contain
    /// it, so renaming a surface without updating this list fails HERE rather
    /// than passing a check of strings nobody uses.
    @Test
    func theEnumeratedNamesAreTheOnesTheFilesUse() throws {
        let workflow = try S.text(".github/workflows/qwen-mtp-ranked-benchmark.yml")
        let provisionWorkflow = try S.text(
            ".github/workflows/qwen-mtp-provision-goldens.yml")
        let probeWorkflow = try S.text(".github/workflows/qwen-mtp-r2-key-probe.yml")
        let benchmarkRunner = try S.text("benchmark-qwen-mtp.sh")
        let setupRunner = try S.text("setup-qwen-mtp.sh")
        let manifest = try S.text("benchmark.qwen-mtp.json")
        let cli = try S.text("Sources/MLXFastCLI/main.swift")
        let workerCLI = try S.text("Sources/MLXFastRuntimeWorkerCLI/main.swift")
        let workerHarness = try S.text(
            "Sources/MLXFastHarness/QwenRuntimeMTPWorker.swift")

        let expectations: [(needle: String, haystack: String, file: String)] = [
            ("qwen3.8-27b-mtp-v1", workflow, "the workflow"),
            ("qwen3.8-27b-mtp-v1", manifest, "the manifest"),
            ("qwen3.8-27b-mtp-v1", benchmarkRunner, "the local benchmark runner"),
            ("mlxfast-challenge-dev-qwen38-mtp", manifest, "the manifest"),
            ("MLXFAST_QWEN_MTP_CALIBRATION_READY", workflow, "the workflow"),
            ("measure-qwen-mtp-job.sh", workflow, "the workflow"),
            ("/opt/bench-runner/state/qwen3.8-27b-mtp-v1", workflow, "the workflow"),
            ("m5-qwen38-27b-mtp", workflow, "the workflow"),
            ("mtp-verify", workflow, "the workflow"),
            ("mtp-timed", workflow, "the workflow"),
            ("mtp-verify", benchmarkRunner, "the local benchmark runner"),
            ("mtp-timed", benchmarkRunner, "the local benchmark runner"),
            ("benchmark-qwen-mtp.sh", manifest, "the manifest"),
            ("setup-qwen-mtp.sh", manifest, "the manifest"),
            ("qwen-mtp-ranked-benchmark.yml", manifest, "the manifest"),
            ("qwen-mtp-ranked-benchmark.yml", benchmarkRunner, "the local benchmark runner"),
            ("setup-qwen-mtp.sh", benchmarkRunner, "the local benchmark runner"),
            ("fixtures/qwen3_8_27b_mtp_head.sha256", setupRunner, "the local setup runner"),
            // Phase 2: the verbs, the worker subcommand and the canonical depth
            // flag have to be the names the shipped code actually uses.
            ("--mtp-depth", workflow, "the workflow"),
            ("--mtp-depth", benchmarkRunner, "the local benchmark runner"),
            ("--mtp-head", workflow, "the workflow"),
            ("--mtp-head", benchmarkRunner, "the local benchmark runner"),
            ("mtp-verify", cli, "the trusted CLI"),
            ("mtp-timed", cli, "the trusted CLI"),
            ("mtp-runtime-worker", workerCLI, "the runtime-worker CLI"),
            ("mtp_decode_round", workerHarness, "the worker harness"),
            ("mtp_reference_rows", workerHarness, "the worker harness"),
            ("uses_native_mtp_head", cli, "the trusted CLI"),
            ("uses_pinned_mtp_head", cli, "the trusted CLI"),
            ("row_ledger", cli, "the trusted CLI"),
            ("parity_all_ok", cli, "the trusted CLI"),
            // The trusted-context guards: each credentialled workflow must name
            // its own guard by path. A rename that misses the `run:` line fails
            // the job with exit 127, which is how both of these shipped before
            // the scripts existed at all.
            (
                ".github/scripts/enforce-trusted-qwen-mtp-provision-workflow.sh",
                provisionWorkflow,
                "the provisioning workflow"
            ),
            (
                ".github/scripts/enforce-trusted-qwen-mtp-r2-probe-workflow.sh",
                probeWorkflow,
                "the r2-key-probe workflow"
            ),
        ]
        for expectation in expectations {
            #expect(
                expectation.haystack.contains(expectation.needle),
                """
                \(expectation.file) no longer contains '\(expectation.needle)'. \
                Either the surface was renamed without updating \
                QwenMTPTrackNamingTests.newSurfaceNames, or this list is \
                checking a string nothing uses -- both make the naming guard \
                vacuous.
                """
            )
        }

        // Every needle above must itself be an enumerated surface name (or a
        // prefix of one), so the two lists cannot drift into checking disjoint
        // sets of strings.
        let enumerated = Set(Self.newSurfaceNames.map(\.name))
        for expectation in expectations {
            #expect(
                enumerated.contains(expectation.needle)
                    || enumerated.contains(where: { $0.hasSuffix(expectation.needle) })
                    || enumerated.contains(where: { $0.hasPrefix(expectation.needle) }),
                """
                '\(expectation.needle)' is checked against \(expectation.file) \
                but is not listed in newSurfaceNames, so it is never checked \
                against the retired-name list.
                """
            )
        }
    }

    /// The manifest and the workflow must agree on the track identity. A
    /// mismatch is not merely cosmetic: the workflow refuses a dispatch whose
    /// contract track_id differs from its own, and the leaderboard namespace is
    /// what keeps this track's scores out of another track's ranking.
    @Test
    func theManifestAndWorkflowAgreeOnTrackIdentity() throws {
        let manifest = try S.json("benchmark.qwen-mtp.json")
        #expect(manifest["trackId"] as? String == "qwen3.8-27b-mtp-v1")
        #expect(manifest["name"] as? String == "mlxfast-challenge-dev-qwen38-mtp")
        #expect(manifest["staticReviewTrackId"] as? String == "qwen3.8-27b-mtp-v1")
        let leaderboard = try #require(manifest["leaderboard"] as? [String: Any])
        #expect(leaderboard["namespace"] as? String == "qwen3.8-27b-mtp-v1")
        let runner = try #require(manifest["runner"] as? [String: Any])
        #expect(runner["workflow"] as? String == "qwen-mtp-ranked-benchmark.yml")
        let benchmarkCommand = try #require(manifest["benchmarkCommand"] as? [String])
        #expect(
            benchmarkCommand.contains { $0.contains("./benchmark-qwen-mtp.sh") },
            "benchmarkCommand must drive ./benchmark-qwen-mtp.sh, got \(benchmarkCommand)"
        )

        let workflowEnvironment = try S.jobEnvironment(
            try S.text(".github/workflows/qwen-mtp-ranked-benchmark.yml")
        )
        #expect(workflowEnvironment["MLXFAST_QWEN_MTP_TRACK_ID"] == "qwen3.8-27b-mtp-v1")
        #expect(
            workflowEnvironment["MLXFAST_QWEN_MTP_TRACK_NAME"]
                == manifest["name"] as? String
        )
        // The editable-surface contract the workflow passes to BOTH the surface
        // enforcement and the overlay must be THIS manifest. Naming a different
        // file in one of the two silently drops files the other accepted.
        #expect(
            workflowEnvironment["MLXFAST_QWEN_MTP_EDITABLE_SURFACE_CONTRACT"]
                == "benchmark.qwen-mtp.json"
        )
    }

    /// The track is ARMED on Qwen 3.8.
    ///
    /// This assertion has now pointed both ways more than once, which is the whole
    /// point of it: it pinned the inert posture through the 3.6 draft, was inverted
    /// to pin the ARMED posture at the 2026-08-13 go-live, was inverted back for the
    /// 3.6 -> 3.8 model cutover, and is inverted again here now that the 3.8
    /// bring-up has completed. Each switch below is a one-character edit away from
    /// flipping, so DISARMING the track means editing this test in the same commit
    /// as the flags -- deliberate friction, in both directions
    /// (NEW-MODEL-BRINGUP 7.5).
    @Test
    func theQwenMTPTrackIsArmedOnQwen38() throws {
        let workflow = try S.text(".github/workflows/qwen-mtp-ranked-benchmark.yml")
        let environment = try S.jobEnvironment(workflow)

        // 1. The calibration-ready interlock is a hard-pinned literal "1" --
        //    never an expression, never a dispatch input. The 3.8 baseline is
        //    installed and calibrated on BOTH ranked boxes (six gated sessions,
        //    all ACCEPT, 2026-08-14), so it now attests to something real.
        #expect(environment["MLXFAST_QWEN_MTP_CALIBRATION_READY"] == "1")
        #expect(!workflow.contains("MLXFAST_QWEN_MTP_CALIBRATION_READY: ${{"))
        // ... and the runbook-shaped refusal message still exists for the closed
        // direction, scanned on the comment-stripped view so a mention in prose
        // cannot satisfy it.
        #expect(
            S.executable(workflow).contains(
                "Qwen 3.8 MTP model-derived fixtures and final-SHA baseline are not calibrated"
            )
        )

        // 2. THE TRUSTED CONTRACT IS ARMED.
        //
        //    "Enforce Qwen-MTP track enablement" refuses a RANKED dispatch while
        //    EITHER of these is false. Both are true again here: the fixture's
        //    measured content is now entirely 3.8 -- the timed pool was re-measured
        //    on the 3.8 tower across six gated sessions on both ranked boxes, and
        //    calibration.expected_raw_median carries that measurement rather than
        //    the 3.6 k-matrix seed. Disarming MUST edit this test in the same commit.
        let contract = try S.json("fixtures/qwen3_8_27b_mtp_track.json")
        #expect(
            contract["official_scoring_enabled"] as? Bool == true,
            "official_scoring_enabled must be true now the 3.8 bring-up has completed"
        )
        let baseline = contract["reference_baseline"] as? [String: Any]
        #expect(
            baseline?["publication_allowed"] as? Bool == true,
            "reference_baseline.publication_allowed must be true now the 3.8 baseline is measured"
        )
        #expect(
            contract["track_id"] as? String == "qwen3.8-27b-mtp-v1",
            "the contract track_id must match the workflow's track"
        )
        // The pool the normalised floor divides by: 8 entries, all distinct.
        let pool = contract["timed_prompt_pool"] as? [[String: Any]]
        #expect(pool?.count == 8, "the timed prompt pool must carry 8 entries")
        #expect(
            Set((pool ?? []).compactMap { $0["sha256"] as? String }).count == 8,
            "the timed prompt pool must name 8 DISTINCT targets"
        )
        #expect(
            (pool ?? []).allSatisfy {
                ($0["noop_decode_speedup"] as? Double).map { $0 > 0 } ?? false
            },
            "every pool entry needs a positive measured no-op reference"
        )

        //    BOTH pending markers must remain spelled in the workflow, because the
        //    pin-completeness gate greps for them by name. QWEN-MTP-PENDING-ORGANIZER
        //    is the 3.6-era marker ("this organizer artifact was never produced")
        //    and is retained unweakened; QWEN38-PENDING-RELEASE is the cutover
        //    marker ("this WAS a resolved value that the model change
        //    invalidated"). Deleting either would delete the gate that catches a
        //    placeholder reaching a download, a comparison or a score.
        //
        //    Two narrower markers (QWEN38-PENDING-HEAD-UPLOAD and
        //    QWEN38-PENDING-BACKBONE-PUBLISH) existed for a few hours on
        //    2026-08-14 while the two checkpoint halves awaited publication;
        //    both RESOLVED to real revisions the same day and their gate
        //    classes were removed. The gate's check_revision guard survives
        //    them: it now holds the published shas to 40-hex exactly.
        let marker = "QWEN-MTP-PENDING-ORGANIZER"
        let releaseMarker = "QWEN38-PENDING-RELEASE"
        #expect(
            workflow.contains("PENDING_MARKER: \(marker)"),
            "the pin-completeness gate lost its organizer marker definition"
        )
        #expect(
            workflow.contains("RELEASE_PENDING_MARKER: \(releaseMarker)"),
            "the pin-completeness gate lost its 3.8 release marker definition"
        )
        // Both markers must land in `pending`, and `pending` must still
        // exit 1. Widening WHICH pins may be pending must never widen what may
        // RUN.
        let gate = try S.stepBody(workflow, "Enforce Qwen-MTP organizer pin completeness")
        #expect(gate.contains("is_pending()"))
        #expect(gate.contains("${PENDING_MARKER}") && gate.contains("${RELEASE_PENDING_MARKER}"))
        #expect(gate.contains("if ((${#pending[@]} + ${#malformed[@]})); then"))
        // Both revision pins stay gate-checked after the publish: check_revision
        // now enforces that each is EXACTLY the 40-hex published sha shape, so
        // a truncated or decorated revision cannot ride in on a later edit.
        for revisionPin in [
            "MLXFAST_QWEN_MTP_CHECKPOINT_REVISION",
            "MLXFAST_QWEN_MTP_HEAD_REVISION",
        ] {
            #expect(
                gate.contains("check_revision \(revisionPin)"),
                "the pin-completeness gate does not check \(revisionPin)"
            )
        }
        // The four whole-manifest shape pins are checked here now that they can
        // carry a marker; before the adoption they were plain integers.
        for shapePin in [
            "MLXFAST_QWEN_MTP_TARGET_MANIFEST_RECORDS",
            "MLXFAST_QWEN_MTP_TARGET_MANIFEST_BYTES",
            "MLXFAST_QWEN_MTP_HEAD_MANIFEST_RECORDS",
            "MLXFAST_QWEN_MTP_HEAD_MANIFEST_BYTES",
        ] {
            #expect(
                gate.contains("check_positive_int \(shapePin)"),
                "the pin-completeness gate does not check \(shapePin)"
            )
        }

        // 3. EVERY MEASURED / UPLOADED PIN IS STILL PENDING. This is the inverse
        //    of the list the armed 3.6 posture pinned: each of these was a real,
        //    uploaded, digest-verified value, and each is reset because it
        //    describes the 3.6 tower. A pin here that is NOT pending is a 3.6
        //    value that would be scored under the 3.8 track id.
        //
        //    The BACKBONE checkpoint pins are deliberately NOT in this list any
        //    more -- they resolved on 2026-08-14 and are asserted RESOLVED just
        //    below. Everything that still needs the 3.8 tower to be run, or
        //    someone else's upload to land, stays here.
        //
        //    The digest and the R2 object key that embeds it move together or not
        //    at all -- that lockstep is what makes a key/digest mismatch
        //    impossible to introduce quietly, and it is preserved here by
        //    requiring the key to be pending too.
        for pin in [
            // Backbone manifest shape: the manifest file is a header-only stub
            // whose per-file body is generated from the pinned snapshot. The
            // adopted snapshot is KNOWN to carry 11 files / ~15.2 GB and those
            // counts are deliberately not typed in, because this pin is defined
            // as what the manifest body sums to.
            "MLXFAST_QWEN_MTP_TARGET_MANIFEST_RECORDS",
            "MLXFAST_QWEN_MTP_TARGET_MANIFEST_BYTES",
            "MLXFAST_QWEN_MTP_CORRECTNESS_GOLDEN_SHA256",
            "MLXFAST_QWEN_MTP_CORRECTNESS_GOLDEN_BYTES",
            "MLXFAST_RAW_CORRECTNESS_GOLDEN_SHA256",
            "MLXFAST_RAW_CORRECTNESS_GOLDEN_BYTES",
            "MLXFAST_GPQA_REFERENCE_SHA256",
            "MLXFAST_GPQA_REFERENCE_BYTES",
            // The public drift pins are NOT here. They were the ONE class the
            // 3.6 gate asserted could never be pending -- computed from bytes in
            // this repository, therefore always computable -- and they joined
            // this list only while those bytes were 3.6 captures. The 3.8
            // goldens are checked in, so they are asserted RESOLVED just below
            // and the gate's own rule is unsuspended.
        ] {
            let value = try #require(
                environment[pin],
                "the workflow declares no \(pin)"
            )
            #expect(
                value.contains(releaseMarker),
                """
                \(pin) is '\(value)', not \(releaseMarker). Every measured or \
                uploaded pin must stay pending until it is re-measured against \
                the Qwen 3.8 tower; a resolved-looking 3.6 value here is a 3.6 \
                artifact about to be scored under the 3.8 track id.
                """
            )
        }

        // 3-PUBLIC. THE PUBLIC DRIFT PINS ARE RESOLVED, and each one must equal
        //     what the checked-in fixture actually hashes to. Deriving the
        //     expectation from the BYTES rather than restating a literal is the
        //     whole point: regenerating a fixture without moving its pin fails
        //     here, and moving a pin without the fixture fails here too.
        for (pathPin, shaPin, bytesPin) in [
            (
                "MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_PATH",
                "MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_SHA256",
                "MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_BYTES"
            ),
            (
                "MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_1024_PATH",
                "MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_1024_SHA256",
                "MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_1024_BYTES"
            ),
        ] {
            let path = try #require(environment[pathPin])
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            let digest = SHA256.hash(data: data)
                .map { String(format: "%02x", $0) }
                .joined()
            #expect(
                environment[shaPin] == digest,
                "\(shaPin) does not match the checked-in \(path) (\(digest))"
            )
            #expect(
                environment[bytesPin] == String(data.count),
                "\(bytesPin) does not match the checked-in \(path) (\(data.count))"
            )
            for forbidden in [marker, releaseMarker] {
                #expect(
                    !(environment[shaPin] ?? "").contains(forbidden)
                        && !(environment[bytesPin] ?? "").contains(forbidden),
                    """
                    a public fixture pin is computed from bytes in this \
                    repository and may never carry \(forbidden).
                    """
                )
            }
        }

        // 3a. THE MTP HEAD'S REVISION AND MANIFEST SHAPE ARE RESOLVED, not
        //     pending. They carried QWEN38-PENDING-HEAD-UPLOAD until the
        //     2026-08-14 publish and were repinned onto 26a328e0 when the head
        //     was republished with the two config files verifyHeadTree
        //     requires. The narrow marker and its gate class are gone; what
        //     replaces them is an EXACT assertion, so a repin that half-lands
        //     (a new revision against an old manifest shape, or the reverse)
        //     fails here rather than at verify_cache on the runner.
        //
        //     `headMarker` and `backboneMarker` were deleted with their gate
        //     classes but their USES survived here and at 3b, so this file did
        //     not compile. Both are replaced with the published values.
        #expect(
            environment["MLXFAST_QWEN_MTP_HEAD_REVISION"]
                == "26a328e070875b0314d652a039b6b59902690f03",
            "the published MTP head revision must be pinned exactly"
        )
        #expect(environment["MLXFAST_QWEN_MTP_HEAD_MANIFEST_RECORDS"] == "4")
        #expect(environment["MLXFAST_QWEN_MTP_HEAD_MANIFEST_BYTES"] == "849406438")

        // 3b. BOTH REPOSITORIES ARE DECIDED AND BOTH REVISIONS ARE PUBLISHED, and
        //     each half is pinned so that "decided" cannot decay into "whatever
        //     the last edit left". The backbone is OUR OWN mlx-0.32.0
        //     conversion of the official bf16 base Qwen/Qwen3.8-27B @
        //     1d4bf0f2, which replaced a third-party conversion adopted earlier
        //     on 2026-08-14 and terminated by the operator's validation
        //     kill-switch (994 of 1,847 tensors differed numerically from the
        //     reconversion). The geometry reading survived that swap, which is
        //     why MLXFAST_EXPECTED_NUM_LAYERS below is still a 3.8 fact rather
        //     than a carried 3.6 one: the terminating cross-check was numerical,
        //     not structural.
        let checkpointRepository = "EigenLabs/Qwen3.8-27B-4bit"
        let checkpointRevision = "eda45ab47f465d08d6558f0353a2346e2eb9d5b3"
        #expect(environment["MLXFAST_QWEN_MTP_CHECKPOINT_REPO"] == checkpointRepository)
        #expect(environment["MLXFAST_QWEN_MTP_CHECKPOINT_REVISION"] == checkpointRevision)
        #expect(
            environment["MLXFAST_QWEN_MTP_HEAD_REPO"] == "EigenLabs/Qwen3.8-27B-MTP-bf16",
            "the MTP head repository is decided and must be named, not left pending"
        )
        // The compiled constants are the SAME artifact. The cutover commit left
        // them at 3.6 while the workflow read a placeholder, deliberately; the
        // adoption closed that split rather than widening it, so a future repin
        // that moves one and not the other fails here.
        #expect(MLXFastConstants.referenceModelRepository == checkpointRepository)
        #expect(MLXFastConstants.referenceModelRevision == checkpointRevision)

        // The cache paths spell the revision, because the host preflight asserts
        // the snapshot leaf IS the revision. Spelled twice, so they must stay
        // equal -- and MLXFAST_REFERENCE_DIR is the third spelling of the same
        // directory, asserted equal by that same preflight. The equality is what
        // matters; a partial repin that moved one leaf and not the others fails
        // right here.
        for pathPin in ["MLXFAST_QWEN_MTP_TARGET_DIR", "MLXFAST_REFERENCE_DIR"] {
            let value = try #require(environment[pathPin])
            #expect(
                value.hasSuffix("/snapshots/\(checkpointRevision)"),
                "\(pathPin) ('\(value)') must end in the pinned revision"
            )
            #expect(
                value.contains("models--EigenLabs--Qwen3.8-27B-4bit/"),
                "\(pathPin) ('\(value)') must name the pinned repository cache dir"
            )
        }
        #expect(
            environment["MLXFAST_QWEN_MTP_TARGET_DIR"] == environment["MLXFAST_REFERENCE_DIR"],
            "the MTP target and the reused serial gates must name the same checkpoint"
        )
        // The goldens-provisioning job FREEZES what the ranked job then scores
        // against, so its snapshot path is the same string or the track is
        // scoring against a checkpoint it did not provision from.
        let provisioning = try S.jobEnvironment(
            try S.text(".github/workflows/qwen-mtp-provision-goldens.yml"))
        #expect(
            provisioning["MLXFAST_QWEN_MTP_TARGET_DIR"]
                == environment["MLXFAST_QWEN_MTP_TARGET_DIR"],
            "the provisioning job must mint goldens from the ranked checkpoint"
        )

        // The hidden R2 object keys are SHA-addressed, so they are pending for
        // the same reason and in the same commit as the digests above.
        //
        // They are declared at STEP level, never at job level -- deliberately, so
        // untrusted bench phases never receive the object names -- which is why
        // they are read out of the step bodies rather than out of `environment`.
        for (step, keyPins) in [
            (
                "Prepare hidden correctness golden",
                ["MLXFAST_CORRECTNESS_GOLDEN_R2_PATH", "MLXFAST_GPQA_R2_PATH"]
            ),
            (
                "Prepare hidden Qwen-MTP goldens",
                ["MLXFAST_QWEN_MTP_CORRECTNESS_GOLDEN_R2_PATH"]
            ),
        ] {
            let body = try S.stepBody(workflow, step)
            for keyPin in keyPins {
                let line = try #require(
                    body
                        .split(separator: "\n")
                        .map(String.init)
                        .first { $0.trimmingCharacters(in: .whitespaces)
                            .hasPrefix("\(keyPin):") },
                    "step '\(step)' declares no \(keyPin)"
                )
                let value = line
                    .split(separator: ":", maxSplits: 1)
                    .map(String.init)[1]
                    .trimmingCharacters(in: .whitespaces)
                #expect(
                    value.contains(releaseMarker),
                    "\(keyPin) is '\(value)', not \(releaseMarker); it must stay pending"
                )
                #expect(
                    value.hasPrefix("correctness_prompts/qwen3.8-27b-mtp-v1/"),
                    """
                    \(keyPin) ('\(value)') must sit under the NEW track \
                    namespace. Resolving a 3.8 object under the retired \
                    correctness_prompts/qwen3.6-27b-mtp-v1/ prefix would mix two \
                    incomparable populations in one bucket.
                    """
                )
            }
        }

        // 4. The values that are NOT pending are the ones a model change cannot
        //    invalidate, plus the ones deliberately carried at their 3.6
        //    measurement behind the closed gate (QWEN38-VERIFY-AT-RELEASE). They
        //    are pinned so that "carried deliberately" cannot decay into
        //    "forgotten": changing any of them is a decision, not a drive-by.
        for (pin, expected) in [
            ("MLXFAST_METAL_TOOLCHAIN_IDENTIFIER", "com.apple.dt.toolchain.Metal.32023.883"),
            // VERIFIED 2026-08-14, no longer a carried 3.6 measurement: the
            // adopted 3.8 config.json was read and the tower is byte-identical
            // in geometry to the 3.6 one -- 64 layers on the same 4-layer
            // hybrid repeat, vocab 248320, hidden 5120, 24 attention heads / 4
            // KV heads, head_dim 256. Equal to MLXFastConstants.numHiddenLayers
            // by the pin below.
            ("MLXFAST_EXPECTED_NUM_LAYERS", "64"),
            // (The four manifest shape pins used to sit here at their 3.6
            // values. They are pending markers now -- see 3 and 3a above.)
            // QWEN38-VERIFY-AT-RELEASE, still a 3.6 measurement:
            ("MLXFAST_SEMANTIC_GPQA_MIN_PASS", "8"),
            // The floor and ceiling are POLICY bounds ("do not regress serial by
            // more than 10%"; "anything above 5.0x is a measurement fault"), not
            // measurements, so they survive the model change as decisions. They
            // are still marked QWEN38-VERIFY-AT-RELEASE in the workflow because
            // the 0.90 choice was justified by a 3.6 launch requirement. The
            // ceiling was raised 3.0 -> 5.0 by operator decision 2026-08-17
            // (per-pair wrapper bound raised 5.0 -> 8.0 in the same decision).
            ("MLXFAST_QWEN_MTP_DECODE_SPEEDUP_FLOOR", "0.90"),
            ("MLXFAST_QWEN_MTP_DECODE_SPEEDUP_CEILING", "5.0"),
        ] {
            #expect(
                environment[pin] == expected,
                "\(pin) is \(environment[pin] ?? "unset"), expected \(expected)"
            )
        }
    }

    /// `benchmark.qwen-mtp.json`'s own `decodeSpeedupFloorNote` says the field
    /// "must be pinned equal to" the workflow env, and that the workflow is the
    /// enforcing site. Nothing checked it, and the manifest sat at `null` while
    /// the workflow had already resolved to 0.95. Modelled on the DFlash
    /// equivalent, which exists because the same gap opened there.
    @Test
    func theManifestDeclaresTheDecodeFloorTheWorkflowEnforces() throws {
        let environment = try S.jobEnvironment(
            try S.text(".github/workflows/qwen-mtp-ranked-benchmark.yml"))
        let raw = try #require(
            environment["MLXFAST_QWEN_MTP_DECODE_SPEEDUP_FLOOR"],
            "the Qwen-MTP workflow must declare the floor it enforces"
        )
        let enforced = try #require(
            Double(raw),
            "MLXFAST_QWEN_MTP_DECODE_SPEEDUP_FLOOR is not a number: \(raw)"
        )

        let manifest = try S.json("benchmark.qwen-mtp.json")
        let scoring = try #require(manifest["scoring"] as? [String: Any])
        let declared = try #require(
            scoring["decodeSpeedupFloor"] as? Double,
            "benchmark.qwen-mtp.json declares no numeric decodeSpeedupFloor"
        )
        #expect(
            enforced == declared,
            """
            MLXFAST_QWEN_MTP_DECODE_SPEEDUP_FLOOR=\(raw) is the only value the \
            workflow actually rejects on, while benchmark.qwen-mtp.json \
            declares \(declared).
            """
        )
    }

    /// A track cannot be ranking submissions while its own manifest says the
    /// token-fidelity gate is unimplemented. The DFlash suite pins exactly this
    /// invariant for its manifest; the Qwen manifest shipped "pending" against a
    /// fixture that already said "implemented", which go-live had to reconcile.
    @Test
    func anEnabledTrackDeclaresAnImplementedTokenFidelityGate() throws {
        let manifest = try S.json("benchmark.qwen-mtp.json")
        let scoring = try #require(manifest["scoring"] as? [String: Any])
        let gateStatus = try #require(scoring["tokenFidelityGateStatus"] as? String)

        let contract = try S.json("fixtures/qwen3_8_27b_mtp_track.json")
        let proposed = try #require(contract["proposed_scoring"] as? [String: Any])
        #expect(
            proposed["token_fidelity_gate_status"] as? String == gateStatus,
            "fixture token_fidelity_gate_status disagrees with the manifest"
        )

        if contract["official_scoring_enabled"] as? Bool == true {
            #expect(
                gateStatus == "implemented",
                """
                Qwen-MTP official scoring is enabled while \
                tokenFidelityGateStatus is '\(gateStatus)'. Ranked scoring \
                without a proven fidelity gate is exactly the state the \
                enablement interlock exists to prevent.
                """
            )
        }
    }
}

/// The Qwen-MTP track carries its own calibrated constants rather than reusing
/// the unprefixed ones, which belong to the Poolside/Laguna serial calibration
/// that the LIVE DFlash track compiles against. That separation is only
/// trustworthy if it cannot silently collapse, in either direction: a future
/// edit that "tidies up" by pointing the Qwen workflow at the shared constant,
/// or that raises the shared constant to match Qwen, would retune a live track
/// that has not been re-derived. Both directions are pinned here.
@Suite
struct QwenMTPTrackScopedConstantsTests {
    private typealias S = DFlashGateTextSupport

    @Test
    func theQwenWorkflowMirrorsTheTrackScopedGPQAFloor() throws {
        let environment = try S.jobEnvironment(
            try S.text(".github/workflows/qwen-mtp-ranked-benchmark.yml"))
        #expect(
            environment["MLXFAST_SEMANTIC_GPQA_MIN_PASS"]
                == String(MLXFastConstants.qwenMTPSemanticGPQAMinPassCount),
            """
            the Qwen workflow's semantic GPQA floor must mirror \
            qwenMTPSemanticGPQAMinPassCount \
            (\(MLXFastConstants.qwenMTPSemanticGPQAMinPassCount)), not the \
            shared DFlash-calibrated semanticGPQAMinPassCount \
            (\(MLXFastConstants.semanticGPQAMinPassCount))
            """
        )
    }

    /// The two floors are ALLOWED to differ — that is the point — but the
    /// shared one must not drift onto the Qwen value by accident.
    @Test
    func theSharedGPQAFloorStaysAtTheDFlashCalibration() throws {
        #expect(
            MLXFastConstants.semanticGPQAMinPassCount == 7,
            """
            the shared semanticGPQAMinPassCount moved. It is the LIVE DFlash \
            track's calibrated floor, and this test is now the only thing tying \
            it to the DFlash workflow env: the suite that used to do that \
            (reusedGateCalibrationInTheDFlashWorkflowMatchesConstants, in \
            DFlashTrackTests.swift) was deleted at the repo split. Re-derive \
            that track before changing it; the Qwen track has its own constant.
            """
        )
        let dflash = try S.jobEnvironment(try S.text(S.dflashWorkflowPath))
        #expect(dflash["MLXFAST_SEMANTIC_GPQA_MIN_PASS"] == "7",
                "the DFlash workflow floor must not follow the Qwen re-derivation")
        // The shared gate script serves both tracks, so its default must stay
        // the conservative inherited value; the binding value is the job env.
        let gate = try S.text(".github/scripts/run-semantic-gpqa-gate.sh")
        #expect(gate.contains("MLXFAST_SEMANTIC_GPQA_MIN_PASS:-7}"),
                "the shared gate script default must stay 7")
    }

    /// The scored decode denominator must not be confused with the shared
    /// Poolside value, and the prefill figure must stay marked unscored.
    @Test
    func theTrackScopedBaselineConstantsAreDistinctAndDocumented() throws {
        #expect(
            MLXFastConstants.qwenMTPOfficialBaselineDecodeSecondsPerToken
                != MLXFastConstants.officialBaselineDecodeSecondsPerToken,
            "the Qwen decode denominator must not collapse onto the Poolside one"
        )
        #expect(MLXFastConstants.qwenMTPOfficialBaselineDecodeSecondsPerToken > 0)
        #expect(MLXFastConstants.qwenMTPOfficialBaselinePrefillSecondsPerToken > 0)

        let constants = try S.text("Sources/MLXFastCore/Constants.swift")
        // Prefill is UNSCORED on this decode-only track; the prose must say so
        // wherever the constant is quoted (RUNBOOK 3.6).
        #expect(constants.contains("prefill_component: \"none\""),
                "the prefill constant's provenance must state the sealed contract")
        #expect(constants.lowercased().contains("unscored"))
    }
}

/// The no-op references were re-measured 2026-08-13 as N-pair means after the
/// single-shot 6a3bee9-era values were found to carry both single-pair noise and
/// a +6.58% thermal-regime delta. Timing tolerance on this track is DERIVED from
/// measured per-prompt spread, never inherited, so every entry must carry the
/// provenance that makes the derivation auditable rather than folklore.
@Suite
struct QwenMTPNoopReferenceProvenanceTests {
    private typealias S = DFlashGateTextSupport

    @Test
    func everyPoolEntryCarriesItsMeasuredSpreadAndPairCount() throws {
        let contract = try S.json("fixtures/qwen3_8_27b_mtp_track.json")
        let pool = try #require(contract["timed_prompt_pool"] as? [[String: Any]])
        #expect(pool.count == 8)
        for e in pool {
            let path = (e["r2_path"] as? String) ?? "<unknown>"
            let pairs = try #require(
                e["noop_decode_speedup_pairs"] as? Int,
                "\(path) has no noop_decode_speedup_pairs"
            )
            #expect(
                pairs >= 3,
                "\(path) was averaged over \(pairs) pairs; the wrapper's min-pairs floor is 3"
            )
            let spread = try #require(
                e["noop_decode_speedup_spread_pct"] as? Double,
                "\(path) has no noop_decode_speedup_spread_pct"
            )
            #expect(
                spread > 0 && spread < 1.0,
                """
                \(path) spread \(spread)% is outside the measured envelope \
                (0.14-0.44% across the pool). A spread at or above 1% means the \
                mean is not trustworthy -- add pairs before pinning it, as \
                medicine required.
                """
            )
        }
    }

    /// The note must record WHY the values moved, because a future reader
    /// comparing them against the 6a3bee9-era numbers will otherwise read the
    /// difference as a regression.
    @Test
    func theNoteRecordsTheRegimeDeltaAndTheRejectionRateFinding() throws {
        let contract = try S.json("fixtures/qwen3_8_27b_mtp_track.json")
        let note = try #require(contract["noop_decode_speedup_note"] as? String)
        #expect(note.contains("6.58%"), "the thermal-regime delta is not recorded")
        #expect(note.contains("0.037994794617407023"),
                "the note must tie the regime to the pinned serial calibration")
        // Token exactness is explicitly out of scope for these tolerances.
        #expect(note.lowercased().contains("token exactness"))
        // The honest version of the rejection-rate finding: the strong
        // correlation is with OLD reference error, not with current spread.
        #expect(note.contains("-0.75") && note.contains("-0.26"))
    }
}

// MARK: - The go-live runbook the workflows point at

/// `.github/workflows/qwen-mtp-provision-goldens.yml` refers operators to a
/// "Qwen-MTP go-live runbook" from four places, including the "step A" and
/// "step B" cited in error messages an operator only ever sees when a dispatch
/// fails closed. That reference dangled at a file that did not exist until
/// go-live on 2026-08-13. This pins its existence and the sections the workflow
/// names, so the pointer cannot rot again -- the same guard the DFlash track
/// grew after its runbook reference dangled for a week.
@Suite
struct QwenMTPGoLiveRunbookTests {
    private static let path = "docs/qwen-mtp-go-live-runbook.md"

    @Test
    func runbookExistsAndCoversTheStepsTheWorkflowsCite() throws {
        let runbook = try String(contentsOfFile: Self.path, encoding: .utf8)
        // Steps named in the provisioning workflow's own error text.
        #expect(runbook.contains("Step A"))
        #expect(runbook.contains("Step B"))
        // The two trusted-contract fields the enablement guard requires.
        #expect(runbook.contains("official_scoring_enabled"))
        #expect(runbook.contains("publication_allowed"))
        // The blocking prerequisite the pool-selection step fails closed on.
        #expect(runbook.contains("timed_prompt_pool"))
        // The interlock this track's go-live actually turns.
        #expect(runbook.contains("MLXFAST_QWEN_MTP_CALIBRATION_READY"))
    }

    /// The runbook must carry the measured facts and the known limits an
    /// operator accepts by flipping the switch, not just the mechanical steps.
    @Test
    func runbookRecordsTheLimitsBeingAcceptedAtGoLive() throws {
        let runbook = try String(contentsOfFile: Self.path, encoding: .utf8)
        // The pool's raw no-op spread -- the reason the score is normalised.
        // Re-measured 2026-08-13 as N-pair means; the single-shot 6a3bee9-era
        // span (0.756-1.0915) is superseded.
        #expect(runbook.contains("0.7623"))
        #expect(runbook.contains("1.0845"))
        // The calibration acceptance window must be DERIVED and shown, not
        // inherited from the serial era.
        #expect(runbook.contains("0.992"))
        #expect(runbook.lowercased().contains("common-mode"))
        // Prefill is measured but UNSCORED on this decode-only track.
        #expect(runbook.contains("prefill_component"))
        // The GPQA floor's honest limitation: a constant-"A" answerer scores
        // exactly the floor, so the floor does not reject it.
        #expect(runbook.lowercased().contains("constant-\"a\"")
            || runbook.lowercased().contains("constant-a"))
        // The guard fails closed before the flags are flipped.
        #expect(runbook.lowercased().contains("fail-closed")
            || runbook.lowercased().contains("fails closed"))
    }

    /// The runbook must describe the scoring the track actually performs. It
    /// documented single-sampled-prompt scoring for as long as that was true;
    /// leaving that text in place after the semantics changed would make the
    /// operator-facing record the most authoritative-looking wrong answer in
    /// the tree.
    @Test
    func runbookDescribesMedianOfEightScoring() throws {
        let runbook = try String(contentsOfFile: Self.path, encoding: .utf8)
        #expect(runbook.contains("median"))
        // The rule, not just the word: 8 is even, so which median matters.
        #expect(
            runbook.contains("two central order statistics"),
            "the runbook must state the even-n median rule the score uses"
        )
        // Pair budget is per prompt now, and the runbook is where an operator
        // reads the wall-clock consequence.
        #expect(runbook.contains("pairs-per-prompt") || runbook.contains("per prompt"))
        // Pooled denominator banding -- the property that lets the INSTALLED
        // calibration keep working across this change.
        #expect(runbook.lowercased().contains("pooled"))
        // And the recorded reason the per-invocation shape was kept.
        #expect(runbook.contains("mtp_decode_begin"))
    }
}

// MARK: - Scoring semantics: median of 8

/// The operator-ratified scoring change of 2026-08-14: a ranked run times ALL
/// eight pool prompts and publishes the MEDIAN of the eight RAW
/// serial-relative speedups. SERIAL = 1.0.
///
/// This supersedes the 2026-08-13 normalised rule, which divided each prompt by
/// its pinned no-op reference and therefore made an UNMODIFIED candidate score
/// exactly 1.0 by construction -- declaring the shipped depth-2 speculative
/// decoder to be the zero point when it is measurably a ~6.5% regression
/// against true serial decode. The no-op references survive as informational
/// diagnostics and divide nothing.
///
/// Everything here reads checked-in text or runs pure arithmetic. The point of
/// the suite is that the rule is stated identically in the four places that
/// have to agree -- the contract fixture, the track manifest, the ranked
/// workflow and the operator runbook -- because the previous single-sample rule
/// was also stated in all four, and a change that updates three of them leaves
/// the fourth as a confident lie.
@Suite
struct QwenMTPScoringSemanticsTests {
    private typealias S = DFlashGateTextSupport

    private static let workflowPath =
        ".github/workflows/qwen-mtp-ranked-benchmark.yml"
    private static let fixturePath = "fixtures/qwen3_8_27b_mtp_track.json"
    private static let manifestPath = "benchmark.qwen-mtp.json"
    private static let aggregation =
        "median_of_per_prompt_raw_serial_relative_speedup"
    /// The retired rule. Pinned by name so a revert is a red test rather than a
    /// silent re-normalisation, and so the cutover-era diagnostic field that
    /// still carries the old number keeps a stable spelling.
    private static let retiredAggregation =
        "median_of_per_prompt_normalized_ratio_of_means"
    private static let medianRule =
        "even_n_mean_of_two_central_order_statistics"

    /// THE RULE ITSELF, executed. 8 is even, so "the median" is ambiguous until
    /// the tie-break is named, and the two candidate rules disagree on every
    /// even-length population. The wrapper and the workflow both implement the
    /// mean-of-two-central rule; this pins what that means so a future
    /// "simplification" onto the lower-median rule used by the per-pair
    /// diagnostic is a red test rather than a quiet score shift.
    @Test
    func theEvenMedianRuleIsTheMeanOfTheTwoCentralValues() throws {
        func median(_ values: [Double]) -> Double {
            let sorted = values.sorted()
            let n = sorted.count
            return n % 2 == 1
                ? sorted[(n - 1) / 2]
                : (sorted[n / 2 - 1] + sorted[n / 2]) / 2
        }
        func lowerMedian(_ values: [Double]) -> Double {
            let sorted = values.sorted()
            return sorted[(sorted.count - 1) / 2]
        }

        let eight = [1.10, 0.90, 1.00, 1.02, 0.98, 1.05, 0.95, 1.01]
        #expect(median(eight) == 1.005)
        // The two rules genuinely differ on this population -- otherwise the
        // assertion above would be pinning nothing.
        #expect(lowerMedian(eight) == 1.00)
        #expect(median(eight) != lowerMedian(eight))
        // Odd n is unambiguous and both rules agree.
        #expect(median([3, 1, 2]) == 2)

        // THE ANCHOR, as arithmetic. Under the RETIRED rule an unmodified
        // candidate normalised to 1.0 on every prompt by construction -- which
        // is exactly the problem, because the pinned references ARE the shipped
        // configuration's own measured per-prompt performance, so the rule made
        // the reference tree the zero point by definition rather than by
        // measurement.
        let fixture = try S.json(Self.fixturePath)
        let pool = try #require(fixture["timed_prompt_pool"] as? [[String: Any]])
        let noops = pool.compactMap { $0["noop_decode_speedup"] as? Double }
        #expect(noops.count == 8)
        #expect(median(noops.map { $0 / $0 }) == 1.0)
        // Under the LIVE rule those same measurements are the score, and the
        // pool's 1.42x spread survives into it: the unmodified tree medians
        // BELOW parity because the shipped depth-2 configuration is slower than
        // serial decode at this window. The seeded calibration expectation says
        // the same thing to ten decimal places.
        let unmodifiedRaw = median(noops)
        #expect(unmodifiedRaw < 1.0)
        #expect(abs(unmodifiedRaw - 0.9438) < 0.01)
        let calibration = try #require(fixture["calibration"] as? [String: Any])
        let expected = try #require(calibration["expected_raw_median"] as? Double)
        #expect(expected < 1.0)
        #expect(abs(expected - 0.9350424303) < 1e-10)
        // A candidate that stops drafting IS the serial control and scores
        // exactly 1.0. This is what makes the opened speculative surface safe:
        // the degenerate submission is legal and unremarkable, not an exploit.
        #expect(median(Array(repeating: 1.0, count: 8)) == 1.0)
        // ... and a candidate that only helps ONE prompt does not score as if
        // it helped all of them. A mean would read 1.05 here; the median does
        // not move at all. That property was always the median's, not the
        // normalisation's, so the re-anchoring did not cost it.
        var oneWin = Array(repeating: 1.0, count: 8)
        oneWin[0] = 1.40
        #expect(median(oneWin) == 1.0)
        #expect(oneWin.reduce(0, +) / 8 > 1.04)
    }

    /// THE SEEDED CALIBRATION EXPECTATION, checked against its own provenance.
    ///
    /// `expected_raw_median` is not a measurement taken under these semantics --
    /// no dispatch has run under them, because the runner for this repository is
    /// not registered yet. It is RE-DERIVED from the k2 reference dispatch's own
    /// sealed per-prompt raw ratios, and the fixture carries those eight numbers
    /// so the derivation can be checked rather than believed. This test IS that
    /// check: recompute the median from the recorded inputs and require it to
    /// equal the recorded output.
    @Test
    func theSeededCalibrationExpectationMatchesItsRecordedProvenance() throws {
        let fixture = try S.json(Self.fixturePath)
        let calibration = try #require(fixture["calibration"] as? [String: Any])
        let expected = try #require(calibration["expected_raw_median"] as? Double)
        let provenance = try #require(
            calibration["expected_raw_median_provenance"] as? [String: Any])
        let ratios = try #require(
            provenance["per_prompt_raw_ratios"] as? [String: Double])
        #expect(ratios.count == 8, "the provenance must record all eight prompts")

        let sorted = ratios.values.sorted()
        let recomputed = (sorted[3] + sorted[4]) / 2
        #expect(
            abs(recomputed - expected) < 1e-9,
            """
            expected_raw_median \(expected) is not the even-n median of the \
            eight raw ratios recorded beside it (\(recomputed)). Either the \
            number was edited without its provenance or the provenance was \
            edited without the number; both make the calibration \
            unattributable.
            """
        )
        // It must be labelled as SEEDED. A number this load-bearing must not be
        // mistaken for one measured under the semantics it calibrates.
        let status = try #require(calibration["status"] as? String)
        #expect(status.contains("seeded"))
        #expect(status.contains("pending"))
        // The band is a percentage around it, and the serial denominator band
        // is explicitly untouched -- they guard different quantities.
        #expect((calibration["band_pct"] as? Double) == 2.0)
        #expect((calibration["serial_denominator_band_unchanged"] as? Bool) == true)
    }

    /// The contract fixture carries the machine-readable rule, in its own
    /// coordinate-free block. It must NOT have disturbed the measured pool: the
    /// eight entries and their pinned no-op references are a separate concern
    /// with a separate owner, and this change adds keys beside them rather than
    /// editing them.
    @Test
    func theFixtureDeclaresTheScoringSemanticsWithoutDisturbingThePool() throws {
        let fixture = try S.json(Self.fixturePath)
        let semantics = try #require(
            fixture["scoring_semantics"] as? [String: Any],
            "the contract fixture must carry a machine-readable scoring_semantics block"
        )
        #expect(semantics["aggregation"] as? String == Self.aggregation)
        #expect(semantics["median_rule"] as? String == Self.medianRule)
        #expect(semantics["pairs_per_prompt"] as? Int == 1)
        // The reference is joined on the golden object's own digest, not on
        // argv order or a filename. That is what makes an unpinned golden fail
        // closed instead of borrowing its neighbour's normalisation.
        let key = try #require(semantics["reference_key"] as? String)
        #expect(key.contains("sha256"))
        // Pooled denominator banding has to be stated, because it is the reason
        // the installed calibration survives the change unmodified.
        let banding = try #require(semantics["serial_denominator_banding"] as? String)
        #expect(banding.lowercased().contains("unchanged"))
        #expect(banding.lowercased().contains("load-bearing"))
        // The floor and the NEW ceiling are both declared, and the ceiling is
        // the thing the published number never had (k-matrix finding 3).
        #expect((semantics["floor"] as? Double) == 0.90)
        #expect((semantics["ceiling"] as? Double) == 5.0)
        #expect((semantics["score_anchor"] as? String) == "serial = 1.0")

        // The pool itself is untouched: 8 distinct entries, each with a
        // positive measured reference.
        let pool = try #require(fixture["timed_prompt_pool"] as? [[String: Any]])
        #expect(pool.count == 8)
        #expect(Set(pool.compactMap { $0["sha256"] as? String }).count == 8)
        #expect(
            pool.allSatisfy { ($0["noop_decode_speedup"] as? Double).map { $0 > 0 } ?? false }
        )
    }

    /// The manifest and the fixture must state the SAME rule. These are the two
    /// documents a participant and an operator respectively read first.
    @Test
    func theManifestAndFixtureAgreeOnTheAggregationRule() throws {
        let manifest = try S.json(Self.manifestPath)
        let scoring = try #require(manifest["scoring"] as? [String: Any])
        #expect(scoring["aggregation"] as? String == Self.aggregation)
        #expect(scoring["medianRule"] as? String == Self.medianRule)
        #expect(scoring["pairsPerPrompt"] as? Int == 1)
        #expect(scoring["minPairsPerPrompt"] as? Int == 1)

        let fixture = try S.json(Self.fixturePath)
        let semantics = try #require(fixture["scoring_semantics"] as? [String: Any])
        #expect(
            scoring["aggregation"] as? String == semantics["aggregation"] as? String,
            "the manifest and the contract fixture disagree about how the score is aggregated"
        )
        #expect(
            scoring["medianRule"] as? String == semantics["median_rule"] as? String,
            "the manifest and the contract fixture disagree about the median rule"
        )
        #expect(
            (scoring["pairsPerPrompt"] as? Int) == (semantics["pairs_per_prompt"] as? Int),
            "the manifest and the contract fixture disagree about the per-prompt pair budget"
        )

        // The floor and the ceiling must agree between the two documents too:
        // a manifest that promised a bound the fixture did not carry would be
        // a participant-facing lie.
        #expect((scoring["decodeSpeedupFloor"] as? Double) == semantics["floor"] as? Double)
        #expect((scoring["decodeSpeedupCeiling"] as? Double) == semantics["ceiling"] as? Double)
        // The manifest must say WHERE 1.0 IS, since that is the first question
        // a participant asks and the one the retired rule answered wrongly.
        let anchor = try #require(scoring["scoreAnchorNote"] as? String)
        #expect(anchor.lowercased().contains("serial"))
        #expect(anchor.contains("1.0"))
        // ... and it must say what became of the no-op references, because they
        // are still pinned and still published and a reader will assume a
        // pinned number is a scored one unless told otherwise.
        let role = try #require(scoring["noopReferenceRole"] as? String)
        #expect(role == "informational_diagnostic_not_scored")
        // The retired rule must not still be claimed anywhere in the manifest's
        // scoring block: three documents updated and one left behind is exactly
        // how a confident lie survives a contract change.
        let scoringText = String(describing: scoring)
        #expect(!scoringText.contains(Self.retiredAggregation))
    }

    /// The workflow's per-prompt pair budget must equal the one both documents
    /// declare. The workflow is the only enforcing site; the other two are
    /// documentation, exactly as with the decode floor.
    @Test
    func theWorkflowPinsThePairBudgetTheDocumentsDeclare() throws {
        let environment = try S.jobEnvironment(try S.text(Self.workflowPath))
        let pairs = try #require(environment["MLXFAST_QWEN_MTP_PAIRS_PER_PROMPT"])
        let minPairs = try #require(environment["MLXFAST_QWEN_MTP_MIN_PAIRS_PER_PROMPT"])
        #expect(pairs == "1")
        #expect(minPairs == "1")

        let manifest = try S.json(Self.manifestPath)
        let scoring = try #require(manifest["scoring"] as? [String: Any])
        #expect(Int(pairs) == scoring["pairsPerPrompt"] as? Int)
        #expect(Int(minPairs) == scoring["minPairsPerPrompt"] as? Int)

        // The run-level budget the per-prompt one replaced must be GONE, not
        // left beside it: two pair budgets in one workflow is an invitation to
        // wire the wrong one into the wrapper.
        let workflow = try S.text(Self.workflowPath)
        #expect(!workflow.contains("MLXFAST_QWEN_MTP_MIN_ACCEPTED_PAIRS"))
        #expect(!workflow.contains("MLXFAST_QWEN_MTP_TARGET_PAIRS"))
    }

    /// The ranked workflow must TIME the whole pool. Scanned on the
    /// comment-stripped view so the historical explanation in the comments
    /// cannot satisfy an assertion about what the job does.
    @Test
    func theRankedWorkflowTimesEveryPoolPromptAndDrawsNothing() throws {
        let workflow = try S.text(Self.workflowPath)
        let executable = S.executable(workflow)

        // 1. NO DRAW. The uniform sampler is the thing that was removed; if it
        //    comes back, the median is a median over one prompt again.
        #expect(
            !executable.contains("/dev/urandom"),
            "the Qwen-MTP workflow still draws a random timed target"
        )
        #expect(!executable.contains("selected_index"))

        // 2. The resolve step publishes the whole set, and the download step
        //    verifies EVERY object rather than one sampled pin.
        let resolve = try S.stepBody(workflow, "Resolve the hidden Qwen-MTP timed prompt set")
        #expect(resolve.contains("qwen_mtp_timed_prompt_set.json"))
        #expect(resolve.contains("pool_size="))
        // The pool validation that predates this change must survive it: the
        // per-entry sweep and the >= 8 DISTINCT floor are what make the median
        // a median over eight real prompts.
        #expect(resolve.contains("required_distinct=8"))
        #expect(resolve.contains("noop_decode_speedup"))

        let prepare = try S.stepBody(workflow, "Prepare hidden Qwen-MTP goldens")
        #expect(prepare.contains("qwen_mtp_benchmark_golden_${index}.json"))
        #expect(
            prepare.contains("hidden Qwen-MTP timed golden pin mismatch for pool entry"),
            "each downloaded pool object must be verified against its own pin"
        )

        // 3. The timing step passes the whole set to the wrapper, per prompt.
        let timed = try S.stepBody(
            workflow, "Timed paired Qwen-MTP benchmark (measure-qwen-mtp-job)")
        #expect(timed.contains("golden_args+=(--golden"))
        #expect(timed.contains("--pairs-per-prompt"))
        #expect(timed.contains("--min-pairs-per-prompt"))
        #expect(
            !timed.contains("--golden \"${MLXFAST_PRIVATE_DIR}/qwen_mtp_benchmark_golden.json\""),
            "the timing step still passes a single sampled golden"
        )
    }

    /// The scoring step must RECOMPUTE the median from the sealed per-prompt
    /// breakdown and the TRUSTED contract's references, and must assert the
    /// breakdown's shape. A results.json carrying fewer per-prompt entries than
    /// the pool has is the prompt lottery returning as a missing-data bug, so
    /// the shape is part of the gate rather than a diagnostic.
    @Test
    func theScoringStepRecomputesTheMedianAndGatesTheBreakdownShape() throws {
        let workflow = try S.text(Self.workflowPath)
        let score = try S.stepBody(workflow, "Compute Qwen-MTP score and enforce floor")

        // Recomputed here, from per-prompt means -- never read out of a
        // pre-aggregated field the wrapper supplied.
        #expect(score.contains(".serial_seconds_per_token_mean / .mtp_seconds_per_token_mean"))
        // ... and the recompute no longer consults the contract AT ALL. Each
        // prompt's ratio is its own two measured means from one thermally-gated
        // session, so the whole join against pinned references -- and every way
        // that join could resolve to the wrong reference -- left the scoring
        // path on 2026-08-14. Asserting its ABSENCE is the point.
        #expect(
            !score.contains("timed_prompt_pool"),
            "the scoring recompute still joins against the pinned no-op pool"
        )
        // The even-n rule, spelled in the jq as well as in the docs.
        #expect(score.contains("(.[length/2 - 1] + .[length/2]) / 2"))

        // Breakdown shape is gated: one entry per pool prompt, each parity-clean
        // and distinct.
        #expect(score.contains("(.per_prompt | length) == $pool_size"))
        #expect(score.contains(".parity_ok == true"))
        #expect(score.contains("unique | length) == $pool_size"))

        // Cross-check against the wrapper's own sealed value: same rule, two
        // implementations, and a disagreement is an error rather than a
        // preference for one of them.
        #expect(score.contains("raw_decode_speedup_median"))
        #expect(score.contains("disagrees with the wrapper's sealed"))

        // Floor AND ceiling, both applied to the published raw median. The
        // ceiling is new: until 2026-08-14 the wrapper bounded per-PAIR ratios
        // and nothing bounded the aggregate that reached the leaderboard.
        #expect(score.contains("MLXFAST_QWEN_MTP_DECODE_SPEEDUP_FLOOR"))
        #expect(score.contains("MLXFAST_QWEN_MTP_DECODE_SPEEDUP_CEILING"))
        #expect(score.contains("plausibility ceiling"))

        // The published payload carries the median AND the breakdown, so a
        // submitter can see which prompts their change helped.
        #expect(score.contains("mtp_decode_speedup_raw_median"))
        #expect(score.contains("score_anchor: \"serial = 1.0\""))
        #expect(score.contains("per_prompt: ["))
        #expect(score.contains("aggregation: \"\(Self.aggregation)\""))
    }
}
