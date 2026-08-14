import Foundation
import Testing

/// A FAILED gates-only run must not upload the raw sealed gates score.
///
/// The defect this pins was introduced by making "Stage DFlash correctness
/// artifacts" `always()`. That was right in itself — a failed gates-only run is
/// exactly when its diagnostics are wanted — but it made the step run in the one
/// state its own comment assumed away: when "Validate sealed gates score" did
/// NOT run.
///
/// On a failed hidden-gates run the gates pass writes `gates-score.json` and
/// then exits nonzero, so validation is skipped and staging would upload the raw
/// file. Its `.metrics` can carry `first_failing_step`, `expected_token` and
/// `actual_token` — values derived FROM THE HIDDEN GOLDEN. Two things made that
/// worse than a one-shot disclosure:
///
///   * `deny-private-artifacts.sh` is NAME- and SIZE-based. It cannot see values
///     inside a permitted filename, and it passes such a file with exit 0.
///   * A gates-only failure charges no attributable-failure budget, so the run
///     is repeatable for free — one hidden token per run, i.e. an ITERATED
///     ORACLE. `redact-benchmark-failure.sh`'s own header states the invariant
///     this broke: "a failing score.json must never be uploaded raw."
///
/// The fix re-asserts the sealed-score predicate inline, in the staging step,
/// and drops a score that fails it rather than failing the step — the run has
/// already failed for its own reason and the other artifacts are still useful.
/// `public-gate-report.json` gets the guard serial has and the DFlash copy
/// dropped: its `.error` can embed sandboxed-worker stderr, which is
/// submitted-code-controlled text.
@Suite("DFlash gates-only artifact leak")
struct DFlashGatesOnlyArtifactLeakTests {
    private static let workflowPath = ".github/workflows/dflash-benchmark.yml"
    private static let stepName = "Stage DFlash correctness artifacts"

    // MARK: - behavioural: run the real step body

    /// The case that was live on `main`: hidden-golden token fields in a failed
    /// sealed score must never reach the upload set.
    @Test
    func aFailedSealedScoreIsWithheldFromStaging() throws {
        let staged = try runStagingStep(
            gatesScore: """
                {"passed":false,"metrics":{"error":"benchmark decode token mismatch",\
                "first_failing_step":61,"expected_token":31337,"actual_token":90210,\
                "passed_correctness":false}}
                """,
            publicGateReport: nil,
            isSubmissionBranch: false
        )

        #expect(
            !staged.stagedNames.contains("gates-score.json"),
            """
            a failed sealed gates score reached the upload set. Its .metrics carry \
            hidden-golden token fields, deny-private-artifacts.sh is name-based and \
            cannot see them, and a gates-only failure charges no failure budget — \
            so this is repeatable for free, one hidden token per run.
            """
        )
        for leaked in ["31337", "90210", "61"] {
            #expect(
                !staged.stagedArgumentText.contains(leaked),
                "hidden-golden value \(leaked) appeared in the staged set"
            )
        }
        // The rest of the diagnostics must survive — withholding is not aborting.
        #expect(staged.stagedNames.contains("candidate.sha"))
        #expect(staged.exitStatus == 0, "withholding must not fail the step")
    }

    /// The guard must not cost the happy path: a score that passes the predicate
    /// is still staged, or the fix has simply disabled the artifact.
    @Test
    func aCleanSealedScoreIsStillStaged() throws {
        let staged = try runStagingStep(
            gatesScore: """
                {"passed":true,"metrics":{"error":"","first_failing_case":null,\
                "first_failing_layer":null,"first_failing_step":null,\
                "expected_token":null,"actual_token":null,"passed_correctness":true}}
                """,
            publicGateReport: nil,
            isSubmissionBranch: false
        )
        #expect(
            staged.stagedNames.contains("gates-score.json"),
            "a clean sealed score must still be published, or the guard is a mute"
        )
        #expect(staged.exitStatus == 0)
    }

    /// `public-gate-report.json`'s `.error` can carry sandboxed-worker stderr,
    /// which submitted code controls. Serial guards this on submission branches.
    @Test
    func aPublicGateReportCarryingWorkerOutputIsWithheldOnSubmissionBranches() throws {
        let onSubmission = try runStagingStep(
            gatesScore: nil,
            publicGateReport: #"{"passed":false,"error":"worker stderr: SENTINEL-42"}"#,
            isSubmissionBranch: true
        )
        #expect(
            !onSubmission.stagedNames.contains("public-gate-report.json"),
            "a public gate report with submitted-code-controlled .error was staged"
        )
        #expect(!onSubmission.stagedArgumentText.contains("SENTINEL-42"))

        // On an organizer ref the same report is organizer output and is useful.
        let offSubmission = try runStagingStep(
            gatesScore: nil,
            publicGateReport: #"{"passed":false,"error":"worker stderr: SENTINEL-42"}"#,
            isSubmissionBranch: false
        )
        #expect(
            offSubmission.stagedNames.contains("public-gate-report.json"),
            "the guard must be scoped to submission branches, as serial scopes it"
        )
    }

    // MARK: - structural: the predicate must match the validation step's

    /// The inline gate is a COPY of "Validate sealed gates score"'s predicate.
    /// If that step ever tightens, this copy must tighten too, or the always()
    /// path silently permits what the happy path forbids.
    @Test
    func theInlineGateChecksEveryFieldTheValidationStepChecks() throws {
        let workflow = try String(contentsOfFile: Self.workflowPath, encoding: .utf8)
        let validation = try Self.stepBody(workflow, "Validate sealed gates score")
        let staging = try Self.stepBody(workflow, Self.stepName)

        // Every `.metrics.<field> == null` (and the two booleans) asserted by the
        // validation step must also be asserted inline.
        let fields = [
            "first_failing_case", "first_failing_layer", "first_failing_step",
            "expected_token", "actual_token",
        ]
        for field in fields where validation.contains(field) {
            #expect(
                staging.contains(field),
                """
                'Validate sealed gates score' checks .metrics.\(field) but the \
                always() staging gate does not. The staging path runs precisely \
                when that validation was skipped, so any field it omits is a field \
                that can be uploaded raw from a failed run.
                """
            )
        }
        #expect(staging.contains("passed_correctness"))
        #expect(staging.contains(".metrics.error"))
    }

    // MARK: - harness

    private struct StagingResult {
        var exitStatus: Int32
        var stagedArgumentText: String
        var stagedNames: [String]
    }

    private func runStagingStep(
        gatesScore: String?,
        publicGateReport: String?,
        isSubmissionBranch: Bool
    ) throws -> StagingResult {
        let workspace = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("dflash-stage-\(UUID().uuidString)")
        let scripts = workspace.appendingPathComponent(".github/scripts")
        try FileManager.default.createDirectory(at: scripts, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: workspace) }

        // Stubs that record rather than act, so the assertion is about WHICH
        // names reach staging, not about the staging script's own behaviour.
        for (name, body) in [
            ("deny-private-artifacts.sh", "#!/bin/bash\nexit 0\n"),
            ("stage-benchmark-artifacts.sh", "#!/bin/bash\nshift\nprintf 'STAGED %s\\n' \"$*\"\n"),
        ] {
            let url = scripts.appendingPathComponent(name)
            try body.write(to: url, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        }

        try "deadbeef\n".write(
            to: workspace.appendingPathComponent("candidate.sha"),
            atomically: true, encoding: .utf8
        )
        try "{}".write(
            to: workspace.appendingPathComponent("dflash-gates-report.json"),
            atomically: true, encoding: .utf8
        )
        if let gatesScore {
            try gatesScore.write(
                to: workspace.appendingPathComponent("gates-score.json"),
                atomically: true, encoding: .utf8
            )
        }
        if let publicGateReport {
            try publicGateReport.write(
                to: workspace.appendingPathComponent("public-gate-report.json"),
                atomically: true, encoding: .utf8
            )
        }

        let workflow = try String(contentsOfFile: Self.workflowPath, encoding: .utf8)
        let body = try Self.stepBody(workflow, Self.stepName, stripComments: false)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["-c", body]
        process.currentDirectoryURL = workspace
        process.environment = [
            "PATH": ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin",
            "MLXFAST_CANDIDATE_SHA": "deadbeef",
            "MLXFAST_ARTIFACT_ROOT": workspace.appendingPathComponent("art").path,
            "MLXFAST_IS_SUBMISSION_BRANCH": isSubmissionBranch ? "1" : "0",
        ]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let output = String(
            decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self
        )
        process.waitUntilExit()

        // Only the STAGED line counts; ::notice:: lines mention withheld names by
        // design and must not be mistaken for staging.
        let stagedLine =
            output
            .split(separator: "\n")
            .first { $0.hasPrefix("STAGED ") }
            .map(String.init) ?? ""
        let names = stagedLine
            .split(separator: " ")
            .dropFirst()
            .compactMap { $0.split(separator: "=").first.map(String.init) }

        return StagingResult(
            exitStatus: process.terminationStatus,
            stagedArgumentText: stagedLine,
            stagedNames: names
        )
    }

    /// The `run:` scalar of a named step, dedented, terminating at the first
    /// line indented less than the block (a YAML block scalar ends there — not
    /// at the next `- name:`, which would swallow the following step's comments).
    private static func stepBody(
        _ workflow: String,
        _ step: String,
        stripComments: Bool = true
    ) throws -> String {
        let marker = "- name: \(step)\n"
        let start = try #require(workflow.range(of: marker), "step '\(step)' not found")
        let rest = workflow[start.upperBound...]
        let runMarker = try #require(rest.range(of: "run: |\n"), "step '\(step)' has no run: block")
        let scalar = String(rest[runMarker.upperBound...])
        let lines = scalar.split(separator: "\n", omittingEmptySubsequences: false)
        let indent =
            lines.first { !$0.trimmingCharacters(in: .whitespaces).isEmpty }?
            .prefix { $0 == " " }.count ?? 0

        var body: [String] = []
        for line in lines {
            let blank = line.trimmingCharacters(in: .whitespaces).isEmpty
            if !blank && line.prefix(while: { $0 == " " }).count < indent { break }
            let dedented = line.count >= indent ? String(line.dropFirst(indent)) : String(line)
            if stripComments && dedented.trimmingCharacters(in: .whitespaces).hasPrefix("#") {
                continue
            }
            body.append(dedented)
        }
        return body.joined(separator: "\n")
    }
}
