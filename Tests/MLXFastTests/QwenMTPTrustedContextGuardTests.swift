import Foundation
import Testing

// THE TWO QWEN-MTP TRUSTED-CONTEXT GUARDS, EXECUTED.
//
// qwen-mtp-provision-goldens.yml and qwen-mtp-r2-key-probe.yml each invoke a
// guard script BY PATH as the first step after checkout, before any credential
// is used and before any other script from the dispatched ref runs. Both
// shipped as a deliberate fail-closed gap -- the `run:` lines named scripts
// that did not exist, so the jobs died with exit 127 -- and this suite is what
// stops them regressing into something that exists but does not refuse.
//
// EVERY ASSERTION THAT CAN BE EXECUTED IS EXECUTED. The DFlash twins are
// covered the same way for a concrete reason recorded in
// DFlashGoldenProvisioningTests: one of them shipped mode 0644, which fails the
// step with exit 126 and names no cause, and only a test that RAN the script
// caught it. Reading workflow text is reserved here for the one thing that has
// no runtime -- which script path each workflow's `run:` line names.
//
// MAIN ONLY. Both guards allowlist refs/heads/main and nothing else. The
// pre-merge migration allowlist entry for refs/heads/qwen36-mtp-track was
// removed at the 2026-08-13 repo split, where main of
// Layr-Labs/qwen-3.6-mtp-challenge-dev BECAME the track; that branch is now in
// the refusal list of `onlyTheAllowlistedRefsMayDispatch` below, which is the
// end state the go-live runbook specified.
@Suite("Qwen-MTP trusted-context guards")
struct QwenMTPTrustedContextGuardTests {
    private typealias S = DFlashGateTextSupport

    private static let provisionGuard =
        ".github/scripts/enforce-trusted-qwen-mtp-provision-workflow.sh"
    private static let probeGuard =
        ".github/scripts/enforce-trusted-qwen-mtp-r2-probe-workflow.sh"

    private static let provisionWorkflow =
        ".github/workflows/qwen-mtp-provision-goldens.yml"
    private static let probeWorkflow =
        ".github/workflows/qwen-mtp-r2-key-probe.yml"

    /// Each guard paired with the workflow it is pinned to.
    private static let guards: [(guardScript: String, workflow: String)] = [
        (provisionGuard, provisionWorkflow),
        (probeGuard, probeWorkflow),
    ]

    // MARK: - The workflows invoke them, and they are runnable

    /// The `run:` line of each workflow's trusted-context step must name that
    /// workflow's own guard. This is the half with no runtime: a workflow that
    /// named the sibling guard would still execute a real script, and the
    /// script's own workflow-path pin would then refuse -- correctly, but only
    /// at dispatch time.
    @Test
    func eachWorkflowInvokesItsOwnGuard() throws {
        for (guardScript, workflow) in Self.guards {
            let text = try S.text(workflow)
            #expect(
                text.contains("run: \(guardScript)"),
                """
                \(workflow) does not invoke \(guardScript) with a literal \
                `run:` line. The trusted-context step is the first thing after \
                checkout for a reason; if it was renamed, rename both ends.
                """
            )
        }
    }

    /// Both guards must be mode 100755 in the git INDEX, not merely on this
    /// checkout's disk. The workflows run them as commands, so a 0644 mode
    /// fails the step with exit 126 before any interlock, naming no cause --
    /// the exact way the DFlash provisioning guard shipped broken once.
    @Test
    func theGuardsAreExecutable() throws {
        for (guardScript, _) in Self.guards {
            let listing = try Self.bash("git ls-files --stage -- \(guardScript)")
            #expect(
                listing.output.hasPrefix("100755"),
                """
                \(guardScript) is mode \(listing.output.prefix(6)) in the git \
                index, not 100755. `chmod +x` it and commit the mode change.
                """
            )
            #expect(
                FileManager.default.isExecutableFile(
                    atPath: Self.repoRoot.appendingPathComponent(guardScript).path
                ),
                "\(guardScript) is not executable in the working tree"
            )
        }
    }

    // MARK: - What each guard admits

    /// The allowlist, executed: refs/heads/main passes; nothing else does. The
    /// ranked job admits `submissions/*`,
    /// `baseline/*` and `yukon/baseline/*` because a submission must run
    /// against trusted main's harness. These two jobs hold R2 credentials and
    /// choose, from the dispatched ref, the code that touches hidden material,
    /// so copying that allowlist would hand every participant-creatable branch
    /// namespace a hidden golden.
    @Test
    func onlyTheAllowlistedRefsMayDispatch() throws {
        for (guardScript, workflow) in Self.guards {
            let allowed = try Self.runGuard(
                guardScript, ref: "refs/heads/main", event: "workflow_dispatch",
                workflow: workflow
            )
            #expect(
                allowed.status == 0,
                "\(guardScript) refused refs/heads/main: \(allowed.output)"
            )

            for ref in [
                "refs/heads/submissions/example",
                "refs/heads/baseline/reference",
                "refs/heads/yukon/baseline/718528521cd7a7df341b750bc3ccb28478ff045b",
                "refs/heads/feature/anything",
                // Retired at the repo split: main IS the track here.
                "refs/heads/qwen36-mtp-track",
                "refs/heads/qwen36-mtp-track-2",
                "refs/tags/v1",
            ] {
                let run = try Self.runGuard(
                    guardScript, ref: ref, event: "workflow_dispatch", workflow: workflow
                )
                #expect(
                    run.status != 0,
                    """
                    \(guardScript) ADMITTED \(ref). This job holds R2 \
                    credentials against hidden competition material; a branch \
                    namespace a participant can create must never reach it.
                    """
                )
            }
        }
    }

    /// `workflow_dispatch` and nothing else. A push trigger reaching either job
    /// would run it on every merge.
    @Test
    func onlyWorkflowDispatchMayDispatch() throws {
        for (guardScript, workflow) in Self.guards {
            for event in ["push", "pull_request", "schedule", "repository_dispatch"] {
                let run = try Self.runGuard(
                    guardScript, ref: "refs/heads/main", event: event, workflow: workflow
                )
                #expect(
                    run.status != 0,
                    "\(guardScript) admitted the \(event) event: \(run.output)"
                )
            }
        }
    }

    /// A fork or mirror is refused even from main with a dispatch. The
    /// credentials do not travel, but the workflow file and every script it
    /// runs do.
    @Test
    func onlyTheTrustedRepositoryMayDispatch() throws {
        for (guardScript, workflow) in Self.guards {
            let run = try Self.runGuard(
                guardScript,
                ref: "refs/heads/main",
                event: "workflow_dispatch",
                workflow: workflow,
                repository: "someone-else/qwen-3.8-mtp-challenge-dev"
            )
            #expect(
                run.status != 0,
                "\(guardScript) ran outside the trusted repository: \(run.output)"
            )
        }
    }

    /// The workflow-path pin, both directions. A guard borrowed by a foreign
    /// workflow is refused, and -- the case a pair of near-twin scripts makes
    /// live -- so is each guard invoked under its SIBLING's workflow ref.
    @Test
    func eachGuardIsPinnedToItsOwnWorkflowPath() throws {
        for (guardScript, workflow) in Self.guards {
            let foreign = try Self.runGuard(
                guardScript,
                ref: "refs/heads/main",
                event: "workflow_dispatch",
                workflow: ".github/workflows/ci.yml"
            )
            #expect(
                foreign.status != 0,
                "\(guardScript) accepted a foreign workflow ref: \(foreign.output)"
            )

            let sibling = Self.guards.first { $0.workflow != workflow }?.workflow
            let siblingWorkflow = try #require(sibling)
            let borrowed = try Self.runGuard(
                guardScript,
                ref: "refs/heads/main",
                event: "workflow_dispatch",
                workflow: siblingWorkflow
            )
            #expect(
                borrowed.status != 0,
                """
                \(guardScript) accepted \(siblingWorkflow)'s workflow ref. The \
                two guards are near-twins; each must still refuse to stand in \
                for the other, or a copy-paste in either workflow silently \
                swaps the allowlist that job is checked against.
                """
            )
        }
    }

    /// Every required environment variable is required. An unset one must
    /// abort, never default to a passing comparison.
    @Test
    func everyRequiredEnvironmentVariableIsRequired() throws {
        let required = [
            "GITHUB_REPOSITORY", "GITHUB_REF", "GITHUB_WORKFLOW_REF", "GITHUB_EVENT_NAME",
        ]
        for (guardScript, workflow) in Self.guards {
            for omitted in required {
                let run = try Self.runGuard(
                    guardScript,
                    ref: "refs/heads/main",
                    event: "workflow_dispatch",
                    workflow: workflow,
                    omitting: omitted
                )
                #expect(
                    run.status != 0,
                    "\(guardScript) ran with \(omitted) unset: \(run.output)"
                )
            }
        }
    }

    // MARK: - Harness

    private static var repoRoot: URL {
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }

    private static func bash(
        _ script: String,
        env: [String: String] = [:]
    ) throws -> (status: Int32, output: String) {
        var environment = env
        let inherited = ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin"
        environment["PATH"] = inherited
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["-c", script]
        process.currentDirectoryURL = repoRoot
        process.environment = environment
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(decoding: data, as: UTF8.self))
    }

    /// Runs a guard with a synthetic GitHub context. `workflow` is the workflow
    /// PATH the synthesized `GITHUB_WORKFLOW_REF` names.
    private static func runGuard(
        _ guardScript: String,
        ref: String,
        event: String,
        workflow: String,
        repository: String = "Layr-Labs/qwen-3.8-mtp-challenge",
        omitting: String? = nil
    ) throws -> (status: Int32, output: String) {
        var environment = [
            "GITHUB_REPOSITORY": repository,
            "GITHUB_REF": ref,
            "GITHUB_WORKFLOW_REF":
                "Layr-Labs/qwen-3.8-mtp-challenge/\(workflow)@\(ref)",
            "GITHUB_EVENT_NAME": event,
        ]
        if let omitting {
            environment.removeValue(forKey: omitting)
        }
        return try bash(
            repoRoot.appendingPathComponent(guardScript).path,
            env: environment
        )
    }
}
