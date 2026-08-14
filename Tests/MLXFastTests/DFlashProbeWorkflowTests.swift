import Foundation
import Testing

/// `.github/workflows/dflash-probe-r2-keys.yml` — the read-only diagnostic that
/// resolves R2 object keys, so a wrong hidden-golden key is a one-minute answer
/// instead of a 404 discovered 25 minutes into a ranked dispatch.
///
/// WHY THIS SUITE EXISTS. The probe binds `benchmark-private-prompts-v2`, runs
/// on the DFlash box, and executes `.github/scripts/download-r2-object.sh` FROM
/// THE DISPATCHED REF against hidden competition material — and it shipped with
/// no repository check, no ref allowlist, no `GITHUB_WORKFLOW_REF` check, no
/// `persist-credentials: false`, no janitor step, and zero tests. Both siblings
/// have all of those. Anyone able to dispatch it could dispatch from a branch
/// whose downloader had been edited to print the object body and read hidden
/// golden bytes out of an Actions log.
///
/// HOW THE ASSERTIONS ARE SHAPED, AND WHY. This file is the third consecutive
/// attempt to fix a defect in this one workflow, and the previous attempt
/// shipped an assertion that could not fire: `supplied` was incremented inside
/// the loop it was meant to audit, so a loop that dropped a key dropped the
/// expected count with it and `probed != supplied` was unreachable. A test that
/// only checked the assertion's TEXT would have passed on that dead code, and a
/// test that only checked it was not DELETED would have passed too.
///
/// So every assertion below that can be executed is executed, and the counter
/// test does not check for the defect's absence — it reintroduces the exact
/// historical defect (`printf '%s'`, no trailing newline) underneath the new
/// counter and requires the workflow to fail. Reading workflow text is reserved
/// for the things that have no runtime: `with:` keys, `if:` conditions, step
/// order, and git index modes.
@Suite("DFlash R2 key probe workflow")
struct DFlashProbeWorkflowTests {
    static let workflowPath = ".github/workflows/dflash-probe-r2-keys.yml"
    static let trustedScript = ".github/scripts/enforce-trusted-dflash-probe-workflow.sh"
    static let downloaderScript = ".github/scripts/download-r2-object.sh"

    static let probeStep = "Probe"
    static let guardStep = "Enforce trusted workflow context"
    static let checkoutStep = "Checkout"
    static let scratchStep = "Remove probe scratch"
    static let janitorStep = "Janitor reset and integrity audit"

    /// The literal the Probe step feeds its loop from. Reintroducing the
    /// historical bug means swapping this for the form without `\n`.
    static let loopFeeder =
        #"done < <(printf '%s\n' "${CANDIDATE_KEYS}" | tr ',' '\n')"#
    static let droppingLoopFeeder =
        #"done < <(printf '%s' "${CANDIDATE_KEYS}" | tr ',' '\n')"#

    static let bodySentinel = "HIDDEN_GOLDEN_BODY_SENTINEL_DO_NOT_LOG"

    // MARK: - F5: the trusted-context guard

    /// main only, executed against every ref that matters — not read.
    ///
    /// The ranked job admits `submissions/*`, `baseline/*` and
    /// `yukon/baseline/*` because a submission must run against trusted main's
    /// harness; those namespaces carry participant-authored content by design.
    /// This job chooses, from the dispatched ref, the code that touches a
    /// hidden golden. `#825`'s guard reasoned that copying the ranked allowlist
    /// "would have handed submissions/* the ability to overwrite a hidden
    /// golden"; a probe that READS the same material needs no less.
    @Test
    func onlyMainMayDispatchTheProbe() throws {
        #expect(
            FileManager.default.fileExists(
                atPath: Self.repoRoot.appendingPathComponent(Self.trustedScript).path
            ),
            """
            \(Self.trustedScript) does not exist. The probe is the only \
            R2-credentialled DFlash workflow without a trusted-context guard; \
            both siblings run one.
            """
        )

        let allowed = try Self.runTrustedGuard(ref: "refs/heads/main")
        #expect(allowed.status == 0, "main was refused: \(allowed.output)")

        for ref in [
            "refs/heads/submissions/example",
            "refs/heads/baseline/reference",
            "refs/heads/yukon/baseline/718528521cd7a7df341b750bc3ccb28478ff045b",
            "refs/heads/feature/anything",
            "refs/heads/main-but-not-really",
            "refs/tags/v1",
        ] {
            let result = try Self.runTrustedGuard(ref: ref)
            #expect(
                result.status != 0,
                """
                the probe guard ADMITTED \(ref). This job runs \
                download-r2-object.sh from the dispatched ref with real R2 \
                credentials against hidden material, so admitting a ref that \
                can carry edited scripts is a path from "dispatch" to "hidden \
                golden bytes in an Actions log".
                """
            )
        }

        // A push on main is refused too, so widening the trigger list cannot
        // quietly turn a merge into a credentialled run.
        let pushed = try Self.runTrustedGuard(ref: "refs/heads/main", event: "push")
        #expect(pushed.status != 0, "the guard admitted a push event: \(pushed.output)")

        // A fork or mirror of this repository is refused.
        let foreignRepo = try Self.runTrustedGuard(
            ref: "refs/heads/main", repository: "attacker/mlxfast-challenge-dev"
        )
        #expect(foreignRepo.status != 0, "the guard admitted a foreign repository")

        // And the guard cannot be borrowed by another workflow file.
        let wrongFile = try Self.runTrustedGuard(ref: "refs/heads/main", workflow: "ci.yml")
        #expect(wrongFile.status != 0, "the guard accepted a foreign workflow ref")

        // Every required variable is required, so a missing one is a refusal
        // rather than an empty-string comparison that happens to pass.
        for missing in [
            "GITHUB_REPOSITORY", "GITHUB_REF", "GITHUB_WORKFLOW_REF", "GITHUB_EVENT_NAME",
        ] {
            let result = try Self.runTrustedGuard(ref: "refs/heads/main", unset: missing)
            #expect(result.status != 0, "the guard ran with \(missing) unset")
        }
    }

    /// The guard must run, and must run before anything that uses the
    /// credentials or executes a script from the dispatched ref. A guard placed
    /// after the Probe step would verify the context of a leak already taken.
    @Test
    func theGuardRunsBeforeAnythingTouchesTheCredentials() throws {
        let steps = try Self.steps()
        let names = steps.map(\.name)

        let guardIndex = try #require(
            names.firstIndex(of: Self.guardStep),
            "the probe declares no '\(Self.guardStep)' step; both siblings run one"
        )
        let probeIndex = try #require(names.firstIndex(of: Self.probeStep))
        let checkoutIndex = try #require(names.firstIndex(of: Self.checkoutStep))

        #expect(
            steps[guardIndex].body.contains(Self.trustedScript),
            """
            the '\(Self.guardStep)' step does not invoke \(Self.trustedScript): \
            \(steps[guardIndex].body)
            """
        )
        #expect(
            checkoutIndex < guardIndex && guardIndex < probeIndex,
            """
            step order is \(names). The guard must sit after Checkout (it is a \
            script in the tree) and before Probe (which binds the R2 secrets).
            """
        )

        // Nothing before the guard may reference a secret.
        for step in steps[..<guardIndex] {
            #expect(
                !step.body.contains("secrets."),
                """
                step '\(step.name)' runs BEFORE the trusted-context guard and \
                references secrets. Move it after the guard.
                """
            )
        }
    }

    /// `persist-credentials: false`. On a self-hosted box the workspace outlives
    /// the job, so the default leaves a credentialled `http.extraheader` in
    /// `.git/config` for whatever runs next. This step had no `with:` block at
    /// all; both siblings set it.
    @Test
    func theCheckoutDoesNotPersistCredentials() throws {
        let steps = try Self.steps()
        let checkout = try #require(steps.first { $0.name == Self.checkoutStep })

        #expect(
            checkout.body.contains("persist-credentials: false"),
            """
            the probe's actions/checkout does not set persist-credentials: \
            false. On a self-hosted runner that leaves a credentialled \
            http.extraheader in the workspace .git/config after the job. \
            Block was:
            \(checkout.body)
            """
        )
        #expect(
            checkout.body.contains("uses: actions/checkout@"),
            "the '\(Self.checkoutStep)' step no longer uses actions/checkout"
        )
    }

    /// Both siblings end with the janitor. Without it the box is not reset or
    /// integrity-audited after a job that held R2 credentials and wrote hidden
    /// bytes to disk.
    @Test
    func theProbeEndsWithAJanitorResetAndIntegrityAudit() throws {
        let steps = try Self.steps()
        let janitor = try #require(
            steps.last,
            "the probe declares no steps"
        )
        #expect(
            janitor.name == Self.janitorStep,
            """
            the probe's last step is '\(janitor.name)', not '\(Self.janitorStep)'. \
            Both dflash-benchmark.yml and dflash-provision-goldens.yml end with \
            the janitor; the probe ended with its Probe step.
            """
        )
        #expect(
            janitor.body.contains("if: always()"),
            "the janitor step is not `if: always()`, so a failed probe skips the reset"
        )
        #expect(janitor.body.contains(#"sudo -n "${MLXFAST_JANITOR}""#))
        #expect(janitor.body.contains("MLXFAST_QUARANTINE_FLAG"))

        // The env the janitor step reads must actually be bound on the job.
        let jobEnv = try Self.jobEnv()
        #expect(
            jobEnv["MLXFAST_JANITOR"] == "/opt/bench/janitor.sh",
            "MLXFAST_JANITOR is \(jobEnv["MLXFAST_JANITOR"] ?? "unset") on the probe job"
        )
        #expect(jobEnv["MLXFAST_QUARANTINE_FLAG"] == "/opt/bench/quarantine.flag")
    }

    /// Index mode, not working-tree mode: the bit that travels is the one git
    /// records. The workflow invokes these BY PATH, so 0644 dies with exit 126
    /// naming no cause — which is exactly how it shipped once for the
    /// provisioning scripts.
    @Test
    func theScriptsTheWorkflowInvokesAreExecutable() throws {
        for script in [Self.trustedScript, Self.downloaderScript] {
            let listing = try Self.bash(
                "git ls-files --stage -- \(script)", cwd: Self.repoRoot, env: [:]
            )
            #expect(
                listing.output.hasPrefix("100755"),
                """
                \(script) is mode \(listing.output.prefix(6)) in the git index, \
                not 100755. dflash-probe-r2-keys.yml invokes it as a command; a \
                non-executable mode fails the job with exit 126 and no \
                diagnosis. `chmod +x` it and commit the mode change.
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

    /// Dispatch only. A push trigger would run a credentialled job on every
    /// merge, and the guard's event check is the backstop for exactly that.
    @Test
    func theProbeIsDispatchOnly() throws {
        let triggers = try Self.triggerKeys()
        #expect(
            triggers == ["workflow_dispatch"],
            "the probe declares triggers \(triggers); it must be workflow_dispatch only"
        )
    }

    // MARK: - F1: probed-vs-supplied must be able to disagree

    /// THE DEGRADATION TEST. The previous fix added a count assertion whose two
    /// operands came from the same place, so it could not fire. Checking that
    /// the assertion still exists would not have caught that, and neither would
    /// checking that it was not deleted.
    ///
    /// This runs the REAL step body twice: once as shipped, once with the exact
    /// historical defect put back underneath the current counter. The second
    /// run must fail. If it passes, the counter is decorative again.
    @Test
    func theSuppliedCountIsDerivedFromTheInputAndCatchesADroppedKey() throws {
        let sandbox = try Sandbox()
        try Self.stageDownloader(sandbox, objects: ["a": "AAA", "b": "BBB", "c": "CCC"])

        let shipped = try Self.runProbeStep(sandbox, keys: "a,b,c")
        #expect(shipped.status == 0, "the shipped probe refused three good keys: \(shipped.output)")
        #expect(
            shipped.output.contains("probed 3 of 3 supplied key(s)"),
            """
            the probe did not report 3 of 3 for input 'a,b,c'. Output:
            \(shipped.output)
            """
        )

        // Reintroduce the defect: the loop feeder without its trailing newline,
        // which is what made `read` discard the final key. Everything else,
        // including the new counter, is untouched.
        let body = try Self.stepBody(Self.probeStep)
        #expect(
            body.components(separatedBy: Self.loopFeeder).count - 1 == 1,
            """
            the Probe step no longer contains exactly one
              \(Self.loopFeeder)
            so this test can no longer reintroduce the historical defect and \
            has stopped proving anything. Re-point it at the new feeder.
            """
        )
        let degraded = body.replacingOccurrences(
            of: Self.loopFeeder, with: Self.droppingLoopFeeder
        )
        let dropped = try Self.runProbeStep(sandbox, keys: "a,b,c", body: degraded)

        #expect(
            dropped.status != 0,
            """
            THE COUNT ASSERTION IS DEAD AGAIN. With the historical dropping \
            feeder restored the probe examined fewer keys than it was handed \
            and still exited 0. `supplied` must be derived from \
            ${CANDIDATE_KEYS} before the loop; deriving it inside the loop \
            makes it fall with the dropped key. Output:
            \(dropped.output)
            """
        )
        #expect(
            dropped.output.contains("probed 2 of 3"),
            """
            expected the degraded run to report 2 of 3; got:
            \(dropped.output)
            """
        )
        #expect(dropped.output.contains("results are incomplete"))
    }

    /// The count must survive the whitespace and newline forms the input
    /// description advertises ("newline- or comma-separated"), because a
    /// `supplied` derivation that disagrees with the loop's own normalisation
    /// fails every honest dispatch instead of the dishonest one.
    ///
    /// This is not hypothetical: the obvious spelling,
    /// `tr -d '[:space:]'` over the whole stream, deletes the NEWLINES too and
    /// collapses every key onto one line, giving `supplied=1` for 'a,b,c'.
    @Test
    func theSuppliedCountAgreesWithTheLoopOnEveryAdvertisedInputForm() throws {
        let sandbox = try Sandbox()
        try Self.stageDownloader(
            sandbox, objects: ["a": "A", "b": "B", "c": "C", "d": "D"]
        )

        for (label, keys, expected) in [
            ("plain commas", "a,b,c", 3),
            ("padded commas", " a , b , c ", 3),
            ("newlines", "a\nb\nc", 3),
            ("mixed, with a blank field", "a, b\n c ,d\n\n", 4),
            ("one key", "a", 1),
            ("trailing comma", "a,b,", 2),
        ] {
            let run = try Self.runProbeStep(sandbox, keys: keys)
            #expect(
                run.status == 0,
                """
                the probe failed its own count on \(label) (\(keys.debugDescription)). \
                A `supplied` derivation that disagrees with the loop's \
                normalisation rejects honest dispatches. Output:
                \(run.output)
                """
            )
            #expect(
                run.output.contains("probed \(expected) of \(expected) supplied key(s)"),
                "\(label): expected \(expected) of \(expected). Output:\n\(run.output)"
            )
        }
    }

    /// A key that 404s still counts as probed — "missing" is a result, not a
    /// skip. Otherwise the assertion fires on the probe's whole reason to exist.
    @Test
    func aMissingKeyCountsAsProbed() throws {
        let sandbox = try Sandbox()
        try Self.stageDownloader(sandbox, objects: ["a": "AAA"])
        let run = try Self.runProbeStep(sandbox, keys: "a,definitely/absent")
        #expect(run.status == 0, "a 404 candidate failed the run: \(run.output)")
        #expect(run.output.contains("probed 2 of 2 supplied key(s)"))
        #expect(run.output.contains("missing"))
    }

    // MARK: - the property the whole workflow exists to preserve

    /// It prints a status, a byte count and a sha256. Never a byte of the
    /// object. (What binds this to MAIN's downloader is
    /// `onlyMainMayDispatchTheProbe` above — the step cannot police a
    /// downloader chosen by the dispatcher.)
    @Test
    func theProbeNeverEchoesObjectBytes() throws {
        let sandbox = try Sandbox()
        try Self.stageDownloader(
            sandbox, objects: ["a": Self.bodySentinel, "control": Self.bodySentinel]
        )
        let run = try Self.runProbeStep(sandbox, keys: "a")
        #expect(run.status == 0, "\(run.output)")
        #expect(
            !run.output.contains(Self.bodySentinel),
            """
            the probe printed object content. Output:
            \(run.output)
            """
        )
        #expect(run.output.contains("FOUND"))
    }

    // MARK: - F12: where the hidden bytes live, and for how long

    /// Run-scoped, deterministic, 0700, and gone when the step returns —
    /// instead of an opaque `mktemp -d` under the runner's `$TMPDIR`, which on
    /// a self-hosted box persists and is not findable by name.
    @Test
    func probeScratchIsRunScopedAndRemovedWhenTheStepEnds() throws {
        let sandbox = try Sandbox()
        try Self.stageDownloader(sandbox, objects: ["a": Self.bodySentinel])

        let scratchParent = try sandbox.dir("scratch")
        let work = scratchParent.appendingPathComponent("mlxfast-dflash-probe-1-1")
        let run = try Self.runProbeStep(sandbox, keys: "a", privateDir: work)
        #expect(run.status == 0, "\(run.output)")

        #expect(
            !FileManager.default.fileExists(atPath: work.path),
            "the probe scratch directory survived the step: \(work.path)"
        )
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: scratchParent.path)
        #expect(leftovers.isEmpty, "scratch residue: \(leftovers)")

        // The job binds a run-scoped path, so a leak is findable by run id.
        let value = try #require(Self.jobEnv()["MLXFAST_PROBE_PRIVATE_DIR"])
        #expect(
            value.contains("github.run_id") && value.contains("github.run_attempt"),
            """
            MLXFAST_PROBE_PRIVATE_DIR is '\(value)'. It must be scoped by run id \
            and attempt, so two probes cannot share a directory and a leaked \
            one names the run that leaked it.
            """
        )
        #expect(
            value.contains("mlxfast-dflash-probe-"),
            "the scratch path must carry the prefix the step's sweep matches on"
        )
    }

    /// The step sweeps scratch a SIGKILLed predecessor left behind — the case
    /// the trap provably does not cover — and touches nothing else.
    @Test
    func theProbeSweepsAnEarlierRunsAbandonedScratch() throws {
        let sandbox = try Sandbox()
        try Self.stageDownloader(sandbox, objects: ["a": "AAA"])
        let fm = FileManager.default
        let scratchParent = try sandbox.dir("scratch")

        let abandoned = scratchParent.appendingPathComponent("mlxfast-dflash-probe-9-1")
        try fm.createDirectory(at: abandoned, withIntermediateDirectories: true)
        try Self.bodySentinel.write(
            to: abandoned.appendingPathComponent("probe.bin"), atomically: true, encoding: .utf8
        )
        let unrelated = scratchParent.appendingPathComponent("someone-elses-dir")
        try fm.createDirectory(at: unrelated, withIntermediateDirectories: true)

        let run = try Self.runProbeStep(
            sandbox, keys: "a",
            privateDir: scratchParent.appendingPathComponent("mlxfast-dflash-probe-10-1")
        )
        #expect(run.status == 0, "\(run.output)")
        #expect(
            !fm.fileExists(atPath: abandoned.path),
            """
            an earlier probe's abandoned scratch survived. A SIGKILL runs no \
            trap and no `if: always()` step, and the janitor purges BENCH-owned \
            scratch, not this runner-owned directory — the sweep is the only \
            thing that reaps it.
            """
        )
        #expect(
            fm.fileExists(atPath: unrelated.path),
            "the sweep removed a directory that is not probe scratch"
        )
    }

    /// The sweep matches a literal prefix, so a renamed
    /// `MLXFAST_PROBE_PRIVATE_DIR` would silently stop reaping. Refuse instead.
    @Test
    func aScratchPathWithoutTheSweptPrefixIsRefused() throws {
        let sandbox = try Sandbox()
        try Self.stageDownloader(sandbox, objects: ["a": "AAA"])
        let run = try Self.runProbeStep(
            sandbox, keys: "a",
            privateDir: try sandbox.dir("scratch").appendingPathComponent("renamed")
        )
        #expect(
            run.status != 0,
            """
            the probe accepted a scratch path the sweep cannot match, so a \
            rename would disable the sweep silently. Output:
            \(run.output)
            """
        )
        #expect(run.output.contains("mlxfast-dflash-probe-"))
    }

    /// The `if: always()` backstop, executed. The in-step trap does not run when
    /// GitHub cancels the job; this step does.
    @Test
    func theCleanupStepRemovesScratchTheStepShellDidNot() throws {
        let steps = try Self.steps()
        let cleanup = try #require(
            steps.first { $0.name == Self.scratchStep },
            "the probe declares no '\(Self.scratchStep)' step"
        )
        #expect(
            cleanup.body.contains("if: always()"),
            """
            '\(Self.scratchStep)' is not `if: always()`, so it does not run when \
            the Probe step fails or the job is cancelled — the only cases it \
            exists for.
            """
        )
        let probeIndex = try #require(steps.firstIndex { $0.name == Self.probeStep })
        let cleanupIndex = try #require(steps.firstIndex { $0.name == Self.scratchStep })
        #expect(cleanupIndex > probeIndex)

        // Executed against a directory the Probe step's trap never got to.
        let sandbox = try Sandbox()
        let work = try sandbox.dir("scratch").appendingPathComponent("mlxfast-dflash-probe-1-1")
        try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        try Self.bodySentinel.write(
            to: work.appendingPathComponent("probe.bin"), atomically: true, encoding: .utf8
        )

        let run = try Self.bash(
            try Self.stepBody(Self.scratchStep),
            cwd: sandbox.root,
            env: ["MLXFAST_PROBE_PRIVATE_DIR": work.path]
        )
        #expect(run.status == 0, "the cleanup step failed: \(run.output)")
        #expect(
            !FileManager.default.fileExists(atPath: work.path),
            "the cleanup step left hidden bytes at \(work.path)"
        )

        // And it fails loudly rather than reporting success on an unset path.
        let unset = try Self.bash(
            try Self.stepBody(Self.scratchStep), cwd: sandbox.root, env: [:]
        )
        #expect(unset.status != 0, "the cleanup step reported success with no path bound")
    }

    /// The comment must not promise what the code cannot do. The file used to
    /// say "Never keep hidden bytes around, even briefly" next to a trap that a
    /// SIGKILL skips entirely.
    @Test
    func theWorkflowDoesNotClaimHiddenBytesAreAlwaysRemoved() throws {
        let text = try Self.workflow()
        #expect(
            !text.contains("Never keep hidden bytes around, even briefly"),
            """
            the overclaiming comment is back. `trap ... EXIT` does not run on \
            SIGKILL and the janitor reaps bench-owned scratch, not this \
            runner-owned directory; say what is guaranteed and what is not.
            """
        )
        #expect(
            text.contains("SIGKILL"),
            "the workflow no longer names the case its cleanup does not cover"
        )
    }
}

// MARK: - harness

extension DFlashProbeWorkflowTests {
    struct Run {
        var status: Int32
        var output: String
    }

    final class Sandbox {
        let root: URL
        init() throws {
            root = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("dflash-probe-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        }
        deinit { try? FileManager.default.removeItem(at: root) }
        func dir(_ name: String) throws -> URL {
            let url = root.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            return url
        }
    }

    struct Step {
        var name: String
        var body: String
    }

    static var repoRoot: URL {
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }

    static func workflow() throws -> String {
        try String(contentsOfFile: workflowPath, encoding: .utf8)
    }

    static func bash(
        _ script: String,
        cwd: URL,
        env: [String: String]
    ) throws -> Run {
        var environment = env
        if environment["PATH"] == nil {
            let inherited = ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin"
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
        return Run(status: process.terminationStatus, output: String(decoding: data, as: UTF8.self))
    }

    /// Executes the REAL guard script with a synthesised Actions context.
    static func runTrustedGuard(
        ref: String,
        event: String = "workflow_dispatch",
        repository: String = "Layr-Labs/mlxfast-challenge-dev",
        workflow: String = "dflash-probe-r2-keys.yml",
        unset: String? = nil
    ) throws -> Run {
        var env = [
            "GITHUB_REPOSITORY": repository,
            "GITHUB_REF": ref,
            "GITHUB_WORKFLOW_REF":
                "\(repository)/.github/workflows/\(workflow)@\(ref)",
            "GITHUB_EVENT_NAME": event,
        ]
        if let unset { env.removeValue(forKey: unset) }
        return try bash(
            repoRoot.appendingPathComponent(trustedScript).path, cwd: repoRoot, env: env
        )
    }

    /// A checkout-shaped sandbox with a stub downloader: it writes the staged
    /// body for a known key and reports `NoSuchKey` otherwise, so the Probe step
    /// can be executed without credentials or a network.
    ///
    /// The control key is always staged unless the caller overrides it, because
    /// the step probes it first and a missing control muddies every assertion.
    @discardableResult
    static func stageDownloader(_ sandbox: Sandbox, objects: [String: String]) throws -> URL {
        let scripts = try sandbox.dir(".github/scripts")
        let objectDir = try sandbox.dir("objects")
        let fm = FileManager.default

        var staged = objects
        if staged["control"] == nil {
            staged[controlKey] = "control-object-body"
        } else {
            staged[controlKey] = staged.removeValue(forKey: "control")
        }
        for (key, body) in staged {
            try body.write(
                to: objectDir.appendingPathComponent(key.replacingOccurrences(of: "/", with: "_")),
                atomically: true,
                encoding: .utf8
            )
        }

        let stub = """
            #!/bin/bash
            set -u
            src="${STUB_OBJECT_DIR}/$(printf '%s' "${1-}" | tr '/' '_')"
            if [[ -z "${1-}" || ! -f "${src}" ]]; then
              echo "stub: HTTP 404 <Code>NoSuchKey</Code> for '${1-}'" >&2
              exit 22
            fi
            cp "${src}" "${2}"
            """
        let stubURL = scripts.appendingPathComponent("download-r2-object.sh")
        try stub.write(to: stubURL, atomically: true, encoding: .utf8)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: stubURL.path)
        return objectDir
    }

    /// The control key the Probe step hardcodes.
    static let controlKey =
        "correctness_prompts/laguna-xs-2.1-serial-v2/"
        + "hidden-correctness-golden-"
        + "94239d59b435eb8f370c82bcf8c86822d1bbc1094e3650aeff3abc5558137023.json"

    /// Runs the Probe step's real `run:` body (or a caller-supplied mutation of
    /// it) in the sandbox.
    static func runProbeStep(
        _ sandbox: Sandbox,
        keys: String,
        body: String? = nil,
        privateDir: URL? = nil
    ) throws -> Run {
        let work =
            try privateDir
            ?? sandbox.dir("scratch").appendingPathComponent("mlxfast-dflash-probe-1-1")
        return try bash(
            body ?? (try stepBody(probeStep)),
            cwd: sandbox.root,
            env: [
                "R2_ACCESS_KEY_ID": "stub-access-key",
                "R2_SECRET_ACCESS_KEY": "stub-secret",
                "R2_BUCKET_ENDPOINT": "https://stub.invalid/bucket",
                "CANDIDATE_KEYS": keys,
                "STUB_OBJECT_DIR": sandbox.root.appendingPathComponent("objects").path,
                "MLXFAST_PROBE_PRIVATE_DIR": work.path,
            ]
        )
    }
}

// MARK: - workflow parsing

extension DFlashProbeWorkflowTests {
    /// The single job's steps, in order, each as its raw YAML block.
    static func steps() throws -> [Step] {
        let lines = try workflow().split(separator: "\n", omittingEmptySubsequences: false)
        guard let start = lines.firstIndex(where: { $0 == "    steps:" }) else {
            Issue.record("no steps: block in \(workflowPath)")
            return []
        }
        var result: [Step] = []
        var current: [Substring] = []
        var currentName: String?

        func flush() {
            if let name = currentName {
                result.append(Step(name: name, body: current.joined(separator: "\n")))
            }
            current = []
        }

        for line in lines[(start + 1)...] {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty, line.prefix(while: { $0 == " " }).count < 6 { break }
            if line.hasPrefix("      - ") {
                flush()
                // `- name: X` or a bare `- uses: X`.
                if let range = line.range(of: "- name: ") {
                    currentName = String(line[range.upperBound...])
                        .trimmingCharacters(in: .whitespaces)
                } else {
                    currentName = trimmed
                }
            }
            if currentName != nil { current.append(line) }
        }
        flush()
        return result
    }

    /// A step's `run:` block scalar, dedented, comments preserved (it is
    /// executed, so stripping them would change what runs).
    static func stepBody(_ step: String) throws -> String {
        let block = try #require(
            try steps().first { $0.name == step },
            "step '\(step)' is missing from \(workflowPath)"
        ).body
        let marker = try #require(
            block.range(of: "run: |\n"), "step '\(step)' has no literal run: block"
        )
        let scalar = String(block[marker.upperBound...])
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

    /// The job-level `env:` map, comments excluded.
    static func jobEnv() throws -> [String: String] {
        let lines = try workflow().split(separator: "\n", omittingEmptySubsequences: false)
        guard let start = lines.firstIndex(where: { $0 == "    env:" }) else { return [:] }
        var result: [String: String] = [:]
        for line in lines[(start + 1)...] {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }
            let indent = line.prefix { $0 == " " }.count
            if indent < 6 { break }
            if trimmed.hasPrefix("#") { continue }
            guard indent == 6, let colon = line.firstIndex(of: ":") else { continue }
            let key = line[line.startIndex ..< colon].trimmingCharacters(in: .whitespaces)
            result[key] = String(line[line.index(after: colon)...])
                .trimmingCharacters(in: .whitespaces)
        }
        return result
    }

    /// Top-level trigger keys under `on:`.
    static func triggerKeys() throws -> [String] {
        let lines = try workflow().split(separator: "\n", omittingEmptySubsequences: false)
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
}
