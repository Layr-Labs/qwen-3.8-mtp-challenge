import Foundation
import Testing

/// Everything `benchmark-dflash.sh` extracts from `benchmark.sh` by name. Each
/// entry has to fail closed ON ITS OWN -- see
/// `everyReusedDefinitionFailsClosedOnItsOwn`.
private let dflashReusedDefinitions = [
    "local_run_lock_path",
    "acquire_local_run_lock",
    "release_local_run_lock",
    "list_resident_model_processes",
    "abort_if_model_already_resident",
]

/// `benchmark-dflash.sh` must take the SAME local run lock as `benchmark.sh`.
///
/// Why: both local modes hold the ~21.6 GB target plus the drafter, so two
/// overlapping local runs can out-of-memory the machine, and two runs sharing
/// one GPU invalidate both timings. benchmark.sh guarded only one direction of
/// this -- its resident-model scan lists the `dflash-*` subcommands, so a
/// serial run refuses to start against a live DFlash one -- while a DFlash run
/// started happily against a live serial run, because this script took no lock
/// at all.
///
/// The lock is only HALF of benchmark.sh's guard, and the other half is the
/// one that covers the case a lock structurally cannot: an ORPHANED
/// model-holding worker left by a run that died without releasing. Nothing
/// holds the lock, so the lock is free, and a DFlash run starts a second
/// ~21.6 GB copy on top of the orphan. `abort_if_model_already_resident` /
/// `list_resident_model_processes` / `RESIDENT_MODEL_PROCESS_PATTERN` are that
/// half, and benchmark-dflash.sh must reuse them too.
///
/// The reuse itself must fail closed PER ARM. Checking sentinel strings against
/// the CONCATENATION of the extractions does not: one broken arm is covered by
/// the others, the eval succeeds, and the missing function surfaces later as
/// `command not found` -- for `release_local_run_lock`, inside the EXIT trap,
/// where `set -e` abandons the rest of cleanup() and strands both the lock and
/// the scratch directory.
///
/// Some tests below pin the reuse as text (the lock PATH and the process
/// PATTERN are what must not drift). The ones that matter most DRIVE THE REAL
/// SCRIPT: `benchmark-dflash.sh` is run in a throwaway repo root, far enough to
/// reach its guard, and the guard's actual behaviour is observed.
@Suite("DFlash local run lock")
struct DFlashLocalRunLockTests {
    private static let dflashScriptPath = "benchmark-dflash.sh"
    private static let serialScriptPath = "benchmark.sh"
    private static let reusedDefinitions = dflashReusedDefinitions

    private static func text(_ path: String) throws -> String {
        try String(contentsOfFile: path, encoding: .utf8)
    }

    /// Lines with the leading `#` comment stripped, so an assertion cannot be
    /// satisfied by prose that merely mentions the thing.
    private static func executable(_ script: String) -> String {
        script
            .split(separator: "\n", omittingEmptySubsequences: false)
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("#") }
            .joined(separator: "\n")
    }

    @Test
    func dflashAcquiresAndReleasesTheLocalRunLock() throws {
        let script = Self.executable(try Self.text(Self.dflashScriptPath))

        #expect(
            script.contains("acquire_local_run_lock"),
            """
            benchmark-dflash.sh does not acquire the local run lock. Both of its \
            modes hold the ~21.6 GB target plus the drafter; an overlapping local \
            run can out-of-memory the machine.
            """
        )
        #expect(
            script.contains("release_local_run_lock"),
            "benchmark-dflash.sh acquires the run lock but never releases it"
        )

        // Released from the EXIT trap, not just on the success path: a run that
        // aborts must not strand the lock and wedge the next one.
        let cleanup = try #require(
            script.range(of: "cleanup() {"),
            "benchmark-dflash.sh has no cleanup() function to release the lock from"
        )
        let trap = try #require(
            script.range(of: "trap cleanup EXIT"),
            "benchmark-dflash.sh never arms its cleanup trap"
        )
        let cleanupBody = String(script[cleanup.upperBound..<trap.lowerBound])
        #expect(
            cleanupBody.contains("release_local_run_lock"),
            """
            benchmark-dflash.sh releases the run lock outside cleanup(), so an \
            aborted run strands it and the next run refuses to start.
            """
        )

        // Acquired only once the trap can release it.
        let acquire = try #require(script.range(of: "acquire_local_run_lock"))
        #expect(
            acquire.lowerBound > trap.lowerBound
                || cleanupBody.contains("release_local_run_lock"),
            "the lock is acquired before its release trap is armed"
        )
    }

    @Test
    func dflashReusesTheSerialLockDefinitionsRatherThanCopyingThem() throws {
        let dflash = Self.executable(try Self.text(Self.dflashScriptPath))

        // Reuse, by the same awk-extract-and-eval idiom the source_hash() reuse
        // uses. A second implementation of the lock PATH is the failure this
        // asserts against: both scripts would hold "a lock" and exclude nothing.
        for definition in Self.reusedDefinitions {
            // Bound to a Bool first: `#expect` renders every sub-expression it
            // evaluates, and `dflash` is the whole script.
            let isExtracted = dflash.contains(
                "awk '/^\(definition)\\(\\) \\{/,/^\\}/' benchmark.sh"
            )
            #expect(
                isExtracted,
                """
                benchmark-dflash.sh does not extract \(definition)() from \
                benchmark.sh. Reuse the single definition -- a copied lock path \
                that drifts excludes nothing.
                """
            )
        }
        // The lock filename may appear ONLY as the string the extraction is
        // sanity-checked against -- never in a path this script builds itself.
        // (That sanity check is the fail-closed guard, so it is the one
        // legitimate mention; excluding it is what keeps this assertion about
        // drift rather than about vocabulary.)
        let selfSpelled = dflash
            .split(separator: "\n", omittingEmptySubsequences: false)
            .filter { $0.contains("mlxfast-local-benchmark-") }
            .filter { !$0.contains("run_lock_definitions") }
        #expect(
            selfSpelled.isEmpty,
            """
            benchmark-dflash.sh spells the lock filename itself instead of \
            reusing local_run_lock_path(): \
            \(selfSpelled.map { $0.trimmingCharacters(in: .whitespaces) }.joined(separator: " | ")). \
            Two spellings of the path is exactly the drift this reuse prevents.
            """
        )
        #expect(
            dflash.contains("eval \"${run_lock_definitions}\""),
            "the extracted lock definitions are never evaluated"
        )

        // Fails CLOSED: if benchmark.sh is refactored and the extraction stops
        // finding the definitions, the script must refuse to run rather than
        // proceed unguarded.
        let guardStart = try #require(
            dflash.range(of: "run_lock_definitions=\"$("),
            "the lock extraction vanished"
        )
        let after = String(dflash[guardStart.lowerBound...])
        let sanityCheck = try #require(
            after.range(of: "refusing to run unguarded"),
            """
            benchmark-dflash.sh does not fail closed when the lock extraction \
            comes up empty -- a refactor of benchmark.sh would silently leave \
            DFlash local runs unguarded.
            """
        )
        let checkTail = String(after[sanityCheck.lowerBound...])
        #expect(
            checkTail.contains("exit 1"),
            "the failed-extraction path warns instead of exiting"
        )
    }

    @Test
    func serialStillDefinesTheFunctionsDFlashExtracts() throws {
        // The other half of the coupling: this is what turns a benchmark.sh
        // refactor into a red test here instead of an unguarded DFlash run
        // discovered at OOM time.
        let serial = try Self.text(Self.serialScriptPath)
        for definition in Self.reusedDefinitions {
            #expect(
                serial.contains("\n\(definition)() {"),
                """
                benchmark.sh no longer defines \(definition)() at the top level, \
                so benchmark-dflash.sh's extraction of it fails and DFlash local \
                runs refuse to start. Update both scripts together.
                """
            )
        }
        // The resident-process pattern is extracted as a whole line, so it has
        // to stay a top-level `readonly NAME=` declaration.
        #expect(
            serial.contains("\nreadonly RESIDENT_MODEL_PROCESS_PATTERN="),
            """
            benchmark.sh no longer declares RESIDENT_MODEL_PROCESS_PATTERN as a \
            top-level `readonly NAME=` line, so benchmark-dflash.sh extracts no \
            pattern and its orphan scan cannot match anything.
            """
        )
        #expect(serial.contains("mlxfast-local-benchmark-"))
        #expect(serial.contains("LOCAL_RUN_LOCK_OWNED="))
    }

    /// The orphan half of the guard: extracted, and CALLED, and called before
    /// the lock is taken (an orphan holds no lock, so the lock would be handed
    /// out and given straight back).
    @Test
    func dflashImportsAndCallsTheResidentModelScan() throws {
        let dflash = Self.executable(try Self.text(Self.dflashScriptPath))

        // Bound to Bools first: `#expect` renders every sub-expression it
        // evaluates, and `dflash` is the whole script.
        let extractsPattern = dflash.contains(
            "awk '/^readonly RESIDENT_MODEL_PROCESS_PATTERN=/' benchmark.sh"
        )
        #expect(
            extractsPattern,
            """
            benchmark-dflash.sh does not extract RESIDENT_MODEL_PROCESS_PATTERN \
            from benchmark.sh. Without it the imported scan matches nothing.
            """
        )
        // Spelling the pattern here instead of extracting it is the drift that
        // makes the scan miss whichever subcommands benchmark.sh learned last.
        let spellsPatternItself = dflash.contains("runtime-worker[[:space:]]")
        #expect(
            !spellsPatternItself,
            """
            benchmark-dflash.sh spells the resident-process pattern itself \
            instead of reusing benchmark.sh's. A second, staler copy misses \
            exactly the subcommands benchmark.sh has since added.
            """
        )

        let lines = dflash.split(separator: "\n", omittingEmptySubsequences: false)
        func callSite(_ name: String) -> Int? {
            lines.firstIndex { $0.trimmingCharacters(in: .whitespaces) == name }
        }
        let scanCall = try #require(
            callSite("abort_if_model_already_resident"),
            """
            benchmark-dflash.sh never CALLS abort_if_model_already_resident. \
            Importing the scan without calling it leaves the orphan case -- the \
            one no lock can catch -- unguarded.
            """
        )
        let acquireCall = try #require(
            callSite("acquire_local_run_lock"),
            "benchmark-dflash.sh never calls acquire_local_run_lock"
        )
        #expect(
            scanCall < acquireCall,
            """
            benchmark-dflash.sh takes the run lock before scanning for a \
            resident model. An orphan holds no lock, so the lock is granted and \
            immediately abandoned, and the operator is told about a lock rather \
            than about the live pid they need to kill.
            """
        )
    }

    /// The lock only helps if the two scripts agree it is the same lock. Drives
    /// the real extraction and asserts the resulting path matches the one
    /// benchmark.sh computes for the same environment.
    @Test
    func extractedLockPathMatchesTheSerialLockPath() throws {
        let lockRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("dflash-lock-parity-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: lockRoot) }

        // Left: the path benchmark.sh's own definition produces. Right: the
        // path that same definition produces after DFlash's awk extraction and
        // eval. Any divergence means the two scripts lock different files.
        let program = """
        set -euo pipefail
        export MLXFAST_LOCAL_RUN_LOCK_DIR="\(lockRoot.path)"
        serial="$(bash -c 'eval "$(awk "/^local_run_lock_path\\(\\) \\{/,/^\\}/" benchmark.sh)"; local_run_lock_path')"
        defs="$(
          awk '/^local_run_lock_path\\(\\) \\{/,/^\\}/' benchmark.sh
          awk '/^acquire_local_run_lock\\(\\) \\{/,/^\\}/' benchmark.sh
          awk '/^release_local_run_lock\\(\\) \\{/,/^\\}/' benchmark.sh
        )"
        LOCAL_RUN_LOCK_OWNED=""
        local_run_guard_enabled() { [[ "${MLXFAST_LOCAL_RUN_GUARD:-1}" != "0" ]]; }
        eval "${defs}"
        dflash="$(local_run_lock_path)"
        [[ "${serial}" == "${dflash}" ]] || { echo "MISMATCH ${serial} != ${dflash}"; exit 1; }
        acquire_local_run_lock >/dev/null 2>&1 || { echo "ACQUIRE-FAILED"; exit 1; }
        [[ -d "${dflash}" ]] || { echo "NO-LOCK-DIR"; exit 1; }
        ( LOCAL_RUN_LOCK_OWNED=""; acquire_local_run_lock ) >/dev/null 2>&1 \\
          && { echo "DOUBLE-ACQUIRED"; exit 1; }
        release_local_run_lock
        [[ -d "${dflash}" ]] && { echo "NOT-RELEASED"; exit 1; }
        echo "OK ${dflash}"
        """

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["-c", program]
        process.currentDirectoryURL = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath
        )
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let output = String(
            decoding: pipe.fileHandleForReading.readDataToEndOfFile(),
            as: UTF8.self
        )
        process.waitUntilExit()

        #expect(
            process.terminationStatus == 0,
            """
            the lock path benchmark-dflash.sh extracts diverges from \
            benchmark.sh's, or the extracted lock does not exclude a second \
            holder: \(output.trimmingCharacters(in: .whitespacesAndNewlines))
            """
        )
        #expect(output.contains("mlxfast-local-benchmark-"), "\(output)")
    }

    // MARK: - Behaviour: the real script, driven to its guard

    /// The case no lock can catch. A model-holding worker orphaned by a run
    /// that died without releasing holds NO lock, so the lock is free; only the
    /// resident scan sees it.
    @Test
    func dflashRefusesToStartAgainstAnOrphanedModelHoldingWorker() throws {
        let sandbox = try GuardSandbox()
        defer { sandbox.destroy() }

        let orphan = "4242 1 22600000 mlxfast-runtime-worker runtime-worker --weights weights"
        let run = try sandbox.runGuard(orphanScan: "printf '\(orphan)\\n'")

        #expect(
            run.status != 0,
            """
            benchmark-dflash.sh started while a model-holding worker was already \
            resident. Nothing holds the lock in the orphan case, so the lock \
            cannot see it -- only the resident scan can, and this run started a \
            second ~21.6 GB residency: \(run.output)
            """
        )
        #expect(
            run.output.contains("a model-holding mlxfast process is already running"),
            "expected benchmark.sh's resident-model abort, got: \(run.output)"
        )
        #expect(
            run.output.contains("4242 1 22600000"),
            "the abort does not name the resident process: \(run.output)"
        )
        #expect(
            !run.reachedTheFirstGate,
            "the run continued into the tripwire despite the orphan: \(run.output)"
        )
        // Aborting is not enough: an abort that strands scratch is what
        // `git add -A` sweeps into a submission diff.
        #expect(!sandbox.workRootExists, "the aborted run left its work root behind")
        #expect(sandbox.lockDirEntries.isEmpty, "the aborted run left a lock behind")
    }

    /// Same abort, but through the REAL pgrep path against a REAL live process,
    /// so the extracted `readonly RESIDENT_MODEL_PROCESS_PATTERN` is what
    /// decides the match. Proves the readonly declaration survives re-evaluation
    /// in benchmark-dflash.sh's shell and carries its value across.
    ///
    /// The sandbox's own copy of benchmark.sh is rewritten to a pattern unique
    /// to this test, and the decoy is a sleeping shell script named after it, so
    /// the test cannot match -- or be matched by -- anything else on the machine.
    @Test
    func dflashOrphanScanMatchesThroughTheExtractedResidentPattern() throws {
        let sandbox = try GuardSandbox()
        defer { sandbox.destroy() }

        let token = "dflash-guard-decoy-\(UUID().uuidString.lowercased())"
        try sandbox.rewriteSerialScript { serial in
            serial
                .split(separator: "\n", omittingEmptySubsequences: false)
                .map {
                    $0.hasPrefix("readonly RESIDENT_MODEL_PROCESS_PATTERN=")
                        ? "readonly RESIDENT_MODEL_PROCESS_PATTERN='\(token)'"
                        : String($0)
                }
                .joined(separator: "\n")
        }

        let decoy = try sandbox.startDecoyProcess(named: token)
        defer { decoy.terminate(); decoy.waitUntilExit() }

        // No MLXFAST_LOCAL_ORPHAN_SCAN_CMD: this exercises pgrep for real.
        let run = try sandbox.runGuard(orphanScan: nil)

        #expect(
            run.status != 0 && !run.reachedTheFirstGate,
            """
            the extracted resident-process pattern matched nothing, so the \
            orphan scan is inert even though it is wired in: \(run.output)
            """
        )
        #expect(
            run.output.contains(token),
            "the abort does not name the live decoy process: \(run.output)"
        )
        #expect(
            run.output.contains("\(decoy.processIdentifier)"),
            "the abort does not name the live decoy pid \(decoy.processIdentifier): \(run.output)"
        )
    }

    /// Control: with nothing resident the guard must let the run through, so
    /// the tests above are measuring a guard and not a script that always dies.
    @Test
    func dflashRunsPastTheGuardWhenNothingIsResident() throws {
        let sandbox = try GuardSandbox()
        defer { sandbox.destroy() }

        let run = try sandbox.runGuard(orphanScan: "true")

        #expect(
            run.reachedTheFirstGate,
            """
            benchmark-dflash.sh did not reach its first gate on a clean machine, \
            so the guard aborts unconditionally: \(run.output)
            """
        )
        #expect(
            !run.output.contains("refusing to run unguarded"),
            "the reuse failed closed on an unmodified benchmark.sh: \(run.output)"
        )
    }

    /// Every extracted name must fail closed BY ITSELF.
    ///
    /// The superseded check tested two sentinel strings against the
    /// CONCATENATION of the extractions. Break one arm and the others still
    /// supply both sentinels, so the eval succeeded and the run continued: for
    /// `release_local_run_lock` the failure landed inside the EXIT trap, where
    /// `set -e` abandoned the rest of cleanup(), stranding the lock and the
    /// scratch directory; for the two scan functions the run simply proceeded
    /// unguarded.
    @Test(arguments: dflashReusedDefinitions)
    func everyReusedDefinitionFailsClosedOnItsOwn(_ definition: String) throws {
        let sandbox = try GuardSandbox()
        defer { sandbox.destroy() }

        // A rename is what a real refactor looks like to the awk extraction:
        // this one arm produces nothing, every other arm is untouched.
        try sandbox.rewriteSerialScript { serial in
            serial.replacingOccurrences(
                of: "\n\(definition)() {",
                with: "\n\(definition)_renamed_by_refactor() {"
            )
        }

        let run = try sandbox.runGuard(orphanScan: "true")

        #expect(
            !run.reachedTheFirstGate,
            """
            benchmark.sh's \(definition)() was renamed, its extraction produced \
            nothing, and benchmark-dflash.sh ran anyway. The other arms covered \
            for it: \(run.output)
            """
        )
        #expect(
            run.status == 1,
            "expected a deliberate exit 1, got status \(run.status): \(run.output)"
        )
        #expect(
            run.output.contains("could not reuse benchmark.sh's \(definition)()"),
            """
            the missing \(definition)() was not reported by name -- so it was \
            not checked by name: \(run.output)
            """
        )
        #expect(
            run.output.contains("refusing to run unguarded"),
            "the failure is not spelled as a fail-closed refusal: \(run.output)"
        )
        #expect(
            !run.output.contains("command not found"),
            """
            the missing \(definition)() surfaced as a runtime `command not \
            found` instead of a pre-flight refusal: \(run.output)
            """
        )
        #expect(!sandbox.workRootExists, "the refusal left its work root behind")
        #expect(sandbox.lockDirEntries.isEmpty, "the refusal left a lock behind")
    }

    /// The pattern arm needs its own check: it is a variable, not a function,
    /// and an EMPTY pattern is worse than a missing one. `pgrep -f ''` under
    /// `set -u` aborts inside the scan's `|| true`, the scan reports a clean
    /// machine, and the guard passes silently.
    @Test
    func aMissingResidentPatternFailsClosedRatherThanScanningForNothing() throws {
        let sandbox = try GuardSandbox()
        defer { sandbox.destroy() }

        try sandbox.rewriteSerialScript { serial in
            serial
                .split(separator: "\n", omittingEmptySubsequences: false)
                .filter { !$0.hasPrefix("readonly RESIDENT_MODEL_PROCESS_PATTERN=") }
                .joined(separator: "\n")
        }

        let run = try sandbox.runGuard(orphanScan: nil)

        #expect(
            !run.reachedTheFirstGate,
            """
            benchmark.sh's RESIDENT_MODEL_PROCESS_PATTERN vanished and \
            benchmark-dflash.sh ran anyway, with a scan that can no longer match \
            anything: \(run.output)
            """
        )
        #expect(
            run.output.contains(
                "could not reuse benchmark.sh's RESIDENT_MODEL_PROCESS_PATTERN"
            ),
            "the empty pattern was not reported: \(run.output)"
        )
    }
}

// MARK: - Executing harness

/// A throwaway repo root that satisfies everything `benchmark-dflash.sh` checks
/// before its local run guard, so the REAL script (copied byte-for-byte, not
/// re-implemented) can be driven as far as the guard and no further.
///
/// Nothing here loads a model: the trusted CLI is a stub that the guard never
/// reaches, and the run always dies at the first gate. benchmark.sh is COPIED
/// rather than symlinked so a test can mutate it to simulate a refactor.
private struct GuardSandbox {
    struct Run {
        let status: Int32
        let output: String

        /// True once execution is past the guard and into step 1 of the script.
        var reachedTheFirstGate: Bool { output.contains("public drift tripwire") }
    }

    let root: URL
    private let repoRoot: URL

    init() throws {
        let fm = FileManager.default
        repoRoot = URL(fileURLWithPath: fm.currentDirectoryPath)
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("dflash-guard-\(UUID().uuidString)")

        for directory in [
            "drafter", "weights", "lockdir", "bin",
            "Sources/MLXFastCore", "Sources/MLXFastTransform",
        ] {
            try fm.createDirectory(
                at: root.appendingPathComponent(directory),
                withIntermediateDirectories: true
            )
        }
        for script in ["benchmark-dflash.sh", "benchmark.sh", "setup-dflash.sh"] {
            try fm.copyItem(
                at: repoRoot.appendingPathComponent(script),
                to: root.appendingPathComponent(script)
            )
            try fm.setAttributes(
                [.posixPermissions: 0o755],
                ofItemAtPath: root.appendingPathComponent(script).path
            )
        }
        // The trusted CLI only has to exist and be executable: the guard runs
        // before any gate invokes it.
        try write("#!/bin/sh\necho \"STUB-SWIFT $*\" >&2\nexit 9\n", to: "bin/mlxfast-swift", executable: true)
        // jq is required by the preamble and unused by the guard. Stubbing it
        // keeps this test off the developer's tool inventory.
        try write("#!/bin/sh\nexit 0\n", to: "bin/jq", executable: true)
        try write("{}\n", to: "drafter/config.json")
        try write("{}\n", to: "weights/config.json")
        // source_hash()'s inputs, so the transformed-weights staleness check
        // has something to digest.
        try write("", to: "Package.swift")
        try write("", to: "Package.resolved")
        try write("", to: "Sources/MLXFastCore/placeholder")
        try write("", to: "Sources/MLXFastTransform/placeholder")
        try sealWeights()
    }

    func destroy() {
        try? FileManager.default.removeItem(at: root)
    }

    var workRootExists: Bool {
        FileManager.default.fileExists(atPath: root.appendingPathComponent("work").path)
    }

    var lockDirEntries: [String] {
        (try? FileManager.default.contentsOfDirectory(
            atPath: root.appendingPathComponent("lockdir").path
        )) ?? []
    }

    /// Mutate the sandbox's copy of benchmark.sh, then re-seal the weights
    /// digest (source_hash() itself is extracted from that file).
    func rewriteSerialScript(_ transform: (String) -> String) throws {
        let path = root.appendingPathComponent("benchmark.sh")
        let updated = transform(try String(contentsOf: path, encoding: .utf8))
        try updated.write(to: path, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: path.path
        )
        try sealWeights()
    }

    /// A live process whose argv contains `name` and nothing else recognisable,
    /// so a real `pgrep -f` can find it without any real mlxfast process being
    /// involved.
    ///
    /// It blocks on a `read` from a pipe rather than sleeping: one process, no
    /// child to outlive the test, and it exits by itself if this test process
    /// dies and the pipe closes.
    func startDecoyProcess(named name: String) throws -> Process {
        let script = "\(name).sh"
        try write("#!/bin/sh\nread ignored\n", to: script, executable: true)
        let process = Process()
        process.executableURL = root.appendingPathComponent(script)
        process.standardInput = Pipe()
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        // pgrep cannot see a process that has not been scheduled yet.
        for _ in 0..<100 {
            if decoyIsVisible(name) { break }
            Thread.sleep(forTimeInterval: 0.02)
        }
        return process
    }

    /// Runs the real `benchmark-dflash.sh --local-iterate` in this sandbox.
    /// `orphanScan` drives benchmark.sh's documented
    /// MLXFAST_LOCAL_ORPHAN_SCAN_CMD seam; pass nil to exercise real pgrep.
    func runGuard(orphanScan: String?) throws -> Run {
        var environment = ProcessInfo.processInfo.environment
        // The developer's own MLXFAST_*/DARKBLOOM_* exports must not decide
        // what this test measures -- MLXFAST_LOCAL_RUN_GUARD=0 in particular
        // would disable the very guard under test.
        let inherited = environment.keys.filter {
            $0.hasPrefix("MLXFAST_") || $0.hasPrefix("DARKBLOOM_")
        }
        for key in inherited {
            environment.removeValue(forKey: key)
        }
        environment["PATH"] =
            root.appendingPathComponent("bin").path + ":"
            + (environment["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin")
        environment["MLXFAST_SWIFT_BIN"] = root.appendingPathComponent("bin/mlxfast-swift").path
        environment["MLXFAST_DFLASH_CONTRACT_PATH"] =
            repoRoot.appendingPathComponent("fixtures/laguna_xs_2_1_dflash_track.json").path
        environment["MLXFAST_DFLASH_LOCAL_GOLDEN_FIXTURE"] =
            repoRoot
            .appendingPathComponent(
                "correctness_prompts/public_longcopy_gate_english_512_256.json"
            ).path
        environment["MLXFAST_DFLASH_DRAFTER_DIR"] = root.appendingPathComponent("drafter").path
        environment["MLXFAST_WEIGHTS_PATH"] = root.appendingPathComponent("weights").path
        environment["MLXFAST_SCORE_PATH"] = root.appendingPathComponent("score.json").path
        environment["MLXFAST_DFLASH_LOCAL_WORK_DIR"] = root.appendingPathComponent("work").path
        environment["MLXFAST_LOCAL_RUN_LOCK_DIR"] = root.appendingPathComponent("lockdir").path
        if let orphanScan {
            environment["MLXFAST_LOCAL_ORPHAN_SCAN_CMD"] = orphanScan
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["./benchmark-dflash.sh", "--local-iterate"]
        process.currentDirectoryURL = root
        process.environment = environment
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let output = String(
            decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self
        )
        process.waitUntilExit()
        return Run(status: process.terminationStatus, output: output)
    }

    // MARK: helpers

    private func write(_ contents: String, to relativePath: String, executable: Bool = false)
        throws
    {
        let path = root.appendingPathComponent(relativePath)
        try contents.write(to: path, atomically: true, encoding: .utf8)
        if executable {
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o755], ofItemAtPath: path.path
            )
        }
    }

    /// Stamp weights/.benchmark-source.sha256 with the digest benchmark-dflash.sh
    /// will compute for this sandbox, using the same extraction it uses itself.
    private func sealWeights() throws {
        let digest = try bash(
            #"eval "$(awk '/^source_hash\(\) \{/,/^\}/' benchmark.sh)"; source_hash"#
        )
        try write(digest, to: "weights/.benchmark-source.sha256")
    }

    private func decoyIsVisible(_ name: String) -> Bool {
        ((try? bash("pgrep -U \"$(id -u)\" -f -- '\(name)' || true")) ?? "").isEmpty == false
    }

    @discardableResult
    private func bash(_ program: String) throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["-c", program]
        process.currentDirectoryURL = root
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        try process.run()
        let out = String(
            decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self
        )
        process.waitUntilExit()
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
