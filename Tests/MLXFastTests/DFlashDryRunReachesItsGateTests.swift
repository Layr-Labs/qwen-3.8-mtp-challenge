import Foundation
import Testing

/// A gates-only dry run must actually reach the DFlash-specific gate.
///
/// #815 split the enablement interlock in two so the DFlash pipeline could be
/// validated WITHOUT flipping the go-live flags: `run_benchmark=false` produces
/// no ranked score, so it is admitted against an inert track. The whole point of
/// that split is to exercise "DFlash correctness and parity gate (untimed)" --
/// the one gate this track owns and the serial pipeline does not.
///
/// It did not work. Two steps sat ABOVE that gate and were NOT gated on
/// `inputs.run_benchmark`:
///
///   * "Select hidden DFlash timed target from the pool", which fails closed on
///     an empty `timed_prompt_pool`; and
///   * the timed-golden half of "Prepare hidden DFlash goldens", which is
///     addressed BY that selection.
///
/// The pool is deliberately empty on main -- pinned by
/// `DFlashEnablementInterlockTests.theTrackFixtureRemainsInertOnMain` -- so
/// before go-live EVERY dispatch died red at pool selection, one step short of
/// the gate the dry run exists to run. The dry run therefore validated nothing
/// the serial pipeline had not already validated, and operators were being
/// trained to read a red dry run as a successful one. A dry run also downloaded
/// a hidden timed golden it would never time against.
///
/// The fix scopes both to the ranked path. The DFlash CORRECTNESS golden is
/// pinned independently of the pool, so the untimed leg can run without it.
///
/// The hazard in the fix is the mirror image: a guard that also lets a RANKED
/// run past an empty pool would destroy contract layer L6 (anti-lottery) --
/// the timed target must be sampled per run or a failed ranked run becomes a
/// free retry and every output-side gate degrades into submit-until-green. So
/// the ranked refusal is executed here, not read, and the ranked path is given
/// a second, independent refusal for the case where the selection step is
/// skipped but the download still runs.
@Suite("DFlash dry run reaches its own gate")
struct DFlashDryRunReachesItsGateTests {
    private static let workflowPath = ".github/workflows/dflash-benchmark.yml"
    private static let jobKey = "dflash-ranked"
    private static let selectionStep = "Select hidden DFlash timed target from the pool"
    private static let goldensStep = "Prepare hidden DFlash goldens"
    private static let parityGate = "DFlash correctness and parity gate (untimed)"
    private static let timedStep = "Timed paired DFlash benchmark (measure-dflash-job)"

    // MARK: - the dry run's step set, computed from the real `if:` expressions

    /// The headline property: a clean gates-only dispatch runs the parity gate
    /// and does not touch the pool.
    @Test
    func aCleanDryRunReachesTheParityGateAndSkipsPoolSelection() throws {
        let workflow = try Self.workflow()
        let steps = try Self.parseSteps(workflow)
        let ran = try Self.stepsThatRun(steps, runBenchmark: false, ref: "refs/heads/main")

        #expect(
            !ran.contains(Self.selectionStep),
            """
            a gates-only dry run still runs '\(Self.selectionStep)'. That step \
            fails closed on the empty timed_prompt_pool that main deliberately \
            ships, so every pre-go-live dry run dies red there -- BEFORE \
            '\(Self.parityGate)'. The interlock split in #815 exists to reach \
            that gate; leaving this step ungated means it never is, and teaches \
            operators to read red as success.
            """
        )
        #expect(
            ran.contains(Self.parityGate),
            """
            a gates-only dry run does not reach '\(Self.parityGate)' -- the only \
            DFlash-specific gate, and the entire reason the dry run exists.
            """
        )
        #expect(
            ran.contains(Self.goldensStep),
            """
            a gates-only dry run no longer prepares the hidden DFlash \
            correctness golden, so '\(Self.parityGate)' would run with no \
            golden installed. Skipping the TIMED golden is the fix; skipping \
            the correctness golden is a different bug wearing its clothes.
            """
        )

        // The ordering hazard is what made the ungated step fatal rather than
        // merely wasteful. Pin it, so a future reordering does not silently
        // remove the reason this guard exists.
        let selectionIndex = try #require(steps.firstIndex { $0.name == Self.selectionStep })
        let gateIndex = try #require(steps.firstIndex { $0.name == Self.parityGate })
        #expect(
            selectionIndex < gateIndex,
            "pool selection no longer precedes the parity gate; re-derive this suite's premise"
        )

        // Nothing a dry run executes may consult the pool at all. This catches
        // the selection being moved into a neighbouring step rather than
        // guarded.
        for name in ran {
            guard let body = Self.stepBlockOrNil(workflow, name) else { continue }
            #expect(
                !body.contains("timed_prompt_pool"),
                """
                dry-run step '\(name)' reads timed_prompt_pool. The pool is \
                empty on main, so any dry-run step that consults it re-creates \
                the fail-closed wall this fix removed.
                """
            )
        }
    }

    /// The mirror: a ranked dispatch must still sample a target and still time.
    @Test
    func aRankedRunStillSelectsATimedTargetAndTimes() throws {
        let steps = try Self.parseSteps(try Self.workflow())
        let ran = try Self.stepsThatRun(
            steps, runBenchmark: true, ref: "refs/heads/submissions/example"
        )

        #expect(
            ran.contains(Self.selectionStep),
            """
            a RANKED dispatch no longer samples the timed target. Contract layer \
            L6 (anti-lottery) requires per-run sampling from the pool: a frozen \
            timed prompt makes a failed ranked run a free retry and turns every \
            output-side gate into submit-until-green.
            """
        )
        #expect(ran.contains(Self.goldensStep))
        #expect(ran.contains(Self.parityGate))
        #expect(ran.contains(Self.timedStep))
    }

    /// The fix's premise: the correctness golden is pinned independently of the
    /// pool. If it were ever addressed by the pool selection, the untimed leg
    /// could not run before the operator uploads pool objects and the guard
    /// above would just move the red wall one step later.
    @Test
    func theCorrectnessGoldenIsPinnedIndependentlyOfThePool() throws {
        let workflow = try Self.workflow()
        let env = try Self.stepEnvironment(workflow, Self.goldensStep)

        let correctnessKey = try #require(
            env["MLXFAST_DFLASH_CORRECTNESS_GOLDEN_R2_PATH"],
            "the correctness golden object key vanished from '\(Self.goldensStep)'"
        )
        #expect(
            !correctnessKey.contains("select_dflash_target") && !correctnessKey.contains("${{"),
            """
            the hidden DFlash CORRECTNESS golden is addressed by an expression \
            (\(correctnessKey)) rather than a fixed pinned key. It must stay \
            pool-independent: it is what lets the untimed leg -- and therefore \
            the parity gate -- run while the pool is still empty.
            """
        )
        // ...while the timed golden is the one that comes from the selection.
        #expect(
            env["MLXFAST_DFLASH_BENCH_GOLDEN_R2_PATH"]?.contains("select_dflash_target") == true,
            "the timed golden is no longer addressed by the per-run pool selection"
        )
    }

    // MARK: - behavioural: the ranked refusal, executed

    /// A ranked run MUST still fail closed on a pool it cannot sample from.
    /// This runs the real step body; it is not a reading of its text.
    @Test
    func aRankedSelectionStillFailsClosedOnAnUnusablePool() throws {
        let cases: [(String, String)] = [
            ("empty pool", #"{"timed_prompt_pool":[]}"#),
            ("pool key absent", #"{"track_id":"laguna-xs-2.1-dflash-v1"}"#),
            ("pool is null", #"{"timed_prompt_pool":null}"#),
            // Not an array: `length` on an object counts its keys, so a mapping
            // with 8 keys would clear a length floor while `[$i]` indexes
            // nothing.
            ("pool is an object", #"{"timed_prompt_pool":{"a":1,"b":2}}"#),
            ("entry is not an object", #"{"timed_prompt_pool":["p/a.json"]}"#),
            (
                "entry sha256 is not a 64-char hex digest",
                #"{"timed_prompt_pool":[{"r2_path":"p/a.json","sha256":"aa","bytes":10}]}"#
            ),
            (
                "entry bytes is zero",
                #"{"timed_prompt_pool":[{"r2_path":"p/a.json","sha256":"\#(String(repeating: "a", count: 64))","bytes":0}]}"#
            ),
            (
                "entry missing sha256",
                #"{"timed_prompt_pool":[{"r2_path":"p/a.json","bytes":10}]}"#
            ),
            (
                "entry with empty r2_path",
                #"{"timed_prompt_pool":[{"r2_path":"","sha256":"aa","bytes":10}]}"#
            ),
        ]

        // Run each case BOTH ways. The 8-distinct-target floor is ranked-only,
        // so asserting refusal on a ranked dispatch alone would not distinguish
        // "this pool is unusable" from "this pool is merely too small" -- every
        // case here has at most one entry. A pool with nothing to sample from is
        // unusable in both modes.
        for (label, contract) in cases {
            for ranked in [true, false] {
                let mode = ranked ? "ranked" : "gates-only"
                let result = try runSelectionStep(contract: contract, ranked: ranked)
                #expect(
                    result.exitStatus != 0,
                    """
                    'Select hidden DFlash timed target from the pool' ADMITTED a \
                    \(mode) run with \(label). Scoping this step to the ranked \
                    path must not relax it: an unusable pool on a ranked dispatch \
                    means there is no per-run sampling, which is contract layer \
                    L6's whole defence against submit-until-green -- and an \
                    unusable pool on a dry run means the dry run validated a \
                    selection that cannot happen.
                    """
                )
                #expect(
                    result.output.contains("::error::"),
                    "\(label) (\(mode)): the refusal must annotate as an error"
                )
                #expect(
                    result.githubOutput.isEmpty,
                    """
                    \(label) (\(mode)): the step emitted step outputs while \
                    refusing, so 'Prepare hidden DFlash goldens' would receive a \
                    half-formed selection.
                    """
                )
            }
        }
    }

    /// ...and still samples successfully from a real pool, privately.
    ///
    /// Eight distinct, well-formed entries, because that is what a ranked
    /// dispatch is now required to have: the anti-lottery floor counts distinct
    /// sha256 digests and distinct r2_path keys, and every entry must carry a
    /// 64-character lowercase hex digest and a positive byte count. The
    /// three-entry pool with `"sha256":"aaa"` this replaces was only accepted
    /// because the case ran with `MLXFAST_RUN_BENCHMARK` unset -- i.e. it was
    /// never testing the ranked path its name claims.
    @Test
    func aRankedSelectionSamplesFromAPopulatedPool() throws {
        let contract = DFlashWorkflowStep.distinctPool(8)
        var seenPaths = Set<String>()
        for _ in 0 ..< 12 {
            let result = try runSelectionStep(contract: contract)
            #expect(result.exitStatus == 0, "selection refused a valid pool: \(result.output)")

            let outputs = Self.parseOutputs(result.githubOutput)
            let path = try #require(outputs["r2_path"], "no r2_path output")
            #expect(outputs["sha256"] != nil && outputs["bytes"] != nil)
            #expect(
                (0 ..< 8).contains(where: { path.hasSuffix("timed-\($0).json") }),
                "selection returned a path that is not in the pool: \(path)"
            )
            // The sampled index must never reach the participant-visible log.
            #expect(
                !result.output.contains(path),
                "the sampled object key was echoed to the job log, handing back the sampling"
            )
            #expect(result.selectionAudit != nil, "the private audit record was not written")
            seenPaths.insert(path)
        }
        #expect(
            seenPaths.count > 1,
            """
            12 ranked selections over an 8-entry pool returned one entry every \
            time (p ~ 1e-10 if uniform). Per-run sampling is contract layer L6; \
            a constant selection is a frozen timed prompt with extra steps.
            """
        )
    }

    // MARK: - behavioural: the golden preparation, executed

    /// A gates-only dispatch installs the correctness golden and never asks for
    /// the timed one.
    @Test
    func aGatesOnlyGoldenPreparationSkipsTheTimedGolden() throws {
        let result = try runGoldensStep(runBenchmark: false, selection: nil)

        #expect(
            result.exitStatus == 0,
            """
            'Prepare hidden DFlash goldens' failed on a gates-only dispatch. \
            With pool selection now skipped its selection outputs are EMPTY, so \
            an unguarded timed-golden download hands download-r2-object.sh an \
            empty key -- the red wall simply moves one step later and the parity \
            gate is still unreachable. Output:
            \(result.output)
            """
        )
        #expect(
            !result.requestedKeys.contains(where: { $0.contains("timed") }),
            """
            a gates-only dispatch downloaded the hidden TIMED golden \
            (\(result.requestedKeys)). It will never time, so this is hidden \
            material fetched onto the box for no gate at all.
            """
        )
        #expect(
            !result.requestedKeys.contains(""),
            "download-r2-object.sh was invoked with an empty object key"
        )
        #expect(
            result.requestedKeys.contains(where: { $0.contains("dflash_correctness_golden") }),
            "the correctness golden was not downloaded, so the parity gate has no golden"
        )
        #expect(
            result.installedGolden,
            "the correctness golden was not installed into the bench-readable staging dir"
        )
    }

    /// The ranked path still downloads, pins and row-checks the timed golden.
    @Test
    func aRankedGoldenPreparationStillPinsTheTimedGolden() throws {
        let good = try runGoldensStep(
            runBenchmark: true,
            selection: .init(key: "correctness_prompts/dflash-timed-a.json", corruptPin: false)
        )
        #expect(good.exitStatus == 0, "ranked golden preparation failed: \(good.output)")
        #expect(
            good.requestedKeys.contains("correctness_prompts/dflash-timed-a.json"),
            "the ranked path stopped downloading the sampled timed golden"
        )
        #expect(good.installedGolden)

        let swapped = try runGoldensStep(
            runBenchmark: true,
            selection: .init(key: "correctness_prompts/dflash-timed-a.json", corruptPin: true)
        )
        #expect(
            swapped.exitStatus != 0,
            """
            a timed golden whose bytes do not match the SAMPLED pool entry's \
            sha256/bytes was accepted. Scoping the download to the ranked path \
            must not drop its pin check.
            """
        )
        #expect(swapped.output.contains("timed golden pin mismatch"))
    }

    /// The second, independent refusal: if the selection step is ever skipped on
    /// a ranked dispatch, the download must refuse BY NAME rather than hand an
    /// empty object key downstream.
    ///
    /// Scope, stated honestly: `download-r2-object.sh` already rejects an empty
    /// key (exit 2), and an empty pin would mismatch a real digest, so the
    /// ranked path did not silently accept a wrong object before this guard. It
    /// died confusingly instead -- in a helper script, on a generic argument
    /// error, ~25 minutes into a ranked job. This asserts that a ranked
    /// dispatch with no selection is refused where the cause is known, and that
    /// the refusal does not depend on either downstream behaviour staying as it
    /// is. Losing per-run sampling is losing contract layer L6, so the failure
    /// mode worth engineering for is "unmistakable", not merely "nonzero".
    @Test
    func aRankedGoldenPreparationRefusesAnEmptySelection() throws {
        let result = try runGoldensStep(runBenchmark: true, selection: nil)

        #expect(
            result.exitStatus != 0,
            """
            a RANKED dispatch reached the golden download with no timed-target \
            selection and did not refuse. A skipped step yields EMPTY outputs, \
            not an error, so widening the guard on '\(Self.selectionStep)' \
            would leave a ranked run with no per-run sampling at all. Output:
            \(result.output)
            """
        )
        #expect(
            !result.requestedKeys.contains(""),
            """
            an empty object key was handed to download-r2-object.sh. That \
            script refuses it today, so the run still fails -- but it fails on \
            a generic argument error in a helper rather than naming the missing \
            pool selection, and the refusal then depends on a behaviour in a \
            different file that nothing here pins.
            """
        )
        #expect(
            result.output.contains("::error::"),
            "the refusal is not annotated as a workflow error, so it will not surface in the run summary"
        )
        #expect(
            result.output.contains("Select hidden DFlash timed target from the pool"),
            "the refusal does not name the step that failed to produce a selection"
        )
    }

    // MARK: - execution harnesses

    /// The selection step's runner lives in `DFlashWorkflowStep` because
    /// `DFlashEnablementInterlockTests` runs the same step for the anti-lottery
    /// floor. `ranked: true` sets `MLXFAST_RUN_BENCHMARK=1` the way the job env
    /// does -- these cases are all named "ranked", and before this they left the
    /// variable unset, which silently exercised the dry-run arm of the body.
    private func runSelectionStep(
        contract: String,
        ranked: Bool = true
    ) throws -> DFlashWorkflowStep.SelectionResult {
        try DFlashWorkflowStep.runSelection(contract: contract, ranked: ranked)
    }

    private struct Selection {
        var key: String
        var corruptPin: Bool
    }

    private struct GoldensResult {
        var exitStatus: Int32
        var output: String
        var requestedKeys: [String]
        var installedGolden: Bool
    }

    /// Runs the real "Prepare hidden DFlash goldens" body against a stub R2
    /// downloader that records every object key it is asked for.
    private func runGoldensStep(
        runBenchmark: Bool,
        selection: Selection?
    ) throws -> GoldensResult {
        let sandbox = try Sandbox()
        defer { sandbox.destroy() }
        let fm = FileManager.default

        let scripts = sandbox.root.appendingPathComponent(".github/scripts")
        let objects = sandbox.root.appendingPathComponent("objects")
        let privateDir = sandbox.root.appendingPathComponent("private")
        let jobWS = sandbox.root.appendingPathComponent("jobws")
        for dir in [scripts, objects, privateDir, jobWS.appendingPathComponent(".dflash-ranked-src")] {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        let requestLog = sandbox.root.appendingPathComponent("requests.log")
        FileManager.default.createFile(atPath: requestLog.path, contents: nil)

        // Records the key (empty keys included, spelled so they are visible) and
        // serves whatever object was staged under it.
        let stub = """
            #!/bin/bash
            set -u
            printf '%s\\n' "${1-}" >> "${STUB_REQUEST_LOG}"
            key="${1-}"
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

        // Goldens with more rows than the token counts requested below, so the
        // step's own row-count preflight is satisfied rather than bypassed.
        let goldenBody = { (tag: String) in
            #"{"tag":"\#(tag)","rows":[{"t":1},{"t":2},{"t":3},{"t":4}]}"#
        }

        let workflow = try Self.workflow()
        let declared = try Self.stepEnvironment(workflow, Self.goldensStep)
        let correctnessKey = try #require(declared["MLXFAST_DFLASH_CORRECTNESS_GOLDEN_R2_PATH"])

        func stage(_ key: String, _ contents: String) throws {
            let name = key.replacingOccurrences(of: "/", with: "_")
            try contents.write(
                to: objects.appendingPathComponent(name), atomically: true, encoding: .utf8
            )
        }
        let correctnessBody = goldenBody("dflash_correctness_golden")
        try stage(correctnessKey, correctnessBody)

        var env: [String: String] = [
            "MLXFAST_PRIVATE_DIR": privateDir.path,
            "MLXFAST_JOB_WS": jobWS.path,
            "MLXFAST_RUN_BENCHMARK": runBenchmark ? "1" : "0",
            "MLXFAST_DFLASH_CORRECTNESS_TOKENS": "3",
            "MLXFAST_DFLASH_DECODE_TOKENS": "3",
            "MLXFAST_DFLASH_CORRECTNESS_GOLDEN_R2_PATH": correctnessKey,
            "MLXFAST_DFLASH_CORRECTNESS_GOLDEN_SHA256": try Self.sha256(correctnessBody),
            "MLXFAST_DFLASH_CORRECTNESS_GOLDEN_BYTES": "\(correctnessBody.utf8.count)",
            "STUB_REQUEST_LOG": requestLog.path,
            "STUB_OBJECT_DIR": objects.path,
        ]

        // A skipped step contributes EMPTY strings for its outputs, which is
        // exactly what `selection: nil` models.
        if let selection {
            let timedBody = goldenBody("dflash_benchmark_golden")
            try stage(selection.key, timedBody)
            env["MLXFAST_DFLASH_BENCH_GOLDEN_R2_PATH"] = selection.key
            env["MLXFAST_DFLASH_BENCH_GOLDEN_SHA256_SELECTED"] =
                selection.corruptPin
                ? String(repeating: "0", count: 64) : try Self.sha256(timedBody)
            env["MLXFAST_DFLASH_BENCH_GOLDEN_BYTES_SELECTED"] = "\(timedBody.utf8.count)"
        } else {
            env["MLXFAST_DFLASH_BENCH_GOLDEN_R2_PATH"] = ""
            env["MLXFAST_DFLASH_BENCH_GOLDEN_SHA256_SELECTED"] = ""
            env["MLXFAST_DFLASH_BENCH_GOLDEN_BYTES_SELECTED"] = ""
        }

        let body = try Self.rawRunBody(workflow, Self.goldensStep)
        let run = try Self.bash(body, cwd: sandbox.root, env: env)
        let requested = ((try? String(contentsOf: requestLog, encoding: .utf8)) ?? "")
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
            .dropLast()  // the trailing newline
            .map { $0 }

        return GoldensResult(
            exitStatus: run.status,
            output: run.output,
            requestedKeys: Array(requested),
            installedGolden: fm.fileExists(
                atPath: jobWS.appendingPathComponent(
                    ".dflash-ranked-src/dflash_correctness_golden.json"
                ).path
            )
        )
    }

    private struct Sandbox {
        let root: URL
        init() throws {
            root = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("dflash-dryrun-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        }
        func destroy() { try? FileManager.default.removeItem(at: root) }
    }

    private static func bash(
        _ script: String,
        cwd: URL,
        env: [String: String]
    ) throws -> (status: Int32, output: String) {
        let inherited = ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin"
        var environment = env
        // The workflow's own steps use jq; Homebrew's bin is not always on a
        // test runner's PATH.
        environment["PATH"] = "\(inherited):/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

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

    private static func sha256(_ text: String) throws -> String {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("sha-\(UUID().uuidString)")
        try text.write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }
        let run = try bash(
            "shasum -a 256 \(url.path) | awk '{print $1}'",
            cwd: URL(fileURLWithPath: NSTemporaryDirectory()),
            env: [:]
        )
        return run.output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func parseOutputs(_ text: String) -> [String: String] {
        var result: [String: String] = [:]
        for line in text.split(separator: "\n") {
            guard let split = line.firstIndex(of: "=") else { continue }
            result[String(line[..<split])] = String(line[line.index(after: split)...])
        }
        return result
    }

    // MARK: - workflow parsing

    private struct WorkflowStep {
        var name: String?
        var id: String?
        var condition: String?
    }

    private static func workflow() throws -> String {
        try String(contentsOfFile: workflowPath, encoding: .utf8)
    }

    /// The job's step list: `name`, `id` and `if` for every list item, including
    /// unnamed (`uses:`-only) steps. Step-level keys sit at exactly eight spaces
    /// and block-scalar bodies at ten or more, so exact-eight is unambiguous.
    /// Cross-checked against a real YAML parse when this suite was written.
    private static func parseSteps(_ workflow: String) throws -> [WorkflowStep] {
        let lines = workflow.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        guard let jobLine = lines.firstIndex(of: "  \(jobKey):") else {
            Issue.record("job '\(jobKey)' not found in \(workflowPath)")
            return []
        }
        var steps: [WorkflowStep] = []
        var current: WorkflowStep?
        var started = false

        for line in lines[(jobLine + 1)...] {
            // The job block ends at the next top-level or job-level key.
            if !line.trimmingCharacters(in: .whitespaces).isEmpty,
                line.prefix(while: { $0 == " " }).count <= 2
            {
                break
            }
            var key: String
            if line.hasPrefix("      - ") {
                if let current { steps.append(current) }
                current = WorkflowStep()
                started = true
                key = "        " + String(line.dropFirst(8))
            } else if started, line.count > 8, line.hasPrefix("        "),
                line[line.index(line.startIndex, offsetBy: 8)] != " "
            {
                key = line
            } else {
                continue
            }

            let scalar = key.dropFirst(8)
            for field in ["name", "id", "if"] where scalar.hasPrefix("\(field): ") {
                let value = String(scalar.dropFirst(field.count + 2))
                    .trimmingCharacters(in: .whitespaces)
                switch field {
                case "name": current?.name = value
                case "id": current?.id = value
                default: current?.condition = value
                }
            }
        }
        if let current { steps.append(current) }
        #expect(steps.count > 40, "step parse produced only \(steps.count) steps")
        return steps
    }

    /// Which steps run, in order, on a CLEAN dispatch (nothing has failed yet).
    ///
    /// Evaluates the real `if:` expressions. Any construct this evaluator does
    /// not recognise is a hard failure rather than a silent `false` -- a test
    /// that quietly mis-parses a condition is worse than no test.
    private static func stepsThatRun(
        _ steps: [WorkflowStep],
        runBenchmark: Bool,
        ref: String
    ) throws -> [String] {
        var outcomes: [String: String] = [:]
        var ran: [String] = []
        for step in steps {
            let runs = try evaluate(
                step.condition, runBenchmark: runBenchmark, ref: ref, outcomes: outcomes
            )
            if let id = step.id { outcomes[id] = runs ? "success" : "skipped" }
            if runs, let name = step.name { ran.append(name) }
        }
        return ran
    }

    private static let statusFunctions = ["always()", "success()", "failure()", "cancelled()"]

    private static func evaluate(
        _ raw: String?,
        runBenchmark: Bool,
        ref: String,
        outcomes: [String: String]
    ) throws -> Bool {
        // No `if:` at all means the implicit `success()`, which on a clean run
        // is true.
        guard var expression = raw?.trimmingCharacters(in: .whitespaces), !expression.isEmpty else {
            return true
        }
        if expression.hasPrefix("${{") {
            guard expression.hasSuffix("}}") else {
                Issue.record("unterminated expression: \(expression)")
                return false
            }
            expression = String(expression.dropFirst(3).dropLast(2))
                .trimmingCharacters(in: .whitespaces)
        }

        var conjuncts = try split(expression)
        // GitHub ANDs an implicit success() unless a status function appears.
        if !statusFunctions.contains(where: { expression.contains($0) }) {
            conjuncts.append("success()")
        }
        for conjunct in conjuncts {
            guard
                let value = atom(
                    conjunct, runBenchmark: runBenchmark, ref: ref, outcomes: outcomes
                )
            else {
                Issue.record(
                    """
                    the dry-run step-set evaluator does not understand the \
                    condition '\(conjunct)' (from '\(expression)'). Teach it \
                    that construct rather than letting this suite guess -- a \
                    mis-parsed `if:` would silently stop asserting anything.
                    """
                )
                return false
            }
            if !value { return false }
        }
        return true
    }

    /// Top-level `&&` split, quote- and paren-aware. A top-level `||` is
    /// rejected: none exists today, and guessing at its precedence is exactly
    /// the silent mis-parse this evaluator refuses to make.
    private static func split(_ expression: String) throws -> [String] {
        var parts: [String] = []
        var current = ""
        var quoted = false
        var depth = 0
        var index = expression.startIndex
        while index < expression.endIndex {
            let character = expression[index]
            let next = expression.index(after: index)
            if character == "'" { quoted.toggle() }
            if !quoted {
                if character == "(" { depth += 1 }
                if character == ")" { depth -= 1 }
                if depth == 0, character == "&" || character == "|", next < expression.endIndex,
                    expression[next] == character
                {
                    if character == "|" {
                        Issue.record("top-level || in '\(expression)' is not supported")
                        return []
                    }
                    parts.append(current.trimmingCharacters(in: .whitespaces))
                    current = ""
                    index = expression.index(after: next)
                    continue
                }
            }
            current.append(character)
            index = next
        }
        parts.append(current.trimmingCharacters(in: .whitespaces))
        return parts.filter { !$0.isEmpty }
    }

    private static func atom(
        _ text: String,
        runBenchmark: Bool,
        ref: String,
        outcomes: [String: String]
    ) -> Bool? {
        switch text {
        case "always()", "success()": return true
        case "failure()", "cancelled()": return false
        case "inputs.run_benchmark": return runBenchmark
        case "!inputs.run_benchmark": return !runBenchmark
        default: break
        }
        if let match = capture(#"^startsWith\(github\.ref, '([^']*)'\)$"#, text) {
            return ref.hasPrefix(match[0])
        }
        if let match = capture(#"^steps\.([A-Za-z0-9_]+)\.outcome == '([^']*)'$"#, text) {
            return outcomes[match[0]] == match[1]
        }
        return nil
    }

    private static func capture(_ pattern: String, _ text: String) -> [String]? {
        guard let regex = try? NSRegularExpression(pattern: pattern),
            let match = regex.firstMatch(
                in: text, range: NSRange(text.startIndex..., in: text)
            )
        else { return nil }
        return (1 ..< match.numberOfRanges).compactMap {
            Range(match.range(at: $0), in: text).map { String(text[$0]) }
        }
    }

    /// A step's `env:` mapping, values verbatim (expressions included).
    private static func stepEnvironment(
        _ workflow: String,
        _ step: String
    ) throws -> [String: String] {
        let block = try stepBlock(workflow, step)
        var result: [String: String] = [:]
        var inEnv = false
        for line in block.split(separator: "\n", omittingEmptySubsequences: false) {
            if line == "        env:" {
                inEnv = true
                continue
            }
            guard inEnv else { continue }
            if !line.trimmingCharacters(in: .whitespaces).isEmpty,
                line.prefix(while: { $0 == " " }).count <= 8
            {
                break
            }
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("#") || trimmed.isEmpty { continue }
            guard let colon = trimmed.firstIndex(of: ":") else { continue }
            result[String(trimmed[..<colon])] =
                String(trimmed[trimmed.index(after: colon)...])
                .trimmingCharacters(in: .whitespaces)
        }
        return result
    }

    /// A step's own mapping, from just after its `- name:` line to the first
    /// line at list-item indentation (six spaces) or shallower.
    ///
    /// Terminating on the next `- name:` instead would swallow the six-space
    /// COMMENT BLOCK that introduces the following step -- and in this workflow
    /// that block is the long note explaining the timed-prompt pool. A
    /// "no dry-run step reads timed_prompt_pool" assertion written that way
    /// fires on the innocent step above the selection step, which is a false
    /// positive that reads exactly like a real one.
    ///
    /// Returns nil rather than recording an issue, so callers that legitimately
    /// probe for absence (`uses:`-only steps have no `- name:` we search for)
    /// do not fail the test merely by asking.
    private static func stepBlockOrNil(_ workflow: String, _ step: String) -> String? {
        guard let start = workflow.range(of: "- name: \(step)\n") else { return nil }
        var body: [Substring] = []
        for line in workflow[start.upperBound...].split(
            separator: "\n", omittingEmptySubsequences: false
        ) {
            let blank = line.trimmingCharacters(in: .whitespaces).isEmpty
            if !blank, line.prefix(while: { $0 == " " }).count <= 6 { break }
            body.append(line)
        }
        return body.joined(separator: "\n")
    }

    private static func stepBlock(_ workflow: String, _ step: String) throws -> String {
        try #require(
            stepBlockOrNil(workflow, step), "step '\(step)' is missing from \(workflowPath)"
        )
    }

    /// A step's `run:` block scalar, dedented, comments preserved -- it is
    /// executed, so stripping them would change what runs. The scalar ends at
    /// the first non-blank line indented less than the block, not at the next
    /// step marker.
    private static func rawRunBody(_ workflow: String, _ step: String) throws -> String {
        let block = try stepBlock(workflow, step)
        let runMarker = try #require(
            block.range(of: "run: |\n"), "step '\(step)' has no literal run: block"
        )
        let scalar = String(block[runMarker.upperBound...])
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
}
