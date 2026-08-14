import Foundation
import Testing

/// Shared machinery for EXECUTING a real `dflash-benchmark.yml` step body.
///
/// Two suites need the same three things -- the workflow text, a step's `run:`
/// block scalar dedented so bash can run it verbatim, and a sandboxed bash
/// invocation -- and both need to run the timed-target selection step:
/// `DFlashDryRunReachesItsGateTests` for the fail-closed cases, and
/// `DFlashEnablementInterlockTests` for the anti-lottery floor.
///
/// It lives in one place because the alternative is two extractors that drift
/// until one of them quietly stops running the real step. That is not a
/// hypothetical here: the interlock suite's floor test used to assert on the
/// step's TEXT, and the assertion `branch.contains("exit 1")` was satisfied by
/// an unrelated `exit 1` further down the same step. Replacing the floor's
/// refusal with a `::warning::` -- so that a one-entry pool RANKS -- left all
/// 582 tests green.
enum DFlashWorkflowStep {
    static let workflowPath = ".github/workflows/dflash-benchmark.yml"
    static let selectionStep = "Select hidden DFlash timed target from the pool"

    static func workflow() throws -> String {
        try String(contentsOfFile: workflowPath, encoding: .utf8)
    }

    // MARK: - extraction

    /// A step's own mapping, from just after its `- name:` line to the first
    /// line at list-item indentation (six spaces) or shallower.
    ///
    /// Terminating on the next `- name:` instead would swallow the six-space
    /// COMMENT BLOCK that introduces the following step, which in this workflow
    /// is the long note explaining the timed-prompt pool.
    ///
    /// Returns nil rather than recording an issue, so callers that legitimately
    /// probe for absence do not fail a test merely by asking.
    static func stepBlockOrNil(_ workflow: String, _ step: String) -> String? {
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

    static func stepBlock(_ workflow: String, _ step: String) throws -> String {
        try #require(
            stepBlockOrNil(workflow, step), "step '\(step)' is missing from \(workflowPath)"
        )
    }

    /// A step's `if:` expression verbatim, or nil when it has none.
    ///
    /// Matched at EXACTLY eight spaces, which is step-key indentation. A `run:`
    /// body sits at ten or more, so a shell line or comment mentioning `if:`
    /// inside the body cannot be mistaken for the step's own condition.
    static func condition(_ workflow: String, _ step: String) throws -> String? {
        let block = try stepBlock(workflow, step)
        for line in block.split(separator: "\n", omittingEmptySubsequences: false) {
            guard line.hasPrefix("        if: "), !line.hasPrefix("         ") else { continue }
            return String(line.dropFirst("        if: ".count))
                .trimmingCharacters(in: .whitespaces)
        }
        return nil
    }

    /// A step's `run:` block scalar, dedented, comments preserved -- it is
    /// executed, so stripping them would change what runs. The scalar ends at
    /// the first non-blank line indented less than the block, not at the next
    /// step marker.
    static func rawRunBody(_ workflow: String, _ step: String) throws -> String {
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

    // MARK: - execution

    struct Sandbox {
        let root: URL
        init(_ label: String) throws {
            root = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("dflash-\(label)-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        }
        func destroy() { try? FileManager.default.removeItem(at: root) }
    }

    static func bash(
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

    // MARK: - the timed-target selection step

    struct SelectionResult {
        var exitStatus: Int32
        var output: String
        var githubOutput: String
        var selectionAudit: String?

        var admitted: Bool { exitStatus == 0 }

        var outputs: [String: String] {
            var result: [String: String] = [:]
            for line in githubOutput.split(separator: "\n") {
                guard let split = line.firstIndex(of: "=") else { continue }
                result[String(line[..<split])] = String(line[line.index(after: split)...])
            }
            return result
        }
    }

    /// Runs the REAL "Select hidden DFlash timed target from the pool" body
    /// against a synthetic contract.
    ///
    /// `ranked` sets `MLXFAST_RUN_BENCHMARK` the way the job env does
    /// (`inputs.run_benchmark && '1' || '0'`). Note what that flag does and does
    /// not model: in CI a gates-only dispatch never reaches this body at all,
    /// because the step carries `if: ${{ inputs.run_benchmark }}`. Running with
    /// `ranked: false` therefore exercises the env conjunct as defence in depth,
    /// not the exemption that actually applies -- assert the `if:` for that.
    static func runSelection(contract: String, ranked: Bool) throws -> SelectionResult {
        let sandbox = try Sandbox("selection")
        defer { sandbox.destroy() }

        let contractURL = sandbox.root.appendingPathComponent("contract.json")
        try contract.write(to: contractURL, atomically: true, encoding: .utf8)
        let privateDir = sandbox.root.appendingPathComponent("private")
        try FileManager.default.createDirectory(at: privateDir, withIntermediateDirectories: true)
        let githubOutput = sandbox.root.appendingPathComponent("github_output")
        FileManager.default.createFile(atPath: githubOutput.path, contents: nil)

        let body = try rawRunBody(try workflow(), selectionStep)
        let run = try bash(
            body,
            cwd: sandbox.root,
            env: [
                "MLXFAST_DFLASH_CONTRACT_PATH": contractURL.path,
                "MLXFAST_PRIVATE_DIR": privateDir.path,
                "GITHUB_OUTPUT": githubOutput.path,
                "MLXFAST_RUN_BENCHMARK": ranked ? "1" : "0",
            ]
        )
        return SelectionResult(
            exitStatus: run.status,
            output: run.output,
            githubOutput: (try? String(contentsOf: githubOutput, encoding: .utf8)) ?? "",
            selectionAudit: try? String(
                contentsOf: privateDir.appendingPathComponent(
                    "dflash_timed_target_selection.json"
                ),
                encoding: .utf8
            )
        )
    }

    // MARK: - synthetic pools

    /// A well-formed pool entry. `seed` varies BOTH coordinates the ranked floor
    /// counts, so `pool(distinctTargets:)` really is N distinct targets.
    static func entry(seed: Int) -> String {
        let digest = String(
            String(repeating: "0123456789abcdef", count: 4).prefix(60) + String(format: "%04x", seed)
        )
        return #"{"r2_path":"correctness_prompts/laguna-xs-2.1-dflash/timed-\#(seed).json","#
            + #""sha256":"\#(digest)","bytes":\#(1000 + seed)}"#
    }

    static func pool(_ entries: [String]) -> String {
        #"{"timed_prompt_pool":[\#(entries.joined(separator: ","))]}"#
    }

    /// `count` genuinely distinct, well-formed targets.
    static func distinctPool(_ count: Int) -> String {
        pool((0 ..< count).map { entry(seed: $0) })
    }

    /// ONE target listed `count` times -- pool LENGTH `count`, one distinct
    /// target. This is what eight dispatches of `dflash-provision-goldens.yml`
    /// against the same `bench_seed_r2_path` produce.
    static func clonedPool(_ count: Int) -> String {
        pool(Array(repeating: entry(seed: 0), count: count))
    }
}
