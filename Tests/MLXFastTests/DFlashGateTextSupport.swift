import Foundation
import Testing

// SHARED TEXT-ASSERTION SUPPORT for the checked-in workflow/script/manifest
// suites. Extracted verbatim from DFlashTrackTests.swift at the 2026-08-13 repo
// split, when that file was deleted: the Qwen-MTP suites
// (QwenMTPTrackNamingTests, QwenMTPTrustedContextGuardTests, QwenMTPVerbTests,
// RankedWorkflowIsolationTests) and BenchmarkScriptTests all typealias
// `S = DFlashGateTextSupport`, so the helpers outlive the DFlash suites that
// used to host them. The name is kept so those call sites are untouched.
//
// One member was dropped rather than moved: `dflashManifestPath` pointed at
// benchmark.json, which in this repository IS the Qwen-MTP manifest, and no
// surviving suite referenced it.
//
// Nothing here needs a model, hidden material, network access or a ranked
// dispatch: every helper reads checked-in text.

enum DFlashGateTextSupport {
    static let dflashWorkflowPath = ".github/workflows/dflash-benchmark.yml"
    /// The Qwen 3.6 native-MTP ranked workflow. It carries the pins that were
    /// repointed to Qwen (layer count, public fixture digests) while
    /// dflash-benchmark.yml keeps Laguna's, so several tests below assert the
    /// SAME pin against BOTH workflows with different expected values.
    static let qwenMTPWorkflowPath = ".github/workflows/qwen-mtp-ranked-benchmark.yml"
    static let qwenMTPManifestPath = "benchmark.qwen-mtp.json"
    /// Laguna's values, still pinned by the DFlash workflow. Spelled as
    /// literals on purpose: after the Qwen identity repoint they can no longer
    /// be derived from Constants or from the checked-in fixtures, and the point
    /// of keeping them is that an edit to the DFlash track's pins must be
    /// deliberate rather than silent.
    static let lagunaNumHiddenLayers = "40"
    static let lagunaPublicGoldenSHA256 =
        "b9509697c08a2cf3c2943a85f0b76e39c485c441794690fa76835b40a58d7a63"
    static let lagunaPublicGoldenBytes = "10686"
    static let dflashFixturePath = "fixtures/laguna_xs_2_1_dflash_track.json"
    static let cliPath = "Sources/MLXFastCLI/main.swift"

    /// The four serial steps that are REUSED verbatim rather than reimplemented.
    /// The names are the reuse contract: a rename here means a second copy was
    /// written instead.
    static let reusedGateStepNames = [
        "Public behavior gate",
        "Attach GPQA gates and verify augmented golden",
        "Correctness and gates (full base case + hidden gates, no timing)",
        "Semantic GPQA gate",
    ]

    static func text(_ path: String) throws -> String {
        try String(contentsOfFile: path, encoding: .utf8)
    }

    static func json(_ path: String) throws -> [String: Any] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
    }

    /// Comment lines carry deliberate documentation of the dead MTP names that
    /// were removed, so every "does the workflow DO x" assertion runs against
    /// this comment-stripped view.
    static func executable(_ yaml: String) -> String {
        yaml
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("#") }
            .joined(separator: "\n")
    }

    /// The job-level `env:` mapping — the only place these workflows declare the
    /// values their steps enforce.
    static func jobEnvironment(_ workflow: String) throws -> [String: String] {
        let stepsMarker = try #require(
            workflow.range(of: "\n    steps:"),
            "workflow has no job `steps:` block"
        )
        let header = String(workflow[workflow.startIndex ..< stepsMarker.lowerBound])
        var out: [String: String] = [:]
        for raw in header.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(raw)
            // Exactly the job-level env indentation (6 spaces, not 7+).
            guard line.hasPrefix("      "), !line.hasPrefix("       ") else { continue }
            let body = line.dropFirst(6)
            guard let colon = body.firstIndex(of: ":") else { continue }
            let key = String(body[body.startIndex ..< colon])
            guard !key.isEmpty,
                key.allSatisfy({ $0.isUppercase || $0.isNumber || $0 == "_" })
            else { continue }
            var value = String(body[body.index(after: colon)...])
                .trimmingCharacters(in: .whitespaces)
            if value.count >= 2, value.hasPrefix("\""), value.hasSuffix("\"") {
                value = String(value.dropFirst().dropLast())
            }
            out[key] = value
        }
        return out
    }

    /// One workflow step's text, from its `- name:` line to the next step.
    static func stepBody(_ workflow: String, _ name: String) throws -> String {
        let start = try #require(
            workflow.range(of: "- name: \(name)\n"),
            "workflow is missing the step '\(name)'"
        )
        let rest = workflow[start.upperBound...]
        if let next = rest.range(of: "\n      - name: ") {
            return String(rest[rest.startIndex ..< next.lowerBound])
        }
        return String(rest)
    }

    /// Byte offset of the first occurrence of `needle`, or nil.
    static func offset(of needle: String, in haystack: String) -> Int? {
        guard let range = haystack.range(of: needle) else { return nil }
        return haystack.distance(from: haystack.startIndex, to: range.lowerBound)
    }

    static func captures(_ pattern: String, in haystack: String, group: Int = 1) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let ns = haystack as NSString
        return regex
            .matches(in: haystack, range: NSRange(location: 0, length: ns.length))
            .compactMap { match in
                guard match.numberOfRanges > group else { return nil }
                let range = match.range(at: group)
                guard range.location != NSNotFound else { return nil }
                return ns.substring(with: range)
            }
    }

    /// The contract a step puts in force, following one level of workflow-env
    /// indirection (`CONTRACT_PATH="${VAR}"` or `CONTRACT_PATH: ${{ env.VAR }}`).
    /// The indirection is the whole point of the wiring, so the test has to
    /// resolve it rather than demand a literal.
    static func resolvedContractPath(
        in stepBody: String, jobEnvironment environment: [String: String]
    ) -> String? {
        guard let raw = captures(#"CONTRACT_PATH[:=]\s*(.+)"#, in: stepBody).first
        else { return nil }
        var token = raw.trimmingCharacters(in: .whitespaces)
        if token.hasSuffix("\\") {
            token = String(token.dropLast()).trimmingCharacters(in: .whitespaces)
        }
        if token.count >= 2, token.hasPrefix("\""), token.hasSuffix("\"") {
            token = String(token.dropFirst().dropLast())
        }
        if let name = captures(#"^\$\{\{\s*env\.([A-Z0-9_]+)\s*\}\}$"#, in: token).first {
            return environment[name]
        }
        if let name = captures(#"^\$\{([A-Z0-9_]+)\}$"#, in: token).first {
            return environment[name]
        }
        return token
    }

    /// Resolve a shell token through at most three `NAME=` assignments so a test
    /// can assert what an argument actually points at rather than which variable
    /// happens to hold it.
    static func resolveShellValue(
        _ token: String, in script: String, depth: Int = 0
    ) -> String {
        var value = token.trimmingCharacters(in: .whitespaces)
        if value.count >= 2, value.hasPrefix("\""), value.hasSuffix("\"") {
            value = String(value.dropFirst().dropLast())
        }
        guard depth < 3,
            let name = captures(#"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$"#, in: value).first,
            let assignment = captures("(?m)^\\s*\(name)=(.+)$", in: script).first
        else { return value }
        return resolveShellValue(assignment, in: script, depth: depth + 1)
    }

    static func containsMatch(_ pattern: String, in haystack: String) -> Bool {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return false }
        let ns = haystack as NSString
        return regex.firstMatch(
            in: haystack, range: NSRange(location: 0, length: ns.length)
        ) != nil
    }

    // MARK: jq assertion / report payload cross-check

    /// Every `jq -e <program> ... >/dev/null` region inside one step.
    static func jqAssertionPrograms(in stepBody: String) -> [String] {
        var programs: [String] = []
        var cursor = stepBody.startIndex
        while let start = stepBody.range(of: "jq -e", range: cursor ..< stepBody.endIndex) {
            let tail = stepBody[start.upperBound...]
            let terminator =
                tail.range(of: ">/dev/null") ?? tail.range(of: "> /dev/null")
            let stop = terminator?.lowerBound ?? tail.endIndex
            programs.append(String(tail[tail.startIndex ..< stop]))
            cursor = terminator?.upperBound ?? stepBody.endIndex
        }
        return programs
    }

    /// Field names a step's jq programs REQUIRE to exist (`.name`) and field
    /// names they require to be ABSENT (`has("name") | not`).
    static func jqAssertedFields(
        in stepBody: String
    ) -> (required: Set<String>, forbidden: Set<String>) {
        var required: Set<String> = []
        var forbidden: Set<String> = []
        for program in jqAssertionPrograms(in: stepBody) {
            for name in captures(
                #"(?<![A-Za-z0-9_/\-.])\.([a-z][a-z0-9_]*)"#, in: program
            ) {
                required.insert(name)
            }
            for name in captures(#"has\("([a-z][a-z0-9_]*)"\)"#, in: program) {
                forbidden.insert(name)
                required.remove(name)
            }
        }
        return (required, forbidden)
    }

    /// The keys `runDFlashBenchmark` actually puts in its report payload — the
    /// authority for what a gate is allowed to assert on.
    static func dflashReportPayloadKeys() throws -> Set<String> {
        let cli = try text(cliPath)
        let function = try #require(
            cli.range(of: "private static func runDFlashBenchmark("),
            "runDFlashBenchmark is gone from the CLI"
        )
        let tail = cli[function.upperBound...]
        let literal = try #require(
            tail.range(of: "var payload: [String: Any] = ["),
            "runDFlashBenchmark no longer builds a `payload` dictionary"
        )
        let serialize = try #require(
            tail.range(
                of: "JSONSerialization.data(",
                range: literal.upperBound ..< tail.endIndex
            ),
            "could not find the end of runDFlashBenchmark's payload construction"
        )
        let region = String(tail[literal.upperBound ..< serialize.lowerBound])
        var keys = Set(captures(#""([a-z][a-z0-9_]*)":"#, in: region))
        for key in captures(#"payload\["([a-z][a-z0-9_]*)"\]"#, in: region) {
            keys.insert(key)
        }
        return keys
    }

    // MARK: score.json producers

    /// Lines that WRITE `score.json` (redirect target, or cp/mv/install/tee
    /// destination). Reading it, hashing it and passing it as an artifact name
    /// are not production.
    static func scoreJSONProducerLines(_ yaml: String) -> [String] {
        // `gates-score.json` must NOT count as a `score.json` writer, so the
        // name is bounded on both sides.
        let head = #"(?<![A-Za-z0-9_.\-])"#
        let tail = #"(?![A-Za-z0-9_.\-])"#
        let patterns = [
            #">\s*"# + head + #"score\.json"# + tail,
            #"\b(?:cp|mv|install|tee)\b[^|;]*"# + head + #"score\.json"# + tail,
        ]
        return executable(yaml)
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
            .filter { line in patterns.contains { containsMatch($0, in: line) } }
    }

    // MARK: CLI surface

    /// `subcommand -> every option the CLI declares for it`, read out of the
    /// dispatch switch plus each handler's `options.validate(...)` call.
    static func cliSubcommandOptions() throws -> [String: Set<String>] {
        let cli = try text(cliPath)
        let lines = cli.split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)

        // 1. dispatch switch: case "name": ... try run<Fn>(
        var handlerFor: [String: String] = [:]
        for (index, line) in lines.enumerated() {
            guard let name = captures(#"^\s*case "([a-z][a-z0-9-]*)":"#, in: line).first
            else { continue }
            for lookahead in index ..< min(index + 6, lines.count) {
                if let handler = captures(
                    #"try (run[A-Za-z0-9]+)\("#, in: lines[lookahead]
                ).first {
                    handlerFor[name] = handler
                    break
                }
            }
        }

        // 2. each handler's declared options. The search is bounded to the
        //    handler's own body, so a handler that declared nothing cannot
        //    silently inherit the next function's option list.
        var optionsFor: [String: Set<String>] = [:]
        for handler in Set(handlerFor.values) {
            guard let declaration = cli.range(
                of: "private static func \(handler)("
            ) else { continue }
            var body = cli[declaration.upperBound...]
            if let nextFunction = body.range(of: "\n    private static func ") {
                body = body[body.startIndex ..< nextFunction.lowerBound]
            }
            guard let validate = body.range(of: "options.validate(") else {
                optionsFor[handler] = []
                continue
            }
            // Options are declared inside the validate(...) call; stop at
            // whichever closing form comes first.
            let window = String(body[validate.upperBound...])
            let closings = [window.range(of: "\n        )"), window.range(of: ")\n")]
                .compactMap { $0?.lowerBound }
            let stop = closings.min() ?? window.endIndex
            let region = String(window[window.startIndex ..< stop])
            optionsFor[handler] = Set(captures(#""(--[a-z][a-z0-9-]*)""#, in: region))
        }

        var result: [String: Set<String>] = [:]
        for (name, handler) in handlerFor {
            result[name] = optionsFor[handler] ?? []
        }
        return result
    }

    /// Every `"${swift_bin}" <subcommand> ... ` invocation in a shell script,
    /// as (subcommand, full command text including continuation lines).
    static func swiftBinaryInvocations(in script: String) -> [(String, String)] {
        // Comment lines are stripped: the scripts document their own pipeline in
        // a header block that names the same subcommands, and a documented
        // invocation is not an executed one.
        let lines = script.split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
            .map { $0.trimmingCharacters(in: .whitespaces).hasPrefix("#") ? "" : $0 }
        var invocations: [(String, String)] = []
        var index = 0
        while index < lines.count {
            let line = lines[index]
            let launchers = [
                #"\$\{swift_bin\}"?\s+([a-z][a-z0-9-]*)"#,
                #"mlxfast-swift"?\s+([a-z][a-z0-9-]*)"#,
            ]
            var subcommand: String?
            for pattern in launchers {
                if let hit = captures(pattern, in: line).first {
                    subcommand = hit
                    break
                }
            }
            guard let name = subcommand else {
                index += 1
                continue
            }
            var command = line
            var cursor = index
            while command.hasSuffix("\\"), cursor + 1 < lines.count {
                cursor += 1
                command += "\n" + lines[cursor]
            }
            invocations.append((name, command))
            index = cursor + 1
        }
        return invocations
    }

    /// Names retired with the MTP track. A local DFlash script naming any of
    /// them is either calling something that does not exist or reading a file
    /// that was deleted.
    static let retiredMTPNames = [
        "mtp-benchmark",
        "mtp-probe",
        "mtp-reference",
        "laguna-xs-2.1-mtp-v1",
        "benchmark.mtp.json",
        "benchmark-mtp.sh",
        "setup-mtp.sh",
        "benchmark-mtp:",
        "fixtures/laguna_xs_2_1_mtp_track.json",
        "fixtures/mtp_laguna_xs_2_1_4bit.sha256",
        "laguna_xs_2_1_mtp_track.json",
        "mtp_laguna_xs_2_1_4bit.sha256",
    ]
}
