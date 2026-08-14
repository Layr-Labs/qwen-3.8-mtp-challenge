import Foundation
import Testing

// The validator types live in the harness twin this target already depends on;
// the trusted twin is byte-identical for everything used here (asserted by the
// twin-parity suite), and Package.swift's dependency graph is frozen.
import MLXFastHarness

/// Every `DFlashContractViolation.Kind` must have its own arm in
/// `.github/scripts/redact-benchmark-failure.sh`.
///
/// Why this suite exists: three kinds shipped without one. The rejected-tail
/// detections from correctness contract Amendment 21 --
/// `fabricatedRejection`, `rejectedRowReadoutMismatch` and
/// `draftTokenBindingMismatch` -- fell through to the generic
/// `dflash_contract_violation` bucket. That was never a leak (the fallback
/// still redacts, and it never reaches the violation's detail text), which is
/// exactly why nobody noticed: the failure mode is a silent loss of
/// resolution, not a visible error. Those three are the detections aimed at a
/// verifier fabricating rows it never computed, so they are precisely the ones
/// an audit most needs named.
///
/// The suite is driven off `Kind.allCases` rather than a hand-kept list, so a
/// kind added to the enum enters this test automatically. That is the whole
/// point -- a list would have to be remembered, and the thing being guarded
/// against here is forgetting.
@Suite("DFlash redactor kind coverage")
struct DFlashRedactorKindCoverageTests {
    private static let redactorPath = ".github/scripts/redact-benchmark-failure.sh"

    /// The arm every unrecognised kind lands in. Reaching it from a real
    /// `Kind` is the defect this suite detects.
    private static let genericCategory = "dflash_contract_violation"

    @Test
    func everyViolationKindRedactsToItsOwnCategory() throws {
        // Guards the guard: a Kind enum that stopped being CaseIterable, or an
        // empty one, would make every assertion below vacuously pass.
        let kinds = DFlashContractViolation.Kind.allCases
        #expect(
            kinds.count >= 14,
            """
            DFlashContractViolation.Kind has \(kinds.count) cases, fewer than the \
            14 that existed when this suite was written. If a kind was \
            deliberately removed, lower this floor in the same commit -- it is \
            here so an enum that silently stopped enumerating cannot make this \
            whole suite vacuous.
            """
        )

        var uncovered: [String] = []
        var categories: [String: String] = [:]

        for kind in kinds {
            let category = try redactedCategory(for: kind)
            categories[kind.rawValue] = category
            if category == Self.genericCategory {
                uncovered.append(kind.rawValue)
            }
        }

        #expect(
            uncovered.isEmpty,
            """
            these DFlashContractViolation kinds have no arm in \
            \(Self.redactorPath) and collapse to '\(Self.genericCategory)': \
            \(uncovered.sorted().joined(separator: ", ")). Add one arm per kind. \
            A kind that redacts to the generic bucket is not a leak, but an \
            auditor cannot tell it apart from an ordinary shape error -- and \
            the rejected-tail kinds are the ones that most need naming.
            """
        )

        // Distinct kinds must not share a category either: two kinds mapped to
        // one string is the same loss of resolution, just spelled differently.
        let collisions = Dictionary(grouping: categories, by: { $0.value })
            .filter { $0.value.count > 1 }
            .map { category, pairs in
                "\(category) <- \(pairs.map(\.key).sorted().joined(separator: "+"))"
            }
            .sorted()
        #expect(
            collisions.isEmpty,
            """
            distinct violation kinds share a redacted category: \
            \(collisions.joined(separator: "; "))
            """
        )
    }

    /// The redacted artifact must never carry the violation's detail text, the
    /// step number, or anything else that could turn a failing run into a query
    /// oracle for the hidden prompt (contract layer L6). Checked against a
    /// violation carrying deliberately distinctive detail.
    @Test
    func redactedCategoryCarriesNoDetailFromTheViolation() throws {
        let workspace = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: workspace) }

        // Reference token ids and logit values are the material that must not
        // escape; DFlashContractViolation is documented as never carrying them,
        // so this asserts the redactor's behaviour if one ever did.
        let violation = DFlashContractViolation(
            kind: .workBindingLogitMismatch,
            step: 61,
            detail: "row 3 top2Logits [24.1875, 24.125] vs reference [31337.0, 90210.0]"
        )
        try writeScore(errorText: violation.description, to: workspace)

        let result = try runRedactor(in: workspace)

        #expect(result.status == 0)
        #expect(result.artifact.contains("dflash_work_binding_logit_mismatch"))
        for leaked in ["31337", "90210", "24.1875", "top2Logits", "row 3"] {
            #expect(
                !result.artifact.contains(leaked),
                "the redacted artifact leaked '\(leaked)' out of the violation detail"
            )
        }
    }

    // MARK: - harness

    /// Drives the real redactor with a score whose `.metrics.error` is the
    /// string the harness actually authors for this kind
    /// (`DFlashContractViolation.description`), and returns the category it
    /// chose. Going through the real script and the real error text is the
    /// point: a text-matching test would have passed against the three missing
    /// arms, because their `case` labels were absent rather than misspelled.
    private func redactedCategory(for kind: DFlashContractViolation.Kind) throws -> String {
        let workspace = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: workspace) }

        let violation = DFlashContractViolation(kind: kind, step: 17)
        try writeScore(errorText: violation.description, to: workspace)

        let result = try runRedactor(in: workspace)
        #expect(result.status == 0, "redactor exited \(result.status) for \(kind.rawValue)")

        guard
            let data = result.artifact.data(using: .utf8),
            let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let category = object["failure_category"] as? String
        else {
            Issue.record("no failure_category in the redacted artifact for \(kind.rawValue)")
            return ""
        }
        return category
    }

    private func writeScore(errorText: String, to workspace: URL) throws {
        let score: [String: Any] = [
            "passed": false,
            "metrics": [
                "error": errorText,
                "passed_correctness": false,
                "first_failing_step": 17,
            ] as [String: Any],
        ]
        let data = try JSONSerialization.data(withJSONObject: score)
        try data.write(to: workspace.appendingPathComponent("score.json"))
    }

    private func runRedactor(in workspace: URL) throws -> (status: Int32, artifact: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(Self.redactorPath).path,
            "benchmark-failure.json",
        ]
        process.currentDirectoryURL = workspace
        // Blank GITHUB_STEP_SUMMARY so a test run inside CI does not append
        // fixture rows to the real job summary.
        process.environment = ProcessInfo.processInfo.environment
            .merging(["GITHUB_STEP_SUMMARY": ""]) { _, new in new }
        let stdout = Pipe()
        process.standardOutput = stdout
        try process.run()
        _ = stdout.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        let artifact =
            (try? String(
                contentsOf: workspace.appendingPathComponent("benchmark-failure.json"),
                encoding: .utf8
            )) ?? ""
        return (process.terminationStatus, artifact)
    }

    private func temporaryDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("dflash-redactor-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
