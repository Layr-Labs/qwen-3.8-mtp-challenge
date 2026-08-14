import Foundation
import Testing

import MLXFastCore
@testable import MLXFastHarness

// THE 2026-08-14 CONTRACT CHANGE, tested.
//
// The competitive surface of the Qwen 3.6 native-MTP track became the UNION of
// MLX runner/kernel optimisation and the whole speculative apparatus: head
// weights, drafting code, and per-round draft depth/schedule. Three mechanisms
// carry that change and each of them can fail silently in a way a reviewer
// would not notice:
//
//   1. HEAD DELIVERY. A declaration that fails to parse must be a REFUSAL, not
//      a quiet fall back to the pinned head -- "your declaration was broken so
//      we scored you on something else" is how a leaderboard number becomes
//      unattributable. Absence, by contrast, must be a clean default, because
//      submission archives have replace semantics and most will not carry one.
//   2. THE OPTIONAL-PATH / BYTE-BUDGET PLUMBING that makes (1) reachable at all.
//      Without it the overlay refuses the default case and the source byte
//      budget refuses the weights.
//   3. DEPTH FREEDOM. Every ledger rule has to key off the drafts a round
//      ACTUALLY proposed, and exactly one requested-depth rule has to survive:
//      the serial control may never draft.
//
// Nothing here loads a model or needs a network.

@Suite
struct QwenMTPHeadDeclarationTests {
    private typealias S = DFlashGateTextSupport

    private func parse(_ json: String) throws -> QwenMTPHeadDeclaration {
        try QwenMTPHeadDeclaration.parse(
            data: Data(json.utf8), origin: "test")
    }

    /// The checked-in declaration is the DEFAULT, and the default is pinned.
    ///
    /// The file itself was MISSING until 2026-08-14 -- the contract change
    /// added `mtp-head.manifest.json` to `editablePaths` and to
    /// `optionalEditablePaths`, and named it in the README, the runbook and the
    /// static-review prompt, but never authored it. That is not a cosmetic gap:
    /// this test read it through `parse(contentsOf:)`, which treats an
    /// unreadable path as a REFUSAL rather than as absence, so the one test
    /// asserting the shipped default threw instead of checking it. Absence
    /// would still have resolved to `pinned` through `resolve(contractRoot:)`
    /// -- asserted separately below -- but the default a participant reads
    /// before editing has to exist to be read.
    @Test
    func theCheckedInDeclarationSelectsThePinnedHead() throws {
        let declaration = try QwenMTPHeadDeclaration.parse(
            contentsOf: URL(fileURLWithPath: QwenMTPHeadDeclaration.relativePath))
        #expect(declaration.source == .pinned)
        #expect(declaration.maxBytes == QwenMTPHeadDeclaration.defaultMaxBytes)
        // 2 GiB, and the same number in all three places that state it.
        #expect(QwenMTPHeadDeclaration.defaultMaxBytes == 2_147_483_648)
        let manifest = try S.json("benchmark.json")
        let budget = try #require(
            manifest["editableSurfaceByteBudget"] as? [String: Any])
        #expect(
            (budget["exemptPathMaxBytes"] as? Int)
                == QwenMTPHeadDeclaration.defaultMaxBytes)

        // It NAMES the head it selects. The parser does not verify these for
        // source `pinned` -- sha256 and bytes bound only a fetched or in-branch
        // head -- so they are recorded, and what makes them honest is that they
        // must equal the enforcing pins: the manifest fixture's own
        // model.safetensors record, and the tensor count asserted at load.
        let declared = try S.json(QwenMTPHeadDeclaration.relativePath)
        #expect(
            declared["repository"] as? String
                == "EigenLabs/Qwen3.8-27B-MTP-bf16")
        #expect(
            declared["revision"] as? String
                == "26a328e070875b0314d652a039b6b59902690f03")
        #expect(
            (declared["tensor_count"] as? Int)
                == MLXFastConstants.qwenMTPHeadTensorCount)
        let weightsRecord = try #require(
            S.text("fixtures/qwen3_8_27b_mtp_head.sha256")
                .split(separator: "\n")
                .first { $0.hasSuffix(" model.safetensors") }
                .map { $0.split(separator: " ").map(String.init) },
            "the head manifest fixture pins no model.safetensors record"
        )
        #expect(declaration.sha256 == weightsRecord[0])
        #expect(declaration.bytes == Int(weightsRecord[1]))
    }

    /// ABSENCE is the common case, not an error: archives replace the editable
    /// surface wholesale and most will not carry a declaration.
    @Test
    func anAbsentDeclarationResolvesToThePinnedHead() throws {
        let empty = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(
            at: empty, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: empty) }
        let resolved = try QwenMTPHeadDeclaration.resolve(contractRoot: empty)
        #expect(resolved == QwenMTPHeadDeclaration.pinnedDefault)
        #expect(resolved.source == .pinned)
    }

    /// A declaration that is PRESENT and broken must fail closed. Each case
    /// below is a distinct way to end up scoring a head nobody declared.
    @Test
    func abrokenDeclarationIsARefusalAndNeverFallsBackToPinned() throws {
        #expect(throws: (any Error).self) { try parse("not json at all") }
        #expect(throws: (any Error).self) {
            try parse(#"{"source":"borrowed"}"#)
        }
        // remote/in_branch without a verifiable digest
        #expect(throws: (any Error).self) {
            try parse(#"{"source":"remote","source_url":"hf:o/r@abc","bytes":10}"#)
        }
        #expect(throws: (any Error).self) {
            try parse(
                #"{"source":"remote","source_url":"hf:o/r@abc","sha256":"nothex","bytes":10}"#)
        }
        // ... without a byte count
        #expect(throws: (any Error).self) {
            try parse(
                #"{"source":"remote","source_url":"hf:o/r@abc","sha256":"\#(String(repeating: "a", count: 64))"}"#)
        }
        // ... over the cap
        #expect(throws: (any Error).self) {
            try parse(
                #"{"source":"remote","source_url":"hf:o/r@abc","sha256":"\#(String(repeating: "a", count: 64))","bytes":3000000000}"#)
        }
        // ... raising the cap itself, which would make the cap meaningless
        #expect(throws: (any Error).self) {
            try parse(
                #"{"source":"pinned","max_bytes":9999999999}"#)
        }
        // remote without a recognised URL scheme
        #expect(throws: (any Error).self) {
            try parse(
                #"{"source":"remote","source_url":"https://example.com/h.bin","sha256":"\#(String(repeating: "a", count: 64))","bytes":10}"#)
        }
        // in_branch with a traversal path
        #expect(throws: (any Error).self) {
            try parse(
                #"{"source":"in_branch","path":"../../etc","sha256":"\#(String(repeating: "a", count: 64))","bytes":10}"#)
        }
    }

    /// The legal shapes parse, and the digest is normalised to lower case so a
    /// participant who pasted an upper-case digest is not silently rejected by
    /// a string comparison later.
    @Test
    func thelegalDeclarationShapesParse() throws {
        let sha = String(repeating: "A", count: 64)
        let remote = try parse(
            #"{"source":"remote","source_url":"hf:mlx-community/X@deadbeef","sha256":"\#(sha)","bytes":1024}"#)
        #expect(remote.source == .remote)
        #expect(remote.sha256 == sha.lowercased())
        #expect(remote.bytes == 1024)

        let r2 = try parse(
            #"{"source":"remote","source_url":"r2:heads/x.safetensors","sha256":"\#(sha)","bytes":1024}"#)
        #expect(r2.source == .remote)

        let inBranch = try parse(
            #"{"source":"in_branch","path":"mtp-head","sha256":"\#(sha)","bytes":1024}"#)
        #expect(inBranch.source == .inBranch)
        #expect(inBranch.path == "mtp-head")
    }

    /// The tree digest is what a participant is asked to DECLARE, so the rule
    /// has to be reproducible from the documentation alone: sha256 over
    /// "<file sha256>  <relative path>\n" lines, LC_ALL=C sorted, README.md
    /// excluded. This computes it on a real directory and checks the properties
    /// that make it usable -- stable, order-independent, README-insensitive,
    /// and sensitive to content.
    @Test
    func theHeadTreeDigestIsStableOrderIndependentAndContentSensitive() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try "one".write(
            to: root.appendingPathComponent("a.safetensors"),
            atomically: true, encoding: .utf8)
        try "two".write(
            to: root.appendingPathComponent("b.safetensors"),
            atomically: true, encoding: .utf8)

        let declaration = QwenMTPHeadDeclaration.pinnedDefault
        let first = computeQwenMTPHeadProvenance(
            headDirectory: root.path, declaration: declaration)
        #expect(first.fileCount == 2)
        #expect(first.bytes == 6)
        #expect(first.sha256.count == 64)
        #expect(first.source == "pinned")

        // Recomputation is stable.
        let second = computeQwenMTPHeadProvenance(
            headDirectory: root.path, declaration: declaration)
        #expect(second.sha256 == first.sha256)

        // A README does NOT move the digest: the docs tell participants to
        // exclude it, so the implementation must too or every declared digest
        // is off by a documentation file.
        try "docs".write(
            to: root.appendingPathComponent("README.md"),
            atomically: true, encoding: .utf8)
        let withReadme = computeQwenMTPHeadProvenance(
            headDirectory: root.path, declaration: declaration)
        #expect(withReadme.sha256 == first.sha256)

        // Content DOES move it.
        try "three".write(
            to: root.appendingPathComponent("b.safetensors"),
            atomically: true, encoding: .utf8)
        let changed = computeQwenMTPHeadProvenance(
            headDirectory: root.path, declaration: declaration)
        #expect(changed.sha256 != first.sha256)
    }

    /// Provenance names the SOURCE the declaration chose, not the directory the
    /// bytes happened to sit in -- otherwise a fetched head and a pinned head
    /// would be indistinguishable in the sealed report.
    @Test
    func provenanceCarriesTheDeclaredSource() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try "w".write(
            to: root.appendingPathComponent("model.safetensors"),
            atomically: true, encoding: .utf8)

        let sha = String(repeating: "b", count: 64)
        let remote = try parse(
            #"{"source":"remote","source_url":"hf:o/r@rev","sha256":"\#(sha)","bytes":1}"#)
        let provenance = computeQwenMTPHeadProvenance(
            headDirectory: root.path, declaration: remote)
        #expect(provenance.source == "remote")
        #expect(provenance.origin == "hf:o/r@rev")
    }
}

@Suite
struct QwenMTPOptionalEditablePathTests {
    private typealias S = DFlashGateTextSupport

    /// THE PLUMBING THAT MAKES THE DEFAULT REACHABLE.
    ///
    /// Submission archives replace the editable surface wholesale: a blob the
    /// archive does not carry is deleted. The overlay's blanket "submitted
    /// editable path is missing -> exit 1" would therefore have turned the
    /// COMMON case -- an archive with no head declaration -- into a refusal. The
    /// two head paths are optional; every source path is not, because there a
    /// missing file really does mean a stale clone and the loud refusal is
    /// right.
    @Test
    func theHeadPathsAreOptionalAndEverySourcePathIsNot() throws {
        let manifest = try S.json("benchmark.json")
        let editable = try #require(manifest["editablePaths"] as? [String])
        let optional = try #require(manifest["optionalEditablePaths"] as? [String])

        #expect(Set(optional) == ["mtp-head.manifest.json", "mtp-head"])
        for path in optional {
            #expect(
                editable.contains(path),
                "\(path) is optional but is not an editable path at all")
        }
        // Nothing else may be optional: a source file quietly falling back to
        // the trusted copy would let a stale archive score trusted code as if
        // the submission had written it.
        #expect(optional.count == 2)

        // And the overlay must actually honour the list rather than merely
        // carrying it.
        let overlay = try S.text(".github/scripts/overlay-editable-paths.sh")
        #expect(overlay.contains("optionalEditablePaths"))
        #expect(overlay.contains("is_optional_editable_path"))
        #expect(
            overlay.contains("keeping the trusted copy"),
            "an absent optional path must keep the trusted copy, not delete it")
        // The refusal for non-optional paths survives.
        #expect(overlay.contains("submitted editable path is missing"))
    }

    /// The head weights path is held out of the SOURCE byte budget and given
    /// its own cap. Without this the 3 MB code budget makes in-branch delivery
    /// unusable; with it applied too widely the lookup-table defence is gone.
    @Test
    func onlyTheHeadWeightsPathIsExemptFromTheSourceByteBudget() throws {
        let manifest = try S.json("benchmark.json")
        let budget = try #require(
            manifest["editableSurfaceByteBudget"] as? [String: Any])
        let exempt = try #require(budget["exemptPaths"] as? [String])
        #expect(exempt == ["mtp-head"])
        #expect((budget["exemptPathMaxBytes"] as? Int) == 2_147_483_648)

        // The reviewer honours it (it cannot read safetensors anyway) ...
        let review = try S.text(".github/scripts/run-submission-static-review.sh")
        #expect(review.contains("editableSurfaceByteBudget.exemptPaths"))
        #expect(review.contains("reviewed_paths"))
        #expect(
            !review.contains("-- \"${editable_paths[@]}\""),
            "a git pathspec still uses the unfiltered editable set, so exempt weights would enter the reviewed diff")
        // ... and so does the launch-time backstop in the trusted harness.
        let swift = try S.text(
            "Sources/MLXFastTrustedHarness/EditableSurfaceByteBudget.swift")
        #expect(swift.contains("exemptPaths"))
        #expect(swift.contains("defaultExemptPathMaxBytes"))
    }

    /// The byte budget, executed: an exempt path's bytes must not count against
    /// the code total, and a non-exempt path's must.
    @Test
    func theByteBudgetChargesExemptPathsSeparately() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(UUID().uuidString)
        let fm = FileManager.default
        try fm.createDirectory(
            at: root.appendingPathComponent("src"),
            withIntermediateDirectories: true)
        try fm.createDirectory(
            at: root.appendingPathComponent("weights"),
            withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        try String(repeating: "s", count: 100).write(
            to: root.appendingPathComponent("src/a.swift"),
            atomically: true, encoding: .utf8)
        // Comfortably above the code total, and nowhere near the exempt cap.
        try String(repeating: "w", count: 5_000).write(
            to: root.appendingPathComponent("weights/model.safetensors"),
            atomically: true, encoding: .utf8)

        let contract = root.appendingPathComponent("benchmark.json")
        try #"""
        {"editablePaths":["src","weights"],
         "editableSurfaceByteBudget":{"exemptPaths":["weights"],
                                      "exemptPathMaxBytes":1000000}}
        """#.write(to: contract, atomically: true, encoding: .utf8)

        let verdict = verifyEditableSurfaceByteBudget(
            contractPath: contract.path,
            maxTotalBytes: 1_000,
            maxFileBytes: 1_000
        )
        guard case let .verified(totalBytes, fileCount) = verdict else {
            Issue.record("exempt weights were charged to the code budget: \(verdict)")
            return
        }
        #expect(totalBytes == 100, "only the source file may count toward the code total")
        #expect(fileCount == 2, "the exempt file is still counted, just not charged")

        // The exempt cap still binds.
        let tight = root.appendingPathComponent("tight.json")
        try #"""
        {"editablePaths":["src","weights"],
         "editableSurfaceByteBudget":{"exemptPaths":["weights"],
                                      "exemptPathMaxBytes":100}}
        """#.write(to: tight, atomically: true, encoding: .utf8)
        guard case .exceeded = verifyEditableSurfaceByteBudget(
            contractPath: tight.path, maxTotalBytes: 1_000, maxFileBytes: 1_000)
        else {
            Issue.record("the exempt-path cap did not bind")
            return
        }
    }
}

@Suite
struct QwenMTPDepthFreedomTests {
    private typealias S = DFlashGateTextSupport

    /// The trusted maximum is ONE number, stated in five places that must
    /// agree: the Swift constant, the model-side mirror, the contract fixture,
    /// the ranked workflow's offered ceiling, and the box wrapper's row-
    /// accounting bound. A change to any one of them alone silently re-tunes
    /// what the ledger will accept.
    @Test
    func theTrustedMaximumIsTheSameEightEverywhere() throws {
        #expect(MLXFastConstants.qwenMTPMaxDraftDepth == 8)
        #expect(MLXFastConstants.qwenMTPMaxDepth
            == MLXFastConstants.qwenMTPMaxDraftDepth)

        let fixture = try S.json("fixtures/qwen3_8_27b_mtp_track.json")
        let proto = try #require(fixture["protocol"] as? [String: Any])
        #expect((proto["maximum_depth"] as? Int) == 8)
        #expect((proto["offered_draft_depth_ceiling"] as? Int) == 8)
        // The retired pin must be GONE, not left beside its replacement.
        #expect(proto["candidate_depth"] == nil)
        #expect((proto["empty_draft_rounds_legal"] as? Bool) == true)
        #expect((proto["candidate_declares_no_depth"] as? Bool) == true)

        let environment = try S.jobEnvironment(
            try S.text(".github/workflows/qwen-mtp-ranked-benchmark.yml"))
        #expect(environment["MLXFAST_QWEN_MTP_DEPTH"] == "8")

        let manifest = try S.json("benchmark.json")
        let scoring = try #require(manifest["scoring"] as? [String: Any])
        #expect((scoring["mtpMaxDraftDepth"] as? Int) == 8)
        #expect(scoring["mtpDepth"] == nil, "the retired pinned depth is still declared")
    }

    /// EVERY ledger rule keys off the drafts a round ACTUALLY proposed. A round
    /// that drafts narrower than the offer, wider than the offer, or not at all
    /// must all close; only the trusted maximum bounds it.
    @Test
    func theLedgerClosesOnActualDraftsRatherThanTheOffer() throws {
        // 4 rounds, offered 2, actually drafted 1 each: 4 primaries + 4
        // accepted drafts = 8 tokens, 2 rows per round.
        #expect(
            QwenMTPRowAccounting.closes(
                emittedTokenTotal: 8,
                configuredTokenTotal: 8,
                declaredRowTotal: 8,
                referenceCheckedRowTotal: 8,
                acceptedDraftTotal: 4,
                rejectedDraftTotal: 0,
                targetTailTotal: 4,
                roundCount: 4,
                depth: 2,
                seedTokenCount: 512,
                targetCacheOffsetFinal: 520
            ),
            "a round narrower than the offer must close -- k=1 is a legal submission")

        // Same 4 rounds, offered 2, actually drafted 5 each. Legal since
        // 2026-08-14: only qwenMTPMaxDraftDepth bounds the width, and using the
        // offer here would reject an adaptive run whose rounds went wider than
        // the parent's tail-narrowed offer.
        #expect(
            QwenMTPRowAccounting.closes(
                emittedTokenTotal: 24,
                configuredTokenTotal: 24,
                declaredRowTotal: 24,
                referenceCheckedRowTotal: 24,
                acceptedDraftTotal: 20,
                rejectedDraftTotal: 0,
                targetTailTotal: 4,
                roundCount: 4,
                depth: 2,
                seedTokenCount: 512,
                targetCacheOffsetFinal: 536
            ),
            "a round wider than the offer must close; the trusted maximum is the bound")

        // Drafted nothing at all: 4 rounds, 4 tokens, one tail row each.
        #expect(
            QwenMTPRowAccounting.closes(
                emittedTokenTotal: 4,
                configuredTokenTotal: 4,
                declaredRowTotal: 4,
                referenceCheckedRowTotal: 4,
                acceptedDraftTotal: 0,
                rejectedDraftTotal: 0,
                targetTailTotal: 4,
                roundCount: 4,
                depth: 8,
                seedTokenCount: 512,
                targetCacheOffsetFinal: 516
            ),
            "a non-drafting candidate is legal: it is the serial control and scores 1.0")
    }

    /// THE ONE REQUESTED-DEPTH RULE THAT SURVIVES. Depth 0 is the denominator
    /// this track divides by; a "serial" leg that drafted anything is an
    /// already-accelerated baseline, which is the mislabel that once produced a
    /// 0.875x "speedup" out of a method measured at 1.34x.
    @Test
    func theSerialControlMayNeverDraft() throws {
        #expect(
            !QwenMTPRowAccounting.closes(
                emittedTokenTotal: 8,
                configuredTokenTotal: 8,
                declaredRowTotal: 12,
                referenceCheckedRowTotal: 12,
                acceptedDraftTotal: 4,
                rejectedDraftTotal: 0,
                targetTailTotal: 8,
                roundCount: 8,
                depth: MLXFastConstants.qwenMTPSerialControlDepth,
                seedTokenCount: 512,
                targetCacheOffsetFinal: 520
            ))
        #expect(
            !QwenMTPRowAccounting.closes(
                emittedTokenTotal: 8,
                configuredTokenTotal: 8,
                declaredRowTotal: 12,
                referenceCheckedRowTotal: 12,
                acceptedDraftTotal: 0,
                rejectedDraftTotal: 4,
                targetTailTotal: 8,
                roundCount: 8,
                depth: MLXFastConstants.qwenMTPSerialControlDepth,
                seedTokenCount: 512,
                targetCacheOffsetFinal: 520
            ),
            "a REJECTED draft on the serial leg is still drafting; the leg did speculative work")
    }

    /// The refusal that used to stop a non-drafting round is gone from the
    /// driver, and the kind it threw is retained only so archived reports and
    /// the wrapper's rejection-class table still resolve.
    @Test
    func theNonDraftingRefusalIsGoneButItsKindSurvives() throws {
        let driver = try S.text(
            "Sources/MLXFastTrustedHarness/QwenRuntimeMTPDriver.swift")
        #expect(
            !driver.contains("kind: .stopTokenInsideWindow"),
            "the driver still throws the non-drafting refusal")
        #expect(driver.contains("NON-DRAFTING ROUNDS ARE LEGAL"))
        // The kind stays enumerable.
        #expect(
            QwenMTPContractViolation.Kind.allCases
                .contains(.stopTokenInsideWindow))

        // What did NOT move: a round must still commit a token, and the window
        // still runs to the parent's own configured total.
        #expect(driver.contains("no tokens returned"))
        #expect(driver.contains("while emitted.count < options.totalTokenCount"))
    }

    /// Effective depth is derived by the PARENT from its own journal and
    /// summarised in the report. This is the requested-vs-effective provenance
    /// gap (k-matrix finding 5): a report naming only the request describes a
    /// run that, under an adaptive schedule, may never have happened.
    @Test
    func effectiveDepthIsSummarisedFromTheParentsOwnJournal() throws {
        let report = QwenMTPReport(
            verb: "mtp-timed",
            depth: 8,
            seedTokenCount: 512,
            decodeTokenCount: 10,
            emittedTokenTotal: 10,
            allTokensMatched: true,
            firstDivergenceIndex: nil,
            firstDivergenceReferenceMargin: nil,
            roundCount: 4,
            acceptedDraftTotal: 6,
            rejectedDraftTotal: 0,
            targetTailTotal: 4,
            declaredRowTotal: 10,
            referenceCheckedRowTotal: 10,
            rejectedRowsReferenceChecked: 0,
            verifyBlockReplayedRoundCount: 0,
            residualDivergenceCount: 0,
            maxRejectedTailLogitDelta: 0,
            targetCacheOffsetFinal: 522,
            decodeSeconds: 1,
            maxRoundRequestSeconds: 1,
            p50RoundRequestSeconds: 1,
            effectiveDraftLengths: [3, 0, 2, 1]
        )
        #expect(report.effectiveMeanDraftLength == 1.5)
        #expect(report.effectiveMaxDraftLength == 3)
        #expect(report.nonDraftingRoundCount == 1)
        #expect(report.maxDraftDepthBound == 8)
        // The offer and the schedule are different quantities and the report
        // says both.
        #expect(report.depth == 8)
        #expect(report.effectiveMaxDraftLength < report.depth)

        // The CLI seals all of it.
        let cli = try S.text("Sources/MLXFastCLI/main.swift")
        for key in [
            "effective_mean_draft_len", "effective_max_draft_len",
            "non_drafting_round_count", "effective_draft_lengths",
            "max_draft_depth_bound", "requested_draft_depth",
            "head_provenance",
        ] {
            #expect(cli.contains("\"\(key)\""), "the payload does not seal \(key)")
        }
    }

    /// The shipped draft policy is EDITABLE and returns min(offer, 2), which is
    /// what keeps an unmodified tree reproducing the behaviour the calibration
    /// was measured against even though the offered ceiling moved to 8.
    @Test
    func theShippedDraftPolicyKeepsAnUnmodifiedTreeAtTwo() throws {
        let session = try S.text(
            "Sources/MLXFastModel/Qwen36MTPBlockSession.swift")
        #expect(session.contains("public var draftPolicy"))
        #expect(session.contains("public static let defaultDraftDepth = 2"))
        #expect(
            session.contains("THE POLICY BELOW IS THE FIRST THING A SUBMISSION SHOULD CHANGE"),
            "the editable policy must be signposted as editable")
        // A zero from the policy takes the same no-draft path the serial
        // control takes -- an adaptive skip costs exactly what serial costs.
        #expect(session.contains("|| draftCount == 0"))

        let fixture = try S.json("fixtures/qwen3_8_27b_mtp_track.json")
        let proto = try #require(fixture["protocol"] as? [String: Any])
        #expect((proto["shipped_default_draft_policy"] as? Int) == 2)
    }
}

@Suite
struct QwenMTPHeadProvenanceGateTests {
    private typealias S = DFlashGateTextSupport

    private static let workflowPath =
        ".github/workflows/qwen-mtp-ranked-benchmark.yml"

    /// The three head booleans stopped being pass-conditions and became
    /// recorded fields. Both halves matter: asserting them true would reject a
    /// legal non-drafting candidate, and dropping them entirely would lose a
    /// schema regression the payload should still be held to.
    @Test
    func theHeadBooleansAreRecordedRatherThanAsserted() throws {
        let workflow = try S.text(Self.workflowPath)
        let gate = try S.stepBody(
            workflow, "Qwen-MTP correctness and parity gate (untimed)")

        for boolean in [
            "uses_pinned_mtp_head", "uses_native_mtp_head", "mtp_head_attached",
        ] {
            #expect(
                !gate.contains("and .\(boolean) == true"),
                "\(boolean) is still a pass-condition; a legal non-drafting candidate would be rejected")
            #expect(
                gate.contains("(.\(boolean) | type == \"boolean\")"),
                "\(boolean) is not recorded at all; a payload that dropped it would pass")
        }
        // The two spellings must still be pinned equal to each other.
        #expect(gate.contains(".uses_native_mtp_head == .uses_pinned_mtp_head"))
        // And WHICH head drafted is answered by a digest, not a boolean.
        #expect(gate.contains("head_provenance.sha256"))
        #expect(gate.contains("head_provenance.source"))
    }

    /// The head is resolved RUNNER-SIDE, pre-sandbox, and the override reaches
    /// the CANDIDATE leg only. The baseline leg keeping the pinned head is what
    /// anchors the paired ratio; if the override leaked to it, the denominator
    /// would move with the submission.
    @Test
    func theHeadOverrideReachesTheCandidateLegOnly() throws {
        let workflow = try S.text(Self.workflowPath)
        let resolve = try S.stepBody(workflow, "Resolve the declared Qwen-MTP head")
        #expect(resolve.contains("mtp-head.manifest.json"))
        #expect(resolve.contains("MLXFAST_QWEN_MTP_CANDIDATE_HEAD_DIR"))
        // Fail-closed on every declared-head failure mode.
        #expect(resolve.contains("digest mismatch"))
        #expect(resolve.contains("byte count mismatch"))
        #expect(resolve.contains("above the ${cap}-byte cap"))
        #expect(resolve.contains("a broken head declaration is a refusal"))

        // The timed step hands the wrapper BOTH: the pinned head for the
        // baseline leg and the resolved one for the candidate leg.
        let timed = try S.stepBody(
            workflow, "Timed paired Qwen-MTP benchmark (measure-qwen-mtp-job)")
        #expect(timed.contains("QMTP_HEAD_DIR=\"${MLXFAST_QWEN_MTP_HEAD_DIR}\""))
        #expect(
            timed.contains(
                "QMTP_CANDIDATE_HEAD_DIR=\"${MLXFAST_QWEN_MTP_CANDIDATE_HEAD_DIR}\""))

        // The untimed gate is a candidate-side run and must use the resolved
        // head too, or a declared head would be gated on the pinned one.
        let gate = try S.stepBody(
            workflow, "Qwen-MTP correctness and parity gate (untimed)")
        #expect(
            gate.contains(
                "MLXFAST_QWEN_MTP_HEAD_DIR=\"${MLXFAST_QWEN_MTP_CANDIDATE_HEAD_DIR}\""))
    }
}

@Suite
struct QwenMTPLaunchInvariantTests {
    private typealias S = DFlashGateTextSupport

    /// THE LAUNCH BLOCKER, as an assertion.
    ///
    /// Yukon's import dispatches a BASELINE RUN OF THE SOURCE TREE and requires
    /// a published score to seed `currentBestScore`. No published score means
    /// the benchmark never opens, which means the track accepts no submissions
    /// at all -- so "the stock tree clears the submission floor" is not a
    /// quality property, it is a precondition for the track existing.
    ///
    /// It stopped being automatic on 2026-08-14. Under the retired normalised
    /// rule an unmodified tree scored 1.0 BY CONSTRUCTION and no floor below
    /// 1.0 could ever reject it. Under serial anchoring the stock tree publishes
    /// what it actually measures -- ~0.935, a real regression -- and the
    /// family's conventional 0.95 floor would have rejected the one run the
    /// whole track bootstraps from. Hence 0.90.
    ///
    /// The consequence for a future re-measurement is the point of pinning it
    /// here: if the stock median ever drifts below the floor, the floor moves.
    /// Shipping is not an option, because there would be nothing to ship to.
    @Test
    func theStockTreeMustClearTheSubmissionFloor() throws {
        let fixture = try S.json("fixtures/qwen3_8_27b_mtp_track.json")
        let semantics = try #require(fixture["scoring_semantics"] as? [String: Any])
        let calibration = try #require(fixture["calibration"] as? [String: Any])
        let floor = try #require(semantics["floor"] as? Double)
        let stock = try #require(calibration["expected_raw_median"] as? Double)

        #expect(
            stock > floor,
            """
            the stock tree's expected raw median \(stock) does not clear the \
            submission floor \(floor). Yukon's import dispatches a baseline run \
            of the source tree and needs a PUBLISHED score to seed \
            currentBestScore; with no score the benchmark never opens and the \
            track accepts nothing. Lower the floor -- do not ship this.
            """
        )
        // Margin, not just sign. A floor sitting a thousandth below the stock
        // median would satisfy the inequality and still fail the first time
        // thermal noise moved the median, so require room for the measured
        // dispersion (0.135% sd on the median, so 0.02 is ~15 sigma).
        #expect(stock - floor > 0.02)
        #expect((calibration["clears_submission_floor"] as? Bool) == true)

        // The enforcing site must carry the same number as the documents.
        let environment = try S.jobEnvironment(
            try S.text(".github/workflows/qwen-mtp-ranked-benchmark.yml"))
        #expect(environment["MLXFAST_QWEN_MTP_DECODE_SPEEDUP_FLOOR"] == "0.90")
        let manifest = try S.json("benchmark.json")
        let scoring = try #require(manifest["scoring"] as? [String: Any])
        #expect((scoring["decodeSpeedupFloor"] as? Double) == floor)

        // And a genuinely slower submission must still be refused: the floor is
        // a regression bound, not an amnesty. k3 (draft 3) medians 0.7769.
        #expect(0.7768609425 < floor)
    }
}
