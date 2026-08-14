import CryptoKit
import Foundation

public struct GoldenCase: Codable, Equatable {
    public let name: String
    public let promptTokens: [Int]
    public let expectedTokens: [Int]

    enum CodingKeys: String, CodingKey {
        case name
        case promptTokens = "prompt_tokens"
        case expectedTokens = "expected_tokens"
    }

    public init(name: String, promptTokens: [Int], expectedTokens: [Int]) {
        self.name = name
        self.promptTokens = promptTokens
        self.expectedTokens = expectedTokens
    }
}

public struct GoldenModelProvenance: Codable, Equatable {
    public let repository: String
    public let revision: String

    public init(repository: String, revision: String) {
        self.repository = repository
        self.revision = revision
    }
}

public struct GoldenAnchorCase: Codable, Equatable {
    public let name: String
    public let contextTokens: [Int]
    public let expectedToken: Int
    public let acceptedTokens: [Int]?
    public let maxExpectedRank: Int?
    public let maxTopLogitDelta: Double?

    enum CodingKeys: String, CodingKey {
        case name
        case contextTokens = "context_tokens"
        case expectedToken = "expected_token"
        case acceptedTokens = "accepted_tokens"
        case maxExpectedRank = "max_expected_rank"
        case maxTopLogitDelta = "max_top_logit_delta"
    }

    public init(
        name: String,
        contextTokens: [Int],
        expectedToken: Int,
        acceptedTokens: [Int]? = nil,
        maxExpectedRank: Int? = nil,
        maxTopLogitDelta: Double? = nil
    ) {
        self.name = name
        self.contextTokens = contextTokens
        self.expectedToken = expectedToken
        self.acceptedTokens = acceptedTokens
        self.maxExpectedRank = maxExpectedRank
        self.maxTopLogitDelta = maxTopLogitDelta
    }
}

public struct GoldenFreeRunCase: Codable, Equatable {
    public let name: String
    public let promptTokens: [Int]
    public let expectedTokens: [Int]
    public let exactPrefixTokens: Int?

    enum CodingKeys: String, CodingKey {
        case name
        case promptTokens = "prompt_tokens"
        case expectedTokens = "expected_tokens"
        case exactPrefixTokens = "exact_prefix_tokens"
    }

    public init(
        name: String,
        promptTokens: [Int],
        expectedTokens: [Int],
        exactPrefixTokens: Int? = nil
    ) {
        self.name = name
        self.promptTokens = promptTokens
        self.expectedTokens = expectedTokens
        self.exactPrefixTokens = exactPrefixTokens
    }
}

public struct GoldenBehaviorCase: Codable, Equatable {
    public let name: String
    public let promptTokens: [Int]
    public let acceptedTokenSequences: [[Int]]
    public let maxNewTokens: Int
    public let semanticPrompt: String?
    public let semanticAnswerKey: String?
    public let semanticReferenceAnswer: String?
    public let semanticDomain: String?
    public let semanticSubdomain: String?

    enum CodingKeys: String, CodingKey {
        case name
        case promptTokens = "prompt_tokens"
        case acceptedTokenSequences = "accepted_token_sequences"
        case maxNewTokens = "max_new_tokens"
        case semanticPrompt = "semantic_prompt"
        case semanticAnswerKey = "semantic_answer_key"
        case semanticReferenceAnswer = "semantic_reference_answer"
        case semanticDomain = "semantic_domain"
        case semanticSubdomain = "semantic_subdomain"
    }

    public init(
        name: String,
        promptTokens: [Int],
        acceptedTokenSequences: [[Int]],
        maxNewTokens: Int,
        semanticPrompt: String? = nil,
        semanticAnswerKey: String? = nil,
        semanticReferenceAnswer: String? = nil,
        semanticDomain: String? = nil,
        semanticSubdomain: String? = nil
    ) {
        self.name = name
        self.promptTokens = promptTokens
        self.acceptedTokenSequences = acceptedTokenSequences
        self.maxNewTokens = maxNewTokens
        self.semanticPrompt = semanticPrompt
        self.semanticAnswerKey = semanticAnswerKey
        self.semanticReferenceAnswer = semanticReferenceAnswer
        self.semanticDomain = semanticDomain
        self.semanticSubdomain = semanticSubdomain
    }
}

public struct GoldenCorrectnessGates: Codable, Equatable {
    public let anchors: [GoldenAnchorCase]?
    public let freeRun: [GoldenFreeRunCase]?
    public let behavior: [GoldenBehaviorCase]?

    enum CodingKeys: String, CodingKey {
        case anchors
        case freeRun = "free_run"
        case behavior
    }

    public init(
        anchors: [GoldenAnchorCase]? = nil,
        freeRun: [GoldenFreeRunCase]? = nil,
        behavior: [GoldenBehaviorCase]? = nil
    ) {
        self.anchors = anchors
        self.freeRun = freeRun
        self.behavior = behavior
    }

    public var anchorCases: [GoldenAnchorCase] {
        anchors ?? []
    }

    public var freeRunCases: [GoldenFreeRunCase] {
        freeRun ?? []
    }

    public var behaviorCases: [GoldenBehaviorCase] {
        behavior ?? []
    }

    public var totalCaseCount: Int {
        anchorCases.count + freeRunCases.count + behaviorCases.count
    }
}

public struct BenchmarkGolden: Codable, Equatable {
    public let prefillPromptTokens: [Int]
    public let expectedPrefillToken: Int
    public let decodeSeedTokens: [Int]
    public let expectedDecodeSeedToken: Int
    public let expectedDecodeTokens: [Int]
    // Per-prompt baselines for prompt-pool rotation (see
    // docs/benchmark-window-freeze.md). A pool golden carries the seconds-per-
    // token the baseline reference measured for THIS prompt on the official
    // runner, so rotating prompts of different intrinsic difficulty keeps
    // speedups comparable. Both must be present together or absent together;
    // absent means "score against the calibrated defaults in MLXFastConstants"
    // (the pre-pool behavior, used by all public fixtures).
    public let baselinePrefillSecondsPerToken: Double?
    public let baselineDecodeSecondsPerToken: Double?

    enum CodingKeys: String, CodingKey {
        case prefillPromptTokens = "prefill_prompt_tokens"
        case expectedPrefillToken = "expected_prefill_token"
        case decodeSeedTokens = "decode_seed_tokens"
        case expectedDecodeSeedToken = "expected_decode_seed_token"
        case expectedDecodeTokens = "expected_decode_tokens"
        case baselinePrefillSecondsPerToken = "baseline_prefill_seconds_per_token"
        case baselineDecodeSecondsPerToken = "baseline_decode_seconds_per_token"
    }

    public init(
        prefillPromptTokens: [Int],
        expectedPrefillToken: Int,
        decodeSeedTokens: [Int],
        expectedDecodeSeedToken: Int,
        expectedDecodeTokens: [Int],
        baselinePrefillSecondsPerToken: Double? = nil,
        baselineDecodeSecondsPerToken: Double? = nil
    ) {
        self.prefillPromptTokens = prefillPromptTokens
        self.expectedPrefillToken = expectedPrefillToken
        self.decodeSeedTokens = decodeSeedTokens
        self.expectedDecodeSeedToken = expectedDecodeSeedToken
        self.expectedDecodeTokens = expectedDecodeTokens
        self.baselinePrefillSecondsPerToken = baselinePrefillSecondsPerToken
        self.baselineDecodeSecondsPerToken = baselineDecodeSecondsPerToken
    }

    public var resolvedBaselinePrefillSecondsPerToken: Double {
        baselinePrefillSecondsPerToken ?? MLXFastConstants.officialBaselinePrefillSecondsPerToken
    }

    public var resolvedBaselineDecodeSecondsPerToken: Double {
        baselineDecodeSecondsPerToken ?? MLXFastConstants.officialBaselineDecodeSecondsPerToken
    }
}

public struct GoldenDocument: Codable, Equatable {
    public let version: Int?
    public let modelProvenance: GoldenModelProvenance?
    public let cases: [GoldenCase]
    public let correctnessGates: GoldenCorrectnessGates?
    public let benchmark: BenchmarkGolden?

    enum CodingKeys: String, CodingKey {
        case version
        case modelProvenance = "model_provenance"
        case cases
        case correctnessGates = "correctness_gates"
        case benchmark
    }

    public init(
        version: Int = 1,
        modelProvenance: GoldenModelProvenance? = nil,
        cases: [GoldenCase],
        correctnessGates: GoldenCorrectnessGates? = nil,
        benchmark: BenchmarkGolden?
    ) {
        self.version = version
        self.modelProvenance = modelProvenance
        self.cases = cases
        self.correctnessGates = correctnessGates
        self.benchmark = benchmark
    }
}

public struct GoldenFixture: Equatable {
    public let cases: [GoldenCase]
    public let correctnessGates: GoldenCorrectnessGates?
    public let benchmark: BenchmarkGolden?
    public let sha256: String

    public init(
        cases: [GoldenCase],
        correctnessGates: GoldenCorrectnessGates? = nil,
        benchmark: BenchmarkGolden?,
        sha256: String
    ) {
        self.cases = cases
        self.correctnessGates = correctnessGates
        self.benchmark = benchmark
        self.sha256 = sha256
    }

    public var totalCorrectnessCaseCount: Int {
        cases.count + (correctnessGates?.totalCaseCount ?? 0)
    }
}

public struct BenchmarkTokenComparison: Equatable {
    public let passed: Bool
    public let label: String
    public let step: Int?
    public let expectedToken: Int?
    public let actualToken: Int?

    public init(
        passed: Bool,
        label: String,
        step: Int?,
        expectedToken: Int?,
        actualToken: Int?
    ) {
        self.passed = passed
        self.label = label
        self.step = step
        self.expectedToken = expectedToken
        self.actualToken = actualToken
    }
}

public func loadGoldenCases(
    from path: String,
    requiredSteps: Int = MLXFastConstants.correctnessSteps,
    requiredPromptTokens: Int = MLXFastConstants.correctnessPromptTokens
) throws -> [GoldenCase] {
    try loadGoldenFixture(
        from: path,
        requiredSteps: requiredSteps,
        requiredPromptTokens: requiredPromptTokens
    ).cases
}

public func loadGoldenFixture(
    from path: String,
    requiredSteps: Int = MLXFastConstants.correctnessSteps,
    requiredPromptTokens: Int = MLXFastConstants.correctnessPromptTokens
) throws -> GoldenFixture {
    guard requiredSteps > 0 else {
        throw MLXFastError.invalidInput("correctness required steps must be positive")
    }
    guard requiredPromptTokens > 0 else {
        throw MLXFastError.invalidInput("correctness required prompt tokens must be positive")
    }
    try requireFile(path, description: "correctness golden file")

    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    try validateGoldenFixtureKeys(data)
    let decoded = try JSONDecoder().decode(GoldenDocument.self, from: data)
    guard decoded.version == 1 else {
        throw MLXFastError.invalidInput("correctness golden file version must be 1")
    }
    if let provenance = decoded.modelProvenance {
        guard provenance.repository == MLXFastConstants.referenceModelRepository,
              provenance.revision == MLXFastConstants.referenceModelRevision
        else {
            throw MLXFastError.invalidInput(
                "correctness golden model_provenance does not match the pinned reference model"
            )
        }
    }
    try validateGoldenCases(
        decoded.cases,
        requiredSteps: requiredSteps,
        requiredPromptTokens: requiredPromptTokens
    )
    if let correctnessGates = decoded.correctnessGates {
        try validateGoldenCorrectnessGates(
            correctnessGates,
            baseCases: decoded.cases,
            requiredPromptTokens: requiredPromptTokens
        )
    }
    if let benchmark = decoded.benchmark {
        try validateBenchmarkGolden(benchmark)
    }
    let digest = SHA256.hash(data: data)
    let hash = digest.map { String(format: "%02x", $0) }.joined()
    return GoldenFixture(
        cases: decoded.cases,
        correctnessGates: decoded.correctnessGates,
        benchmark: decoded.benchmark,
        sha256: hash
    )
}

public enum GoldenSequenceMatcher {
    public static func firstPrefixMismatch(
        expected: [Int],
        actual: [Int],
        prefixTokens: Int
    ) -> BenchmarkTokenComparison {
        guard prefixTokens > 0 else {
            return BenchmarkTokenComparison(
                passed: true,
                label: "token prefix",
                step: nil,
                expectedToken: nil,
                actualToken: nil
            )
        }
        for step in 0..<prefixTokens {
            let expectedToken = step < expected.count ? expected[step] : nil
            let actualToken = step < actual.count ? actual[step] : nil
            if expectedToken != actualToken {
                return BenchmarkTokenComparison(
                    passed: false,
                    label: "token prefix",
                    step: step,
                    expectedToken: expectedToken,
                    actualToken: actualToken
                )
            }
        }
        return BenchmarkTokenComparison(
            passed: true,
            label: "token prefix",
            step: nil,
            expectedToken: nil,
            actualToken: nil
        )
    }

    public static func matchesAnyExactSequence(
        acceptedSequences: [[Int]],
        actual: [Int]
    ) -> BenchmarkTokenComparison {
        for sequence in acceptedSequences where actual == sequence {
            return BenchmarkTokenComparison(
                passed: true,
                label: "accepted answer token sequence",
                step: nil,
                expectedToken: nil,
                actualToken: nil
            )
        }

        let firstSequence = acceptedSequences.first ?? []
        let closestMismatch = acceptedSequences
            .map { firstPrefixMismatch(expected: $0, actual: actual, prefixTokens: max($0.count, actual.count)) }
            .filter { !$0.passed }
            .max { lhs, rhs in
                (lhs.step ?? 0) < (rhs.step ?? 0)
            }
        return BenchmarkTokenComparison(
            passed: false,
            label: "accepted answer token sequence",
            step: closestMismatch?.step ?? 0,
            expectedToken: closestMismatch?.expectedToken ?? firstSequence.first,
            actualToken: closestMismatch?.actualToken ?? actual.first
        )
    }

    public static func matchesAnyAcceptedPrefix(
        acceptedSequences: [[Int]],
        actual: [Int]
    ) -> BenchmarkTokenComparison {
        for sequence in acceptedSequences {
            let comparison = firstPrefixMismatch(
                expected: sequence,
                actual: actual,
                prefixTokens: sequence.count
            )
            if comparison.passed {
                return BenchmarkTokenComparison(
                    passed: true,
                    label: "accepted answer token prefix",
                    step: nil,
                    expectedToken: nil,
                    actualToken: nil
                )
            }
        }

        let firstSequence = acceptedSequences.first ?? []
        let closestMismatch = acceptedSequences
            .map { firstPrefixMismatch(expected: $0, actual: actual, prefixTokens: $0.count) }
            .filter { !$0.passed }
            .max { lhs, rhs in
                (lhs.step ?? 0) < (rhs.step ?? 0)
            }
        return BenchmarkTokenComparison(
            passed: false,
            label: "accepted answer token prefix",
            step: closestMismatch?.step ?? 0,
            expectedToken: closestMismatch?.expectedToken ?? firstSequence.first,
            actualToken: closestMismatch?.actualToken ?? actual.first
        )
    }
}

public enum BenchmarkOutputValidator {
    public static func comparePrefillToken(
        expected: BenchmarkGolden,
        actualToken: Int
    ) -> BenchmarkTokenComparison {
        comparePrefillToken(
            expectedToken: expected.expectedPrefillToken,
            actualToken: actualToken
        )
    }

    public static func comparePrefillToken(
        expectedToken: Int,
        actualToken: Int
    ) -> BenchmarkTokenComparison {
        compareOne(
            label: "benchmark prefill token",
            expectedToken: expectedToken,
            actualToken: actualToken
        )
    }

    public static func compareDecodeSeedToken(
        expected: BenchmarkGolden,
        actualToken: Int
    ) -> BenchmarkTokenComparison {
        compareDecodeSeedToken(
            expectedToken: expected.expectedDecodeSeedToken,
            actualToken: actualToken
        )
    }

    public static func compareDecodeSeedToken(
        expectedToken: Int,
        actualToken: Int
    ) -> BenchmarkTokenComparison {
        compareOne(
            label: "benchmark decode seed token",
            expectedToken: expectedToken,
            actualToken: actualToken
        )
    }

    public static func compareDecodeTokens(
        expected: BenchmarkGolden,
        actualTokens: [Int]
    ) -> BenchmarkTokenComparison {
        compareDecodeTokens(
            expectedTokens: expected.expectedDecodeTokens,
            actualTokens: actualTokens
        )
    }

    public static func compareDecodeTokens(
        expectedTokens: [Int],
        actualTokens: [Int]
    ) -> BenchmarkTokenComparison {
        let steps = max(expectedTokens.count, actualTokens.count)
        for step in 0..<steps {
            let expectedToken = step < expectedTokens.count ? expectedTokens[step] : nil
            let actualToken = step < actualTokens.count ? actualTokens[step] : nil
            if expectedToken != actualToken {
                return BenchmarkTokenComparison(
                    passed: false,
                    label: "benchmark decode token",
                    step: step,
                    expectedToken: expectedToken,
                    actualToken: actualToken
                )
            }
        }
        return BenchmarkTokenComparison(
            passed: true,
            label: "benchmark decode token",
            step: nil,
            expectedToken: nil,
            actualToken: nil
        )
    }

    private static func compareOne(
        label: String,
        expectedToken: Int,
        actualToken: Int
    ) -> BenchmarkTokenComparison {
        if expectedToken == actualToken {
            return BenchmarkTokenComparison(
                passed: true,
                label: label,
                step: nil,
                expectedToken: nil,
                actualToken: nil
            )
        }
        return BenchmarkTokenComparison(
            passed: false,
            label: label,
            step: nil,
            expectedToken: expectedToken,
            actualToken: actualToken
        )
    }
}

public func validateBenchmarkGolden(_ benchmark: BenchmarkGolden) throws {
    guard benchmark.prefillPromptTokens.count == MLXFastConstants.benchmarkPrefillPromptTokens else {
        throw MLXFastError.invalidInput(
            "benchmark.prefill_prompt_tokens has \(benchmark.prefillPromptTokens.count) tokens; need exactly \(MLXFastConstants.benchmarkPrefillPromptTokens)"
        )
    }
    guard benchmark.decodeSeedTokens.count == MLXFastConstants.benchmarkDecodeSeedTokens else {
        throw MLXFastError.invalidInput(
            "benchmark.decode_seed_tokens has \(benchmark.decodeSeedTokens.count) tokens; need exactly \(MLXFastConstants.benchmarkDecodeSeedTokens). Replace stale local goldens with an updated precomputed golden fixture."
        )
    }
    guard benchmark.expectedDecodeTokens.count >= MLXFastConstants.benchmarkDecodeSteps else {
        throw MLXFastError.invalidInput(
            "benchmark.expected_decode_tokens has \(benchmark.expectedDecodeTokens.count) tokens; need at least \(MLXFastConstants.benchmarkDecodeSteps). Replace stale local goldens with an updated precomputed golden fixture."
        )
    }
    try validateTokens(benchmark.prefillPromptTokens, field: "benchmark.prefill_prompt_tokens")
    try validateTokens([benchmark.expectedPrefillToken], field: "benchmark.expected_prefill_token")
    try validateTokens(benchmark.decodeSeedTokens, field: "benchmark.decode_seed_tokens")
    try validateTokens([benchmark.expectedDecodeSeedToken], field: "benchmark.expected_decode_seed_token")
    try validateTokens(benchmark.expectedDecodeTokens, field: "benchmark.expected_decode_tokens")
    try validateBenchmarkGoldenBaselines(benchmark)
}

private func validateBenchmarkGoldenBaselines(_ benchmark: BenchmarkGolden) throws {
    // A half-calibrated oracle (one axis carried, one falling back to the
    // constants) would silently mix two calibration regimes in one score, so
    // the pair is all-or-nothing.
    switch (benchmark.baselinePrefillSecondsPerToken, benchmark.baselineDecodeSecondsPerToken) {
    case (nil, nil):
        return
    case (nil, _), (_, nil):
        throw MLXFastError.invalidInput(
            "benchmark.baseline_prefill_seconds_per_token and benchmark.baseline_decode_seconds_per_token must be provided together"
        )
    case let (prefill?, decode?):
        guard prefill.isFinite, prefill > 0 else {
            throw MLXFastError.invalidInput(
                "benchmark.baseline_prefill_seconds_per_token must be finite and positive"
            )
        }
        guard decode.isFinite, decode > 0 else {
            throw MLXFastError.invalidInput(
                "benchmark.baseline_decode_seconds_per_token must be finite and positive"
            )
        }
    }
}

// Derive the timed-benchmark oracle a ranked golden must carry from the
// golden's OWN hidden base case, and return the document with it attached.
//
// WHY THIS EXISTS. QwenRuntime.benchmark requires `.benchmark` unconditionally
// -- including on the gates-only phase (checkGates=1, skipTimedBenchmark=1),
// where the oracle supplies only the baseline seconds-per-token placeholders.
// `generate-golden` writes `benchmark: nil` and both attach verbs pass the
// section through unchanged, so before this function existed no in-repo tool
// could author a golden the ranked gates phase would accept.
//
// WHY IT DERIVES RATHER THAN MEASURES. The shape is not invented here: it is
// read off the serial-era hidden golden that the DFlash ranked pipeline has
// been consuming (correctness_prompts/laguna-xs-2.1-serial-v2/...94239d59),
// whose oracle satisfies exactly the five identities below against its own
// cases[0]. So the oracle restates tokens the golden already carries -- it
// introduces no token the reference model did not produce for this prompt,
// and it needs no model, no weights, and no timing run to author.
//
// Deliberately NO baseline_*_seconds_per_token: those two fields are the
// per-prompt pool-rotation baselines (see BenchmarkGolden), and a hidden
// correctness golden is not a pool golden. The precedent carries neither, and
// on the gates-only path they are inert anyway -- the placeholder branch sets
// measured := baseline, so both speedups are exactly 1.0 whatever they hold,
// and overlay-paired-timing.sh overwrites them with the measured pair.
public func goldenDocumentAttachingDerivedBenchmarkOracle(
    _ golden: GoldenDocument
) throws -> GoldenDocument {
    // Never silently rewrite an oracle a golden already carries: the existing
    // one may have been measured rather than derived, and the caller's
    // --output defaults to the input path.
    guard golden.benchmark == nil else {
        throw MLXFastError.invalidInput(
            "golden already contains a benchmark oracle; refusing to overwrite it"
        )
    }
    guard let baseCase = golden.cases.first else {
        throw MLXFastError.invalidInput(
            "golden contains no base cases to derive a benchmark oracle from"
        )
    }
    guard let firstExpectedToken = baseCase.expectedTokens.first else {
        throw MLXFastError.invalidInput(
            "golden base case \(baseCase.name) has no expected tokens to derive a benchmark oracle from"
        )
    }
    let oracle = BenchmarkGolden(
        prefillPromptTokens: baseCase.promptTokens,
        expectedPrefillToken: firstExpectedToken,
        decodeSeedTokens: baseCase.promptTokens,
        expectedDecodeSeedToken: firstExpectedToken,
        expectedDecodeTokens: Array(baseCase.expectedTokens.dropFirst())
    )
    // Fail here, before anything is written, on a base case too short to cover
    // the timed decode window (expected_tokens must be >= benchmarkDecodeSteps
    // + 1, the seed next-token plus the checked decode tokens).
    try validateBenchmarkGolden(oracle)
    return GoldenDocument(
        version: golden.version ?? 1,
        modelProvenance: golden.modelProvenance,
        cases: golden.cases,
        correctnessGates: golden.correctnessGates,
        benchmark: oracle
    )
}

private func validateGoldenFixtureKeys(_ data: Data) throws {
    let json = try JSONSerialization.jsonObject(with: data)
    guard let root = json as? [String: Any] else {
        throw MLXFastError.invalidInput("correctness golden file must be a JSON object")
    }
    try rejectUnknownKeys(
        Set(root.keys),
        allowed: [
            "version",
            "model_provenance",
            "cases",
            "correctness_gates",
            "benchmark",
        ],
        field: "correctness golden file"
    )
    guard root["version"] != nil else {
        return
    }
    guard let cases = root["cases"] as? [Any], !cases.isEmpty else {
        throw MLXFastError.invalidInput("correctness golden file must contain a non-empty cases array")
    }
    if let provenance = root["model_provenance"] {
        try validateGoldenModelProvenanceKeys(provenance)
    }
    if let gates = root["correctness_gates"] {
        try validateGoldenCorrectnessGateKeys(gates)
    }
    if let benchmark = root["benchmark"] {
        try validateGoldenBenchmarkKeys(benchmark)
    }
}

private func validateGoldenModelProvenanceKeys(_ provenance: Any) throws {
    guard !(provenance is NSNull),
          let object = provenance as? [String: Any]
    else {
        throw MLXFastError.invalidInput(
            "model_provenance must be a JSON object"
        )
    }
    try rejectUnknownKeys(
        Set(object.keys),
        allowed: ["repository", "revision"],
        field: "model_provenance"
    )
    guard Set(object.keys) == ["repository", "revision"],
          let repository = object["repository"] as? String,
          !repository.isEmpty,
          let revision = object["revision"] as? String,
          revision.range(
            of: "^[0-9a-f]{40}$",
            options: .regularExpression
          ) != nil
    else {
        throw MLXFastError.invalidInput(
            "model_provenance requires repository and a 40-character lowercase revision"
        )
    }
}

private func validateGoldenBenchmarkKeys(_ benchmark: Any) throws {
    guard !(benchmark is NSNull) else {
        throw MLXFastError.invalidInput("benchmark must not be null")
    }
    guard let object = benchmark as? [String: Any] else {
        throw MLXFastError.invalidInput("benchmark must be a JSON object")
    }
    // The baseline_* fields are scoring-critical: JSONDecoder silently drops
    // unknown keys, so a typo'd baseline key would make the run fall back to
    // the calibrated constants and score against the wrong baseline. Reject
    // anything not in the benchmark oracle contract.
    try rejectUnknownKeys(
        Set(object.keys),
        allowed: [
            "prefill_prompt_tokens",
            "expected_prefill_token",
            "decode_seed_tokens",
            "expected_decode_seed_token",
            "expected_decode_tokens",
            "baseline_prefill_seconds_per_token",
            "baseline_decode_seconds_per_token",
        ],
        field: "benchmark"
    )
}

private func validateGoldenCorrectnessGateKeys(_ gates: Any) throws {
    guard !(gates is NSNull) else {
        throw MLXFastError.invalidInput("correctness_gates must not be null")
    }
    guard let object = gates as? [String: Any] else {
        throw MLXFastError.invalidInput("correctness_gates must be a JSON object")
    }
    try rejectUnknownKeys(
        Set(object.keys),
        allowed: ["anchors", "free_run", "behavior"],
        field: "correctness_gates"
    )
    guard !object.isEmpty else {
        throw MLXFastError.invalidInput("correctness_gates must contain at least one gate section")
    }
    for key in ["anchors", "free_run", "behavior"] where object.keys.contains(key) {
        guard let value = object[key], !(value is NSNull) else {
            throw MLXFastError.invalidInput("correctness_gates.\(key) must not be null")
        }
        guard let array = value as? [Any] else {
            throw MLXFastError.invalidInput("correctness_gates.\(key) must be an array")
        }
        guard !array.isEmpty else {
            throw MLXFastError.invalidInput("correctness_gates.\(key) must not be empty when present")
        }
    }
}

private func rejectUnknownKeys(
    _ keys: Set<String>,
    allowed: Set<String>,
    field: String
) throws {
    let unknown = keys.subtracting(allowed)
    guard unknown.isEmpty else {
        throw MLXFastError.invalidInput(
            "\(field) contains unknown key \(unknown.sorted().joined(separator: ", "))"
        )
    }
}

private func validateGoldenCases(
    _ cases: [GoldenCase],
    requiredSteps: Int,
    requiredPromptTokens: Int
) throws {
    guard !cases.isEmpty else {
        throw MLXFastError.invalidInput("correctness golden file must contain at least one case")
    }

    var names = Set<String>()
    for testCase in cases {
        try validateCaseName(testCase.name, field: "correctness golden case name")
        guard names.insert(testCase.name).inserted else {
            throw MLXFastError.invalidInput("duplicate correctness golden case name \(testCase.name)")
        }
        if testCase.promptTokens.count != requiredPromptTokens {
            throw MLXFastError.invalidInput(
                "\(testCase.name).prompt_tokens has \(testCase.promptTokens.count) tokens; need exactly \(requiredPromptTokens)"
            )
        }
        if testCase.expectedTokens.count < requiredSteps {
            throw MLXFastError.invalidInput(
                "\(testCase.name).expected_tokens has \(testCase.expectedTokens.count) tokens; need at least \(requiredSteps)"
            )
        }
        try validateTokens(testCase.promptTokens, field: "\(testCase.name).prompt_tokens")
        try validateTokens(testCase.expectedTokens, field: "\(testCase.name).expected_tokens")
    }
}

private func validateGoldenCorrectnessGates(
    _ gates: GoldenCorrectnessGates,
    baseCases: [GoldenCase],
    requiredPromptTokens: Int
) throws {
    try validateGoldenAnchorCases(gates.anchorCases)
    try validateGoldenFreeRunCases(gates.freeRunCases, requiredPromptTokens: requiredPromptTokens)
    try validateGoldenBehaviorCases(gates.behaviorCases, requiredPromptTokens: requiredPromptTokens)
    try validateUniqueLayeredCaseNames(baseCases: baseCases, gates: gates)
}

private func validateGoldenAnchorCases(_ cases: [GoldenAnchorCase]) throws {
    var names = Set<String>()
    for testCase in cases {
        try validateCaseName(testCase.name, field: "correctness anchor case name")
        guard names.insert(testCase.name).inserted else {
            throw MLXFastError.invalidInput("duplicate correctness anchor case name \(testCase.name)")
        }
        guard !testCase.contextTokens.isEmpty else {
            throw MLXFastError.invalidInput("\(testCase.name).context_tokens must not be empty")
        }
        guard testCase.contextTokens.count <= MLXFastConstants.correctnessMaxAnchorContextTokens else {
            throw MLXFastError.invalidInput(
                "\(testCase.name).context_tokens has \(testCase.contextTokens.count) tokens; maximum is \(MLXFastConstants.correctnessMaxAnchorContextTokens)"
            )
        }
        try validateTokens(testCase.contextTokens, field: "\(testCase.name).context_tokens")
        try validateTokens([testCase.expectedToken], field: "\(testCase.name).expected_token")
        if let acceptedTokens = testCase.acceptedTokens {
            guard !acceptedTokens.isEmpty else {
                throw MLXFastError.invalidInput("\(testCase.name).accepted_tokens must not be empty when present")
            }
            try validateTokens(acceptedTokens, field: "\(testCase.name).accepted_tokens")
        }
        if let maxExpectedRank = testCase.maxExpectedRank {
            guard maxExpectedRank > 0, maxExpectedRank <= MLXFastConstants.correctnessTopLogits else {
                throw MLXFastError.invalidInput(
                    "\(testCase.name).max_expected_rank must be in 1...\(MLXFastConstants.correctnessTopLogits)"
                )
            }
        }
        if let maxTopLogitDelta = testCase.maxTopLogitDelta {
            guard maxTopLogitDelta.isFinite, maxTopLogitDelta >= 0 else {
                throw MLXFastError.invalidInput("\(testCase.name).max_top_logit_delta must be finite and non-negative")
            }
            guard testCase.maxExpectedRank != nil else {
                throw MLXFastError.invalidInput(
                    "\(testCase.name).max_top_logit_delta requires max_expected_rank"
                )
            }
        }
    }
}

private func validateUniqueLayeredCaseNames(
    baseCases: [GoldenCase],
    gates: GoldenCorrectnessGates
) throws {
    var names = Set<String>()
    for testCase in baseCases {
        names.insert(testCase.name)
    }
    for name in gates.anchorCases.map(\.name) + gates.freeRunCases.map(\.name) + gates.behaviorCases.map(\.name) {
        guard names.insert(name).inserted else {
            throw MLXFastError.invalidInput("duplicate layered correctness case name \(name)")
        }
    }
}

private func validateGoldenFreeRunCases(
    _ cases: [GoldenFreeRunCase],
    requiredPromptTokens: Int
) throws {
    var names = Set<String>()
    for testCase in cases {
        try validateCaseName(testCase.name, field: "correctness free-run case name")
        guard names.insert(testCase.name).inserted else {
            throw MLXFastError.invalidInput("duplicate correctness free-run case name \(testCase.name)")
        }
        guard testCase.promptTokens.count == requiredPromptTokens else {
            throw MLXFastError.invalidInput(
                "\(testCase.name).prompt_tokens has \(testCase.promptTokens.count) tokens; need exactly \(requiredPromptTokens)"
            )
        }
        guard !testCase.expectedTokens.isEmpty else {
            throw MLXFastError.invalidInput("\(testCase.name).expected_tokens must not be empty")
        }
        guard testCase.expectedTokens.count <= MLXFastConstants.correctnessMaxFreeRunSteps else {
            throw MLXFastError.invalidInput(
                "\(testCase.name).expected_tokens has \(testCase.expectedTokens.count) tokens; maximum is \(MLXFastConstants.correctnessMaxFreeRunSteps)"
            )
        }
        if let exactPrefixTokens = testCase.exactPrefixTokens {
            guard exactPrefixTokens > 0, exactPrefixTokens <= testCase.expectedTokens.count else {
                throw MLXFastError.invalidInput(
                    "\(testCase.name).exact_prefix_tokens must be in 1...\(testCase.expectedTokens.count)"
                )
            }
        }
        try validateTokens(testCase.promptTokens, field: "\(testCase.name).prompt_tokens")
        try validateTokens(testCase.expectedTokens, field: "\(testCase.name).expected_tokens")
    }
}

private func validateGoldenBehaviorCases(
    _ cases: [GoldenBehaviorCase],
    requiredPromptTokens _: Int
) throws {
    var names = Set<String>()
    for testCase in cases {
        try validateCaseName(testCase.name, field: "correctness behavior case name")
        guard names.insert(testCase.name).inserted else {
            throw MLXFastError.invalidInput("duplicate correctness behavior case name \(testCase.name)")
        }
        guard !testCase.promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("\(testCase.name).prompt_tokens must not be empty")
        }
        guard testCase.promptTokens.count <= MLXFastConstants.correctnessMaxBehaviorPromptTokens else {
            throw MLXFastError.invalidInput(
                "\(testCase.name).prompt_tokens has \(testCase.promptTokens.count) tokens; maximum is \(MLXFastConstants.correctnessMaxBehaviorPromptTokens)"
            )
        }
        guard testCase.maxNewTokens > 0,
              testCase.maxNewTokens <= MLXFastConstants.correctnessMaxBehaviorSteps
        else {
            throw MLXFastError.invalidInput(
                "\(testCase.name).max_new_tokens must be in 1...\(MLXFastConstants.correctnessMaxBehaviorSteps)"
            )
        }
        guard !testCase.acceptedTokenSequences.isEmpty else {
            throw MLXFastError.invalidInput("\(testCase.name).accepted_token_sequences must not be empty")
        }
        for (index, sequence) in testCase.acceptedTokenSequences.enumerated() {
            guard !sequence.isEmpty else {
                throw MLXFastError.invalidInput(
                    "\(testCase.name).accepted_token_sequences[\(index)] must not be empty"
                )
            }
            guard sequence.count <= testCase.maxNewTokens else {
                throw MLXFastError.invalidInput(
                    "\(testCase.name).accepted_token_sequences[\(index)] has \(sequence.count) tokens; maximum is max_new_tokens \(testCase.maxNewTokens)"
                )
            }
            try validateTokens(sequence, field: "\(testCase.name).accepted_token_sequences[\(index)]")
        }
        try validateTokens(testCase.promptTokens, field: "\(testCase.name).prompt_tokens")
    }
}

private func validateCaseName(_ name: String, field: String) throws {
    let caseNameDescription = String(reflecting: name)
    let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmedName.isEmpty {
        throw MLXFastError.invalidInput("\(field) must not be empty")
    }
    if name != trimmedName {
        throw MLXFastError.invalidInput(
            "\(field) \(caseNameDescription) must not have leading or trailing whitespace"
        )
    }
    if name.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }) {
        throw MLXFastError.invalidInput(
            "\(field) \(caseNameDescription) must not contain control characters"
        )
    }
}

private func validateTokens(_ tokens: [Int], field: String) throws {
    for (index, token) in tokens.enumerated() {
        if token < 0 || token >= MLXFastConstants.vocabSize {
            throw MLXFastError.invalidInput(
                "\(field)[\(index)]=\(token) is outside Laguna vocab range 0..<\(MLXFastConstants.vocabSize)"
            )
        }
    }
}
