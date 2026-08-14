import Foundation

public enum BenchmarkScore {
    public static func speedup(
        baselineSecondsPerToken: Double,
        candidateSecondsPerToken: Double
    ) -> Double {
        guard baselineSecondsPerToken.isFinite,
              candidateSecondsPerToken.isFinite,
              baselineSecondsPerToken > 0,
              candidateSecondsPerToken > 0
        else {
            return 0
        }
        return baselineSecondsPerToken / candidateSecondsPerToken
    }

    public static func score(
        decodeSecondsPerToken: Double,
        prefillSecondsPerToken: Double,
        baselineDecodeSecondsPerToken: Double = MLXFastConstants.officialBaselineDecodeSecondsPerToken,
        baselinePrefillSecondsPerToken: Double = MLXFastConstants.officialBaselinePrefillSecondsPerToken,
        decodeWeight: Double = MLXFastConstants.scoreDecodeWeight,
        prefillWeight: Double = MLXFastConstants.scorePrefillWeight
    ) -> Double {
        let decodeSpeedup = speedup(
            baselineSecondsPerToken: baselineDecodeSecondsPerToken,
            candidateSecondsPerToken: decodeSecondsPerToken
        )
        let prefillSpeedup = speedup(
            baselineSecondsPerToken: baselinePrefillSecondsPerToken,
            candidateSecondsPerToken: prefillSecondsPerToken
        )
        guard decodeSpeedup > 0,
              prefillSpeedup > 0,
              decodeWeight.isFinite,
              prefillWeight.isFinite,
              decodeWeight >= 0,
              prefillWeight >= 0,
              decodeWeight + prefillWeight > 0
        else {
            return .nan
        }

        let totalWeight = decodeWeight + prefillWeight
        return pow(decodeSpeedup, decodeWeight / totalWeight)
            * pow(prefillSpeedup, prefillWeight / totalWeight)
    }

    public static func passesSpeedupFloors(
        decodeSpeedup: Double,
        prefillSpeedup: Double,
        decodeFloor: Double = MLXFastConstants.scoreDecodeSpeedupFloor,
        prefillFloor: Double = MLXFastConstants.scorePrefillSpeedupFloor
    ) -> Bool {
        guard decodeSpeedup.isFinite,
              prefillSpeedup.isFinite,
              decodeFloor.isFinite,
              prefillFloor.isFinite
        else {
            return false
        }
        return decodeSpeedup >= decodeFloor && prefillSpeedup >= prefillFloor
    }

    public static func speedupFloorFailureMessage(
        decodeSpeedup: Double,
        prefillSpeedup: Double,
        decodeFloor: Double = MLXFastConstants.scoreDecodeSpeedupFloor,
        prefillFloor: Double = MLXFastConstants.scorePrefillSpeedupFloor
    ) -> String {
        func format(_ value: Double) -> String {
            String(format: "%.6f", locale: Locale(identifier: "en_US_POSIX"), value)
        }
        return "performance floor failed: decode_speedup=\(format(decodeSpeedup)) "
            + "floor=\(format(decodeFloor)) "
            + "prefill_speedup=\(format(prefillSpeedup)) "
            + "floor=\(format(prefillFloor))"
    }

    public static func evaluateTimedRun(
        decodeSecondsPerToken: Double,
        prefillSecondsPerToken: Double,
        baselineDecodeSecondsPerToken: Double,
        baselinePrefillSecondsPerToken: Double
    ) -> TimedRunScoreEvaluation {
        let score = score(
            decodeSecondsPerToken: decodeSecondsPerToken,
            prefillSecondsPerToken: prefillSecondsPerToken,
            baselineDecodeSecondsPerToken: baselineDecodeSecondsPerToken,
            baselinePrefillSecondsPerToken: baselinePrefillSecondsPerToken
        )
        let decodeSpeedup = speedup(
            baselineSecondsPerToken: baselineDecodeSecondsPerToken,
            candidateSecondsPerToken: decodeSecondsPerToken
        )
        let prefillSpeedup = speedup(
            baselineSecondsPerToken: baselinePrefillSecondsPerToken,
            candidateSecondsPerToken: prefillSecondsPerToken
        )
        let prefillBand = AcceptanceBand.check(
            value: prefillSecondsPerToken,
            reference: baselinePrefillSecondsPerToken,
            upTolerance: MLXFastConstants.prefillBandUpTolerance,
            downTolerance: MLXFastConstants.prefillBandDownTolerance,
            label: "prefill"
        )
        let decodeBand = AcceptanceBand.check(
            value: decodeSecondsPerToken,
            reference: baselineDecodeSecondsPerToken,
            upTolerance: MLXFastConstants.decodeBandUpTolerance,
            downTolerance: MLXFastConstants.decodeBandDownTolerance,
            label: "decode"
        )
        return TimedRunScoreEvaluation(
            score: score,
            decodeSpeedup: decodeSpeedup,
            prefillSpeedup: prefillSpeedup,
            passesFloors: passesSpeedupFloors(
                decodeSpeedup: decodeSpeedup,
                prefillSpeedup: prefillSpeedup
            ),
            prefillBand: prefillBand,
            decodeBand: decodeBand
        )
    }
}

public struct TimedRunScoreEvaluation: Equatable {
    public let score: Double
    public let decodeSpeedup: Double
    public let prefillSpeedup: Double
    public let passesFloors: Bool
    public let prefillBand: AcceptanceBandResult
    public let decodeBand: AcceptanceBandResult

    public var hasFiniteScore: Bool {
        score.isFinite && score >= 0
    }

    public var passesAcceptanceBands: Bool {
        prefillBand.passed && decodeBand.passed
    }

    public func firstFailureReason() -> String? {
        if !hasFiniteScore {
            return "computed score was not finite"
        }
        if !passesFloors {
            return BenchmarkScore.speedupFloorFailureMessage(
                decodeSpeedup: decodeSpeedup,
                prefillSpeedup: prefillSpeedup
            )
        }
        if !passesAcceptanceBands {
            return "acceptance band failed: \(prefillBand.passed ? decodeBand.reason : prefillBand.reason)"
        }
        return nil
    }
}

public struct ScorePayload: Codable, Equatable {
    public let score: Double?
    public let passed: Bool
    public let metrics: ScoreMetrics

    enum CodingKeys: String, CodingKey {
        case score
        case passed
        case metrics
    }

    public init(score: Double?, passed: Bool, metrics: ScoreMetrics) {
        self.score = score
        self.passed = passed
        self.metrics = metrics
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.score = try container.decodeIfPresent(Double.self, forKey: .score)
        self.passed = try container.decode(Bool.self, forKey: .passed)
        self.metrics = try container.decode(ScoreMetrics.self, forKey: .metrics)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        if let score {
            try container.encode(score, forKey: .score)
        } else {
            try container.encodeNil(forKey: .score)
        }
        try container.encode(passed, forKey: .passed)
        try container.encode(metrics, forKey: .metrics)
    }
}

public struct ScoreMetrics: Codable, Equatable {
    public let peakRamGB: Double
    public let bandwidthGBPerToken: Double
    public let decodeSecondsPerToken: Double
    public let prefillSecondsPerToken: Double
    public let baselineDecodeSecondsPerToken: Double
    public let baselinePrefillSecondsPerToken: Double
    public let decodeSpeedup: Double
    public let prefillSpeedup: Double
    public let decodeSpeedupFloor: Double
    public let prefillSpeedupFloor: Double
    public let passedDecodeSpeedupFloor: Bool
    public let passedPrefillSpeedupFloor: Bool
    public let benchmarkWallSeconds: Double
    public let preflightSeconds: Double
    public let correctnessSeconds: Double
    public let timedBenchmarkSeconds: Double
    public let gpqaTTFTPassed: Bool
    public let gpqaTTFTPassCount: Int
    public let gpqaTTFTCaseCount: Int
    public let gpqaTTFTSeconds: Double
    public let gpqaTTFTP50Seconds: Double
    public let gpqaTTFTMaxSeconds: Double
    public let gpqaTTFTSource: String
    public let semanticGPQAPassed: Bool
    public let semanticGPQAPassCount: Int
    public let semanticGPQACaseCount: Int
    public let semanticGPQAModel: String
    public let processResidentMemoryGB: Double
    public let passedCorrectness: Bool
    public let numLayers: Int
    public let checkedSteps: Int
    public let caseCount: Int
    public let expertCacheHits: UInt64
    public let expertCacheMisses: UInt64
    public let expertCacheEvictions: UInt64
    public let expertBytesRead: UInt64
    public let expertReadSeconds: Double
    public let expertPeakCachedTensors: UInt64
    public let expertHitRate: Double
    public let firstFailingLayer: Int?
    public let firstFailingCase: String?
    public let firstFailingStep: Int?
    public let expectedToken: Int?
    public let actualToken: Int?
    public let maxAbsDiff: Double
    public let goldenHash: String
    public let bandwidthSource: String
    public let error: String
    public let commit: String
    public let timestamp: String
    public let harnessHash: String
    public let weightsHash: String
    public let weightsByteCount: Int
    public let weightsFileCount: Int
    public let runtime: String
    // True whenever this metrics payload still contains baseline-placeholder
    // values for the fields a gates-only or timing-only run did not itself
    // measure (see BenchmarkOptions.checkGates/skipTimedBenchmark). The ONLY
    // thing that clears this to false today is overlay-paired-timing.sh, when
    // it overlays measure-job's paired timing into the sealed gates score --
    // a defense-in-depth marker so a future regression there (or anywhere the
    // final score is assembled) has a structural signal to check, instead of
    // relying solely on the YAML wiring being correct.
    public let partialResult: Bool

    enum CodingKeys: String, CodingKey {
        case peakRamGB = "peak_ram_gb"
        case bandwidthGBPerToken = "bandwidth_gb_per_token"
        case decodeSecondsPerToken = "decode_seconds_per_token"
        case prefillSecondsPerToken = "prefill_seconds_per_token"
        case baselineDecodeSecondsPerToken = "baseline_decode_seconds_per_token"
        case baselinePrefillSecondsPerToken = "baseline_prefill_seconds_per_token"
        case decodeSpeedup = "decode_speedup"
        case prefillSpeedup = "prefill_speedup"
        case decodeSpeedupFloor = "decode_speedup_floor"
        case prefillSpeedupFloor = "prefill_speedup_floor"
        case passedDecodeSpeedupFloor = "passed_decode_speedup_floor"
        case passedPrefillSpeedupFloor = "passed_prefill_speedup_floor"
        case benchmarkWallSeconds = "benchmark_wall_seconds"
        case preflightSeconds = "preflight_seconds"
        case correctnessSeconds = "correctness_seconds"
        case timedBenchmarkSeconds = "timed_benchmark_seconds"
        case gpqaTTFTPassed = "gpqa_ttft_passed"
        case gpqaTTFTPassCount = "gpqa_ttft_pass_count"
        case gpqaTTFTCaseCount = "gpqa_ttft_case_count"
        case gpqaTTFTSeconds = "gpqa_ttft_seconds"
        case gpqaTTFTP50Seconds = "gpqa_ttft_p50_seconds"
        case gpqaTTFTMaxSeconds = "gpqa_ttft_max_seconds"
        case gpqaTTFTSource = "gpqa_ttft_source"
        case semanticGPQAPassed = "semantic_gpqa_passed"
        case semanticGPQAPassCount = "semantic_gpqa_pass_count"
        case semanticGPQACaseCount = "semantic_gpqa_case_count"
        case semanticGPQAModel = "semantic_gpqa_model"
        case processResidentMemoryGB = "process_resident_memory_gb"
        case passedCorrectness = "passed_correctness"
        case numLayers = "num_layers"
        case checkedSteps = "checked_steps"
        case caseCount = "case_count"
        case expertCacheHits = "expert_cache_hits"
        case expertCacheMisses = "expert_cache_misses"
        case expertCacheEvictions = "expert_cache_evictions"
        case expertBytesRead = "expert_bytes_read"
        case expertReadSeconds = "expert_read_seconds"
        case expertPeakCachedTensors = "expert_peak_cached_tensors"
        case expertHitRate = "expert_hit_rate"
        case firstFailingLayer = "first_failing_layer"
        case firstFailingCase = "first_failing_case"
        case firstFailingStep = "first_failing_step"
        case expectedToken = "expected_token"
        case actualToken = "actual_token"
        case maxAbsDiff = "max_abs_diff"
        case goldenHash = "golden_hash"
        case bandwidthSource = "bandwidth_source"
        case error
        case commit
        case timestamp
        case harnessHash = "harness_hash"
        case weightsHash = "weights_hash"
        case weightsByteCount = "weights_byte_count"
        case weightsFileCount = "weights_file_count"
        case runtime
        case partialResult = "partial_result"
    }

    public init(
        peakRamGB: Double,
        bandwidthGBPerToken: Double,
        decodeSecondsPerToken: Double,
        prefillSecondsPerToken: Double,
        baselineDecodeSecondsPerToken: Double = MLXFastConstants.officialBaselineDecodeSecondsPerToken,
        baselinePrefillSecondsPerToken: Double = MLXFastConstants.officialBaselinePrefillSecondsPerToken,
        decodeSpeedup: Double? = nil,
        prefillSpeedup: Double? = nil,
        decodeSpeedupFloor: Double = MLXFastConstants.scoreDecodeSpeedupFloor,
        prefillSpeedupFloor: Double = MLXFastConstants.scorePrefillSpeedupFloor,
        passedDecodeSpeedupFloor: Bool? = nil,
        passedPrefillSpeedupFloor: Bool? = nil,
        benchmarkWallSeconds: Double = 0,
        preflightSeconds: Double = 0,
        correctnessSeconds: Double = 0,
        timedBenchmarkSeconds: Double = 0,
        gpqaTTFTPassed: Bool = false,
        gpqaTTFTPassCount: Int = 0,
        gpqaTTFTCaseCount: Int = 0,
        gpqaTTFTSeconds: Double = 0,
        gpqaTTFTP50Seconds: Double = 0,
        gpqaTTFTMaxSeconds: Double = 0,
        gpqaTTFTSource: String = "",
        semanticGPQAPassed: Bool = false,
        semanticGPQAPassCount: Int = 0,
        semanticGPQACaseCount: Int = 0,
        semanticGPQAModel: String = "",
        processResidentMemoryGB: Double = 0,
        passedCorrectness: Bool,
        numLayers: Int,
        checkedSteps: Int,
        caseCount: Int,
        expertCacheHits: UInt64 = 0,
        expertCacheMisses: UInt64 = 0,
        expertCacheEvictions: UInt64 = 0,
        expertBytesRead: UInt64 = 0,
        expertReadSeconds: Double = 0,
        expertPeakCachedTensors: UInt64 = 0,
        expertHitRate: Double = 0,
        firstFailingLayer: Int?,
        firstFailingCase: String?,
        firstFailingStep: Int?,
        expectedToken: Int?,
        actualToken: Int?,
        maxAbsDiff: Double,
        goldenHash: String,
        bandwidthSource: String,
        error: String,
        commit: String,
        timestamp: String,
        harnessHash: String,
        weightsHash: String = "",
        weightsByteCount: Int = 0,
        weightsFileCount: Int = 0,
        runtime: String,
        partialResult: Bool = false
    ) {
        self.peakRamGB = peakRamGB
        self.bandwidthGBPerToken = bandwidthGBPerToken
        self.decodeSecondsPerToken = decodeSecondsPerToken
        self.prefillSecondsPerToken = prefillSecondsPerToken
        self.baselineDecodeSecondsPerToken = baselineDecodeSecondsPerToken
        self.baselinePrefillSecondsPerToken = baselinePrefillSecondsPerToken
        self.decodeSpeedup = decodeSpeedup ?? BenchmarkScore.speedup(
            baselineSecondsPerToken: baselineDecodeSecondsPerToken,
            candidateSecondsPerToken: decodeSecondsPerToken
        )
        self.prefillSpeedup = prefillSpeedup ?? BenchmarkScore.speedup(
            baselineSecondsPerToken: baselinePrefillSecondsPerToken,
            candidateSecondsPerToken: prefillSecondsPerToken
        )
        self.decodeSpeedupFloor = decodeSpeedupFloor
        self.prefillSpeedupFloor = prefillSpeedupFloor
        self.passedDecodeSpeedupFloor = passedDecodeSpeedupFloor ?? (self.decodeSpeedup >= decodeSpeedupFloor)
        self.passedPrefillSpeedupFloor = passedPrefillSpeedupFloor ?? (self.prefillSpeedup >= prefillSpeedupFloor)
        self.benchmarkWallSeconds = benchmarkWallSeconds
        self.preflightSeconds = preflightSeconds
        self.correctnessSeconds = correctnessSeconds
        self.timedBenchmarkSeconds = timedBenchmarkSeconds
        self.gpqaTTFTPassed = gpqaTTFTPassed
        self.gpqaTTFTPassCount = gpqaTTFTPassCount
        self.gpqaTTFTCaseCount = gpqaTTFTCaseCount
        self.gpqaTTFTSeconds = gpqaTTFTSeconds
        self.gpqaTTFTP50Seconds = gpqaTTFTP50Seconds
        self.gpqaTTFTMaxSeconds = gpqaTTFTMaxSeconds
        self.gpqaTTFTSource = gpqaTTFTSource
        self.semanticGPQAPassed = semanticGPQAPassed
        self.semanticGPQAPassCount = semanticGPQAPassCount
        self.semanticGPQACaseCount = semanticGPQACaseCount
        self.semanticGPQAModel = semanticGPQAModel
        self.processResidentMemoryGB = processResidentMemoryGB
        self.passedCorrectness = passedCorrectness
        self.numLayers = numLayers
        self.checkedSteps = checkedSteps
        self.caseCount = caseCount
        self.expertCacheHits = expertCacheHits
        self.expertCacheMisses = expertCacheMisses
        self.expertCacheEvictions = expertCacheEvictions
        self.expertBytesRead = expertBytesRead
        self.expertReadSeconds = expertReadSeconds
        self.expertPeakCachedTensors = expertPeakCachedTensors
        self.expertHitRate = expertHitRate
        self.firstFailingLayer = firstFailingLayer
        self.firstFailingCase = firstFailingCase
        self.firstFailingStep = firstFailingStep
        self.expectedToken = expectedToken
        self.actualToken = actualToken
        self.maxAbsDiff = maxAbsDiff
        self.goldenHash = goldenHash
        self.bandwidthSource = bandwidthSource
        self.error = error
        self.commit = commit
        self.timestamp = timestamp
        self.harnessHash = harnessHash
        self.weightsHash = weightsHash
        self.weightsByteCount = weightsByteCount
        self.weightsFileCount = weightsFileCount
        self.runtime = runtime
        self.partialResult = partialResult
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.peakRamGB = try container.decode(Double.self, forKey: .peakRamGB)
        self.bandwidthGBPerToken = try container.decode(Double.self, forKey: .bandwidthGBPerToken)
        self.decodeSecondsPerToken = try container.decode(Double.self, forKey: .decodeSecondsPerToken)
        self.prefillSecondsPerToken = try container.decode(Double.self, forKey: .prefillSecondsPerToken)
        self.baselineDecodeSecondsPerToken = try container.decodeIfPresent(
            Double.self,
            forKey: .baselineDecodeSecondsPerToken
        ) ?? MLXFastConstants.officialBaselineDecodeSecondsPerToken
        self.baselinePrefillSecondsPerToken = try container.decodeIfPresent(
            Double.self,
            forKey: .baselinePrefillSecondsPerToken
        ) ?? MLXFastConstants.officialBaselinePrefillSecondsPerToken
        self.decodeSpeedup = try container.decodeIfPresent(Double.self, forKey: .decodeSpeedup)
            ?? BenchmarkScore.speedup(
                baselineSecondsPerToken: baselineDecodeSecondsPerToken,
                candidateSecondsPerToken: decodeSecondsPerToken
            )
        self.prefillSpeedup = try container.decodeIfPresent(Double.self, forKey: .prefillSpeedup)
            ?? BenchmarkScore.speedup(
                baselineSecondsPerToken: baselinePrefillSecondsPerToken,
                candidateSecondsPerToken: prefillSecondsPerToken
            )
        self.decodeSpeedupFloor = try container.decodeIfPresent(
            Double.self,
            forKey: .decodeSpeedupFloor
        ) ?? MLXFastConstants.scoreDecodeSpeedupFloor
        self.prefillSpeedupFloor = try container.decodeIfPresent(
            Double.self,
            forKey: .prefillSpeedupFloor
        ) ?? MLXFastConstants.scorePrefillSpeedupFloor
        self.passedDecodeSpeedupFloor = try container.decodeIfPresent(
            Bool.self,
            forKey: .passedDecodeSpeedupFloor
        ) ?? (self.decodeSpeedup >= self.decodeSpeedupFloor)
        self.passedPrefillSpeedupFloor = try container.decodeIfPresent(
            Bool.self,
            forKey: .passedPrefillSpeedupFloor
        ) ?? (self.prefillSpeedup >= self.prefillSpeedupFloor)
        self.benchmarkWallSeconds = try container.decodeIfPresent(Double.self, forKey: .benchmarkWallSeconds) ?? 0
        self.preflightSeconds = try container.decodeIfPresent(Double.self, forKey: .preflightSeconds) ?? 0
        self.correctnessSeconds = try container.decodeIfPresent(Double.self, forKey: .correctnessSeconds) ?? 0
        self.timedBenchmarkSeconds = try container.decodeIfPresent(Double.self, forKey: .timedBenchmarkSeconds) ?? 0
        self.gpqaTTFTPassed = try container.decodeIfPresent(Bool.self, forKey: .gpqaTTFTPassed) ?? false
        self.gpqaTTFTPassCount = try container.decodeIfPresent(Int.self, forKey: .gpqaTTFTPassCount) ?? 0
        self.gpqaTTFTCaseCount = try container.decodeIfPresent(Int.self, forKey: .gpqaTTFTCaseCount) ?? 0
        self.gpqaTTFTSeconds = try container.decodeIfPresent(Double.self, forKey: .gpqaTTFTSeconds) ?? 0
        self.gpqaTTFTP50Seconds = try container.decodeIfPresent(Double.self, forKey: .gpqaTTFTP50Seconds) ?? 0
        self.gpqaTTFTMaxSeconds = try container.decodeIfPresent(Double.self, forKey: .gpqaTTFTMaxSeconds) ?? 0
        self.gpqaTTFTSource = try container.decodeIfPresent(String.self, forKey: .gpqaTTFTSource) ?? ""
        self.semanticGPQAPassed = try container.decodeIfPresent(Bool.self, forKey: .semanticGPQAPassed) ?? false
        self.semanticGPQAPassCount = try container.decodeIfPresent(Int.self, forKey: .semanticGPQAPassCount) ?? 0
        self.semanticGPQACaseCount = try container.decodeIfPresent(Int.self, forKey: .semanticGPQACaseCount) ?? 0
        self.semanticGPQAModel = try container.decodeIfPresent(String.self, forKey: .semanticGPQAModel) ?? ""
        self.processResidentMemoryGB = try container.decodeIfPresent(Double.self, forKey: .processResidentMemoryGB) ?? 0
        self.passedCorrectness = try container.decode(Bool.self, forKey: .passedCorrectness)
        self.numLayers = try container.decode(Int.self, forKey: .numLayers)
        self.checkedSteps = try container.decode(Int.self, forKey: .checkedSteps)
        self.caseCount = try container.decode(Int.self, forKey: .caseCount)
        self.expertCacheHits = try container.decode(UInt64.self, forKey: .expertCacheHits)
        self.expertCacheMisses = try container.decode(UInt64.self, forKey: .expertCacheMisses)
        self.expertCacheEvictions = try container.decode(UInt64.self, forKey: .expertCacheEvictions)
        self.expertBytesRead = try container.decode(UInt64.self, forKey: .expertBytesRead)
        self.expertReadSeconds = try container.decode(Double.self, forKey: .expertReadSeconds)
        self.expertPeakCachedTensors = try container.decode(UInt64.self, forKey: .expertPeakCachedTensors)
        self.expertHitRate = try container.decode(Double.self, forKey: .expertHitRate)
        self.firstFailingLayer = try container.decodeIfPresent(Int.self, forKey: .firstFailingLayer)
        self.firstFailingCase = try container.decodeIfPresent(String.self, forKey: .firstFailingCase)
        self.firstFailingStep = try container.decodeIfPresent(Int.self, forKey: .firstFailingStep)
        self.expectedToken = try container.decodeIfPresent(Int.self, forKey: .expectedToken)
        self.actualToken = try container.decodeIfPresent(Int.self, forKey: .actualToken)
        self.maxAbsDiff = try container.decode(Double.self, forKey: .maxAbsDiff)
        self.goldenHash = try container.decode(String.self, forKey: .goldenHash)
        self.bandwidthSource = try container.decode(String.self, forKey: .bandwidthSource)
        self.error = try container.decode(String.self, forKey: .error)
        self.commit = try container.decode(String.self, forKey: .commit)
        self.timestamp = try container.decode(String.self, forKey: .timestamp)
        self.harnessHash = try container.decode(String.self, forKey: .harnessHash)
        self.weightsHash = try container.decodeIfPresent(String.self, forKey: .weightsHash) ?? ""
        self.weightsByteCount = try container.decodeIfPresent(Int.self, forKey: .weightsByteCount) ?? 0
        self.weightsFileCount = try container.decodeIfPresent(Int.self, forKey: .weightsFileCount) ?? 0
        self.runtime = try container.decode(String.self, forKey: .runtime)
        self.partialResult = try container.decodeIfPresent(Bool.self, forKey: .partialResult) ?? false
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(peakRamGB, forKey: .peakRamGB)
        try container.encode(bandwidthGBPerToken, forKey: .bandwidthGBPerToken)
        try container.encode(decodeSecondsPerToken, forKey: .decodeSecondsPerToken)
        try container.encode(prefillSecondsPerToken, forKey: .prefillSecondsPerToken)
        try container.encode(baselineDecodeSecondsPerToken, forKey: .baselineDecodeSecondsPerToken)
        try container.encode(baselinePrefillSecondsPerToken, forKey: .baselinePrefillSecondsPerToken)
        try container.encode(decodeSpeedup, forKey: .decodeSpeedup)
        try container.encode(prefillSpeedup, forKey: .prefillSpeedup)
        try container.encode(decodeSpeedupFloor, forKey: .decodeSpeedupFloor)
        try container.encode(prefillSpeedupFloor, forKey: .prefillSpeedupFloor)
        try container.encode(passedDecodeSpeedupFloor, forKey: .passedDecodeSpeedupFloor)
        try container.encode(passedPrefillSpeedupFloor, forKey: .passedPrefillSpeedupFloor)
        try container.encode(benchmarkWallSeconds, forKey: .benchmarkWallSeconds)
        try container.encode(preflightSeconds, forKey: .preflightSeconds)
        try container.encode(correctnessSeconds, forKey: .correctnessSeconds)
        try container.encode(timedBenchmarkSeconds, forKey: .timedBenchmarkSeconds)
        try container.encode(gpqaTTFTPassed, forKey: .gpqaTTFTPassed)
        try container.encode(gpqaTTFTPassCount, forKey: .gpqaTTFTPassCount)
        try container.encode(gpqaTTFTCaseCount, forKey: .gpqaTTFTCaseCount)
        try container.encode(gpqaTTFTSeconds, forKey: .gpqaTTFTSeconds)
        try container.encode(gpqaTTFTP50Seconds, forKey: .gpqaTTFTP50Seconds)
        try container.encode(gpqaTTFTMaxSeconds, forKey: .gpqaTTFTMaxSeconds)
        try container.encode(gpqaTTFTSource, forKey: .gpqaTTFTSource)
        try container.encode(semanticGPQAPassed, forKey: .semanticGPQAPassed)
        try container.encode(semanticGPQAPassCount, forKey: .semanticGPQAPassCount)
        try container.encode(semanticGPQACaseCount, forKey: .semanticGPQACaseCount)
        try container.encode(semanticGPQAModel, forKey: .semanticGPQAModel)
        try container.encode(processResidentMemoryGB, forKey: .processResidentMemoryGB)
        try container.encode(passedCorrectness, forKey: .passedCorrectness)
        try container.encode(numLayers, forKey: .numLayers)
        try container.encode(checkedSteps, forKey: .checkedSteps)
        try container.encode(caseCount, forKey: .caseCount)
        try container.encode(expertCacheHits, forKey: .expertCacheHits)
        try container.encode(expertCacheMisses, forKey: .expertCacheMisses)
        try container.encode(expertCacheEvictions, forKey: .expertCacheEvictions)
        try container.encode(expertBytesRead, forKey: .expertBytesRead)
        try container.encode(expertReadSeconds, forKey: .expertReadSeconds)
        try container.encode(expertPeakCachedTensors, forKey: .expertPeakCachedTensors)
        try container.encode(expertHitRate, forKey: .expertHitRate)
        if let firstFailingLayer {
            try container.encode(firstFailingLayer, forKey: .firstFailingLayer)
        } else {
            try container.encodeNil(forKey: .firstFailingLayer)
        }
        if let firstFailingCase {
            try container.encode(firstFailingCase, forKey: .firstFailingCase)
        } else {
            try container.encodeNil(forKey: .firstFailingCase)
        }
        if let firstFailingStep {
            try container.encode(firstFailingStep, forKey: .firstFailingStep)
        } else {
            try container.encodeNil(forKey: .firstFailingStep)
        }
        if let expectedToken {
            try container.encode(expectedToken, forKey: .expectedToken)
        } else {
            try container.encodeNil(forKey: .expectedToken)
        }
        if let actualToken {
            try container.encode(actualToken, forKey: .actualToken)
        } else {
            try container.encodeNil(forKey: .actualToken)
        }
        try container.encode(maxAbsDiff, forKey: .maxAbsDiff)
        try container.encode(goldenHash, forKey: .goldenHash)
        try container.encode(bandwidthSource, forKey: .bandwidthSource)
        try container.encode(error, forKey: .error)
        try container.encode(commit, forKey: .commit)
        try container.encode(timestamp, forKey: .timestamp)
        try container.encode(harnessHash, forKey: .harnessHash)
        try container.encode(weightsHash, forKey: .weightsHash)
        try container.encode(weightsByteCount, forKey: .weightsByteCount)
        try container.encode(weightsFileCount, forKey: .weightsFileCount)
        try container.encode(runtime, forKey: .runtime)
        try container.encode(partialResult, forKey: .partialResult)
    }
}

extension ScorePayload {
    public static func failed(
        error: String,
        commit: String = "",
        harnessHash: String = ""
    ) -> ScorePayload {
        ScorePayload(
            score: nil,
            passed: false,
            metrics: ScoreMetrics(
                peakRamGB: 0,
                bandwidthGBPerToken: 0,
                decodeSecondsPerToken: 0,
                prefillSecondsPerToken: 0,
                passedCorrectness: false,
                numLayers: MLXFastConstants.numHiddenLayers,
                checkedSteps: 0,
                caseCount: 0,
                firstFailingLayer: nil,
                firstFailingCase: nil,
                firstFailingStep: nil,
                expectedToken: nil,
                actualToken: nil,
                maxAbsDiff: 0,
                goldenHash: "",
                bandwidthSource: "",
                error: error,
                commit: commit,
                timestamp: ISO8601DateFormatter().string(from: Date()),
                harnessHash: harnessHash,
                runtime: "swift"
            )
        )
    }
}

/// Monotone significant-figure rounding. Non-decreasing over the reals, so any
/// ordering the scoring pipeline relies on (e.g. wall >= timed, ttft_max >= p50)
/// is preserved when both sides are rounded to the same number of figures, and a
/// positive value never rounds to zero (relative, not absolute, grid). Rounds via
/// a formatted round-trip so the result is the clean nearest double to the
/// N-significant-figure decimal (e.g. 0.38, not 0.38000000000000006) -- otherwise
/// the low-order float noise would both survive into the JSON and hand back the
/// precision this is meant to remove.
public func roundedToSignificantFigures(_ value: Double, _ figures: Int) -> Double {
    guard value.isFinite, value != 0, figures > 0 else {
        return value
    }
    let formatted = String(format: "%.\(figures)g", locale: Locale(identifier: "en_US_POSIX"), value)
    return Double(formatted) ?? value
}

extension ScoreMetrics {
    /// Returns a copy with the diagnostic (non-ranking) real-valued fields rounded
    /// to `figures` significant figures, to shrink the timing/memory covert channel
    /// that submitted model code can drive. Ranking- and floor-critical fields
    /// (decode/prefill seconds-per-token, speedups, floors, baselines) are left
    /// untouched so scoring, the speedup-floor gate, and the paired-timing overlay
    /// are all bit-unaffected. Ordering pairs the validators assert are re-clamped
    /// after rounding as belt-and-suspenders.
    public func withCoarsenedPublicDiagnostics(
        figures: Int = MLXFastConstants.publicDiagnosticSignificantFigures
    ) -> ScoreMetrics {
        func r(_ value: Double) -> Double { roundedToSignificantFigures(value, figures) }

        let roundedTimed = r(timedBenchmarkSeconds)
        let roundedWall = max(r(benchmarkWallSeconds), roundedTimed)
        let roundedP50 = r(gpqaTTFTP50Seconds)
        let roundedTTFTMax = max(r(gpqaTTFTMaxSeconds), roundedP50)

        return ScoreMetrics(
            peakRamGB: r(peakRamGB),
            bandwidthGBPerToken: r(bandwidthGBPerToken),
            decodeSecondsPerToken: decodeSecondsPerToken,
            prefillSecondsPerToken: prefillSecondsPerToken,
            baselineDecodeSecondsPerToken: baselineDecodeSecondsPerToken,
            baselinePrefillSecondsPerToken: baselinePrefillSecondsPerToken,
            decodeSpeedup: decodeSpeedup,
            prefillSpeedup: prefillSpeedup,
            decodeSpeedupFloor: decodeSpeedupFloor,
            prefillSpeedupFloor: prefillSpeedupFloor,
            passedDecodeSpeedupFloor: passedDecodeSpeedupFloor,
            passedPrefillSpeedupFloor: passedPrefillSpeedupFloor,
            benchmarkWallSeconds: roundedWall,
            preflightSeconds: r(preflightSeconds),
            correctnessSeconds: r(correctnessSeconds),
            timedBenchmarkSeconds: roundedTimed,
            gpqaTTFTPassed: gpqaTTFTPassed,
            gpqaTTFTPassCount: gpqaTTFTPassCount,
            gpqaTTFTCaseCount: gpqaTTFTCaseCount,
            gpqaTTFTSeconds: r(gpqaTTFTSeconds),
            gpqaTTFTP50Seconds: roundedP50,
            gpqaTTFTMaxSeconds: roundedTTFTMax,
            gpqaTTFTSource: gpqaTTFTSource,
            semanticGPQAPassed: semanticGPQAPassed,
            semanticGPQAPassCount: semanticGPQAPassCount,
            semanticGPQACaseCount: semanticGPQACaseCount,
            semanticGPQAModel: semanticGPQAModel,
            processResidentMemoryGB: r(processResidentMemoryGB),
            passedCorrectness: passedCorrectness,
            numLayers: numLayers,
            checkedSteps: checkedSteps,
            caseCount: caseCount,
            expertCacheHits: expertCacheHits,
            expertCacheMisses: expertCacheMisses,
            expertCacheEvictions: expertCacheEvictions,
            expertBytesRead: expertBytesRead,
            expertReadSeconds: r(expertReadSeconds),
            expertPeakCachedTensors: expertPeakCachedTensors,
            expertHitRate: r(expertHitRate),
            firstFailingLayer: firstFailingLayer,
            firstFailingCase: firstFailingCase,
            firstFailingStep: firstFailingStep,
            expectedToken: expectedToken,
            actualToken: actualToken,
            maxAbsDiff: r(maxAbsDiff),
            goldenHash: goldenHash,
            bandwidthSource: bandwidthSource,
            error: error,
            commit: commit,
            timestamp: timestamp,
            harnessHash: harnessHash,
            weightsHash: weightsHash,
            weightsByteCount: weightsByteCount,
            weightsFileCount: weightsFileCount,
            runtime: runtime,
            partialResult: partialResult
        )
    }
}

public func writeScorePayload(_ payload: ScorePayload, to path: String) throws {
    let url = URL(fileURLWithPath: path)
    let parent = url.deletingLastPathComponent()
    if !parent.path.isEmpty {
        try FileManager.default.createDirectory(
            at: parent,
            withIntermediateDirectories: true
        )
    }

    // Coarsen diagnostic analog fields before the payload is written/published so
    // submitted model code cannot use them as a fine-grained timing/memory covert
    // channel. Ranking fields are unchanged, so scoring and downstream validation
    // are unaffected.
    let publishedPayload = ScorePayload(
        score: payload.score,
        passed: payload.passed,
        metrics: payload.metrics.withCoarsenedPublicDiagnostics()
    )

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    let data = try encoder.encode(publishedPayload)
    try data.write(to: url)
}
