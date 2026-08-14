import Foundation
import Testing
@testable import MLXFastCore

@Test
func benchmarkScoreUsesWeightedBaselineSpeedups() {
    let score = BenchmarkScore.score(
        decodeSecondsPerToken: 1.5,
        prefillSecondsPerToken: 0.25,
        baselineDecodeSecondsPerToken: 3,
        baselinePrefillSecondsPerToken: 0.25,
        decodeWeight: 0.75,
        prefillWeight: 0.25
    )
    let decodeSpeedup = BenchmarkScore.speedup(
        baselineSecondsPerToken: 3,
        candidateSecondsPerToken: 1.5
    )
    let prefillSpeedup = BenchmarkScore.speedup(
        baselineSecondsPerToken: 0.25,
        candidateSecondsPerToken: 0.25
    )

    #expect(abs(decodeSpeedup - 2) < 1e-12)
    #expect(abs(prefillSpeedup - 1) < 1e-12)
    #expect(abs(score - pow(2, 0.75)) < 1e-12)
}

@Test
func benchmarkScoreChecksComponentFloors() {
    #expect(BenchmarkScore.passesSpeedupFloors(decodeSpeedup: 1.01, prefillSpeedup: 0.95))
    #expect(!BenchmarkScore.passesSpeedupFloors(decodeSpeedup: 0.94, prefillSpeedup: 1.20))
    #expect(!BenchmarkScore.passesSpeedupFloors(decodeSpeedup: 1.20, prefillSpeedup: 0.94))
}

@Test
func evaluateTimedRunMatchesSeparateScoreSpeedupAndBandChecks() {
    let evaluation = BenchmarkScore.evaluateTimedRun(
        decodeSecondsPerToken: 0.10,
        prefillSecondsPerToken: 0.01,
        baselineDecodeSecondsPerToken: 0.10,
        baselinePrefillSecondsPerToken: 0.01
    )

    #expect(evaluation.firstFailureReason() == nil)
    #expect(evaluation.hasFiniteScore)
    #expect(evaluation.passesFloors)
    #expect(evaluation.passesAcceptanceBands)
    #expect(abs(evaluation.score - 1) < 1e-12)
    #expect(abs(evaluation.decodeSpeedup - 1) < 1e-12)
    #expect(abs(evaluation.prefillSpeedup - 1) < 1e-12)

    let failingFloor = BenchmarkScore.evaluateTimedRun(
        decodeSecondsPerToken: 0.20,
        prefillSecondsPerToken: 0.01,
        baselineDecodeSecondsPerToken: 0.10,
        baselinePrefillSecondsPerToken: 0.01
    )
    #expect(failingFloor.passesFloors == false)
    #expect(failingFloor.firstFailureReason()?.hasPrefix("performance floor failed:") == true)
    #expect(failingFloor.hasFiniteScore)

    let nonFinite = BenchmarkScore.evaluateTimedRun(
        decodeSecondsPerToken: 0,
        prefillSecondsPerToken: 0.01,
        baselineDecodeSecondsPerToken: 0.10,
        baselinePrefillSecondsPerToken: 0.01
    )
    #expect(nonFinite.hasFiniteScore == false)
    #expect(nonFinite.firstFailureReason() == "computed score was not finite")

    // Decode fast enough that the speedup (1.25) clears the 0.95 floor, but the
    // value lands below the decode acceptance band's -5% edge (0.10 * 0.95 =
    // 0.095), so the band -- not the floor -- is the first failure reported.
    let outsideBand = BenchmarkScore.evaluateTimedRun(
        decodeSecondsPerToken: 0.08,
        prefillSecondsPerToken: 0.01,
        baselineDecodeSecondsPerToken: 0.10,
        baselinePrefillSecondsPerToken: 0.01
    )
    #expect(outsideBand.hasFiniteScore)
    #expect(outsideBand.passesFloors)
    #expect(outsideBand.prefillBand.passed)
    #expect(outsideBand.decodeBand.passed == false)
    #expect(outsideBand.passesAcceptanceBands == false)
    #expect(outsideBand.firstFailureReason()?.hasPrefix("acceptance band failed: decode") == true)
    #expect(outsideBand.firstFailureReason()?.contains("below") == true)
}

@Test
func writeScorePayloadEmitsDarkbloomShape() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("score.json")

    try writeScorePayload(
        .failed(error: "runtime unavailable"),
        to: path.path
    )

    let data = try Data(contentsOf: path)
    let raw = String(decoding: data, as: UTF8.self)
    let decoded = try JSONDecoder().decode(ScorePayload.self, from: data)

    #expect(raw.contains("\"score\" : null"))
    #expect(decoded.score == nil)
    #expect(decoded.passed == false)
    #expect(decoded.metrics.passedCorrectness == false)
    #expect(decoded.metrics.checkedSteps == 0)
    #expect(decoded.metrics.caseCount == 0)
    #expect(decoded.metrics.expertCacheHits == 0)
    #expect(decoded.metrics.expertCacheMisses == 0)
    #expect(decoded.metrics.expertCacheEvictions == 0)
    #expect(decoded.metrics.expertBytesRead == 0)
    #expect(decoded.metrics.expertReadSeconds == 0)
    #expect(decoded.metrics.expertPeakCachedTensors == 0)
    #expect(decoded.metrics.expertHitRate == 0)
    #expect(decoded.metrics.weightsHash == "")
    #expect(decoded.metrics.weightsByteCount == 0)
    #expect(decoded.metrics.weightsFileCount == 0)
    #expect(decoded.metrics.partialResult == false)
    #expect(decoded.metrics.baselineDecodeSecondsPerToken == MLXFastConstants.officialBaselineDecodeSecondsPerToken)
    #expect(decoded.metrics.baselinePrefillSecondsPerToken == MLXFastConstants.officialBaselinePrefillSecondsPerToken)
    #expect(decoded.metrics.decodeSpeedup == 0)
    #expect(decoded.metrics.prefillSpeedup == 0)
    #expect(decoded.metrics.decodeSpeedupFloor == MLXFastConstants.scoreDecodeSpeedupFloor)
    #expect(decoded.metrics.prefillSpeedupFloor == MLXFastConstants.scorePrefillSpeedupFloor)
    #expect(decoded.metrics.passedDecodeSpeedupFloor == false)
    #expect(decoded.metrics.passedPrefillSpeedupFloor == false)
    #expect(decoded.metrics.benchmarkWallSeconds == 0)
    #expect(decoded.metrics.preflightSeconds == 0)
    #expect(decoded.metrics.correctnessSeconds == 0)
    #expect(decoded.metrics.timedBenchmarkSeconds == 0)
    #expect(decoded.metrics.gpqaTTFTPassed == false)
    #expect(decoded.metrics.gpqaTTFTPassCount == 0)
    #expect(decoded.metrics.gpqaTTFTCaseCount == 0)
    #expect(decoded.metrics.gpqaTTFTSeconds == 0)
    #expect(decoded.metrics.gpqaTTFTP50Seconds == 0)
    #expect(decoded.metrics.gpqaTTFTMaxSeconds == 0)
    #expect(decoded.metrics.gpqaTTFTSource == "")
    #expect(decoded.metrics.semanticGPQAPassed == false)
    #expect(decoded.metrics.semanticGPQAPassCount == 0)
    #expect(decoded.metrics.semanticGPQACaseCount == 0)
    #expect(decoded.metrics.semanticGPQAModel == "")
    #expect(decoded.metrics.processResidentMemoryGB == 0)
    #expect(decoded.metrics.firstFailingLayer == nil)
    #expect(decoded.metrics.firstFailingCase == nil)
    #expect(decoded.metrics.firstFailingStep == nil)
    #expect(decoded.metrics.expectedToken == nil)
    #expect(decoded.metrics.actualToken == nil)
    #expect(decoded.metrics.goldenHash == "")
    #expect(decoded.metrics.error == "runtime unavailable")
    #expect(decoded.metrics.runtime == "swift")
}

@Test
func writeScorePayloadKeepsTokenStepSeparateFromLayerFailures() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("score.json")

    try writeScorePayload(
        ScorePayload(
            score: nil,
            passed: false,
            metrics: ScoreMetrics(
                peakRamGB: 0,
                bandwidthGBPerToken: 0,
                decodeSecondsPerToken: 0,
                prefillSecondsPerToken: 0,
                benchmarkWallSeconds: 11,
                preflightSeconds: 1,
                correctnessSeconds: 2,
                timedBenchmarkSeconds: 8,
                gpqaTTFTPassed: true,
                gpqaTTFTPassCount: 9,
                gpqaTTFTCaseCount: 9,
                gpqaTTFTSeconds: 72,
                gpqaTTFTP50Seconds: 72,
                gpqaTTFTMaxSeconds: 75,
                gpqaTTFTSource: "hidden_gpqa_first_token",
                semanticGPQAPassed: true,
                semanticGPQAPassCount: 8,
                semanticGPQACaseCount: 9,
                semanticGPQAModel: "claude-opus-4-8",
                processResidentMemoryGB: 3.5,
                passedCorrectness: false,
                numLayers: MLXFastConstants.numHiddenLayers,
                checkedSteps: 13,
                caseCount: 2,
                expertCacheHits: 3,
                expertCacheMisses: 5,
                expertCacheEvictions: 2,
                expertBytesRead: 1024,
                expertReadSeconds: 0.25,
                expertPeakCachedTensors: 4,
                expertHitRate: 0.38,
                firstFailingLayer: nil,
                firstFailingCase: "case-b",
                firstFailingStep: 12,
                expectedToken: 42,
                actualToken: 17,
                maxAbsDiff: 0,
                goldenHash: "golden-hash",
                bandwidthSource: "",
                error: "generated token mismatch",
                commit: "abc123",
                timestamp: "2026-06-18T00:00:00Z",
                harnessHash: "hash",
                weightsHash: "weights-hash",
                weightsByteCount: 4096,
                weightsFileCount: 7,
                runtime: "swift",
                partialResult: true
            )
        ),
        to: path.path
    )

    let data = try Data(contentsOf: path)
    let raw = String(decoding: data, as: UTF8.self)
    let decoded = try JSONDecoder().decode(ScorePayload.self, from: data)

    #expect(raw.contains("\"first_failing_layer\" : null"))
    #expect(raw.contains("\"first_failing_case\" : \"case-b\""))
    #expect(raw.contains("\"first_failing_step\" : 12"))
    #expect(raw.contains("\"expected_token\" : 42"))
    #expect(raw.contains("\"actual_token\" : 17"))
    #expect(raw.contains("\"checked_steps\" : 13"))
    #expect(raw.contains("\"case_count\" : 2"))
    #expect(raw.contains("\"expert_cache_hits\" : 3"))
    #expect(raw.contains("\"expert_cache_misses\" : 5"))
    #expect(raw.contains("\"expert_cache_evictions\" : 2"))
    #expect(raw.contains("\"expert_bytes_read\" : 1024"))
    #expect(raw.contains("\"expert_read_seconds\" : 0.25"))
    #expect(raw.contains("\"expert_peak_cached_tensors\" : 4"))
    #expect(raw.contains("\"expert_hit_rate\" : 0.38"))
    #expect(raw.contains("\"golden_hash\" : \"golden-hash\""))
    #expect(raw.contains("\"weights_hash\" : \"weights-hash\""))
    #expect(raw.contains("\"weights_byte_count\" : 4096"))
    #expect(raw.contains("\"weights_file_count\" : 7"))
    #expect(raw.contains("\"partial_result\" : true"))
    #expect(raw.contains("\"baseline_decode_seconds_per_token\" : \(MLXFastConstants.officialBaselineDecodeSecondsPerToken)"))
    #expect(raw.contains("\"baseline_prefill_seconds_per_token\" : \(MLXFastConstants.officialBaselinePrefillSecondsPerToken)"))
    #expect(raw.contains("\"decode_speedup\" : 0"))
    #expect(raw.contains("\"prefill_speedup\" : 0"))
    #expect(raw.contains("\"decode_speedup_floor\" : \(MLXFastConstants.scoreDecodeSpeedupFloor)"))
    #expect(raw.contains("\"prefill_speedup_floor\" : \(MLXFastConstants.scorePrefillSpeedupFloor)"))
    #expect(raw.contains("\"passed_decode_speedup_floor\" : false"))
    #expect(raw.contains("\"passed_prefill_speedup_floor\" : false"))
    #expect(raw.contains("\"benchmark_wall_seconds\" : 11"))
    #expect(raw.contains("\"preflight_seconds\" : 1"))
    #expect(raw.contains("\"correctness_seconds\" : 2"))
    #expect(raw.contains("\"timed_benchmark_seconds\" : 8"))
    #expect(raw.contains("\"gpqa_ttft_passed\" : true"))
    #expect(raw.contains("\"gpqa_ttft_pass_count\" : 9"))
    #expect(raw.contains("\"gpqa_ttft_case_count\" : 9"))
    #expect(raw.contains("\"gpqa_ttft_seconds\" : 72"))
    #expect(raw.contains("\"gpqa_ttft_p50_seconds\" : 72"))
    #expect(raw.contains("\"gpqa_ttft_max_seconds\" : 75"))
    #expect(raw.contains("\"gpqa_ttft_source\" : \"hidden_gpqa_first_token\""))
    #expect(raw.contains("\"semantic_gpqa_passed\" : true"))
    #expect(raw.contains("\"semantic_gpqa_pass_count\" : 8"))
    #expect(raw.contains("\"semantic_gpqa_case_count\" : 9"))
    #expect(raw.contains("\"semantic_gpqa_model\" : \"claude-opus-4-8\""))
    #expect(raw.contains("\"process_resident_memory_gb\" : 3.5"))
    #expect(decoded.metrics.firstFailingLayer == nil)
    #expect(decoded.metrics.firstFailingCase == "case-b")
    #expect(decoded.metrics.firstFailingStep == 12)
    #expect(decoded.metrics.expectedToken == 42)
    #expect(decoded.metrics.actualToken == 17)
    #expect(decoded.metrics.checkedSteps == 13)
    #expect(decoded.metrics.caseCount == 2)
    #expect(decoded.metrics.expertCacheHits == 3)
    #expect(decoded.metrics.expertCacheMisses == 5)
    #expect(decoded.metrics.expertCacheEvictions == 2)
    #expect(decoded.metrics.expertBytesRead == 1024)
    #expect(decoded.metrics.expertReadSeconds == 0.25)
    #expect(decoded.metrics.expertPeakCachedTensors == 4)
    #expect(decoded.metrics.expertHitRate == 0.38)
    #expect(decoded.metrics.goldenHash == "golden-hash")
    #expect(decoded.metrics.weightsHash == "weights-hash")
    #expect(decoded.metrics.weightsByteCount == 4096)
    #expect(decoded.metrics.weightsFileCount == 7)
    #expect(decoded.metrics.partialResult == true)
    #expect(decoded.metrics.baselineDecodeSecondsPerToken == MLXFastConstants.officialBaselineDecodeSecondsPerToken)
    #expect(decoded.metrics.baselinePrefillSecondsPerToken == MLXFastConstants.officialBaselinePrefillSecondsPerToken)
    #expect(decoded.metrics.decodeSpeedup == 0)
    #expect(decoded.metrics.prefillSpeedup == 0)
    #expect(decoded.metrics.decodeSpeedupFloor == MLXFastConstants.scoreDecodeSpeedupFloor)
    #expect(decoded.metrics.prefillSpeedupFloor == MLXFastConstants.scorePrefillSpeedupFloor)
    #expect(decoded.metrics.passedDecodeSpeedupFloor == false)
    #expect(decoded.metrics.passedPrefillSpeedupFloor == false)
    #expect(decoded.metrics.benchmarkWallSeconds == 11)
    #expect(decoded.metrics.preflightSeconds == 1)
    #expect(decoded.metrics.correctnessSeconds == 2)
    #expect(decoded.metrics.timedBenchmarkSeconds == 8)
    #expect(decoded.metrics.gpqaTTFTPassed == true)
    #expect(decoded.metrics.gpqaTTFTPassCount == 9)
    #expect(decoded.metrics.gpqaTTFTCaseCount == 9)
    #expect(decoded.metrics.gpqaTTFTSeconds == 72)
    #expect(decoded.metrics.gpqaTTFTP50Seconds == 72)
    #expect(decoded.metrics.gpqaTTFTMaxSeconds == 75)
    #expect(decoded.metrics.gpqaTTFTSource == "hidden_gpqa_first_token")
    #expect(decoded.metrics.semanticGPQAPassed == true)
    #expect(decoded.metrics.semanticGPQAPassCount == 8)
    #expect(decoded.metrics.semanticGPQACaseCount == 9)
    #expect(decoded.metrics.semanticGPQAModel == "claude-opus-4-8")
    #expect(decoded.metrics.processResidentMemoryGB == 3.5)
}

@Test
func publicDiagnosticsAreCoarsenedWhileRankingStaysPrecise() throws {
    let directory = try temporaryDirectory()
    let path = directory.appendingPathComponent("score.json")

    try writeScorePayload(
        ScorePayload(
            score: 1.947063198,
            passed: true,
            metrics: ScoreMetrics(
                peakRamGB: 15.927597,
                bandwidthGBPerToken: 2.882812,
                decodeSecondsPerToken: 2.0953410813828124,
                prefillSecondsPerToken: 0.0985421517,
                decodeSpeedup: 2.0142336773,
                prefillSpeedup: 1.7586954276,
                benchmarkWallSeconds: 484.7341,
                preflightSeconds: 0.0270376,
                correctnessSeconds: 1076.0285,
                timedBenchmarkSeconds: 283.10626,
                gpqaTTFTPassed: true,
                gpqaTTFTPassCount: 5,
                gpqaTTFTCaseCount: 5,
                gpqaTTFTSeconds: 58.3755152,
                gpqaTTFTP50Seconds: 59.6692053,
                gpqaTTFTMaxSeconds: 63.3736865,
                gpqaTTFTSource: "hidden_gpqa_first_token",
                semanticGPQAPassed: true,
                semanticGPQAPassCount: 3,
                semanticGPQACaseCount: 5,
                semanticGPQAModel: "m",
                processResidentMemoryGB: 0.30030822,
                passedCorrectness: true,
                numLayers: MLXFastConstants.numHiddenLayers,
                checkedSteps: 114,
                caseCount: 6,
                expertReadSeconds: 95.48715,
                expertHitRate: 0.173790069,
                firstFailingLayer: nil,
                firstFailingCase: nil,
                firstFailingStep: nil,
                expectedToken: nil,
                actualToken: nil,
                maxAbsDiff: 0,
                goldenHash: "g",
                bandwidthSource: "ram_resident_model",
                error: "",
                commit: "c",
                timestamp: "t",
                harnessHash: "h",
                runtime: "swift"
            )
        ),
        to: path.path
    )

    let decoded = try JSONDecoder().decode(ScorePayload.self, from: Data(contentsOf: path))

    // Ranking- and floor-critical fields are published at full precision so
    // scoring, the speedup floor, and paired timing overlay are unaffected.
    #expect(decoded.score == 1.947063198)
    #expect(decoded.metrics.decodeSecondsPerToken == 2.0953410813828124)
    #expect(decoded.metrics.prefillSecondsPerToken == 0.0985421517)
    #expect(decoded.metrics.decodeSpeedup == 2.0142336773)
    #expect(decoded.metrics.prefillSpeedup == 1.7586954276)

    // Diagnostic analog fields are coarsened to 2 significant figures, shrinking
    // the timing/memory covert channel submitted code can drive.
    #expect(decoded.metrics.peakRamGB == 16)
    #expect(decoded.metrics.bandwidthGBPerToken == 2.9)
    #expect(decoded.metrics.benchmarkWallSeconds == 480)
    #expect(decoded.metrics.correctnessSeconds == 1100)
    #expect(decoded.metrics.timedBenchmarkSeconds == 280)
    #expect(decoded.metrics.gpqaTTFTSeconds == 58)
    #expect(decoded.metrics.gpqaTTFTP50Seconds == 60)
    #expect(decoded.metrics.gpqaTTFTMaxSeconds == 63)
    #expect(decoded.metrics.processResidentMemoryGB == 0.3)
    #expect(decoded.metrics.expertReadSeconds == 95)

    // Ordering invariants the artifact validators assert survive rounding.
    #expect(decoded.metrics.benchmarkWallSeconds >= decoded.metrics.timedBenchmarkSeconds)
    #expect(decoded.metrics.gpqaTTFTMaxSeconds >= decoded.metrics.gpqaTTFTP50Seconds)
}

@Test
func scoreMetricsDecodeOlderPayloadWithoutWeightsIntegrityFields() throws {
    let data = """
    {
      "score": null,
      "passed": false,
      "metrics": {
        "peak_ram_gb": 0,
        "bandwidth_gb_per_token": 0,
        "decode_seconds_per_token": 0,
        "prefill_seconds_per_token": 0,
        "passed_correctness": false,
        "num_layers": \(MLXFastConstants.numHiddenLayers),
        "checked_steps": 0,
        "case_count": 0,
        "expert_cache_hits": 0,
        "expert_cache_misses": 0,
        "expert_cache_evictions": 0,
        "expert_bytes_read": 0,
        "expert_read_seconds": 0,
        "expert_peak_cached_tensors": 0,
        "expert_hit_rate": 0,
        "first_failing_layer": null,
        "first_failing_case": null,
        "first_failing_step": null,
        "expected_token": null,
        "actual_token": null,
        "max_abs_diff": 0,
        "golden_hash": "",
        "bandwidth_source": "",
        "error": "old payload",
        "commit": "",
        "timestamp": "2026-06-18T00:00:00Z",
        "harness_hash": "",
        "runtime": "swift"
      }
    }
    """.data(using: .utf8)!

    let decoded = try JSONDecoder().decode(ScorePayload.self, from: data)

    #expect(decoded.metrics.weightsHash == "")
    #expect(decoded.metrics.weightsByteCount == 0)
    #expect(decoded.metrics.weightsFileCount == 0)
    #expect(decoded.metrics.partialResult == false)
    #expect(decoded.metrics.baselineDecodeSecondsPerToken == MLXFastConstants.officialBaselineDecodeSecondsPerToken)
    #expect(decoded.metrics.baselinePrefillSecondsPerToken == MLXFastConstants.officialBaselinePrefillSecondsPerToken)
    #expect(decoded.metrics.decodeSpeedup == 0)
    #expect(decoded.metrics.prefillSpeedup == 0)
    #expect(decoded.metrics.decodeSpeedupFloor == MLXFastConstants.scoreDecodeSpeedupFloor)
    #expect(decoded.metrics.prefillSpeedupFloor == MLXFastConstants.scorePrefillSpeedupFloor)
    #expect(decoded.metrics.passedDecodeSpeedupFloor == false)
    #expect(decoded.metrics.passedPrefillSpeedupFloor == false)
    #expect(decoded.metrics.benchmarkWallSeconds == 0)
    #expect(decoded.metrics.preflightSeconds == 0)
    #expect(decoded.metrics.correctnessSeconds == 0)
    #expect(decoded.metrics.timedBenchmarkSeconds == 0)
    #expect(decoded.metrics.gpqaTTFTPassed == false)
    #expect(decoded.metrics.gpqaTTFTPassCount == 0)
    #expect(decoded.metrics.gpqaTTFTCaseCount == 0)
    #expect(decoded.metrics.gpqaTTFTSeconds == 0)
    #expect(decoded.metrics.gpqaTTFTP50Seconds == 0)
    #expect(decoded.metrics.gpqaTTFTMaxSeconds == 0)
    #expect(decoded.metrics.gpqaTTFTSource == "")
    #expect(decoded.metrics.semanticGPQAPassed == false)
    #expect(decoded.metrics.semanticGPQAPassCount == 0)
    #expect(decoded.metrics.semanticGPQACaseCount == 0)
    #expect(decoded.metrics.semanticGPQAModel == "")
    #expect(decoded.metrics.processResidentMemoryGB == 0)
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
