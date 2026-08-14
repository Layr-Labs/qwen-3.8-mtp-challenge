import Foundation
import MLXFastCore
#if !MLXFAST_TRUSTED_HARNESS
import MLXFastModel
#endif
import Tokenizers

// QwenRuntime is split across QwenRuntime*.swift for auditability.
// Generated split; behavior identical to the original single file.

extension QwenRuntime {
    public static func generateGreedyTokens(
        _ options: GreedyGenerationOptions,
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> [Int] {
        try generateGreedyTokens(options, worker: nil, progress: progress)
    }

    public static func generateGreedyTokens(
        _ options: GreedyGenerationOptions,
        worker workerOptions: RuntimeWorkerOptions?,
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> [Int] {
        if let workerOptions {
            let worker = try RuntimeWorkerClient(options: workerOptions, weightsPath: options.weightsPath)
            defer {
                worker.close()
            }
            let response = try worker.generateCorrectness(
                promptTokens: options.promptTokens,
                steps: options.steps
            )
            guard let tokens = response.tokens else {
                throw MLXFastError.invalidInput("runtime worker greedy generation response missing tokens")
            }
            try requireGeneratedTokenCount(tokens.count, expected: options.steps, label: "greedy generation")
            progress?(tokens.count, options.steps)
            return tokens
        }

        #if !MLXFAST_TRUSTED_HARNESS
            let config = try Qwen35Config.load(from: options.weightsPath)
            let loader = try Qwen35WeightLoader(
                weightsPath: options.weightsPath
            )
            let weightCache = Qwen35RuntimeWeightCache(
                loader: loader,
                config: config
            )
            return try generateGreedyCached(
                promptTokens: options.promptTokens,
                steps: options.steps,
                weightCache: weightCache,
                progressIntervalSteps: 1,
                progress: progress
            )
        #else
            throw MLXFastError.invalidInput(
                "trusted greedy generation requires the participant worker"
            )
        #endif
    }

    public static func runCorrectness(
        _ options: CorrectnessOptions,
        worker: RuntimeWorkerOptions? = nil
    ) throws -> CorrectnessReport {
        if let worker {
            return runCorrectnessWithWorker(options, worker: worker)
        }

        #if !MLXFAST_TRUSTED_HARNESS
            var loadedGolden: GoldenFixture?
            var loader: Qwen35WeightLoader?
            do {
                try requireFile(
                    options.goldenPath,
                    description: "correctness golden file"
                )
                let golden = try loadGoldenFixture(from: options.goldenPath)
                loadedGolden = golden
                _ = try BenchmarkPreflight.checkCorrectnessArtifacts(
                    weightsPath: options.weightsPath,
                    goldenPath: options.goldenPath
                )
                let config = try Qwen35Config.load(from: options.weightsPath)
                let runtimeLoader = try Qwen35WeightLoader(
                    weightsPath: options.weightsPath
                )
                loader = runtimeLoader
                let weightCache = Qwen35RuntimeWeightCache(
                    loader: runtimeLoader,
                    config: config
                )
                return runLayeredCorrectness(
                    golden: golden,
                    weightCache: weightCache,
                    steps: MLXFastConstants.correctnessSteps
                )
            } catch {
                return failedCorrectnessReport(
                    checkedSteps: 0,
                    caseCount: loadedGolden?.totalCorrectnessCaseCount ?? 0,
                    goldenHash: loadedGolden?.sha256 ?? "",
                    expertStats: expertStats(from: loader),
                    error: "\(error)"
                )
            }
        #else
            var loadedGolden: GoldenFixture?
            do {
                try requireFile(
                    options.goldenPath,
                    description: "correctness golden file"
                )
                let golden = try loadGoldenFixture(
                    from: options.goldenPath
                )
                loadedGolden = golden
                _ = try BenchmarkPreflight.checkCorrectnessArtifacts(
                    weightsPath: options.weightsPath,
                    goldenPath: options.goldenPath
                )
                return failedCorrectnessReport(
                    checkedSteps: 0,
                    caseCount: golden.totalCorrectnessCaseCount,
                    goldenHash: golden.sha256,
                    expertStats: .zero,
                    error:
                        "trusted correctness requires the participant worker"
                )
            } catch {
                return failedCorrectnessReport(
                    checkedSteps: 0,
                    caseCount:
                        loadedGolden?.totalCorrectnessCaseCount ?? 0,
                    goldenHash: loadedGolden?.sha256 ?? "",
                    expertStats: .zero,
                    error: "\(error)"
                )
            }
        #endif
    }

    static func runCorrectnessWithWorker(
        _ options: CorrectnessOptions,
        worker workerOptions: RuntimeWorkerOptions
    ) -> CorrectnessReport {
        var loadedGolden: GoldenFixture?
        var lastExpertStats = ExpertStreamingStats.zero
        var checkedSteps = 0
        do {
            try requireFile(options.goldenPath, description: "correctness golden file")
            let golden = try loadGoldenFixture(from: options.goldenPath)
            loadedGolden = golden
            _ = try BenchmarkPreflight.checkCorrectnessArtifacts(
                weightsPath: options.weightsPath,
                goldenPath: options.goldenPath
            )
            let worker = try RuntimeWorkerClient(
                options: workerOptions,
                weightsPath: options.weightsPath
            )
            defer {
                worker.close()
            }
            let result = runLayeredCorrectnessWithWorker(
                golden: golden,
                worker: worker,
                steps: MLXFastConstants.correctnessSteps
            )
            checkedSteps = result.report.checkedSteps
            lastExpertStats = result.expertStats
            return result.report
        } catch {
            return failedCorrectnessReport(
                checkedSteps: checkedSteps,
                caseCount: loadedGolden?.totalCorrectnessCaseCount ?? 0,
                goldenHash: loadedGolden?.sha256 ?? "",
                expertStats: lastExpertStats,
                error: "\(error)"
            )
        }
    }

    struct WorkerLayeredCorrectnessResult {
        let report: CorrectnessReport
        let expertStats: ExpertStreamingStats
        let peakRamGB: Double
        let gpqaTTFT: GPQATTFTSummary
        // Collected + validated hidden-GPQA answers, returned to the trusted
        // caller so it can write the capture AFTER the correctness worker (the
        // only process running editable model code) is closed. Non-nil only on
        // a fully-successful capture run; nil when no capture was requested or
        // correctness failed before the capture completed. See the write site
        // in benchmarkWithWorker and writeSemanticGPQAAnswers.
        let semanticGPQAAnswers: [SemanticGPQAAnswerCase]?
    }

    struct GPQATTFTSummary {
        static let zero = GPQATTFTSummary(passCount: 0, caseCount: 0, seconds: [])

        let passCount: Int
        let caseCount: Int
        let seconds: [Double]

        var passed: Bool {
            caseCount > 0 && passCount == caseCount && seconds.count == caseCount
        }

        var source: String {
            caseCount > 0 ? "hidden_gpqa_first_token" : ""
        }

        var meanSeconds: Double {
            guard !seconds.isEmpty else {
                return 0
            }
            return seconds.reduce(0, +) / Double(seconds.count)
        }

        var p50Seconds: Double {
            guard !seconds.isEmpty else {
                return 0
            }
            let sortedSeconds = seconds.sorted()
            return sortedSeconds[sortedSeconds.count / 2]
        }

        var maxSeconds: Double {
            seconds.max() ?? 0
        }
    }

    #if !MLXFAST_TRUSTED_HARNESS
        static func runLayeredCorrectness(
        golden: GoldenFixture,
        weightCache: Qwen35RuntimeWeightCache,
        steps: Int = MLXFastConstants.correctnessSteps,
        progress: ((String) -> Void)? = nil
    ) -> CorrectnessReport {
        // This in-process path always checks every gate the golden carries.
        // Only the worker variant supports the gates-skipped timing phase
        // (see runLayeredCorrectnessWithWorker's checkGates).
        let caseCount = golden.totalCorrectnessCaseCount
        var checkedSteps = 0
        var currentCase: String?

        func failure(
            caseName: String,
            comparison: CorrectnessTokenComparison,
            error: String
        ) -> CorrectnessReport {
            let stats = expertStats(from: weightCache)
            return CorrectnessReport(
                passed: false,
                checkedSteps: checkedSteps + comparison.checkedSteps,
                caseCount: caseCount,
                expertCacheHits: stats.cacheHits,
                expertCacheMisses: stats.cacheMisses,
                expertCacheEvictions: stats.cacheEvictions,
                expertBytesRead: stats.bytesRead,
                expertReadSeconds: stats.readSeconds,
                expertPeakCachedTensors: stats.peakCachedTensors,
                expertHitRate: stats.hitRate,
                firstFailingCase: caseName,
                firstFailingStep: comparison.firstFailingStep,
                expectedToken: comparison.expectedToken,
                actualToken: comparison.actualToken,
                goldenHash: golden.sha256,
                error: error
            )
        }

        do {
            for (caseIndex, testCase) in golden.cases.enumerated() {
                currentCase = testCase.name
                let caseLabel = "\(caseIndex + 1)/\(golden.cases.count)"
                // See the matching guard in runLayeredCorrectnessWithWorker: steps == 0
                // intentionally skips the base case for this run.
                guard steps > 0 else {
                    progress?("correctness case \(caseLabel) skipped (steps=0)")
                    continue
                }
                progress?("correctness case \(caseLabel) start prompt_tokens=\(testCase.promptTokens.count)")
                let comparison = try compareTeacherForcedCached(
                    testCase: testCase,
                    weightCache: weightCache,
                    steps: steps,
                    progressIntervalSteps: 64,
                    progress: { step, total in
                        progress?("correctness case \(caseLabel) checked \(step)/\(total) tokens")
                    }
                )
                if !comparison.passed {
                    progress?("correctness case \(caseLabel) failed step=\(comparison.firstFailingStep ?? -1)")
                    return failure(
                        caseName: testCase.name,
                        comparison: comparison,
                        error: "teacher-forced token mismatch"
                    )
                }
                progress?("correctness case \(caseLabel) complete checked_steps=\(comparison.checkedSteps)")
                checkedSteps += comparison.checkedSteps
            }

            let gates = golden.correctnessGates
            for (caseIndex, anchor) in (gates?.anchorCases ?? []).enumerated() {
                currentCase = anchor.name
                let caseLabel = "\(caseIndex + 1)/\(gates?.anchorCases.count ?? 0)"
                progress?("correctness anchor \(caseLabel) start context_tokens=\(anchor.contextTokens.count)")
                let comparison = try compareAnchorCached(anchor: anchor, weightCache: weightCache)
                if !comparison.passed {
                    progress?("correctness anchor \(caseLabel) failed")
                    return failure(
                        caseName: anchor.name,
                        comparison: comparison,
                        error: "anchor token mismatch"
                    )
                }
                checkedSteps += comparison.checkedSteps
                progress?("correctness anchor \(caseLabel) complete")
            }

            for (caseIndex, freeRun) in (gates?.freeRunCases ?? []).enumerated() {
                currentCase = freeRun.name
                let caseLabel = "\(caseIndex + 1)/\(gates?.freeRunCases.count ?? 0)"
                progress?("correctness free-run \(caseLabel) start tokens=\(freeRun.expectedTokens.count)")
                let comparison = try compareFreeRunCached(
                    testCase: freeRun,
                    weightCache: weightCache,
                    progressIntervalSteps: 64,
                    progress: { step, total in
                        progress?("correctness free-run \(caseLabel) generated \(step)/\(total) tokens")
                    }
                )
                if !comparison.passed {
                    progress?("correctness free-run \(caseLabel) failed step=\(comparison.firstFailingStep ?? -1)")
                    return failure(
                        caseName: freeRun.name,
                        comparison: comparison,
                        error: "free-run token mismatch"
                    )
                }
                checkedSteps += comparison.checkedSteps
                progress?("correctness free-run \(caseLabel) complete checked_steps=\(comparison.checkedSteps)")
            }

            for (caseIndex, behavior) in (gates?.behaviorCases ?? []).enumerated() {
                currentCase = behavior.name
                let caseLabel = "\(caseIndex + 1)/\(gates?.behaviorCases.count ?? 0)"
                progress?("correctness behavior \(caseLabel) start max_new_tokens=\(behavior.maxNewTokens)")
                let comparison = try compareBehaviorCached(
                    testCase: behavior,
                    weightCache: weightCache
                )
                if !comparison.passed {
                    progress?("correctness behavior \(caseLabel) failed step=\(comparison.firstFailingStep ?? -1)")
                    return failure(
                        caseName: behavior.name,
                        comparison: comparison,
                        error: "behavior answer mismatch"
                    )
                }
                checkedSteps += comparison.checkedSteps
                progress?("correctness behavior \(caseLabel) complete checked_steps=\(comparison.checkedSteps)")
            }
        } catch {
            progress?("correctness error=\(redactedProgressError("\(error)"))")
            return failedCorrectnessReport(
                checkedSteps: checkedSteps,
                caseCount: caseCount,
                firstFailingCase: currentCase,
                goldenHash: golden.sha256,
                expertStats: expertStats(from: weightCache),
                error: "\(error)"
            )
        }

        let stats = expertStats(from: weightCache)
        return CorrectnessReport(
            passed: true,
            checkedSteps: checkedSteps,
            caseCount: caseCount,
            expertCacheHits: stats.cacheHits,
            expertCacheMisses: stats.cacheMisses,
            expertCacheEvictions: stats.cacheEvictions,
            expertBytesRead: stats.bytesRead,
            expertReadSeconds: stats.readSeconds,
            expertPeakCachedTensors: stats.peakCachedTensors,
            expertHitRate: stats.hitRate,
            firstFailingCase: nil,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil,
            goldenHash: golden.sha256,
            error: ""
        )
        }
    #endif

    static func runLayeredCorrectnessWithWorker(
        golden: GoldenFixture,
        worker: RuntimeWorkerClient,
        steps: Int = MLXFastConstants.correctnessSteps,
        checkGates: Bool = true,
        semanticCapture: SemanticGPQACaptureOptions? = nil,
        progress: ((String) -> Void)? = nil
    ) -> WorkerLayeredCorrectnessResult {
        // checkGates: false is the gates-skipped timing phase of the phased
        // ranked run (gates were checked by an earlier invocation; see
        // BenchmarkOptions.checkGates): it skips anchors/free-run/behavior/GPQA
        // entirely and reports caseCount/checkedSteps for golden.cases alone.
        let caseCount = checkGates ? golden.totalCorrectnessCaseCount : golden.cases.count
        var checkedSteps = 0
        var currentCase: String?
        var lastExpertStats = ExpertStreamingStats.zero
        var peakRamGB = 0.0
        var gpqaTTFTPassCount = 0
        var gpqaTTFTCaseCount = 0
        var gpqaTTFTSeconds: [Double] = []
        var semanticAnswers: [SemanticGPQAAnswerCase] = []
        // Set only on the fully-successful capture path (see the write site
        // below); left nil on every failure/no-capture path so the trusted
        // caller never writes a partial or absent capture.
        var capturedSemanticAnswers: [SemanticGPQAAnswerCase]?

        func result(report: CorrectnessReport) -> WorkerLayeredCorrectnessResult {
            WorkerLayeredCorrectnessResult(
                report: report,
                expertStats: lastExpertStats,
                peakRamGB: peakRamGB,
                gpqaTTFT: GPQATTFTSummary(
                    passCount: gpqaTTFTPassCount,
                    caseCount: gpqaTTFTCaseCount,
                    seconds: gpqaTTFTSeconds
                ),
                semanticGPQAAnswers: capturedSemanticAnswers
            )
        }

        func failure(
            caseName: String,
            comparison: CorrectnessTokenComparison,
            error: String
        ) -> WorkerLayeredCorrectnessResult {
            result(report: CorrectnessReport(
                passed: false,
                checkedSteps: checkedSteps + comparison.checkedSteps,
                caseCount: caseCount,
                expertCacheHits: lastExpertStats.cacheHits,
                expertCacheMisses: lastExpertStats.cacheMisses,
                expertCacheEvictions: lastExpertStats.cacheEvictions,
                expertBytesRead: lastExpertStats.bytesRead,
                expertReadSeconds: lastExpertStats.readSeconds,
                expertPeakCachedTensors: lastExpertStats.peakCachedTensors,
                expertHitRate: lastExpertStats.hitRate,
                firstFailingCase: caseName,
                firstFailingStep: comparison.firstFailingStep,
                expectedToken: comparison.expectedToken,
                actualToken: comparison.actualToken,
                goldenHash: golden.sha256,
                error: error
            ))
        }

        do {
            let semanticTokenizer: (any Tokenizer)? = try semanticCapture.map {
                try loadLocalTokenizer(at: $0.tokenizerPath)
            }
            for (caseIndex, testCase) in golden.cases.enumerated() {
                currentCase = testCase.name
                let caseLabel = "\(caseIndex + 1)/\(golden.cases.count)"
                // steps == 0 means this run intentionally does not verify the base
                // teacher-forced case itself, because a separate phase of the ranked
                // pipeline verifies it. Skipping here does NOT mean correctness was
                // checked; only the trusted pipeline that assembles the final score
                // may treat the base case as covered.
                guard steps > 0 else {
                    progress?("correctness case \(caseLabel) skipped (steps=0)")
                    continue
                }
                progress?("correctness case \(caseLabel) start prompt_tokens=\(testCase.promptTokens.count)")
                let check = try compareTeacherForcedWithWorker(
                    testCase: testCase,
                    worker: worker,
                    steps: steps,
                    progressIntervalSteps: 64,
                    progress: { step, total in
                        progress?("correctness case \(caseLabel) checked \(step)/\(total) tokens")
                    }
                )
                lastExpertStats = check.expertStats
                peakRamGB = max(peakRamGB, check.peakRamGB)
                if !check.comparison.passed {
                    progress?("correctness case \(caseLabel) failed step=\(check.comparison.firstFailingStep ?? -1)")
                    return failure(
                        caseName: testCase.name,
                        comparison: check.comparison,
                        error: "teacher-forced token mismatch"
                    )
                }
                checkedSteps += check.comparison.checkedSteps
                progress?("correctness case \(caseLabel) complete checked_steps=\(check.comparison.checkedSteps)")
            }

            let gates = checkGates ? golden.correctnessGates : nil
            for (caseIndex, anchor) in (gates?.anchorCases ?? []).enumerated() {
                currentCase = anchor.name
                let caseLabel = "\(caseIndex + 1)/\(gates?.anchorCases.count ?? 0)"
                progress?("correctness anchor \(caseLabel) start context_tokens=\(anchor.contextTokens.count)")
                let check = try compareAnchorWithWorker(anchor: anchor, worker: worker)
                lastExpertStats = check.expertStats
                peakRamGB = max(peakRamGB, check.peakRamGB)
                if !check.comparison.passed {
                    progress?("correctness anchor \(caseLabel) failed")
                    return failure(
                        caseName: anchor.name,
                        comparison: check.comparison,
                        error: "anchor token mismatch"
                    )
                }
                checkedSteps += check.comparison.checkedSteps
                progress?("correctness anchor \(caseLabel) complete")
            }

            for (caseIndex, freeRun) in (gates?.freeRunCases ?? []).enumerated() {
                currentCase = freeRun.name
                let caseLabel = "\(caseIndex + 1)/\(gates?.freeRunCases.count ?? 0)"
                progress?("correctness free-run \(caseLabel) start tokens=\(freeRun.expectedTokens.count)")
                let check = try compareFreeRunWithWorker(testCase: freeRun, worker: worker)
                lastExpertStats = check.expertStats
                peakRamGB = max(peakRamGB, check.peakRamGB)
                if !check.comparison.passed {
                    progress?("correctness free-run \(caseLabel) failed step=\(check.comparison.firstFailingStep ?? -1)")
                    return failure(
                        caseName: freeRun.name,
                        comparison: check.comparison,
                        error: "free-run token mismatch"
                    )
                }
                checkedSteps += check.comparison.checkedSteps
                progress?("correctness free-run \(caseLabel) complete checked_steps=\(check.comparison.checkedSteps)")
            }

            for (caseIndex, behavior) in (gates?.behaviorCases ?? []).enumerated() {
                currentCase = behavior.name
                let caseLabel = "\(caseIndex + 1)/\(gates?.behaviorCases.count ?? 0)"
                progress?("correctness behavior \(caseLabel) start max_new_tokens=\(behavior.maxNewTokens)")
                let check = try compareBehaviorWithWorker(testCase: behavior, worker: worker)
                lastExpertStats = check.expertStats
                peakRamGB = max(peakRamGB, check.peakRamGB)
                if check.ttftSeconds != nil {
                    gpqaTTFTCaseCount += 1
                    if check.comparison.passed, let ttftSeconds = check.ttftSeconds, ttftSeconds > 0 {
                        gpqaTTFTPassCount += 1
                        gpqaTTFTSeconds.append(ttftSeconds)
                    }
                }
                if let semanticCapture,
                   let semanticTokenizer,
                   semanticAnswers.count < semanticCapture.caseCount,
                   let generatedTokens = check.generatedTokens,
                   let answer = try semanticAnswerCase(
                       behavior: behavior,
                       generatedTokens: Array(generatedTokens.prefix(semanticCapture.maxNewTokens)),
                       tokenizer: semanticTokenizer,
                       maxNewTokens: semanticCapture.maxNewTokens
                   )
                {
                    semanticAnswers.append(answer)
                }
                if !check.comparison.passed {
                    progress?("correctness behavior \(caseLabel) failed step=\(check.comparison.firstFailingStep ?? -1)")
                    return failure(
                        caseName: behavior.name,
                        comparison: check.comparison,
                        error: "behavior answer mismatch"
                    )
                }
                checkedSteps += check.comparison.checkedSteps
                progress?("correctness behavior \(caseLabel) complete checked_steps=\(check.comparison.checkedSteps)")
            }
            // checkGates false means the behavior loop above never ran, so
            // semanticAnswers is always empty here regardless of caseCount --
            // that is not a capture failure, there was simply nothing to
            // capture in this phase, and enforcing the count would turn a
            // valid gates-skipped timing phase into a spurious hard failure.
            if checkGates, let semanticCapture {
                guard semanticAnswers.count == semanticCapture.caseCount else {
                    throw MLXFastError.invalidInput(
                        "captured \(semanticAnswers.count) semantic GPQA answers; expected \(semanticCapture.caseCount)"
                    )
                }
                // Do NOT write the capture here: this runs while the correctness
                // worker (which executes editable model code) is still alive. The
                // capture embeds hidden prompt text, answer keys, and reference
                // answers, so it must not exist on disk while submitted code can
                // run. Return the validated answers to the trusted caller, which
                // writes them only after closing the worker.
                capturedSemanticAnswers = semanticAnswers
                progress?("correctness semantic GPQA answers captured cases=\(semanticAnswers.count)")
            }
        } catch {
            progress?("correctness error=\(redactedProgressError("\(error)"))")
            return result(report: failedCorrectnessReport(
                checkedSteps: checkedSteps,
                caseCount: caseCount,
                firstFailingCase: currentCase,
                goldenHash: golden.sha256,
                expertStats: lastExpertStats,
                error: "\(error)"
            ))
        }

        return result(report: CorrectnessReport(
            passed: true,
            checkedSteps: checkedSteps,
            caseCount: caseCount,
            expertCacheHits: lastExpertStats.cacheHits,
            expertCacheMisses: lastExpertStats.cacheMisses,
            expertCacheEvictions: lastExpertStats.cacheEvictions,
            expertBytesRead: lastExpertStats.bytesRead,
            expertReadSeconds: lastExpertStats.readSeconds,
            expertPeakCachedTensors: lastExpertStats.peakCachedTensors,
            expertHitRate: lastExpertStats.hitRate,
            firstFailingCase: nil,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil,
            goldenHash: golden.sha256,
            error: ""
        ))
    }

    static func failedCorrectnessReport(
        checkedSteps: Int,
        caseCount: Int = 0,
        firstFailingCase: String? = nil,
        goldenHash: String = "",
        expertStats: ExpertStreamingStats = .zero,
        error: String
    ) -> CorrectnessReport {
        CorrectnessReport(
            passed: false,
            checkedSteps: checkedSteps,
            caseCount: caseCount,
            expertCacheHits: expertStats.cacheHits,
            expertCacheMisses: expertStats.cacheMisses,
            expertCacheEvictions: expertStats.cacheEvictions,
            expertBytesRead: expertStats.bytesRead,
            expertReadSeconds: expertStats.readSeconds,
            expertPeakCachedTensors: expertStats.peakCachedTensors,
            expertHitRate: expertStats.hitRate,
            firstFailingCase: firstFailingCase,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil,
            goldenHash: goldenHash,
            error: error
        )
    }

}
