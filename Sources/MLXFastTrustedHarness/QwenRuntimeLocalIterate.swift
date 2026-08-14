import Foundation
#if !MLXFAST_TRUSTED_HARNESS
import MLX
#endif
import MLXFastCore
#if !MLXFAST_TRUSTED_HARNESS
import MLXFastModel
#endif

extension QwenRuntime {
    public static func localIterate(
        _ options: LocalIterateOptions,
        worker: RuntimeWorkerOptions? = nil
    ) -> ScorePayload {
        let startedAt = DispatchTime.now().uptimeNanoseconds
        let progress = makeBenchmarkProgressReporter(startedAt: startedAt)
        var correctnessReport: CorrectnessReport?
        var transformedWeightsDigest: DirectoryDigest?
        var validationSeconds = 0.0
        var correctnessSeconds = 0.0
        var timedSeconds = 0.0
        var expertStats = ExpertStreamingStats.zero
        var peakRamGB = 0.0
        let modeName = options.modeName
        let checkedStepsPerPass = options.benchmarkDecodeSteps + 2
        let totalCheckedSteps = checkedStepsPerPass * options.timingRepeats

        progress(
            "\(modeName) start checked_tokens=\(totalCheckedSteps) "
                + "decode_steps=\(options.benchmarkDecodeSteps) repeats=\(options.timingRepeats)"
        )
        progress(
            "\(modeName) baseline prefill_seconds_per_token="
                + formatDouble(MLXFastConstants.officialBaselinePrefillSecondsPerToken)
                + " decode_seconds_per_token="
                + formatDouble(MLXFastConstants.officialBaselineDecodeSecondsPerToken)
                + " (official-runner constants; local speedups are directional)"
        )

        func failed(
            _ error: String,
            passedCorrectness: Bool = false,
            decodeSecondsPerToken: Double = 0,
            prefillSecondsPerToken: Double = 0,
            bandwidthGBPerToken: Double = 0,
            bandwidthSource: String = ""
        ) -> ScorePayload {
            progress("\(modeName) failed error=\(redactedProgressError(error))")
            let payload = failedScore(
                error: error,
                correctness: correctnessReport,
                passedCorrectness: passedCorrectness,
                expertStats: expertStats,
                expectedToken: correctnessReport?.expectedToken,
                actualToken: correctnessReport?.actualToken,
                weightsDigest: transformedWeightsDigest,
                benchmarkWallSeconds: secondsSince(startedAt),
                preflightSeconds: validationSeconds,
                correctnessSeconds: correctnessSeconds,
                timedBenchmarkSeconds: timedSeconds,
                processResidentMemoryGB: currentResidentMemoryGB(),
                peakRamGB: peakRamGB,
                bandwidthGBPerToken: bandwidthGBPerToken,
                decodeSecondsPerToken: decodeSecondsPerToken,
                prefillSecondsPerToken: prefillSecondsPerToken,
                bandwidthSource: bandwidthSource,
                runtime: options.runtime
            )
            return localModeFailedPayloadWithEstimatedScore(
                payload,
                modeName: modeName,
                progress: progress
            )
        }

        do {
            try validateLocalIterateOptions(options)

            let validationStart = DispatchTime.now().uptimeNanoseconds
            progress("\(modeName) validation start")
            try checkWorkerBenchmarkInputs(weightsPath: options.weightsPath, goldenPath: options.goldenPath)
            let digestHeartbeat = startPhaseHeartbeat(
                label: "\(modeName) weights digest",
                progress: progress
            )
            defer {
                digestHeartbeat?.cancel()
            }
            transformedWeightsDigest = try directoryDigest(
                rootPath: options.weightsPath,
                ignoredRelativePaths: [".benchmark-source.sha256", ".gitkeep"]
            )
            digestHeartbeat?.cancel()
            if let transformedWeightsDigest {
                try enforceTransformedWeightsByteLimit(transformedWeightsDigest.byteCount)
                progress(
                    "\(modeName) weights digest complete files=\(transformedWeightsDigest.fileCount) "
                        + "bytes=\(transformedWeightsDigest.byteCount)"
                )
            }
            let golden = try loadGoldenFixture(
                from: options.goldenPath,
                requiredSteps: options.benchmarkDecodeSteps + 1,
                requiredPromptTokens: MLXFastConstants.correctnessPromptTokens
            )
            guard let localCase = golden.cases.first else {
                throw MLXFastError.invalidInput("\(modeName) public golden must contain at least one case")
            }
            validationSeconds = secondsSince(validationStart)
            progress(
                "\(modeName) validation complete prompt_tokens=\(localCase.promptTokens.count) "
                    + "seconds=\(formatSeconds(validationSeconds))"
            )

            let timingWallStart = DispatchTime.now().uptimeNanoseconds
            progress("\(modeName) checked timing start")
            let timing: LocalIterateTimingResult
            if let worker {
                timing = try runLocalIterateCheckedTimingWithWorker(
                    weightsPath: options.weightsPath,
                    testCase: localCase,
                    goldenHash: golden.sha256,
                    decodeSteps: options.benchmarkDecodeSteps,
                    timingRepeats: options.timingRepeats,
                    modeName: modeName,
                    workerOptions: worker,
                    progress: progress
                )
            } else {
                #if !MLXFAST_TRUSTED_HARNESS
                    timing = try runLocalIterateCheckedTiming(
                        weightsPath: options.weightsPath,
                        testCase: localCase,
                        goldenHash: golden.sha256,
                        decodeSteps: options.benchmarkDecodeSteps,
                        timingRepeats: options.timingRepeats,
                        modeName: modeName,
                        progress: progress
                    )
                #else
                    throw MLXFastError.invalidInput(
                        "trusted local benchmark requires the participant worker"
                    )
                #endif
            }
            let timingWallSeconds = secondsSince(timingWallStart)
            timedSeconds = timing.prefillSecondsPerToken
                * Double(localCase.promptTokens.count * options.timingRepeats)
                + timing.decode.secondsPerToken
                * Double(options.benchmarkDecodeSteps * options.timingRepeats)
            correctnessSeconds = timedSeconds
            correctnessReport = timing.correctness
            expertStats = timing.expertStats
            peakRamGB = timing.peakRamGB
            progress(
                "\(modeName) checked timing complete passed=\(timing.correctness.passed) "
                    + "checked_steps=\(timing.correctness.checkedSteps) "
                    + "prefill_seconds_per_token=\(formatDouble(timing.prefillSecondsPerToken)) "
                    + "decode_seconds_per_token=\(formatDouble(timing.decode.secondsPerToken)) "
                    + "measured_seconds=\(formatSeconds(timedSeconds)) "
                    + "wall_seconds=\(formatSeconds(timingWallSeconds))"
            )
            emitLocalIterateSummary(modeName: modeName, timing: timing, progress: progress)
            if !timing.correctness.passed {
                guard localGoldenDriftAllowed() else {
                    return failed(
                        timing.correctness.error.isEmpty ? "\(modeName) correctness failed" : timing.correctness.error,
                        passedCorrectness: false,
                        decodeSecondsPerToken: timing.decode.secondsPerToken,
                        prefillSecondsPerToken: timing.prefillSecondsPerToken,
                        bandwidthGBPerToken: timing.decode.bandwidthGBPerToken,
                        bandwidthSource: timing.decode.bandwidthSource
                    )
                }
                emitLocalGoldenDriftOverrideNotice(
                    modeName: modeName,
                    correctness: timing.correctness,
                    progress: progress
                )
            }

            return localIterateScore(
                peakRamGB: peakRamGB,
                bandwidthGBPerToken: timing.decode.bandwidthGBPerToken,
                decodeSecondsPerToken: timing.decode.secondsPerToken,
                prefillSecondsPerToken: timing.prefillSecondsPerToken,
                wallSeconds: secondsSince(startedAt),
                validationSeconds: validationSeconds,
                correctnessSeconds: correctnessSeconds,
                timedSeconds: timedSeconds,
                correctness: timing.correctness,
                expertStats: expertStats,
                bandwidthSource: timing.decode.bandwidthSource,
                weightsDigest: transformedWeightsDigest,
                runtime: options.runtime
            )
        } catch {
            return failed("\(error)", passedCorrectness: correctnessReport?.passed == true)
        }
    }

    static func validateLocalIterateOptions(_ options: LocalIterateOptions) throws {
        guard options.benchmarkDecodeSteps > 0 else {
            throw MLXFastError.invalidInput("\(options.modeName) decode steps must be positive")
        }
        guard options.timingRepeats > 0 else {
            throw MLXFastError.invalidInput("\(options.modeName) timing repeats must be positive")
        }
    }

    // MARK: - Live progress helpers
    //
    // Local modes are the participant edit loop, so the run streams every
    // useful number to stderr the moment it exists instead of holding
    // everything for the final score JSON. All lines below are progress-only:
    // they never feed the sealed score payload, and they never include
    // expected/actual token values (mismatch details stay in the score JSON,
    // matching the redaction the shared progress reporter applies to errors).

    static func formatRatio(_ value: Double) -> String {
        String(format: "%.3f", value)
    }

    /// Short runs print running numbers on every decoded token; longer runs
    /// (local-submit's 1023 steps) keep the historical 8/64-step cadence so the
    /// log stays readable.
    static func localIterateDecodeProgressInterval(
        totalDecodeSteps: Int,
        timingRepeats: Int
    ) -> Int {
        if totalDecodeSteps <= 32 {
            return 1
        }
        return timingRepeats > 1 ? 64 : 8
    }

    /// Projects the final charged decode seconds-per-token from the phase so
    /// far: seed prefill/setup already charged, plus remaining steps assumed to
    /// run at the mean per-step rate observed so far. Converges to the exact
    /// final metric at the last step.
    static func localIterateProjectedDecodeSecondsPerToken(
        chargedSecondsSoFar: Double,
        stepOnlySecondsSoFar: Double,
        decodedTokens: Int,
        totalDecodeTokens: Int
    ) -> Double {
        guard decodedTokens > 0, totalDecodeTokens > 0 else {
            return 0
        }
        guard decodedTokens < totalDecodeTokens else {
            return chargedSecondsSoFar / Double(totalDecodeTokens)
        }
        let meanStepSeconds = stepOnlySecondsSoFar / Double(decodedTokens)
        let remainingTokens = Double(totalDecodeTokens - decodedTokens)
        return (chargedSecondsSoFar + remainingTokens * meanStepSeconds) / Double(totalDecodeTokens)
    }

    /// Builds the per-step live status suffix: last step latency, mean step
    /// latency, ETA for the remaining decode tokens, projected charged decode
    /// seconds-per-token, and -- once prefill has been measured -- the
    /// projected decode speedup and estimated score under the official
    /// formula. The RAM-resident dense runtime has no expert-streaming
    /// machinery, so this no longer reports expert bandwidth/hit-rate fields.
    static func localIterateLiveDecodeStatus(
        lastStepSeconds: Double,
        chargedSecondsSoFar: Double,
        stepOnlySecondsSoFar: Double,
        decodedTokens: Int,
        totalDecodeTokens: Int,
        prefillSecondsPerToken: Double?
    ) -> String {
        guard decodedTokens > 0 else {
            return ""
        }
        let meanStepSeconds = stepOnlySecondsSoFar / Double(decodedTokens)
        let remainingTokens = totalDecodeTokens - decodedTokens
        let projected = localIterateProjectedDecodeSecondsPerToken(
            chargedSecondsSoFar: chargedSecondsSoFar,
            stepOnlySecondsSoFar: stepOnlySecondsSoFar,
            decodedTokens: decodedTokens,
            totalDecodeTokens: totalDecodeTokens
        )
        var status = "last_step_seconds=\(formatDouble(lastStepSeconds))"
            + " mean_step_seconds=\(formatDouble(meanStepSeconds))"
        if remainingTokens > 0 {
            status += " decode_eta_seconds=\(formatSeconds(Double(remainingTokens) * meanStepSeconds))"
        }
        status += " projected_decode_seconds_per_token=\(formatDouble(projected))"
        if projected > 0 {
            let decodeSpeedup = BenchmarkScore.speedup(
                baselineSecondsPerToken: MLXFastConstants.officialBaselineDecodeSecondsPerToken,
                candidateSecondsPerToken: projected
            )
            status += " projected_decode_speedup=\(formatRatio(decodeSpeedup))"
            if let prefillSecondsPerToken, prefillSecondsPerToken > 0 {
                let estScore = BenchmarkScore.score(
                    decodeSecondsPerToken: projected,
                    prefillSecondsPerToken: prefillSecondsPerToken
                )
                if estScore.isFinite {
                    status += " projected_score=\(formatRatio(estScore))"
                }
            }
        }
        return status
    }

    static func localIteratePrefillStatus(
        elapsedSeconds: Double,
        promptTokens: Int
    ) -> String {
        guard promptTokens > 0, elapsedSeconds > 0 else {
            return "seconds=\(formatSeconds(elapsedSeconds))"
        }
        let secondsPerToken = elapsedSeconds / Double(promptTokens)
        let speedup = BenchmarkScore.speedup(
            baselineSecondsPerToken: MLXFastConstants.officialBaselinePrefillSecondsPerToken,
            candidateSecondsPerToken: secondsPerToken
        )
        return "seconds=\(formatSeconds(elapsedSeconds))"
            + " seconds_per_token=\(formatDouble(secondsPerToken))"
            + " prefill_speedup=\(formatRatio(speedup))"
    }

    /// One-sentence caveat printed wherever a local public-gate token mismatch
    /// is reported: the checked-in goldens are greedy continuations captured on
    /// the ranked M5 hardware, so on other Apple Silicon generations near-tie
    /// argmaxes can diverge deterministically for a perfectly correct build.
    /// Messaging only -- the gate itself still fails.
    public static let nonM5GoldenMismatchCaveat =
        "note: the public goldens are M5-generated, so on non-M5 Apple Silicon "
        + "a deterministic near-tie token mismatch is expected for a correct build "
        + "(see \"Correctness fixtures are M5-generated\" in README.md); "
        + "the ranked M5 runner is the source of truth"

    static func reportFirstTokenMismatch(
        _ progress: ((String) -> Void)?,
        modeName: String,
        checkedStep: Int
    ) {
        progress?(
            "\(modeName) correctness FAIL first token mismatch at checked_step=\(checkedStep) "
                + "(run continues; expected/actual tokens are in the score JSON)"
        )
        progress?("\(modeName) \(nonM5GoldenMismatchCaveat)")
    }

    /// Emits a heartbeat line every `intervalSeconds` while a long silent phase
    /// (weights digest, measured prefill forward, decode seed prefill) is
    /// blocking, so the console shows liveness before the phase completes.
    /// Callers must `cancel()` the returned timer when the phase ends.
    static func startPhaseHeartbeat(
        label: String,
        intervalSeconds: Double = 10,
        progress: ((String) -> Void)?
    ) -> DispatchSourceTimer? {
        guard let progress, intervalSeconds > 0 else {
            return nil
        }
        let phaseStart = DispatchTime.now().uptimeNanoseconds
        let timer = DispatchSource.makeTimerSource(
            queue: DispatchQueue(label: "mlxfast.local-iterate.heartbeat")
        )
        timer.schedule(
            deadline: .now() + intervalSeconds,
            repeating: intervalSeconds,
            leeway: .milliseconds(250)
        )
        timer.setEventHandler {
            progress("\(label) still running phase_seconds=\(formatSeconds(secondsSince(phaseStart)))")
        }
        timer.resume()
        return timer
    }

    static func emitLocalIterateSummary(
        modeName: String,
        timing: LocalIterateTimingResult,
        progress: (String) -> Void
    ) {
        let prefillSpeedup = BenchmarkScore.speedup(
            baselineSecondsPerToken: MLXFastConstants.officialBaselinePrefillSecondsPerToken,
            candidateSecondsPerToken: timing.prefillSecondsPerToken
        )
        let decodeSpeedup = BenchmarkScore.speedup(
            baselineSecondsPerToken: MLXFastConstants.officialBaselineDecodeSecondsPerToken,
            candidateSecondsPerToken: timing.decode.secondsPerToken
        )
        let estScore = BenchmarkScore.score(
            decodeSecondsPerToken: timing.decode.secondsPerToken,
            prefillSecondsPerToken: timing.prefillSecondsPerToken
        )
        progress(
            "\(modeName) summary prefill_seconds_per_token=\(formatDouble(timing.prefillSecondsPerToken)) "
                + "prefill_speedup=\(formatRatio(prefillSpeedup))"
        )
        progress(
            "\(modeName) summary decode_seconds_per_token=\(formatDouble(timing.decode.secondsPerToken)) "
                + "decode_speedup=\(formatRatio(decodeSpeedup))"
        )
        if estScore.isFinite {
            progress(
                "\(modeName) summary est_score=\(formatRatio(estScore)) "
                    + "(decode_speedup^0.75 * prefill_speedup^0.25 vs official baseline; "
                    + "published as this run's local estimated score, not a ranked score)"
            )
        }
        progress(
            "\(modeName) summary decode_bandwidth_gb_per_token=\(formatDouble(timing.decode.bandwidthGBPerToken)) "
                + "peak_ram_gb=\(formatRatio(timing.peakRamGB))"
        )
        emitLocalAcceptanceBandNotice(
            modeName: modeName,
            decodeSecondsPerToken: timing.decode.secondsPerToken,
            prefillSecondsPerToken: timing.prefillSecondsPerToken,
            progress: progress
        )
    }

    /// Opt-in escape hatch for the documented non-M5 golden divergence.
    ///
    /// The public fixtures are greedy continuations captured on the ranked M5
    /// box, so a near-tie argmax diverges deterministically on other Apple
    /// Silicon generations even for a correct build. Without this, one such
    /// token in 128 ends the local edit loop: the payload carries
    /// `score: null`, which the Yukon participant CLI rejects outright, so the
    /// contestant gets no estimate and no baseline delta -- on a machine where
    /// nothing is actually wrong. With it set, the run publishes its timing
    /// estimate and stays honest about what was not verified
    /// (`passed_correctness: false`, the diverging tokens, and an explanatory
    /// `metrics.error`).
    ///
    /// Local-only by construction: this file is never on the ranked path,
    /// which routes to QwenRuntime.benchmark and evaluates the hidden
    /// goldens. The M5 runner remains the fidelity authority.
    static let localGoldenDriftEnvironmentName = "MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT"

    static func localGoldenDriftAllowed() -> Bool {
        ProcessInfo.processInfo.environment[localGoldenDriftEnvironmentName] == "1"
    }

    static func emitLocalGoldenDriftOverrideNotice(
        modeName: String,
        correctness: CorrectnessReport,
        progress: (String) -> Void
    ) {
        let step = correctness.firstFailingStep.map(String.init) ?? "unknown"
        let expected = correctness.expectedToken.map(String.init) ?? "unknown"
        let actual = correctness.actualToken.map(String.init) ?? "unknown"
        progress(
            "\(modeName) WARNING golden drift ACCEPTED via \(localGoldenDriftEnvironmentName)=1: "
                + "checked_step=\(step) expected_token=\(expected) actual_token=\(actual)"
        )
        progress(
            "\(modeName) WARNING this run's tokens are NOT verified. The estimate below is usable "
                + "for A/B timing on this machine ONLY if unmodified main diverges identically "
                + "here -- confirm that first, or you are timing a genuinely broken build. The "
                + "ranked M5 runner enforces exact tokens and is the fidelity authority."
        )
    }

    /// The ranked path applies a two-sided acceptance band on top of the
    /// speedup floors: a single submission's GAIN is capped at about 5%, and
    /// larger wins must be chunked across submissions. Local modes publish an
    /// unbounded estimate, so without this notice the edit loop silently
    /// rewards exceeding a bound that then fails the ranked run with a category
    /// the participant cannot act on.
    ///
    /// Only the too-fast edge is reported. The slow edge is already covered by
    /// the published 0.95 floors, and the reference here is the pinned M5
    /// baseline rather than a same-session paired measurement -- so on any
    /// machine slower than the ranked box a two-sided check would fire on
    /// literally every run and train participants to ignore it.
    static func emitLocalAcceptanceBandNotice(
        modeName: String,
        decodeSecondsPerToken: Double,
        prefillSecondsPerToken: Double,
        progress: (String) -> Void
    ) {
        let decodeFastEdge = MLXFastConstants.officialBaselineDecodeSecondsPerToken
            * (1.0 - MLXFastConstants.decodeBandDownTolerance)
        let prefillFastEdge = MLXFastConstants.officialBaselinePrefillSecondsPerToken
            * (1.0 - MLXFastConstants.prefillBandDownTolerance)
        var axes: [String] = []
        if decodeSecondsPerToken.isFinite, decodeSecondsPerToken > 0,
           decodeSecondsPerToken < decodeFastEdge
        {
            axes.append("decode")
        }
        if prefillSecondsPerToken.isFinite, prefillSecondsPerToken > 0,
           prefillSecondsPerToken < prefillFastEdge
        {
            axes.append("prefill")
        }
        guard !axes.isEmpty else {
            return
        }
        progress(
            "\(modeName) WARNING \(axes.joined(separator: "+")) is more than "
                + "\(Int((MLXFastConstants.decodeBandDownTolerance * 100).rounded()))% faster than "
                + "the pinned baseline, which is OUTSIDE the ranked acceptance band"
        )
        progress(
            "\(modeName) WARNING the ranked run requires decode_speedup in [0.980, 1.053] and "
                + "prefill_speedup in [0.952, 1.053] on top of the 0.95 floors, so a gain this "
                + "large must be CHUNKED across submissions or the ranked run fails with "
                + "failure_category=acceptance_band_failed. Local modes never enforce it, and this "
                + "estimate uses the pinned baseline constants rather than the M5 paired baseline, "
                + "so on a fast local box it can fire spuriously. See "
                + "docs/benchmark-window-freeze.md."
        )
    }

    struct LocalIterateTimingResult {
        let correctness: CorrectnessReport
        let prefillSecondsPerToken: Double
        let decode: DecodeMeasurement
        let expertStats: ExpertStreamingStats
        let peakRamGB: Double
    }

    static func runLocalPhaseCoolGate(
        phase: String,
        progress: ((String) -> Void)? = nil
    ) throws {
        guard let helper = ProcessInfo.processInfo.environment["MLXFAST_LOCAL_COOL_GATE_HELPER"],
              !helper.isEmpty
        else {
            return
        }
        guard FileManager.default.isExecutableFile(atPath: helper) else {
            throw MLXFastError.invalidInput("local GPU cool-down helper is not executable: \(helper)")
        }

        progress?("local thermal gate start phase=\(phase)")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: helper)
        process.arguments = ["--local-cool-gate-only"]
        var environment = ProcessInfo.processInfo.environment
        environment["MLXFAST_LOCAL_COOL_GATE_PHASE"] = phase
        process.environment = environment
        try process.run()
        process.waitUntilExit()
        guard process.terminationReason == .exit, process.terminationStatus == 0 else {
            throw MLXFastError.invalidInput(
                "local GPU cool-down gate failed for \(phase) with status \(process.terminationStatus)"
            )
        }
        progress?("local thermal gate complete phase=\(phase)")
    }

    #if !MLXFAST_TRUSTED_HARNESS
        static func runLocalIterateCheckedTiming(
        weightsPath: String,
        testCase: GoldenCase,
        goldenHash: String,
        decodeSteps: Int,
        timingRepeats: Int,
        modeName: String,
        progress: ((String) -> Void)?
    ) throws -> LocalIterateTimingResult {
        let config = try Qwen35Config.load(from: weightsPath)
        let loader = try Qwen35WeightLoader(weightsPath: weightsPath)
        let weightCache = Qwen35RuntimeWeightCache(loader: loader, config: config)
        let model = try weightCache.requireLibraryModel()
        guard !testCase.promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("\(modeName) prompt must not be empty")
        }
        guard testCase.expectedTokens.count > decodeSteps else {
            throw MLXFastError.invalidInput(
                "\(testCase.name).expected_tokens has \(testCase.expectedTokens.count) tokens; "
                    + "need at least \(decodeSteps + 1)"
            )
        }

        var totalPrefillSeconds = 0.0
        var totalDecodeSeconds = 0.0
        var totalStepOnlySeconds = 0.0
        var latestStats = ExpertStreamingStats.zero
        var failureStep: Int?
        var failureExpected: Int?
        var failureActual: Int?
        let checkedStepsPerPass = decodeSteps + 2
        let totalDecodeSteps = decodeSteps * timingRepeats
        let decodeProgressInterval = localIterateDecodeProgressInterval(
            totalDecodeSteps: totalDecodeSteps,
            timingRepeats: timingRepeats
        )
        let expectedSeedToken = testCase.expectedTokens[0]
        let expectedDecodeTokens = Array(testCase.expectedTokens.dropFirst().prefix(decodeSteps))

        for repeatIndex in 0..<timingRepeats {
            if timingRepeats > 1 {
                progress?("\(modeName) checked pass \(repeatIndex + 1)/\(timingRepeats) start")
            }
            try runLocalPhaseCoolGate(phase: "prefill", progress: progress)
            progress?("\(modeName) prefill measured start prompt_tokens=\(testCase.promptTokens.count)")
            let prefillHeartbeat = startPhaseHeartbeat(
                label: "\(modeName) prefill measured",
                progress: progress
            )
            defer {
                prefillHeartbeat?.cancel()
            }
            let prefillCache = model.newCache(parameters: nil)
            let prefillStart = DispatchTime.now().uptimeNanoseconds
            let prefillLogits = try qwenLogits(
                inputIDs: inputIDsArray(testCase.promptTokens),
                model: model,
                cache: prefillCache,
                positionOffset: 0
            )
            eval(prefillLogits)
            let prefillToken = try QwenCorrectness.greedyToken(from: prefillLogits)
            let prefillElapsed = secondsSince(prefillStart)
            prefillHeartbeat?.cancel()
            totalPrefillSeconds += prefillElapsed
            Memory.clearCache()
            latestStats = QwenRuntime.expertStats(from: weightCache)
            if failureStep == nil, prefillToken != expectedSeedToken {
                failureStep = repeatIndex * checkedStepsPerPass
                failureExpected = expectedSeedToken
                failureActual = prefillToken
                reportFirstTokenMismatch(progress, modeName: modeName, checkedStep: failureStep! + 1)
            }
            progress?(
                "\(modeName) prefill measured complete "
                    + localIteratePrefillStatus(
                        elapsedSeconds: prefillElapsed,
                        promptTokens: testCase.promptTokens.count
                    )
            )
            let runningPrefillSecondsPerToken =
                totalPrefillSeconds / Double(testCase.promptTokens.count * (repeatIndex + 1))

            // Match official benchmark decode semantics: charge prompt-specific
            // setup, seed prefill, cache materialization, and checked token steps
            // to decode_seconds_per_token so local signals cannot hide work that
            // the official benchmark charges.
            try runLocalPhaseCoolGate(phase: "decode", progress: progress)
            let decodePhaseStart = DispatchTime.now().uptimeNanoseconds
            progress?("\(modeName) decode measured start tokens=\(decodeSteps) includes_seed_prefill=true")
            let seedHeartbeat = startPhaseHeartbeat(
                label: "\(modeName) decode seed prefill",
                progress: progress
            )
            defer {
                seedHeartbeat?.cancel()
            }
            let cache = model.newCache(parameters: nil)
            var logits = try qwenLogits(
                inputIDs: inputIDsArray(testCase.promptTokens),
                model: model,
                cache: cache,
                positionOffset: 0
            )
            var actualToken = try QwenCorrectness.greedyToken(from: logits)
            latestStats = QwenRuntime.expertStats(from: weightCache)
            if failureStep == nil, actualToken != expectedSeedToken {
                failureStep = repeatIndex * checkedStepsPerPass + 1
                failureExpected = expectedSeedToken
                failureActual = actualToken
                reportFirstTokenMismatch(progress, modeName: modeName, checkedStep: failureStep! + 1)
            }
            materializeQwenCacheState(cache)
            seedHeartbeat?.cancel()
            progress?(
                "\(modeName) decode seed prefill complete "
                    + "seconds=\(formatSeconds(secondsSince(decodePhaseStart))) (charged to decode)"
            )
            for decodedStep in 0..<decodeSteps {
                let previousToken = decodedStep == 0 ? expectedSeedToken : expectedDecodeTokens[decodedStep - 1]
                let stepStart = DispatchTime.now().uptimeNanoseconds
                logits = try qwenLogits(
                    inputIDs: inputIDsArray([previousToken]),
                    model: model,
                    cache: cache,
                    positionOffset: testCase.promptTokens.count + decodedStep
                )
                actualToken = try QwenCorrectness.greedyToken(from: logits)
                let expectedToken = expectedDecodeTokens[decodedStep]
                if failureStep == nil, actualToken != expectedToken {
                    failureStep = repeatIndex * checkedStepsPerPass + decodedStep + 2
                    failureExpected = expectedToken
                    failureActual = actualToken
                    reportFirstTokenMismatch(progress, modeName: modeName, checkedStep: failureStep! + 1)
                }
                let stepElapsed = secondsSince(stepStart)
                totalStepOnlySeconds += stepElapsed
                latestStats = QwenRuntime.expertStats(from: weightCache)
                reportProgress(
                    step: repeatIndex * decodeSteps + decodedStep + 1,
                    total: totalDecodeSteps,
                    intervalSteps: decodeProgressInterval,
                    progress: { step, total in
                        progress?(
                            "\(modeName) checked decode \(step)/\(total) tokens "
                                + localIterateLiveDecodeStatus(
                                    lastStepSeconds: stepElapsed,
                                    chargedSecondsSoFar: totalDecodeSeconds
                                        + secondsSince(decodePhaseStart),
                                    stepOnlySecondsSoFar: totalStepOnlySeconds,
                                    decodedTokens: step,
                                    totalDecodeTokens: total,
                                    prefillSecondsPerToken: runningPrefillSecondsPerToken
                                )
                        )
                    }
                )
            }
            totalDecodeSeconds += secondsSince(decodePhaseStart)
        }

        let bandwidth = (gbPerToken: 0.0, source: QwenRuntime.bandwidthSource)
        latestStats = QwenRuntime.expertStats(from: weightCache)
        let correctness = localIterateCorrectnessReport(
            passed: failureStep == nil,
            checkedSteps: failureStep.map { $0 + 1 } ?? checkedStepsPerPass * timingRepeats,
            caseCount: timingRepeats,
            firstFailingStep: failureStep,
            expectedToken: failureExpected,
            actualToken: failureActual,
            goldenHash: goldenHash,
            expertStats: latestStats,
            error: failureStep == nil ? "" : "\(modeName) teacher-forced token mismatch",
            modeName: modeName
        )
        return LocalIterateTimingResult(
            correctness: correctness,
            prefillSecondsPerToken: totalPrefillSeconds / Double(testCase.promptTokens.count * timingRepeats),
            decode: DecodeMeasurement(
                secondsPerToken: totalDecodeSeconds / Double(totalDecodeSteps),
                bandwidthGBPerToken: bandwidth.gbPerToken,
                bandwidthSource: bandwidth.source
            ),
            expertStats: latestStats,
            peakRamGB: Double(Memory.peakMemory) / Double(1 << 30)
        )
        }
    #endif

    static func runLocalIterateCheckedTimingWithWorker(
        weightsPath: String,
        testCase: GoldenCase,
        goldenHash: String,
        decodeSteps: Int,
        timingRepeats: Int,
        modeName: String,
        workerOptions: RuntimeWorkerOptions,
        progress: ((String) -> Void)?
    ) throws -> LocalIterateTimingResult {
        guard !testCase.promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("\(modeName) prompt must not be empty")
        }
        guard testCase.expectedTokens.count > decodeSteps else {
            throw MLXFastError.invalidInput(
                "\(testCase.name).expected_tokens has \(testCase.expectedTokens.count) tokens; "
                    + "need at least \(decodeSteps + 1)"
            )
        }

        var totalPrefillSeconds = 0.0
        var totalDecodeSeconds = 0.0
        var totalStepOnlySeconds = 0.0
        var latestStats = ExpertStreamingStats.zero
        var peakRamGB = 0.0
        var failureStep: Int?
        var failureExpected: Int?
        var failureActual: Int?
        let checkedStepsPerPass = decodeSteps + 2
        let totalDecodeSteps = decodeSteps * timingRepeats
        let decodeProgressInterval = localIterateDecodeProgressInterval(
            totalDecodeSteps: totalDecodeSteps,
            timingRepeats: timingRepeats
        )
        let expectedSeedToken = testCase.expectedTokens[0]
        let expectedDecodeTokens = Array(testCase.expectedTokens.dropFirst().prefix(decodeSteps))

        for repeatIndex in 0..<timingRepeats {
            if timingRepeats > 1 {
                progress?("\(modeName) checked pass \(repeatIndex + 1)/\(timingRepeats) start")
            }
            do {
                let prefillWorker = try RuntimeWorkerClient(options: workerOptions, weightsPath: weightsPath)
                defer {
                    prefillWorker.close()
                }
                try runLocalPhaseCoolGate(phase: "prefill", progress: progress)
                let prefillStart = DispatchTime.now().uptimeNanoseconds
                progress?("\(modeName) prefill measured start prompt_tokens=\(testCase.promptTokens.count)")
                let prefillHeartbeat = startPhaseHeartbeat(
                    label: "\(modeName) prefill measured",
                    progress: progress
                )
                defer {
                    prefillHeartbeat?.cancel()
                }
                let response = try prefillWorker.prefill(promptTokens: testCase.promptTokens)
                let prefillElapsed = secondsSince(prefillStart)
                prefillHeartbeat?.cancel()
                totalPrefillSeconds += prefillElapsed
                guard let prefillToken = response.token else {
                    throw MLXFastError.invalidInput("runtime worker \(modeName) prefill response missing token")
                }
                if failureStep == nil, prefillToken != expectedSeedToken {
                    failureStep = repeatIndex * checkedStepsPerPass
                    failureExpected = expectedSeedToken
                    failureActual = prefillToken
                    reportFirstTokenMismatch(progress, modeName: modeName, checkedStep: failureStep! + 1)
                }
                let diagnostics = try prefillWorker.phaseDiagnostics()
                latestStats = diagnostics.expertStats ?? latestStats
                peakRamGB = max(peakRamGB, diagnostics.peakRamGB ?? 0)
                progress?(
                    "\(modeName) prefill measured complete "
                        + localIteratePrefillStatus(
                            elapsedSeconds: prefillElapsed,
                            promptTokens: testCase.promptTokens.count
                        )
                )
            }
            let runningPrefillSecondsPerToken =
                totalPrefillSeconds / Double(testCase.promptTokens.count * (repeatIndex + 1))

            // Use the same decode protocol as the official benchmark and start
            // the parent timer before decode_begin so seed prefill/setup is
            // charged to decode_seconds_per_token.
            let decodeWorker = try RuntimeWorkerClient(options: workerOptions, weightsPath: weightsPath)
            defer {
                decodeWorker.close()
            }
            try runLocalPhaseCoolGate(phase: "decode", progress: progress)
            let decodePhaseStart = DispatchTime.now().uptimeNanoseconds
            progress?("\(modeName) decode measured start tokens=\(decodeSteps) includes_seed_prefill=true")
            let seedHeartbeat = startPhaseHeartbeat(
                label: "\(modeName) decode seed prefill",
                progress: progress
            )
            // Cancel eagerly after the blocking call; the defer only covers the
            // throwing path (defers run at end of the loop iteration, which
            // would otherwise leave the heartbeat firing through decode steps).
            defer {
                seedHeartbeat?.cancel()
            }
            var response = try decodeWorker.beginDecode(seedTokens: testCase.promptTokens)
            seedHeartbeat?.cancel()
            progress?(
                "\(modeName) decode seed prefill complete "
                    + "seconds=\(formatSeconds(secondsSince(decodePhaseStart))) (charged to decode)"
            )
            guard let seedToken = response.seedToken else {
                throw MLXFastError.invalidInput("runtime worker \(modeName) decode_begin response missing seed token")
            }
            if failureStep == nil, seedToken != expectedSeedToken {
                failureStep = repeatIndex * checkedStepsPerPass + 1
                failureExpected = expectedSeedToken
                failureActual = seedToken
                reportFirstTokenMismatch(progress, modeName: modeName, checkedStep: failureStep! + 1)
            }

            for decodedStep in 0..<decodeSteps {
                let inputToken = decodedStep == 0 ? expectedSeedToken : expectedDecodeTokens[decodedStep - 1]
                let stepStart = DispatchTime.now().uptimeNanoseconds
                response = try decodeWorker.decodeStep(inputToken: inputToken)
                let stepElapsed = secondsSince(stepStart)
                totalStepOnlySeconds += stepElapsed
                guard let token = response.token else {
                    throw MLXFastError.invalidInput("runtime worker \(modeName) decode response missing token")
                }
                let expectedToken = expectedDecodeTokens[decodedStep]
                if failureStep == nil, token != expectedToken {
                    failureStep = repeatIndex * checkedStepsPerPass + decodedStep + 2
                    failureExpected = expectedToken
                    failureActual = token
                    reportFirstTokenMismatch(progress, modeName: modeName, checkedStep: failureStep! + 1)
                }
                reportProgress(
                    step: repeatIndex * decodeSteps + decodedStep + 1,
                    total: totalDecodeSteps,
                    intervalSteps: decodeProgressInterval,
                    progress: { step, total in
                        progress?(
                            "\(modeName) checked decode \(step)/\(total) tokens "
                                + localIterateLiveDecodeStatus(
                                    lastStepSeconds: stepElapsed,
                                    chargedSecondsSoFar: totalDecodeSeconds
                                        + secondsSince(decodePhaseStart),
                                    stepOnlySecondsSoFar: totalStepOnlySeconds,
                                    decodedTokens: step,
                                    totalDecodeTokens: total,
                                    prefillSecondsPerToken: runningPrefillSecondsPerToken
                                )
                        )
                    }
                )
            }
            totalDecodeSeconds += secondsSince(decodePhaseStart)
            let diagnostics = try decodeWorker.phaseDiagnostics()
            latestStats = diagnostics.expertStats ?? latestStats
            peakRamGB = max(peakRamGB, diagnostics.peakRamGB ?? 0)
            if timingRepeats > 1 {
                progress?("\(modeName) checked pass \(repeatIndex + 1)/\(timingRepeats) complete")
            }
        }
        let bandwidth = (gbPerToken: 0.0, source: QwenRuntime.bandwidthSource)
        let correctness = localIterateCorrectnessReport(
            passed: failureStep == nil,
            checkedSteps: failureStep.map { $0 + 1 } ?? checkedStepsPerPass * timingRepeats,
            caseCount: timingRepeats,
            firstFailingStep: failureStep,
            expectedToken: failureExpected,
            actualToken: failureActual,
            goldenHash: goldenHash,
            expertStats: latestStats,
            error: failureStep == nil ? "" : "\(modeName) teacher-forced token mismatch",
            modeName: modeName
        )
        return LocalIterateTimingResult(
            correctness: correctness,
            prefillSecondsPerToken: totalPrefillSeconds / Double(testCase.promptTokens.count * timingRepeats),
            decode: DecodeMeasurement(
                secondsPerToken: totalDecodeSeconds / Double(totalDecodeSteps),
                bandwidthGBPerToken: bandwidth.gbPerToken,
                bandwidthSource: bandwidth.source
            ),
            expertStats: latestStats,
            peakRamGB: peakRamGB
        )
    }

    static func localIterateCorrectnessReport(
        passed: Bool,
        checkedSteps: Int,
        caseCount: Int,
        firstFailingStep: Int?,
        expectedToken: Int?,
        actualToken: Int?,
        goldenHash: String,
        expertStats: ExpertStreamingStats,
        error: String,
        modeName: String
    ) -> CorrectnessReport {
        CorrectnessReport(
            passed: passed,
            checkedSteps: checkedSteps,
            caseCount: caseCount,
            expertCacheHits: expertStats.cacheHits,
            expertCacheMisses: expertStats.cacheMisses,
            expertCacheEvictions: expertStats.cacheEvictions,
            expertBytesRead: expertStats.bytesRead,
            expertReadSeconds: expertStats.readSeconds,
            expertPeakCachedTensors: expertStats.peakCachedTensors,
            expertHitRate: expertStats.hitRate,
            firstFailingCase: firstFailingStep == nil ? nil : modeName,
            firstFailingStep: firstFailingStep,
            expectedToken: expectedToken,
            actualToken: actualToken,
            goldenHash: goldenHash,
            error: error
        )
    }

    /// LOCAL MODES ONLY: attach the finite estimated score to a FAILED local
    /// payload whose timings were still measured (the common case: the public
    /// gate's teacher-forced check failed -- e.g. the deterministic non-M5
    /// near-tie divergence -- but prefill/decode completed). The Yukon
    /// participant CLI (`mlxfast run`) validates the contract scorePath as
    /// `{ "score": <finite number>, ... }`, so a `score: null` failure payload
    /// makes every local run on non-M5 hardware error at the contract layer
    /// instead of reporting the measured timings. `passed` stays false and
    /// every failure field (error, passed_correctness, first_failing_step,
    /// expected/actual token) is preserved, so the failure remains
    /// unmistakable; only the score becomes the same directional
    /// decode_speedup^0.75 * prefill_speedup^0.25 estimate that passing local
    /// runs publish. Official/ranked scoring is untouched: this is reached
    /// exclusively from localIterate's failure path, and QwenRuntime.benchmark
    /// keeps publishing score: null on failure (see
    /// rankedScoreSemanticsAreUnchangedByLocalEstimatedScore).
    static func localModeFailedPayloadWithEstimatedScore(
        _ payload: ScorePayload,
        modeName: String,
        progress: ((String) -> Void)? = nil
    ) -> ScorePayload {
        guard payload.score == nil, !payload.passed else {
            return payload
        }
        let decodeSecondsPerToken = payload.metrics.decodeSecondsPerToken
        let prefillSecondsPerToken = payload.metrics.prefillSecondsPerToken
        guard decodeSecondsPerToken.isFinite,
              decodeSecondsPerToken > 0,
              prefillSecondsPerToken.isFinite,
              prefillSecondsPerToken > 0
        else {
            return payload
        }
        let estimatedScore = BenchmarkScore.score(
            decodeSecondsPerToken: decodeSecondsPerToken,
            prefillSecondsPerToken: prefillSecondsPerToken
        )
        guard estimatedScore.isFinite, estimatedScore > 0 else {
            return payload
        }
        progress?(
            "\(modeName) publishing est_score=\(formatRatio(estimatedScore)) with passed=false "
                + "(local modes always carry the numeric local estimate for the CLI contract; "
                + "the run still failed -- see the error field in the score JSON)"
        )
        return ScorePayload(
            score: estimatedScore,
            passed: false,
            metrics: payload.metrics
        )
    }

    static func localIterateScore(
        peakRamGB: Double,
        bandwidthGBPerToken: Double,
        decodeSecondsPerToken: Double,
        prefillSecondsPerToken: Double,
        wallSeconds: Double,
        validationSeconds: Double,
        correctnessSeconds: Double,
        timedSeconds: Double,
        correctness: CorrectnessReport,
        expertStats: ExpertStreamingStats,
        bandwidthSource: String,
        weightsDigest: DirectoryDigest?,
        runtime: String
    ) -> ScorePayload {
        // Local modes publish the ESTIMATED score: the same decode_speedup^0.75
        // * prefill_speedup^0.25 estimate against the pinned officialBaseline*
        // constants that the live progress stream and shell summary already
        // print. The Yukon participant CLI (`mlxfast run`) executes
        // benchmarkCommand and then validates the contract scorePath as
        // `{ "score": <finite number>, ... }` -- `score: null` is rejected --
        // so local modes must carry a numeric score to be consumable at all.
        // This stays a directional local signal, never an official ranking:
        // metrics.runtime ("swift-local-iterate"/"swift-local-submit") marks
        // the payload as local-mode, submit never uploads local score files
        // (only editablePaths), the ranked pipeline never runs this code path
        // (benchmark.yml and measure-job.sh invoke --official, which routes to
        // QwenRuntime.benchmark), and the ranked artifact validator rejects
        // this shape (runtime must be "swift" with the hidden-gate fields
        // populated). The official score remains the paired-ratio overlay
        // computed by the trusted ranked workflow, exactly as before.
        let estimatedScore = BenchmarkScore.score(
            decodeSecondsPerToken: decodeSecondsPerToken,
            prefillSecondsPerToken: prefillSecondsPerToken
        )
        guard decodeSecondsPerToken.isFinite,
              decodeSecondsPerToken > 0,
              prefillSecondsPerToken.isFinite,
              prefillSecondsPerToken > 0,
              estimatedScore.isFinite,
              estimatedScore > 0
        else {
            return failedScore(
                error: "local estimated score is invalid: timing metrics must be finite and positive",
                correctness: correctness,
                passedCorrectness: correctness.passed,
                expertStats: expertStats,
                weightsDigest: weightsDigest,
                benchmarkWallSeconds: finiteNonnegativeMetric(wallSeconds),
                preflightSeconds: finiteNonnegativeMetric(validationSeconds),
                correctnessSeconds: finiteNonnegativeMetric(correctnessSeconds),
                timedBenchmarkSeconds: finiteNonnegativeMetric(timedSeconds),
                processResidentMemoryGB: finiteNonnegativeMetric(currentResidentMemoryGB()),
                peakRamGB: finiteNonnegativeMetric(peakRamGB),
                bandwidthGBPerToken: finiteNonnegativeMetric(bandwidthGBPerToken),
                decodeSecondsPerToken: finiteNonnegativeMetric(decodeSecondsPerToken),
                prefillSecondsPerToken: finiteNonnegativeMetric(prefillSecondsPerToken),
                bandwidthSource: bandwidthSource,
                runtime: runtime
            )
        }
        return ScorePayload(
            score: estimatedScore,
            passed: true,
            metrics: ScoreMetrics(
                peakRamGB: peakRamGB,
                bandwidthGBPerToken: bandwidthGBPerToken,
                decodeSecondsPerToken: decodeSecondsPerToken,
                prefillSecondsPerToken: prefillSecondsPerToken,
                benchmarkWallSeconds: wallSeconds,
                preflightSeconds: validationSeconds,
                correctnessSeconds: correctnessSeconds,
                timedBenchmarkSeconds: timedSeconds,
                processResidentMemoryGB: currentResidentMemoryGB(),
                // Normally true (the caller returns early on a mismatch); false
                // only under MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT, where the payload
                // must stay self-describing about the tokens that diverged.
                passedCorrectness: correctness.passed,
                numLayers: MLXFastConstants.numHiddenLayers,
                checkedSteps: correctness.checkedSteps,
                caseCount: correctness.caseCount,
                expertCacheHits: expertStats.cacheHits,
                expertCacheMisses: expertStats.cacheMisses,
                expertCacheEvictions: expertStats.cacheEvictions,
                expertBytesRead: expertStats.bytesRead,
                expertReadSeconds: expertStats.readSeconds,
                expertPeakCachedTensors: expertStats.peakCachedTensors,
                expertHitRate: expertStats.hitRate,
                firstFailingLayer: nil,
                firstFailingCase: nil,
                firstFailingStep: correctness.firstFailingStep,
                expectedToken: correctness.expectedToken,
                actualToken: correctness.actualToken,
                maxAbsDiff: 0,
                goldenHash: correctness.goldenHash,
                bandwidthSource: bandwidthSource,
                error: correctness.passed
                    ? ""
                    : "golden drift accepted by \(localGoldenDriftEnvironmentName)=1: "
                        + "timings are usable, tokens are NOT verified",
                commit: commitIdentifier(),
                timestamp: ISO8601DateFormatter().string(from: Date()),
                harnessHash: harnessHash(),
                weightsHash: weightsDigest?.sha256 ?? "",
                weightsByteCount: weightsDigest?.byteCount ?? 0,
                weightsFileCount: weightsDigest?.fileCount ?? 0,
                runtime: runtime
            )
        )
    }
}

private func finiteNonnegativeMetric(_ value: Double) -> Double {
    value.isFinite && value >= 0 ? value : 0
}
