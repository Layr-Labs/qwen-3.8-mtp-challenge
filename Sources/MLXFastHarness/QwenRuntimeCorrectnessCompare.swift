import Foundation
import MLX
import MLXFastCore
import MLXFastModel

// QwenRuntime is split across QwenRuntime*.swift for auditability.
// Generated split; behavior identical to the original single file.

extension QwenRuntime {
    public static func traceCorrectness(_ options: CorrectnessTraceOptions) throws -> CorrectnessTraceReport {
        let golden = try loadGoldenFixture(from: options.goldenPath)
        let selectedCase: GoldenCase
        if let caseName = options.caseName, !caseName.isEmpty {
            guard let match = golden.cases.first(where: { $0.name == caseName }) else {
                throw MLXFastError.invalidInput("correctness golden does not contain case \(caseName)")
            }
            selectedCase = match
        } else {
            guard let first = golden.cases.first else {
                throw MLXFastError.invalidInput("correctness golden contains no cases")
            }
            selectedCase = first
        }

        let config = try Qwen35Config.load(from: options.weightsPath)
        let loader = try Qwen35WeightLoader(weightsPath: options.weightsPath)
        let weightCache = Qwen35RuntimeWeightCache(loader: loader, config: config)
        return try traceGreedyCached(
            testCase: selectedCase,
            step: options.step,
            topK: options.topK,
            weightCache: weightCache,
            goldenHash: golden.sha256
        )
    }

    struct WorkerCorrectnessResult {
        let comparison: CorrectnessTokenComparison
        let expertStats: ExpertStreamingStats
        let peakRamGB: Double
        let ttftSeconds: Double?
        let generatedTokens: [Int]?

        init(
            comparison: CorrectnessTokenComparison,
            expertStats: ExpertStreamingStats,
            peakRamGB: Double,
            ttftSeconds: Double? = nil,
            generatedTokens: [Int]? = nil
        ) {
            self.comparison = comparison
            self.expertStats = expertStats
            self.peakRamGB = peakRamGB
            self.ttftSeconds = ttftSeconds
            self.generatedTokens = generatedTokens
        }
    }

    static func compareTeacherForcedWithWorker(
        testCase: GoldenCase,
        worker: RuntimeWorkerClient,
        steps: Int = MLXFastConstants.correctnessSteps,
        progressIntervalSteps: Int = 0,
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> WorkerCorrectnessResult {
        guard !testCase.promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("teacher-forced correctness prompt must not be empty")
        }
        guard steps > 0 else {
            throw MLXFastError.invalidInput("teacher-forced correctness steps must be positive")
        }
        guard testCase.expectedTokens.count >= steps else {
            throw MLXFastError.invalidInput(
                "\(testCase.name).expected_tokens has \(testCase.expectedTokens.count) tokens; "
                    + "need at least \(steps)"
            )
        }

        var response = try worker.beginTeacherForcedCorrectness(promptTokens: testCase.promptTokens)
        var latestExpertStats = response.expertStats ?? .zero
        var peakRamGB = response.peakRamGB ?? 0
        for step in 0..<steps {
            if step > 0 {
                response = try worker.teacherForcedCorrectnessStep(previousToken: testCase.expectedTokens[step - 1])
                latestExpertStats = response.expertStats ?? latestExpertStats
                peakRamGB = max(peakRamGB, response.peakRamGB ?? 0)
            }
            guard let actualToken = response.token else {
                throw MLXFastError.invalidInput("runtime worker teacher-forced correctness response missing token")
            }
            let expectedToken = testCase.expectedTokens[step]
            // Same acceptance rule as the non-worker path (compareTeacherForcedCached):
            // exact match, or a true top-logit tie within correctnessLogitTieTolerance.
            // Cross-hardware Apple GPU/Metal builds break exact argmax near-ties
            // differently, and a golden generated on one Apple Silicon generation
            // can deterministically flip single near-tie steps on another (issue
            // #83). The worker's reported top logits are validated for internal
            // consistency first (top-1 must equal the returned token, sorted,
            // finite, deduped, capped) and discarded if malformed, so a protocol
            // violation degrades to the strict comparison, never to a wider
            // acceptance. The worker cannot target the tolerance at the golden
            // token: it answers before the expected token is ever sent to it, and
            // fabricating flat logits to widen every step is caught by the strict
            // free-run, behavior, and benchmark-oracle gates plus the static
            // review. Either way the next teacher-forced input stays the GOLDEN
            // token, so acceptance here never changes downstream state.
            if actualToken != expectedToken {
                let tieTopLogits = try? validatedWorkerTopLogits(
                    response.topLogits,
                    actualToken: actualToken
                )
                if !correctnessTokenAccepted(
                    expectedToken: expectedToken,
                    actualToken: actualToken,
                    topLogits: tieTopLogits
                ) {
                    return WorkerCorrectnessResult(
                        comparison: CorrectnessTokenComparison(
                            passed: false,
                            checkedSteps: step + 1,
                            firstFailingStep: step,
                            expectedToken: expectedToken,
                            actualToken: actualToken
                        ),
                        expertStats: latestExpertStats,
                        peakRamGB: peakRamGB
                    )
                }
            }
            reportProgress(
                step: step + 1,
                total: steps,
                intervalSteps: progressIntervalSteps,
                progress: progress
            )
        }

        return WorkerCorrectnessResult(
            comparison: CorrectnessTokenComparison(
                passed: true,
                checkedSteps: steps,
                firstFailingStep: nil,
                expectedToken: nil,
                actualToken: nil
            ),
            expertStats: latestExpertStats,
            peakRamGB: peakRamGB
        )
    }

    static func compareAnchorWithWorker(
        anchor: GoldenAnchorCase,
        worker: RuntimeWorkerClient
    ) throws -> WorkerCorrectnessResult {
        let response = try worker.beginTeacherForcedCorrectness(promptTokens: anchor.contextTokens)
        guard let actualToken = response.token else {
            throw MLXFastError.invalidInput("runtime worker anchor response missing token")
        }
        // Parity with compareAnchorCached: without the validated worker top
        // logits, the anchor's tie tolerance and max_expected_rank /
        // max_top_logit_delta acceptance fields were silently inert on the
        // worker path even when the golden carries them. Malformed logits
        // degrade to the strict comparison (see compareTeacherForcedWithWorker).
        return WorkerCorrectnessResult(
            comparison: compareAnchorToken(
                anchor: anchor,
                actualToken: actualToken,
                topLogits: try? validatedWorkerTopLogits(response.topLogits, actualToken: actualToken)
            ),
            expertStats: response.expertStats ?? .zero,
            peakRamGB: response.peakRamGB ?? 0
        )
    }

    static func compareFreeRunWithWorker(
        testCase: GoldenFreeRunCase,
        worker: RuntimeWorkerClient
    ) throws -> WorkerCorrectnessResult {
        let response = try worker.generateCorrectness(
            promptTokens: testCase.promptTokens,
            steps: testCase.expectedTokens.count
        )
        guard let generated = response.tokens else {
            throw MLXFastError.invalidInput("runtime worker free-run response missing tokens")
        }
        try requireGeneratedTokenCount(
            generated.count,
            expected: testCase.expectedTokens.count,
            label: "free-run"
        )
        return WorkerCorrectnessResult(
            comparison: compareFreeRunTokens(testCase: testCase, generated: generated),
            expertStats: response.expertStats ?? .zero,
            peakRamGB: response.peakRamGB ?? 0
        )
    }

    static func compareBehaviorWithWorker(
        testCase: GoldenBehaviorCase,
        worker: RuntimeWorkerClient
    ) throws -> WorkerCorrectnessResult {
        // Hidden GPQA TTFT is a timing gate, so measure it in the trusted parent
        // instead of trusting the submitted-code worker's reported seconds.
        let ttftStart = DispatchTime.now().uptimeNanoseconds
        let beginResponse = try worker.beginTeacherForcedCorrectness(promptTokens: testCase.promptTokens)
        let ttftSeconds = secondsSince(ttftStart)
        guard let firstToken = beginResponse.token else {
            throw MLXFastError.invalidInput("runtime worker behavior response missing token")
        }
        let usesSemanticJudge = behaviorUsesSemanticJudge(testCase)
        if !usesSemanticJudge {
            // Parity with compareBehaviorCached's maxNewTokens == 1 path: the
            // tie tolerance applies against each accepted first token instead
            // of being silently disabled on the worker path.
            let firstTokenComparison = compareBehaviorFirstToken(
                testCase: testCase,
                actualToken: firstToken,
                topLogits: try? validatedWorkerTopLogits(beginResponse.topLogits, actualToken: firstToken)
            )
            if !firstTokenComparison.passed {
                return WorkerCorrectnessResult(
                    comparison: firstTokenComparison,
                    expertStats: beginResponse.expertStats ?? .zero,
                    peakRamGB: beginResponse.peakRamGB ?? 0,
                    ttftSeconds: ttftSeconds,
                    generatedTokens: [firstToken]
                )
            }
        }

        var generated = [firstToken]
        generated.reserveCapacity(testCase.maxNewTokens)
        var expertStats = beginResponse.expertStats ?? .zero
        var peakRamGB = beginResponse.peakRamGB ?? 0
        while generated.count < testCase.maxNewTokens {
            let response = try worker.teacherForcedCorrectnessStep(previousToken: generated[generated.count - 1])
            guard let token = response.token else {
                throw MLXFastError.invalidInput("runtime worker behavior continuation response missing token")
            }
            generated.append(token)
            expertStats = response.expertStats ?? expertStats
            peakRamGB = max(peakRamGB, response.peakRamGB ?? 0)
        }

        let comparison: CorrectnessTokenComparison
        // Hidden semantic GPQA uses the external judge for answer validity.
        // Exact token checks on these prompts are brittle across Apple
        // Silicon/MLX versions, including the first generated token.
        if usesSemanticJudge || testCase.acceptedTokenSequences.allSatisfy({ $0.count <= 1 }) {
            comparison = CorrectnessTokenComparison(
                passed: true,
                checkedSteps: testCase.maxNewTokens,
                firstFailingStep: nil,
                expectedToken: nil,
                actualToken: nil
            )
        } else {
            comparison = compareBehaviorTokens(testCase: testCase, generated: generated)
        }
        return WorkerCorrectnessResult(
            comparison: comparison,
            expertStats: expertStats,
            peakRamGB: peakRamGB,
            ttftSeconds: ttftSeconds,
            generatedTokens: generated
        )
    }

    static func behaviorUsesSemanticJudge(_ testCase: GoldenBehaviorCase) -> Bool {
        trimmedNonEmpty(testCase.semanticPrompt) != nil
            && trimmedNonEmpty(testCase.semanticReferenceAnswer) != nil
    }

    static func validatedWorkerTopLogits(
        _ topLogits: [CorrectnessTraceLogit]?,
        actualToken: Int
    ) throws -> [CorrectnessTraceLogit] {
        guard let topLogits, !topLogits.isEmpty else {
            throw MLXFastError.invalidInput("runtime worker response missing top_logits")
        }
        guard topLogits.count <= MLXFastConstants.correctnessTopLogits else {
            throw MLXFastError.invalidInput(
                "runtime worker returned \(topLogits.count) top_logits; maximum is \(MLXFastConstants.correctnessTopLogits)"
            )
        }
        guard topLogits[0].token == actualToken else {
            throw MLXFastError.invalidInput("runtime worker top_logits[0] does not match returned token")
        }

        var seen = Set<Int>()
        for (index, item) in topLogits.enumerated() {
            guard item.token >= 0, item.token < MLXFastConstants.vocabSize else {
                throw MLXFastError.invalidInput("runtime worker top_logits[\(index)].token is outside vocab")
            }
            guard item.logit.isFinite else {
                throw MLXFastError.invalidInput("runtime worker top_logits[\(index)].logit must be finite")
            }
            guard seen.insert(item.token).inserted else {
                throw MLXFastError.invalidInput("runtime worker top_logits contains duplicate token \(item.token)")
            }
            if index > 0 {
                let previous = topLogits[index - 1]
                guard previous.logit > item.logit
                    || (previous.logit == item.logit && previous.token < item.token)
                else {
                    throw MLXFastError.invalidInput("runtime worker top_logits must be sorted by logit descending")
                }
            }
        }
        return topLogits
    }

    static func requireGeneratedTokenCount(
        _ actual: Int,
        expected: Int,
        label: String
    ) throws {
        guard actual == expected else {
            throw MLXFastError.invalidInput(
                "runtime worker \(label) returned \(actual) tokens; expected \(expected)"
            )
        }
    }

    static func compareTeacherForcedCached(
        testCase: GoldenCase,
        weightCache: Qwen35RuntimeWeightCache,
        steps: Int = MLXFastConstants.correctnessSteps,
        progressIntervalSteps: Int = 0,
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> CorrectnessTokenComparison {
        guard !testCase.promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("teacher-forced correctness prompt must not be empty")
        }
        guard testCase.expectedTokens.count >= steps else {
            throw MLXFastError.invalidInput(
                "\(testCase.name).expected_tokens has \(testCase.expectedTokens.count) tokens; need at least \(steps)"
            )
        }

        let model = try weightCache.requireLibraryModel()
        let cache = model.newCache(parameters: nil)
        var logits = try qwenLogits(
            inputIDs: inputIDsArray(testCase.promptTokens),
            model: model,
            cache: cache,
            positionOffset: 0
        )
        var actualToken = try QwenCorrectness.greedyToken(from: logits)

        for step in 0..<steps {
            let expectedToken = testCase.expectedTokens[step]
            if !correctnessTokenAccepted(
                expectedToken: expectedToken,
                actualToken: actualToken,
                topLogits: try topLogits(from: logits, topK: MLXFastConstants.correctnessTopLogits)
            ) {
                return CorrectnessTokenComparison(
                    passed: false,
                    checkedSteps: step + 1,
                    firstFailingStep: step,
                    expectedToken: expectedToken,
                    actualToken: actualToken
                )
            }
            reportProgress(
                step: step + 1,
                total: steps,
                intervalSteps: progressIntervalSteps,
                progress: progress
            )

            if step == steps - 1 {
                break
            }

            logits = try qwenLogits(
                inputIDs: inputIDsArray([expectedToken]),
                model: model,
                cache: cache,
                positionOffset: testCase.promptTokens.count + step
            )
            actualToken = try QwenCorrectness.greedyToken(from: logits)
        }

        return CorrectnessTokenComparison(
            passed: true,
            checkedSteps: steps,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil
        )
    }

    static func compareAnchorCached(
        anchor: GoldenAnchorCase,
        weightCache: Qwen35RuntimeWeightCache
    ) throws -> CorrectnessTokenComparison {
        let model = try weightCache.requireLibraryModel()
        let cache = model.newCache(parameters: nil)
        let logits = try qwenLogits(
            inputIDs: inputIDsArray(anchor.contextTokens),
            model: model,
            cache: cache,
            positionOffset: 0
        )
        let actualToken = try QwenCorrectness.greedyToken(from: logits)
        return compareAnchorToken(
            anchor: anchor,
            actualToken: actualToken,
            topLogits: try topLogits(from: logits, topK: MLXFastConstants.correctnessTopLogits)
        )
    }

    static func compareFreeRunCached(
        testCase: GoldenFreeRunCase,
        weightCache: Qwen35RuntimeWeightCache,
        progressIntervalSteps: Int = 0,
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> CorrectnessTokenComparison {
        let generated = try generateGreedyCached(
            promptTokens: testCase.promptTokens,
            steps: testCase.expectedTokens.count,
            weightCache: weightCache,
            progressIntervalSteps: progressIntervalSteps,
            progress: progress
        )
        return compareFreeRunTokens(testCase: testCase, generated: generated)
    }

    static func compareBehaviorCached(
        testCase: GoldenBehaviorCase,
        weightCache: Qwen35RuntimeWeightCache
    ) throws -> CorrectnessTokenComparison {
        if testCase.maxNewTokens == 1 {
            let model = try weightCache.requireLibraryModel()
            let cache = model.newCache(parameters: nil)
            let logits = try qwenLogits(
                inputIDs: inputIDsArray(testCase.promptTokens),
                model: model,
                cache: cache,
                positionOffset: 0
            )
            let actualToken = try QwenCorrectness.greedyToken(from: logits)
            return compareBehaviorFirstToken(
                testCase: testCase,
                actualToken: actualToken,
                topLogits: try topLogits(from: logits, topK: MLXFastConstants.correctnessTopLogits)
            )
        }

        let generated = try generateGreedyCached(
            promptTokens: testCase.promptTokens,
            steps: testCase.maxNewTokens,
            weightCache: weightCache
        )
        return compareBehaviorTokens(testCase: testCase, generated: generated)
    }

    static func compareAnchorToken(
        anchor: GoldenAnchorCase,
        actualToken: Int,
        topLogits: [CorrectnessTraceLogit]?
    ) -> CorrectnessTokenComparison {
        if anchorTokenAccepted(anchor: anchor, actualToken: actualToken, topLogits: topLogits) {
            return CorrectnessTokenComparison(
                passed: true,
                checkedSteps: 1,
                firstFailingStep: nil,
                expectedToken: nil,
                actualToken: nil
            )
        }
        return CorrectnessTokenComparison(
            passed: false,
            checkedSteps: 1,
            firstFailingStep: 0,
            expectedToken: anchor.expectedToken,
            actualToken: actualToken
        )
    }

    static func compareFreeRunTokens(
        testCase: GoldenFreeRunCase,
        generated: [Int]
    ) -> CorrectnessTokenComparison {
        let prefixTokens = testCase.exactPrefixTokens ?? testCase.expectedTokens.count
        let comparison = GoldenSequenceMatcher.firstPrefixMismatch(
            expected: testCase.expectedTokens,
            actual: generated,
            prefixTokens: prefixTokens
        )
        return CorrectnessTokenComparison(
            passed: comparison.passed,
            checkedSteps: comparison.passed ? prefixTokens : (comparison.step ?? 0) + 1,
            firstFailingStep: comparison.step,
            expectedToken: comparison.expectedToken,
            actualToken: comparison.actualToken
        )
    }

    static func compareBehaviorTokens(
        testCase: GoldenBehaviorCase,
        generated: [Int]
    ) -> CorrectnessTokenComparison {
        let comparison = GoldenSequenceMatcher.matchesAnyAcceptedPrefix(
            acceptedSequences: testCase.acceptedTokenSequences,
            actual: generated
        )
        return CorrectnessTokenComparison(
            passed: comparison.passed,
            checkedSteps: comparison.passed ? testCase.maxNewTokens : (comparison.step ?? 0) + 1,
            firstFailingStep: comparison.step,
            expectedToken: comparison.expectedToken,
            actualToken: comparison.actualToken
        )
    }

    static func compareBehaviorFirstToken(
        testCase: GoldenBehaviorCase,
        actualToken: Int,
        topLogits: [CorrectnessTraceLogit]?
    ) -> CorrectnessTokenComparison {
        let acceptedTokens = Set(testCase.acceptedTokenSequences.compactMap(\.first))
        if acceptedTokens.contains(actualToken) {
            return CorrectnessTokenComparison(
                passed: true,
                checkedSteps: 1,
                firstFailingStep: nil,
                expectedToken: nil,
                actualToken: nil
            )
        }
        for acceptedToken in acceptedTokens where correctnessTokenAccepted(
            expectedToken: acceptedToken,
            actualToken: actualToken,
            topLogits: topLogits
        ) {
            return CorrectnessTokenComparison(
                passed: true,
                checkedSteps: 1,
                firstFailingStep: nil,
                expectedToken: nil,
                actualToken: nil
            )
        }
        return CorrectnessTokenComparison(
            passed: false,
            checkedSteps: 1,
            firstFailingStep: 0,
            expectedToken: acceptedTokens.sorted().first,
            actualToken: actualToken
        )
    }

    static func anchorTokenAccepted(
        anchor: GoldenAnchorCase,
        actualToken: Int,
        topLogits: [CorrectnessTraceLogit]?
    ) -> Bool {
        var acceptedTokens = Set(anchor.acceptedTokens ?? [])
        acceptedTokens.insert(anchor.expectedToken)
        if acceptedTokens.contains(actualToken) {
            return true
        }
        for acceptedToken in acceptedTokens where correctnessTokenAccepted(
            expectedToken: acceptedToken,
            actualToken: actualToken,
            topLogits: topLogits
        ) {
            return true
        }
        guard let maxExpectedRank = anchor.maxExpectedRank,
              let topLogits,
              let topLogit = topLogits.first?.logit,
              let expectedIndex = topLogits.firstIndex(where: { $0.token == anchor.expectedToken })
        else {
            return false
        }
        let expectedRank = expectedIndex + 1
        let expectedDelta = topLogit - topLogits[expectedIndex].logit
        let maxDelta = anchor.maxTopLogitDelta ?? MLXFastConstants.correctnessLogitTieTolerance
        return expectedRank <= maxExpectedRank && expectedDelta <= maxDelta
    }

    struct CorrectnessLogitDiagnostics {
        let topLogits: [CorrectnessTraceLogit]
        let expectedTokenLogit: Double?
        let expectedTokenRank: Int?
        let topLogitMargin: Double?
    }

    static func topLogits(
        from logits: MLXArray,
        topK: Int
    ) throws -> [CorrectnessTraceLogit] {
        try correctnessLogitDiagnostics(
            from: logits,
            topK: topK,
            expectedToken: nil
        ).topLogits
    }

    static func correctnessLogitDiagnostics(
        from logits: MLXArray,
        topK: Int,
        expectedToken: Int?
    ) throws -> CorrectnessLogitDiagnostics {
        guard let vocabSize = logits.shape.last, vocabSize > 0 else {
            throw MLXFastError.invalidInput("correctness logits must have a non-empty vocab dimension")
        }
        let rows = logits.reshaped([-1, vocabSize])
        return try correctnessLogitDiagnostics(
            fromRows: rows,
            row: rows.shape[0] - 1,
            vocabSize: vocabSize,
            topK: topK,
            expectedToken: expectedToken
        )
    }

    static func correctnessLogitDiagnostics(
        fromRows rows: MLXArray,
        row: Int,
        vocabSize: Int,
        topK: Int,
        expectedToken: Int?
    ) throws -> CorrectnessLogitDiagnostics {
        guard row >= 0, row < rows.shape[0] else {
            throw MLXFastError.invalidInput("correctness logits row \(row) is outside available rows \(rows.shape[0])")
        }
        guard topK > 0 else {
            throw MLXFastError.invalidInput("correctness topK must be positive")
        }
        if let expectedToken {
            guard expectedToken >= 0, expectedToken < vocabSize else {
                throw MLXFastError.invalidInput(
                    "correctness expected token \(expectedToken) is outside vocab size \(vocabSize)"
                )
            }
        }
        let selected = rows[row]
        eval(selected)
        let values = selected.asArray(Float.self).map(Double.init)
        guard values.count == vocabSize else {
            throw MLXFastError.invalidInput(
                "correctness logits materialized \(values.count) values, expected \(vocabSize)"
            )
        }
        return try correctnessLogitDiagnostics(
            values: values,
            topK: topK,
            expectedToken: expectedToken
        )
    }

    static func correctnessLogitDiagnostics(
        values: [Double],
        topK: Int,
        expectedToken: Int?
    ) throws -> CorrectnessLogitDiagnostics {
        guard !values.isEmpty else {
            throw MLXFastError.invalidInput(
                "correctness logits must not be empty"
            )
        }
        guard topK > 0 else {
            throw MLXFastError.invalidInput(
                "correctness topK must be positive"
            )
        }
        if let expectedToken {
            guard expectedToken >= 0,
                  expectedToken < values.count
            else {
                throw MLXFastError.invalidInput(
                    "correctness expected token \(expectedToken) is outside vocab size \(values.count)"
                )
            }
        }
        let sortedIndices = values.indices.sorted {
            let lhs = values[$0]
            let rhs = values[$1]
            return lhs == rhs ? $0 < $1 : lhs > rhs
        }
        let topLogits = sortedIndices.prefix(min(topK, sortedIndices.count)).map {
            CorrectnessTraceLogit(token: $0, logit: values[$0])
        }
        let expectedTokenLogit = expectedToken.map { values[$0] }
        let expectedTokenRank = expectedToken.map {
            (sortedIndices.firstIndex(of: $0) ?? sortedIndices.count - 1) + 1
        }
        let topLogitMargin = sortedIndices.count >= 2
            ? values[sortedIndices[0]] - values[sortedIndices[1]]
            : nil
        return CorrectnessLogitDiagnostics(
            topLogits: topLogits,
            expectedTokenLogit: expectedTokenLogit,
            expectedTokenRank: expectedTokenRank,
            topLogitMargin: topLogitMargin
        )
    }

    static func traceGreedyCached(
        testCase: GoldenCase,
        step: Int,
        topK: Int,
        weightCache: Qwen35RuntimeWeightCache,
        goldenHash: String
    ) throws -> CorrectnessTraceReport {
        guard !testCase.promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("greedy correctness prompt must not be empty")
        }
        guard step >= 0, step < testCase.expectedTokens.count else {
            throw MLXFastError.invalidInput(
                "trace step \(step) is outside expected token range 0..<\(testCase.expectedTokens.count)"
            )
        }
        guard topK > 0 else {
            throw MLXFastError.invalidInput("trace topK must be positive")
        }

        let model = try weightCache.requireLibraryModel()
        let cache = model.newCache(parameters: nil)
        var logits = try qwenLogits(
            inputIDs: inputIDsArray(testCase.promptTokens),
            model: model,
            cache: cache,
            positionOffset: 0
        )
        var token = try QwenCorrectness.greedyToken(from: logits)
        var generated: [Int] = []
        generated.reserveCapacity(step + 1)

        for currentStep in 0...step {
            generated.append(token)
            if currentStep == step {
                return try traceReport(
                    logits: logits,
                    testCase: testCase,
                    step: step,
                    topK: topK,
                    generated: generated,
                    goldenHash: goldenHash
                )
            }

            logits = try qwenLogits(
                inputIDs: inputIDsArray([token]),
                model: model,
                cache: cache,
                positionOffset: testCase.promptTokens.count + currentStep
            )
            token = try QwenCorrectness.greedyToken(from: logits)
        }

        throw MLXFastError.invalidInput("trace failed to reach step \(step)")
    }

    static func traceReport(
        logits: MLXArray,
        testCase: GoldenCase,
        step: Int,
        topK: Int,
        generated: [Int],
        goldenHash: String
    ) throws -> CorrectnessTraceReport {
        guard let vocabSize = logits.shape.last, vocabSize > 0 else {
            throw MLXFastError.invalidInput("trace logits must have a non-empty vocab dimension")
        }
        let rows = logits.reshaped([-1, vocabSize])
        let last = rows[-1]
        eval(last)
        let values = last.asArray(Float.self).map(Double.init)
        guard values.count == vocabSize else {
            throw MLXFastError.invalidInput(
                "trace logits materialized \(values.count) values, expected \(vocabSize)"
            )
        }

        let expectedToken = testCase.expectedTokens[step]
        let actualToken = generated[step]
        guard expectedToken >= 0, expectedToken < values.count else {
            throw MLXFastError.invalidInput("expected token \(expectedToken) is outside vocab size \(values.count)")
        }
        guard actualToken >= 0, actualToken < values.count else {
            throw MLXFastError.invalidInput("actual token \(actualToken) is outside vocab size \(values.count)")
        }

        let sortedIndices = values.indices.sorted {
            let lhs = values[$0]
            let rhs = values[$1]
            return lhs == rhs ? $0 < $1 : lhs > rhs
        }
        let requestedTopK = min(topK, sortedIndices.count)
        let topLogits = sortedIndices.prefix(requestedTopK).map {
            CorrectnessTraceLogit(token: $0, logit: values[$0])
        }
        let expectedRank = (sortedIndices.firstIndex(of: expectedToken) ?? sortedIndices.count - 1) + 1
        let topMargin: Double?
        if sortedIndices.count >= 2 {
            topMargin = values[sortedIndices[0]] - values[sortedIndices[1]]
        } else {
            topMargin = nil
        }
        let matchedPrefixSteps = zip(generated, testCase.expectedTokens)
            .prefix { pair in pair.0 == pair.1 }
            .count

        return CorrectnessTraceReport(
            caseName: testCase.name,
            step: step,
            promptTokenCount: testCase.promptTokens.count,
            expectedToken: expectedToken,
            actualToken: actualToken,
            matchedPrefixSteps: matchedPrefixSteps,
            generatedPrefix: generated,
            actualTokenLogit: values[actualToken],
            expectedTokenLogit: values[expectedToken],
            actualExpectedLogitDelta: values[actualToken] - values[expectedToken],
            expectedTokenRank: expectedRank,
            topLogitMargin: topMargin,
            topLogits: topLogits,
            goldenHash: goldenHash
        )
    }

    static func generateGreedyCached(
        promptTokens: [Int],
        steps: Int,
        weightCache: Qwen35RuntimeWeightCache,
        progressIntervalSteps: Int = 0,
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> [Int] {
        guard !promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("greedy correctness prompt must not be empty")
        }
        guard steps >= 0 else {
            throw MLXFastError.invalidInput("greedy correctness steps must be non-negative")
        }

        let model = try weightCache.requireLibraryModel()
        let cache = model.newCache(parameters: nil)
        var logits = try qwenLogits(
            inputIDs: inputIDsArray(promptTokens),
            model: model,
            cache: cache,
            positionOffset: 0
        )
        var token = try QwenCorrectness.greedyToken(from: logits)
        var generated: [Int] = []
        generated.reserveCapacity(steps)

        for step in 0..<steps {
            generated.append(token)
            reportProgress(
                step: step + 1,
                total: steps,
                intervalSteps: progressIntervalSteps,
                progress: progress
            )
            if step == steps - 1 {
                break
            }
            logits = try qwenLogits(
                inputIDs: inputIDsArray([token]),
                model: model,
                cache: cache,
                positionOffset: promptTokens.count + step
            )
            token = try QwenCorrectness.greedyToken(from: logits)
        }
        return generated
    }

}
