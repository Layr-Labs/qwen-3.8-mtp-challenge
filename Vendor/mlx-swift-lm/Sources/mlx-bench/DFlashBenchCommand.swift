// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXRandom
import MLXSpeculative
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

extension MLXBench {
    static func runDFlashBenchmark(args: BenchArguments, mode: BenchMode) async throws {
        let targetURL = try requiredURL(
            explicit: args.targetPath,
            envNames: ["TARGET_DIR", "\(mode.envPrefix)_TARGET_DIR", "MLX_SWIFT_LM_DFLASH_TARGET_DIR"],
            description: "target model directory")
        let drafterURL = try requiredURL(
            explicit: args.drafterPath,
            envNames: ["DRAFTER_DIR", "\(mode.envPrefix)_DRAFTER_DIR", "MLX_SWIFT_LM_DFLASH_DRAFTER_DIR"],
            description: "DFlash drafter directory")

        print("loading target: \(targetURL.path)")
        let context = try await LLMModelFactory.shared.load(
            from: targetURL,
            using: #huggingFaceTokenizerLoader())
        guard let target = context.model as? any DFlashTargetModel else {
            throw CLIError("Target model does not conform to DFlashTargetModel: \(type(of: context.model))")
        }
        let enableVerifyQMM =
            args.enableVerifyQMM
            ?? envBoolOverride(names: [
                "DFLASH_BENCH_VERIFY_QMM",
                "\(mode.envPrefix)_VERIFY_QMM",
                "MLX_DFLASH_VERIFY_QMM",
            ])
            ?? false
        let verifyQMMInclude =
            args.verifyQMMInclude
            ?? envString(
                names: [
                    "DFLASH_BENCH_VERIFY_QMM_INCLUDE",
                    "\(mode.envPrefix)_VERIFY_QMM_INCLUDE",
                    "MLX_DFLASH_VERIFY_QMM_INCLUDE",
                ])
            ?? "all"
        let verifyQMMCount =
            enableVerifyQMM
            ? DFlashVerifyLinear.install(
                on: context.model, enableQMM: true, include: verifyQMMInclude)
            : 0

        print("loading DFlash drafter: \(drafterURL.path)")
        let drafter = try await DFlashDraftModel.load(
            from: drafterURL,
            bindTo: target)
        eval(context.model, drafter)

        let prompts = try benchmarkPrompts(args: args, mode: mode, tokenizer: context.tokenizer)
        let maxTokens = args.maxTokens
            ?? envInt(names: ["MAX_TOKENS", "\(mode.envPrefix)_MAX_TOKENS"], default: 128)
        let warmupTokens = min(
            args.warmupTokens
                ?? envInt(
                    names: ["WARMUP_TOKENS", "\(mode.envPrefix)_WARMUP_TOKENS"],
                    default: min(16, maxTokens)),
            maxTokens)
        let blockSizes = args.blockSizes
            ?? envIntList(
                names: ["BLOCK_SIZES", "\(mode.envPrefix)_BLOCK_SIZES"],
                default: defaultDFlashBlockSizes(
                    configured: drafter.config.blockSize,
                    recommended: drafter.config.recommendedBlockSize),
                minimum: 2)
        let batchSize = args.batchSize
            ?? envInt(names: ["BATCH_SIZE", "\(mode.envPrefix)_BATCH_SIZE"], default: 1)
        let collectPhaseTimings = args.phaseTimings
            || envBool(names: ["DFLASH_BENCH_PHASES", "\(mode.envPrefix)_PHASES"], default: false)
        let collectVerifySubphaseTimings = args.verifySubphaseTimings
            || envBool(
                names: ["DFLASH_BENCH_VERIFY_SUBPHASES", "\(mode.envPrefix)_VERIFY_SUBPHASES"],
                default: false)
        let officialFastMode = envBool(names: ["MLX_GEMMA4_DFLASH_OFFICIAL_FAST"], default: false)
        let stopOnEOS = args.stopOnEOS
            || envBool(names: ["DFLASH_BENCH_STOP_ON_EOS", "\(mode.envPrefix)_STOP_ON_EOS"], default: false)
        let stopPredicate = stopOnEOS ? dFlashEOSStopPredicate(context: context) : nil
        let parityCheck = args.parityCheck
            || envBool(names: ["DFLASH_BENCH_PARITY", "\(mode.envPrefix)_PARITY"], default: false)
        let parityBaselineScan = args.parityBaselineScan
            || envBool(
                names: ["DFLASH_BENCH_PARITY_BASELINE_SCAN", "\(mode.envPrefix)_PARITY_BASELINE_SCAN"],
                default: false)
        let parityMaxFailures = args.parityMaxFailures
            ?? envInt(
                names: ["DFLASH_BENCH_PARITY_MAX_FAILURES", "\(mode.envPrefix)_PARITY_MAX_FAILURES"],
                default: 1)
        let parityLayerDrift = args.parityLayerDrift
            || envBool(
                names: ["DFLASH_BENCH_PARITY_LAYER_DRIFT", "\(mode.envPrefix)_PARITY_LAYER_DRIFT"],
                default: false)
        let paritySummaryOnly = args.paritySummaryOnly
            || envBool(
                names: ["DFLASH_BENCH_PARITY_SUMMARY_ONLY", "\(mode.envPrefix)_PARITY_SUMMARY_ONLY"],
                default: false)
        let parityTopK = args.parityTopK
            ?? envInt(names: ["DFLASH_BENCH_PARITY_TOP_K", "\(mode.envPrefix)_PARITY_TOP_K"], default: 5)

        if parityCheck {
            if parityBaselineScan {
                try runDFlashBaselineBlockScan(
                    target: target,
                    drafter: drafter,
                    prompts: prompts,
                    maxTokens: maxTokens,
                    blockSizes: orderedUnique(blockSizes),
                    maxFailures: parityMaxFailures,
                    printLayerDrift: parityLayerDrift,
                    summaryOnly: paritySummaryOnly,
                    topK: parityTopK
                )
            } else {
                try runDFlashParityCheck(
                    target: target,
                    drafter: drafter,
                    prompts: prompts,
                    maxTokens: maxTokens,
                    blockSizes: orderedUnique(blockSizes),
                    printLayerDrift: parityLayerDrift,
                    topK: parityTopK
                )
            }
            return
        }

        let answerRegex = args.answerRegex
            ?? envString(names: ["DFLASH_BENCH_ANSWER_REGEX", "\(mode.envPrefix)_ANSWER_REGEX"])
        if let answerRegex {
            try runDFlashAnswerBenchmark(
                targetURL: targetURL,
                drafterURL: drafterURL,
                target: target,
                drafter: drafter,
                tokenizer: context.tokenizer,
                prompts: prompts,
                maxTokens: maxTokens,
                warmupTokens: warmupTokens,
                blockSizes: orderedUnique(blockSizes),
                answerRegex: answerRegex,
                expectedAnswer: args.expectedAnswer
                    ?? envString(names: ["DFLASH_BENCH_EXPECTED_ANSWER", "\(mode.envPrefix)_EXPECTED_ANSWER"]),
                repetitions: args.repetitions
                    ?? envInt(names: ["DFLASH_BENCH_REPETITIONS", "\(mode.envPrefix)_REPETITIONS"], default: 1),
                collectPhaseTimings: collectPhaseTimings,
                collectVerifySubphaseTimings: collectVerifySubphaseTimings
            )
            return
        }

        if batchSize > 1 {
            try runBatchedDFlashBenchmark(
                targetURL: targetURL,
                drafterURL: drafterURL,
                target: target,
                drafter: drafter,
                tokenizer: context.tokenizer,
                prompts: prompts,
                batchSize: batchSize,
                maxTokens: maxTokens,
                warmupTokens: warmupTokens,
                blockSizes: orderedUnique(blockSizes),
                enableVerifyQMM: enableVerifyQMM,
                verifyQMMInclude: verifyQMMInclude,
                verifyQMMCount: verifyQMMCount,
                printOutput: args.printOutput,
                tokenHashes: args.tokenHashes
            )
            return
        }

        print("")
        print("=== DFlash benchmark ===")
        print("target=\(targetURL.lastPathComponent)")
        print("drafter=\(drafterURL.lastPathComponent)")
        print("prompts=\(prompts.count), max_tokens=\(maxTokens), warmup=\(warmupTokens)")
        print(
            "verify_qmm=\(enableVerifyQMM ? "enabled" : "disabled")"
                + " include=\(verifyQMMInclude) linears=\(verifyQMMCount)"
        )
        if officialFastMode {
            print("dflash_mode=official_fast (vector verify; token-hash parity not guaranteed)")
        }
        if stopOnEOS {
            print("stop_on_eos=enabled")
        }
        if collectPhaseTimings || collectVerifySubphaseTimings {
            print("phase_timing=enabled (diagnostic wall-clock phases)")
        }
        if collectVerifySubphaseTimings {
            print("verify_subphases=enabled (adds target eval barriers)")
        }
        if args.printOutput {
            print("print_output=enabled")
        }
        print("K  base tok/s  dflash tok/s  speedup  accept_avg  emit_avg  generated")
        let dumpTokens = envBool(names: ["DFLASH_BENCH_DUMP_TOKENS"], default: false)

        let warmupPrompt = MLXArray(prompts[0])
        _ = measureDFlashBaselineThroughput(
            target: target,
            promptTokens: warmupPrompt,
            maxTokens: warmupTokens)
        for blockSize in orderedUnique(blockSizes) {
            _ = try measureDFlashThroughput(
                target: target,
                drafter: drafter,
                promptTokens: warmupPrompt,
                maxTokens: warmupTokens,
                blockSize: blockSize)
            MLX.Memory.clearCache()
        }

        var baselineRates = [Double]()
        var baselineHashes = [String]()
        for (promptIndex, prompt) in prompts.enumerated() {
            let result = measureDFlashBaselineThroughput(
                target: target,
                promptTokens: MLXArray(prompt),
                maxTokens: maxTokens,
                stopAfterGeneratedTokenCount: stopPredicate)
            baselineRates.append(result.tokensPerSecond)
            baselineHashes.append(tokenHash(result.generatedTokenIds))
            if result.prefillSeconds > 0 {
                print(
                    "   prefill[\(promptIndex)]: tokens=\(prompt.count)"
                        + " seconds=\(String(format: "%.3f", result.prefillSeconds))"
                        + " tok_s=\(String(format: "%.1f", Double(prompt.count) / result.prefillSeconds))")
            }
            if dumpTokens {
                print(
                    "   baseline_tokens[\(promptIndex)]="
                        + result.generatedTokenIds.map(String.init).joined(separator: ","))
            }
            if args.printOutput {
                printDecodedOutput(
                    label: "baseline_output[\(promptIndex)]",
                    tokenIds: result.generatedTokenIds,
                    tokenizer: context.tokenizer)
            }
            MLX.Memory.clearCache()
        }
        let baselineAverage = average(baselineRates)
        if args.tokenHashes {
            print("   baseline_token_hashes=" + baselineHashes.joined(separator: ","))
        }

        for blockSize in blockSizes {
            var dflashRates = [Double]()
            var dflashSeconds = 0.0
            var generated = [String]()
            var tokenHashes = [String]()
            var accepts = [Int]()
            var phaseTotals = DFlashPhaseTotals()
            for (promptIndex, prompt) in prompts.enumerated() {
                let result = try measureDFlashThroughput(
                    target: target,
                    drafter: drafter,
                    promptTokens: MLXArray(prompt),
                    maxTokens: maxTokens,
                    blockSize: blockSize,
                    collectPhaseTimings: collectPhaseTimings,
                    collectVerifySubphaseTimings: collectVerifySubphaseTimings,
                    stopAfterGeneratedTokenCount: stopPredicate)
                dflashRates.append(result.tokensPerSecond)
                dflashSeconds += result.generationSeconds
                generated.append(String(result.generatedTokens))
                tokenHashes.append(tokenHash(result.generatedTokenIds))
                accepts.append(contentsOf: result.acceptLengths ?? [])
                if let phases = result.phaseTimings {
                    phaseTotals.add(phases)
                }
                if dumpTokens {
                    print(
                        "   dflash_tokens[\(promptIndex),K=\(blockSize)]="
                            + result.generatedTokenIds.map(String.init).joined(separator: ","))
                }
                if args.printOutput {
                    printDecodedOutput(
                        label: "dflash_output[\(promptIndex),K=\(blockSize)]",
                        tokenIds: result.generatedTokenIds,
                        tokenizer: context.tokenizer)
                }
                MLX.Memory.clearCache()
            }

            let dflashAverage = average(dflashRates)
            let speedup = dflashAverage / max(baselineAverage, 1e-9)
            let acceptAverage = average(accepts.map(Double.init))
            let emitAverage = acceptAverage + 1
            print(
                "\(blockSize)  "
                    + "\(String(format: "%10.1f", baselineAverage)) "
                    + "\(String(format: "%12.1f", dflashAverage))   "
                    + "\(String(format: "%.2fx", speedup))   "
                    + "\(String(format: "%.2f", acceptAverage))/\(blockSize - 1)   "
                    + "\(String(format: "%.2f", emitAverage))/\(blockSize)   "
                    + generated.joined(separator: ",")
            )
            if collectPhaseTimings || collectVerifySubphaseTimings, phaseTotals.rounds > 0 {
                print("   " + phaseTotals.summary(generationSeconds: dflashSeconds))
            }
            if args.tokenHashes {
                print("   token_hashes=" + tokenHashes.joined(separator: ","))
                let matchingHashes = zip(tokenHashes, baselineHashes).filter { $0 == $1 }.count
                print("   token_hash_matches_baseline=\(matchingHashes)/\(baselineHashes.count)")
            }
        }
    }

    private static func dFlashEOSStopPredicate(context: ModelContext) -> DFlashStopPredicate {
        var stopTokenIds = context.configuration.eosTokenIds
        if let tokenizerEOS = context.tokenizer.eosTokenId {
            stopTokenIds.insert(tokenizerEOS)
        }
        for token in context.configuration.extraEOSTokens {
            if let id = context.tokenizer.convertTokenToId(token) {
                stopTokenIds.insert(id)
            }
        }
        return { generatedTokenIds in
            for (index, tokenId) in generatedTokenIds.enumerated()
            where stopTokenIds.contains(tokenId) {
                return index + 1
            }
            return nil
        }
    }

    private static func runBatchedDFlashBenchmark(
        targetURL: URL,
        drafterURL: URL,
        target: any DFlashTargetModel,
        drafter: DFlashDraftModel,
        tokenizer: MLXLMCommon.Tokenizer,
        prompts: [[Int32]],
        batchSize: Int,
        maxTokens: Int,
        warmupTokens: Int,
        blockSizes: [Int],
        enableVerifyQMM: Bool,
        verifyQMMInclude: String,
        verifyQMMCount: Int,
        printOutput: Bool,
        tokenHashes shouldPrintTokenHashes: Bool
    ) throws {
        let (batchPrompts, promptMode) = batchedPromptSet(prompts, batchSize: batchSize)

        print("")
        print("=== DFlash batched benchmark ===")
        print("target=\(targetURL.lastPathComponent)")
        print("drafter=\(drafterURL.lastPathComponent)")
        print(
            "batch_size=\(batchSize), prompt_mode=\(promptMode), max_tokens=\(maxTokens), warmup=\(warmupTokens)"
        )
        print(
            "verify_qmm=\(enableVerifyQMM ? "enabled" : "disabled")"
                + " include=\(verifyQMMInclude) linears=\(verifyQMMCount)"
        )
        print(
            "req_K  eff_K  base agg tok/s  dflash agg tok/s  dflash req tok/s  speedup  accept_avg  emit_avg  generated/row"
        )

        _ = try measureBatchedDFlashBaselineThroughput(
            target: target,
            promptTokens: batchPrompts,
            maxTokens: warmupTokens)
        for blockSize in blockSizes {
            _ = try measureBatchedDFlashThroughput(
                target: target,
                drafter: drafter,
                promptTokens: batchPrompts,
                maxTokens: warmupTokens,
                blockSize: blockSize)
            MLX.Memory.clearCache()
        }

        let baseline = try measureBatchedDFlashBaselineThroughput(
            target: target,
            promptTokens: batchPrompts,
            maxTokens: maxTokens)
        let baselineHashes = baseline.generatedTokenIds.map(tokenHash)
        if shouldPrintTokenHashes {
            print("   baseline_token_hashes=" + baselineHashes.joined(separator: ","))
        }
        if printOutput {
            for (row, tokens) in baseline.generatedTokenIds.enumerated() {
                printDecodedOutput(
                    label: "baseline_output[row=\(row)]",
                    tokenIds: tokens,
                    tokenizer: tokenizer)
            }
        }
        MLX.Memory.clearCache()

        for blockSize in blockSizes {
            let dflash = try measureBatchedDFlashThroughput(
                target: target,
                drafter: drafter,
                promptTokens: batchPrompts,
                maxTokens: maxTokens,
                blockSize: blockSize)
            let effectiveBlockSize = dflash.effectiveBlockSizes.max() ?? blockSize
            let effectiveBlockSummary = effectiveBlockSizeRange(
                dflash.effectiveBlockSizes,
                fallback: blockSize)
            let speedup = dflash.tokensPerSecond / max(baseline.tokensPerSecond, 1e-9)
            let acceptAverage = average((dflash.acceptLengths ?? []).map(Double.init))
            let emitAverage = acceptAverage + 1
            print(
                "\(blockSize)  "
                    + effectiveBlockSummary.padding(toLength: 5, withPad: " ", startingAt: 0)
                    + " "
                    + "\(String(format: "%14.1f", baseline.tokensPerSecond)) "
                    + "\(String(format: "%16.1f", dflash.tokensPerSecond)) "
                    + "\(String(format: "%16.1f", dflash.averageTokensPerSecondPerRequest))   "
                    + "\(String(format: "%.2fx", speedup))   "
                    + "\(String(format: "%.2f", acceptAverage))/\(effectiveBlockSize - 1)   "
                    + "\(String(format: "%.2f", emitAverage))/\(effectiveBlockSize)   "
                    + dflash.generatedTokensPerRow.map(String.init).joined(separator: ",")
            )
            print(
                "   effective_block_sizes="
                    + effectiveBlockSizeHistogram(dflash.effectiveBlockSizes, fallback: blockSize)
            )

            if shouldPrintTokenHashes {
                let hashes = dflash.generatedTokenIds.map(tokenHash)
                let matchingHashes = zip(hashes, baselineHashes).filter { $0 == $1 }.count
                print("   token_hashes=" + hashes.joined(separator: ","))
                print("   token_hash_matches_baseline=\(matchingHashes)/\(baselineHashes.count)")
                for row in dflash.generatedTokenIds.indices
                    where row < baseline.generatedTokenIds.count
                        && hashes[row] != baselineHashes[row]
                {
                    if let mismatch = firstTokenMismatch(
                        baseline.generatedTokenIds[row],
                        dflash.generatedTokenIds[row])
                    {
                        print("   token_mismatch[row=\(row)] \(mismatch)")
                    }
                }
            }
            if printOutput {
                for (row, tokens) in dflash.generatedTokenIds.enumerated() {
                    printDecodedOutput(
                        label: "dflash_output[row=\(row),K=\(blockSize)]",
                        tokenIds: tokens,
                        tokenizer: tokenizer)
                }
            }
            MLX.Memory.clearCache()
        }
    }

    private static func effectiveBlockSizeRange(
        _ blockSizes: [Int],
        fallback: Int
    ) -> String {
        guard let minSize = blockSizes.min(), let maxSize = blockSizes.max() else {
            return String(fallback)
        }
        return minSize == maxSize ? String(minSize) : "\(minSize)-\(maxSize)"
    }

    private static func effectiveBlockSizeHistogram(
        _ blockSizes: [Int],
        fallback: Int
    ) -> String {
        guard !blockSizes.isEmpty else {
            return "\(fallback):0"
        }
        var counts: [Int: Int] = [:]
        for blockSize in blockSizes {
            counts[blockSize, default: 0] += 1
        }
        return counts.keys.sorted()
            .map { "\($0):\(counts[$0] ?? 0)" }
            .joined(separator: ",")
    }

    private static func batchedPromptSet(
        _ prompts: [[Int32]],
        batchSize: Int
    ) -> (prompts: [[Int32]], mode: String) {
        precondition(batchSize > 0, "batchSize must be positive")
        if prompts.count == batchSize {
            return (prompts, "provided")
        }
        if prompts.count == 1, let first = prompts.first {
            return (Array(repeating: first, count: batchSize), "repeat_first")
        }
        if prompts.count > batchSize {
            return (Array(prompts.prefix(batchSize)), "prefix")
        }

        var output: [[Int32]] = []
        output.reserveCapacity(batchSize)
        for i in 0 ..< batchSize {
            output.append(prompts[i % prompts.count])
        }
        return (output, "cycled")
    }

    private static func runDFlashAnswerBenchmark(
        targetURL: URL,
        drafterURL: URL,
        target: any DFlashTargetModel,
        drafter: DFlashDraftModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        prompts: [[Int32]],
        maxTokens: Int,
        warmupTokens: Int,
        blockSizes: [Int],
        answerRegex: String,
        expectedAnswer: String?,
        repetitions: Int,
        collectPhaseTimings: Bool,
        collectVerifySubphaseTimings: Bool
    ) throws {
        let regex = try NSRegularExpression(pattern: answerRegex)
        let expected = expectedAnswer?.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()

        func decoded(_ tokenIds: [Int]) -> String {
            tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: true)
                .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
        }

        func capturedAnswer(in text: String) -> String? {
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            guard let match = regex.firstMatch(in: text, range: range) else {
                return nil
            }
            let captureIndex = match.numberOfRanges > 1 ? 1 : 0
            guard let captureRange = Range(match.range(at: captureIndex), in: text) else {
                return nil
            }
            return String(text[captureRange])
                .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
                .uppercased()
        }

        func stopAfterAnswer(_ tokenIds: [Int]) -> Int? {
            capturedAnswer(in: decoded(tokenIds)) == nil ? nil : tokenIds.count
        }

        struct Stats {
            var runs = 0
            var reached = 0
            var correct = 0
            var generatedTokens = 0
            var generationTokensPerSecond = 0.0
            var endToEndTokensPerSecond = 0.0

            mutating func add(_ result: DFlashBenchmarkResult, answer: String?, expected: String?) {
                runs += 1
                generatedTokens += result.generatedTokens
                if answer != nil { reached += 1 }
                if let answer, let expected, answer == expected { correct += 1 }
                generationTokensPerSecond += result.tokensPerSecond
                let totalSeconds = result.prefillSeconds + result.generationSeconds
                if totalSeconds > 0 {
                    endToEndTokensPerSecond += Double(result.generatedTokens) / totalSeconds
                }
            }

            var averageGeneratedTokens: Double {
                runs == 0 ? 0 : Double(generatedTokens) / Double(runs)
            }

            var averageGenerationTokensPerSecond: Double {
                runs == 0 ? 0 : generationTokensPerSecond / Double(runs)
            }

            var averageEndToEndTokensPerSecond: Double {
                runs == 0 ? 0 : endToEndTokensPerSecond / Double(runs)
            }
        }

        func printRun(
            label: String,
            run: Int,
            result: DFlashBenchmarkResult,
            text: String,
            answer: String?
        ) {
            let totalSeconds = result.prefillSeconds + result.generationSeconds
            let totalTPS = totalSeconds > 0 ? Double(result.generatedTokens) / totalSeconds : 0
            let correctness =
                if let expected {
                    answer == expected ? " correct=1" : " correct=0 expected=\(expected)"
                } else {
                    ""
                }
            print(
                "   \(label) run=\(run) reached=\(answer == nil ? 0 : 1)"
                    + " answer=\(answer ?? "<none>")\(correctness)"
                    + " tokens=\(result.generatedTokens)"
                    + " gen_tok_s=\(String(format: "%.1f", result.tokensPerSecond))"
                    + " e2e_tok_s=\(String(format: "%.1f", totalTPS))"
            )
            let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
            if let last = lines.last {
                print("      last_line=\(last)")
            }
        }

        print("")
        print("=== DFlash answer benchmark ===")
        print("target=\(targetURL.lastPathComponent)")
        print("drafter=\(drafterURL.lastPathComponent)")
        print("prompts=\(prompts.count), max_tokens=\(maxTokens), warmup=\(warmupTokens), repetitions=\(repetitions)")
        print("answer_regex=\(answerRegex)")
        if let expected {
            print("expected_answer=\(expected)")
        }

        let warmupPrompt = MLXArray(prompts[0])
        _ = measureDFlashBaselineThroughput(
            target: target,
            promptTokens: warmupPrompt,
            maxTokens: warmupTokens)
        for blockSize in blockSizes {
            _ = try measureDFlashThroughput(
                target: target,
                drafter: drafter,
                promptTokens: warmupPrompt,
                maxTokens: warmupTokens,
                blockSize: blockSize)
            MLX.Memory.clearCache()
        }

        var baselineStats = Stats()
        var dflashStats = Dictionary(uniqueKeysWithValues: blockSizes.map { ($0, Stats()) })

        for run in 1...repetitions {
            for (promptIndex, prompt) in prompts.enumerated() {
                let baseline = measureDFlashBaselineThroughput(
                    target: target,
                    promptTokens: MLXArray(prompt),
                    maxTokens: maxTokens,
                    stopAfterGeneratedTokenCount: stopAfterAnswer)
                let baselineText = decoded(baseline.generatedTokenIds)
                let baselineAnswer = capturedAnswer(in: baselineText)
                baselineStats.add(baseline, answer: baselineAnswer, expected: expected)
                printRun(
                    label: "baseline[\(promptIndex)]",
                    run: run,
                    result: baseline,
                    text: baselineText,
                    answer: baselineAnswer)
                MLX.Memory.clearCache()

                for blockSize in blockSizes {
                    let dflash = try measureDFlashThroughput(
                        target: target,
                        drafter: drafter,
                        promptTokens: MLXArray(prompt),
                        maxTokens: maxTokens,
                        blockSize: blockSize,
                        collectPhaseTimings: collectPhaseTimings,
                        collectVerifySubphaseTimings: collectVerifySubphaseTimings,
                        stopAfterGeneratedTokenCount: stopAfterAnswer)
                    let dflashText = decoded(dflash.generatedTokenIds)
                    let dflashAnswer = capturedAnswer(in: dflashText)
                    dflashStats[blockSize]?.add(dflash, answer: dflashAnswer, expected: expected)
                    printRun(
                        label: "dflash[\(promptIndex),K=\(blockSize)]",
                        run: run,
                        result: dflash,
                        text: dflashText,
                        answer: dflashAnswer)
                    MLX.Memory.clearCache()
                }
            }
        }

        print("")
        print("answer_summary")
        print(
            "   baseline avg_tokens=\(String(format: "%.1f", baselineStats.averageGeneratedTokens))"
                + " avg_gen_tok_s=\(String(format: "%.1f", baselineStats.averageGenerationTokensPerSecond))"
                + " avg_e2e_tok_s=\(String(format: "%.1f", baselineStats.averageEndToEndTokensPerSecond))"
                + " reached=\(baselineStats.reached)/\(baselineStats.runs)"
                + (expected == nil ? "" : " correct=\(baselineStats.correct)/\(baselineStats.runs)")
        )
        for blockSize in blockSizes {
            let stats = dflashStats[blockSize] ?? Stats()
            let speedup = stats.averageGenerationTokensPerSecond
                / max(baselineStats.averageGenerationTokensPerSecond, 1e-9)
            print(
                "   dflash K=\(blockSize)"
                    + " avg_tokens=\(String(format: "%.1f", stats.averageGeneratedTokens))"
                    + " avg_gen_tok_s=\(String(format: "%.1f", stats.averageGenerationTokensPerSecond))"
                    + " avg_e2e_tok_s=\(String(format: "%.1f", stats.averageEndToEndTokensPerSecond))"
                    + " speedup=\(String(format: "%.2fx", speedup))"
                    + " reached=\(stats.reached)/\(stats.runs)"
                    + (expected == nil ? "" : " correct=\(stats.correct)/\(stats.runs)")
            )
        }
    }

    private static func defaultDFlashBlockSizes(configured: Int, recommended: Int) -> [Int] {
        var seen = Set<Int>()
        var values = [Int]()
        let candidates =
            recommended == configured
            ? [4, 5, 6, 8, configured]
            : [recommended, 4, 5, 6, 8]
        for candidate in candidates {
            guard candidate >= 2,
                candidate <= configured,
                candidate <= recommended,
                !seen.contains(candidate)
            else {
                continue
            }
            seen.insert(candidate)
            values.append(candidate)
        }
        if values.isEmpty, configured >= 2 {
            values.append(configured)
        }
        return values
    }

    private static func runDFlashParityCheck(
        target: any DFlashTargetModel,
        drafter: DFlashDraftModel,
        prompts: [[Int32]],
        maxTokens: Int,
        blockSizes: [Int],
        printLayerDrift: Bool,
        topK: Int
    ) throws {
        print("")
        print("=== DFlash parity check ===")
        print("prompts=\(prompts.count), max_tokens=\(maxTokens), block_sizes=\(blockSizes.map(String.init).joined(separator: ","))")
        print("top_k=\(topK)")

        var failures = 0
        let baselineTokenBudget = maxTokens + (blockSizes.max() ?? 0)
        for (promptIndex, prompt) in prompts.enumerated() {
            let baseline = measureDFlashBaselineThroughput(
                target: target,
                promptTokens: MLXArray(prompt),
                maxTokens: baselineTokenBudget
            )
            MLX.Memory.clearCache()

            for blockSize in blockSizes {
                let result = try runDFlashParityCheck(
                    target: target,
                    drafter: drafter,
                    promptTokens: prompt,
                    baselineTokenIds: baseline.generatedTokenIds,
                    promptIndex: promptIndex,
                    maxTokens: maxTokens,
                    blockSize: blockSize,
                    printLayerDrift: printLayerDrift,
                    topK: topK
                )
                if !result.passed {
                    failures += 1
                }
                MLX.Memory.clearCache()
            }
        }

        if failures == 0 {
            print("parity=pass")
        } else {
            print("parity=fail failures=\(failures)")
            throw CLIError("DFlash parity check failed: \(failures) failure(s)")
        }
    }

    private static func runDFlashBaselineBlockScan(
        target: any DFlashTargetModel,
        drafter: DFlashDraftModel,
        prompts: [[Int32]],
        maxTokens: Int,
        blockSizes: [Int],
        maxFailures: Int,
        printLayerDrift: Bool,
        summaryOnly: Bool,
        topK: Int
    ) throws {
        print("")
        print("=== DFlash baseline block scan ===")
        print(
            "prompts=\(prompts.count), max_tokens=\(maxTokens), block_sizes=\(blockSizes.map(String.init).joined(separator: ","))"
                + " max_failures=\(maxFailures) top_k=\(topK)"
        )

        let baselineTokenBudget = maxTokens + (blockSizes.max() ?? 0)
        var totalFailures = 0
        var totalRounds = 0
        for (promptIndex, prompt) in prompts.enumerated() {
            let baseline = measureDFlashBaselineThroughput(
                target: target,
                promptTokens: MLXArray(prompt),
                maxTokens: baselineTokenBudget
            )
            MLX.Memory.clearCache()

            for blockSize in blockSizes {
                let result = try runDFlashBaselineBlockScan(
                    target: target,
                    drafter: drafter,
                    promptTokens: prompt,
                    baselineTokenIds: baseline.generatedTokenIds,
                    promptIndex: promptIndex,
                    maxTokens: maxTokens,
                    blockSize: blockSize,
                    remainingFailureBudget: maxFailures - totalFailures,
                    printLayerDrift: printLayerDrift,
                    summaryOnly: summaryOnly,
                    topK: topK
                )
                totalFailures += result.failures
                totalRounds += result.rounds
                MLX.Memory.clearCache()
                if totalFailures >= maxFailures {
                    print(
                        "baseline_scan=fail failures=\(totalFailures) rounds=\(totalRounds)"
                    )
                    throw CLIError("DFlash baseline block scan failed: \(totalFailures) failure(s)")
                }
            }
        }

        if totalFailures == 0 {
            print("baseline_scan=pass rounds=\(totalRounds)")
        } else {
            print("baseline_scan=fail failures=\(totalFailures) rounds=\(totalRounds)")
            throw CLIError("DFlash baseline block scan failed: \(totalFailures) failure(s)")
        }
    }

    private static func runDFlashBaselineBlockScan(
        target: any DFlashTargetModel,
        drafter: DFlashDraftModel,
        promptTokens: [Int32],
        baselineTokenIds: [Int],
        promptIndex: Int,
        maxTokens: Int,
        blockSize: Int,
        remainingFailureBudget: Int,
        printLayerDrift: Bool,
        summaryOnly: Bool,
        topK: Int
    ) throws -> DFlashBaselineBlockScanResult {
        var prompt = MLXArray(promptTokens)
        if prompt.ndim == 1 {
            prompt = prompt[.newAxis, .ellipsis]
        }

        var generationParameters = GenerateParameters(temperature: 0)
        generationParameters.maxTokens = maxTokens
        generationParameters.temperature = 0

        let cache = target.newCache(parameters: generationParameters)
        let prefill = try target.forwardForDFlash(
            prompt,
            cache: cache,
            targetLayerIds: [0]
        )
        eval(prefill.hiddenStates[0])

        var generated = 1
        var failures = 0
        var rounds = 0
        while generated < maxTokens {
            let remaining = maxTokens - generated
            let roundBlockSize = Swift.min(blockSize, remaining + 1)
            if roundBlockSize < 2 { break }

            let inputIds = [baselineTokenIds[generated - 1]]
                + Array(baselineTokenIds.dropFirst(generated).prefix(roundBlockSize - 1))
            guard inputIds.count == roundBlockSize else { break }

            rounds += 1
            let verifyInput = tokenRow(inputIds)
            let verifyCache = cache.map { $0.copy() }
            let blockOut = try target.forwardGreedyTokensForDFlash(
                verifyInput,
                cache: verifyCache,
                targetLayerIds: drafter.config.targetLayerIds
            )
            let blockTokens = blockOut.tokens.squeezed(axis: 0)
            eval(blockTokens, blockOut.targetHidden)
            let actual = blockTokens.asArray(Int32.self).map { Int($0) }
            let expected = Array(baselineTokenIds.dropFirst(generated).prefix(actual.count))

            if let mismatch = firstMismatch(actual: actual, expected: expected) {
                failures += 1
                if !summaryOnly {
                    printDFlashParityMismatch(
                        kind: "baseline_scan_block_vs_sequential",
                        promptIndex: promptIndex,
                        blockSize: blockSize,
                        roundIndex: rounds,
                        generatedOffset: generated + mismatch,
                        actual: actual,
                        expected: expected,
                        draft: Array(inputIds.dropFirst()),
                        mismatchOffset: mismatch
                    )
                    if topK > 0 {
                        try printDFlashParityTopK(
                            target: target,
                            prompt: prompt,
                            baselineTokenIds: baselineTokenIds,
                            generatedOffset: generated + mismatch,
                            verifyInput: verifyInput,
                            verifyCache: cache.map { $0.copy() },
                            verifyOffset: mismatch,
                            targetLayerIds: drafter.config.targetLayerIds,
                            topK: topK
                        )
                    }
                }
                if printLayerDrift {
                    try printDFlashLayerDrift(
                        target: target,
                        prompt: prompt,
                        baselineTokenIds: baselineTokenIds,
                        generatedOffset: generated + mismatch,
                        verifyInput: verifyInput,
                        verifyCache: cache.map { $0.copy() },
                        verifyOffset: mismatch
                    )
                }
                if failures >= remainingFailureBudget {
                    break
                }
            }

            for token in inputIds {
                let step = try target.forwardForDFlash(
                    tokenRow([token]),
                    cache: cache,
                    targetLayerIds: [0]
                )
                eval(step.hiddenStates[0])
            }
            generated += roundBlockSize
        }

        print(
            "prompt=\(promptIndex) K=\(blockSize) baseline_scan rounds=\(rounds)"
                + " failures=\(failures)"
        )
        return DFlashBaselineBlockScanResult(rounds: rounds, failures: failures)
    }

    private static func runDFlashParityCheck(
        target: any DFlashTargetModel,
        drafter: DFlashDraftModel,
        promptTokens: [Int32],
        baselineTokenIds: [Int],
        promptIndex: Int,
        maxTokens: Int,
        blockSize: Int,
        printLayerDrift: Bool,
        topK: Int
    ) throws -> DFlashParityResult {
        var generationParameters = GenerateParameters(temperature: 0)
        generationParameters.maxTokens = maxTokens
        generationParameters.temperature = 0

        try drafter.bind(target: target)

        var prompt = MLXArray(promptTokens)
        if prompt.ndim == 1 {
            prompt = prompt[.newAxis, .ellipsis]
        }

        var targetCache = target.newCache(parameters: generationParameters)
        let draftCache = try drafter.makeCache()
        guard canTrimPromptCache(draftCache) else {
            throw DFlashError.untrimmableCache
        }

        let prefillOut = try target.forwardGreedyTokensForDFlash(
            prompt,
            cache: targetCache,
            targetLayerIds: drafter.config.targetLayerIds
        )
        let firstBonusArray = prefillOut.tokens[0..., -1]
        eval(firstBonusArray, prefillOut.targetHidden)

        var bonus = Int(firstBonusArray.item(Int32.self))
        var targetHidden = prefillOut.targetHidden
        let baselineFirst = baselineTokenIds.first
        let prefillOK = baselineFirst == bonus

        print(
            "prompt=\(promptIndex) K=\(blockSize) prefill="
                + "\(prefillOK ? "pass" : "mismatch")"
                + " baseline_first=\(baselineFirst.map(String.init) ?? "nil") dflash_first=\(bonus)"
        )
        if !prefillOK {
            return DFlashParityResult(passed: false)
        }

        var generated = 1
        var emittedIds = [bonus]
        var roundIndex = 0
        while generated < maxTokens {
            let remaining = maxTokens - generated
            let roundBlockSize = Swift.min(blockSize, remaining + 1)
            if roundBlockSize < 2 {
                break
            }
            roundIndex += 1

            let rollbackProvider = target as? any DFlashTargetCacheRollbackProvider
            let targetRollbackState =
                rollbackProvider?.makeDFlashCacheRollbackState(cache: targetCache)
                ?? target.makeDefaultDFlashCacheRollbackState(cache: targetCache)
            let preVerifyCache = targetCache.map { $0.copy() }
            let baselineBlockInputIds =
                [bonus] + Array(baselineTokenIds.dropFirst(generated).prefix(roundBlockSize - 1))
            if baselineBlockInputIds.count == roundBlockSize {
                let baselineBlockInput = tokenRow(baselineBlockInputIds)
                let baselineBlockOut = try target.forwardGreedyTokensForDFlash(
                    baselineBlockInput,
                    cache: preVerifyCache.map { $0.copy() },
                    targetLayerIds: drafter.config.targetLayerIds
                )
                let baselineBlockTargetIds = baselineBlockOut.tokens.squeezed(axis: 0)
                eval(baselineBlockTargetIds, baselineBlockOut.targetHidden)
                let actualBaselineBlock = baselineBlockTargetIds.asArray(Int32.self).map { Int($0) }
                let expectedBaselineBlock = Array(
                    baselineTokenIds.dropFirst(generated).prefix(actualBaselineBlock.count))
                if let mismatch = firstMismatch(
                    actual: actualBaselineBlock, expected: expectedBaselineBlock)
                {
                    printDFlashParityMismatch(
                        kind: "baseline_path_block_vs_sequential",
                        promptIndex: promptIndex,
                        blockSize: blockSize,
                        roundIndex: roundIndex,
                        generatedOffset: generated + mismatch,
                        actual: actualBaselineBlock,
                        expected: expectedBaselineBlock,
                        draft: Array(baselineBlockInputIds.dropFirst()),
                        mismatchOffset: mismatch
                    )
                    if topK > 0 {
                        try printDFlashParityTopK(
                            target: target,
                            prompt: prompt,
                            baselineTokenIds: baselineTokenIds,
                            generatedOffset: generated + mismatch,
                            verifyInput: baselineBlockInput,
                            verifyCache: preVerifyCache.map { $0.copy() },
                            verifyOffset: mismatch,
                            targetLayerIds: drafter.config.targetLayerIds,
                            topK: topK
                        )
                    }
                    if printLayerDrift {
                        try printDFlashLayerDrift(
                            target: target,
                            prompt: prompt,
                            baselineTokenIds: baselineTokenIds,
                            generatedOffset: generated + mismatch,
                            verifyInput: baselineBlockInput,
                            verifyCache: preVerifyCache.map { $0.copy() },
                            verifyOffset: mismatch
                        )
                    }
                    return DFlashParityResult(passed: false)
                }
            }

            let draftTokens = try drafter.draftBlock(
                bonus: bonus,
                targetHidden: targetHidden,
                cache: draftCache,
                blockSize: roundBlockSize
            )
            eval(draftTokens)
            let draftTokenIds = draftTokens.squeezed(axis: 0)
                .asArray(Int32.self)
                .map { Int($0) }

            let bonusColumn = MLXArray([Int32(bonus)])[.newAxis, .ellipsis]
            let verifyInput = concatenated([bonusColumn, draftTokens], axis: 1)
            let verifyOut = try DFlashTargetRuntimeOptions.withSmallRowVerifyFusionsEnabled {
                try target.forwardGreedyTokensForDFlash(
                    verifyInput,
                    cache: targetCache,
                    targetLayerIds: drafter.config.targetLayerIds
                )
            }
            let targetTokenIds = verifyOut.tokens.squeezed(axis: 0)
            eval(targetTokenIds, verifyOut.targetHidden)
            let verifyTokenIds = targetTokenIds.asArray(Int32.self).map { Int($0) }
            let expectedTokenIds = Array(
                baselineTokenIds.dropFirst(generated).prefix(verifyTokenIds.count))

            let comparableCount = sequentialComparableOutputCount(
                bonus: bonus,
                draftTokenIds: draftTokenIds,
                baselineTokenIds: baselineTokenIds,
                generated: generated
            )
            let comparableActual = Array(verifyTokenIds.prefix(comparableCount))
            let comparableExpected = Array(expectedTokenIds.prefix(comparableCount))
            if let mismatch = firstMismatch(actual: comparableActual, expected: comparableExpected) {
                printDFlashParityMismatch(
                    kind: "target_block_prefix_vs_sequential",
                    promptIndex: promptIndex,
                    blockSize: blockSize,
                    roundIndex: roundIndex,
                    generatedOffset: generated + mismatch,
                    actual: verifyTokenIds,
                    expected: expectedTokenIds,
                    draft: draftTokenIds,
                    mismatchOffset: mismatch
                )
                if topK > 0 {
                    try printDFlashParityTopK(
                        target: target,
                        prompt: prompt,
                        baselineTokenIds: baselineTokenIds,
                        generatedOffset: generated + mismatch,
                        verifyInput: verifyInput,
                        verifyCache: preVerifyCache.map { $0.copy() },
                        verifyOffset: mismatch,
                        targetLayerIds: drafter.config.targetLayerIds,
                        topK: topK
                    )
                }
                if printLayerDrift {
                    try printDFlashLayerDrift(
                        target: target,
                        prompt: prompt,
                        baselineTokenIds: baselineTokenIds,
                        generatedOffset: generated + mismatch,
                        verifyInput: verifyInput,
                        verifyCache: preVerifyCache.map { $0.copy() },
                        verifyOffset: mismatch
                    )
                }
                return DFlashParityResult(passed: false)
            }

            let proposedCount = Swift.max(0, roundBlockSize - 1)
            var accepted = 0
            while accepted < proposedCount
                && accepted < draftTokenIds.count
                && accepted < verifyTokenIds.count
                && draftTokenIds[accepted] == verifyTokenIds[accepted]
            {
                accepted += 1
            }

            let walkedTokenCount = accepted + 1
            let emittedCount = Swift.min(remaining, walkedTokenCount)
            let emitted = Array(verifyTokenIds.prefix(emittedCount))
            let expectedEmitted = Array(baselineTokenIds.dropFirst(generated).prefix(emitted.count))
            if let mismatch = firstMismatch(actual: emitted, expected: expectedEmitted) {
                printDFlashParityMismatch(
                    kind: "dflash_emitted_vs_sequential",
                    promptIndex: promptIndex,
                    blockSize: blockSize,
                    roundIndex: roundIndex,
                    generatedOffset: generated + mismatch,
                    actual: emitted,
                    expected: expectedEmitted,
                    draft: draftTokenIds,
                    mismatchOffset: mismatch
                )
                return DFlashParityResult(passed: false)
            }

            let trim = roundBlockSize - accepted - 1
            if let rollbackProvider {
                targetHidden = try rollbackProvider.rollbackDFlashCache(
                    &targetCache,
                    state: targetRollbackState,
                    verifyInput: verifyInput,
                    acceptedTokenCount: accepted,
                    rejectedTokenCount: trim,
                    targetLayerIds: drafter.config.targetLayerIds,
                    verifiedTargetHidden: verifyOut.targetHidden
                )
            } else {
                targetHidden = try target.rollbackDFlashCacheUsingDefault(
                    &targetCache,
                    state: targetRollbackState,
                    verifyInput: verifyInput,
                    acceptedTokenCount: accepted,
                    rejectedTokenCount: trim,
                    targetLayerIds: drafter.config.targetLayerIds,
                    verifiedTargetHidden: verifyOut.targetHidden
                )
            }

            emittedIds.append(contentsOf: emitted)
            generated += emitted.count
            if emitted.isEmpty {
                print("prompt=\(promptIndex) K=\(blockSize) round=\(roundIndex) emitted=0")
                return DFlashParityResult(passed: false)
            }
            bonus = emitted.last ?? bonus
        }

        print(
            "prompt=\(promptIndex) K=\(blockSize) parity=pass generated=\(generated)"
                + " token_hash=\(tokenHash(emittedIds))"
        )
        return DFlashParityResult(passed: true)
    }

    private static func printDFlashParityMismatch(
        kind: String,
        promptIndex: Int,
        blockSize: Int,
        roundIndex: Int,
        generatedOffset: Int,
        actual: [Int],
        expected: [Int],
        draft: [Int],
        mismatchOffset: Int
    ) {
        print(
            "prompt=\(promptIndex) K=\(blockSize) round=\(roundIndex)"
                + " parity=mismatch kind=\(kind)"
                + " generated_offset=\(generatedOffset)"
                + " block_offset=\(mismatchOffset)"
        )
        print("   expected=" + expected.map(String.init).joined(separator: ","))
        print("   actual=" + actual.map(String.init).joined(separator: ","))
        print("   draft=" + draft.map(String.init).joined(separator: ","))
    }

    private static func firstMismatch(actual: [Int], expected: [Int]) -> Int? {
        let count = Swift.min(actual.count, expected.count)
        for index in 0 ..< count where actual[index] != expected[index] {
            return index
        }
        return actual.count == expected.count ? nil : count
    }

    private static func printDFlashParityTopK(
        target: any DFlashTargetModel,
        prompt: MLXArray,
        baselineTokenIds: [Int],
        generatedOffset: Int,
        verifyInput: MLXArray,
        verifyCache: [KVCache],
        verifyOffset: Int,
        targetLayerIds: [Int],
        topK: Int
    ) throws {
        let sequentialRow = targetSequentialLogitsRow(
            target: target,
            prompt: prompt,
            baselineTokenIds: baselineTokenIds,
            generatedOffset: generatedOffset
        )
        let blockForward = try target.forwardForDFlash(
            verifyInput,
            cache: verifyCache,
            targetLayerIds: targetLayerIds
        )
        let blockRow = blockForward.logits[0, verifyOffset, 0...]
        eval(sequentialRow, blockRow)

        let expected = generatedOffset < baselineTokenIds.count
            ? baselineTokenIds[generatedOffset]
            : nil
        print("   sequential_top\(topK)=" + formatTopK(sequentialRow, k: topK))
        print("   block_top\(topK)=" + formatTopK(blockRow, k: topK))
        if let expected {
            print(
                "   expected_token_scores sequential="
                    + formatScore(sequentialRow, token: expected)
                    + " block="
                    + formatScore(blockRow, token: expected)
            )
        }
    }

    private static func printDFlashLayerDrift(
        target: any DFlashTargetModel,
        prompt: MLXArray,
        baselineTokenIds: [Int],
        generatedOffset: Int,
        verifyInput: MLXArray,
        verifyCache: [KVCache],
        verifyOffset: Int
    ) throws {
        let layerIds = Array(0 ..< target.dFlashLayerCount)
        guard !layerIds.isEmpty else { return }

        let sequentialRows = try targetSequentialHiddenRowsForDFlash(
            target: target,
            prompt: prompt,
            baselineTokenIds: baselineTokenIds,
            generatedOffset: generatedOffset,
            targetLayerIds: layerIds
        )
        let blockForward = try target.forwardForDFlash(
            verifyInput,
            cache: verifyCache,
            targetLayerIds: layerIds
        )
        let blockRows = blockForward.hiddenStates.map { $0[0, verifyOffset, 0...] }
        eval(sequentialRows + blockRows)

        var metrics = [DFlashLayerDriftMetric]()
        metrics.reserveCapacity(layerIds.count)
        for index in layerIds.indices {
            let diff = (sequentialRows[index] - blockRows[index]).abs()
            let maxDiff = diff.max().item(Float.self)
            let meanDiff = diff.mean().item(Float.self)
            metrics.append(
                DFlashLayerDriftMetric(
                    layer: layerIds[index],
                    maxAbs: maxDiff,
                    meanAbs: meanDiff
                ))
        }

        let first1e3 = metrics.first { $0.maxAbs > 1e-3 }?.layer
        let first1e2 = metrics.first { $0.maxAbs > 1e-2 }?.layer
        let first1e1 = metrics.first { $0.maxAbs > 1e-1 }?.layer
        print(
            "   layer_drift first_gt_1e-3=\(first1e3.map(String.init) ?? "none")"
                + " first_gt_1e-2=\(first1e2.map(String.init) ?? "none")"
                + " first_gt_1e-1=\(first1e1.map(String.init) ?? "none")"
        )

        let top = metrics.sorted { lhs, rhs in
            if lhs.maxAbs == rhs.maxAbs {
                return lhs.meanAbs > rhs.meanAbs
            }
            return lhs.maxAbs > rhs.maxAbs
        }
        .prefix(8)
        .map {
            "\($0.layer):max=\(String(format: "%.5f", $0.maxAbs)),mean=\(String(format: "%.5f", $0.meanAbs))"
        }
        .joined(separator: " ")
        print("   layer_drift_top=" + top)
    }

    private static func dFlashDriftMetric(
        _ lhs: MLXArray,
        _ rhs: MLXArray
    ) -> (maxAbs: Float, meanAbs: Float) {
        let diff = (lhs - rhs).abs()
        return (
            maxAbs: diff.max().item(Float.self),
            meanAbs: diff.mean().item(Float.self)
        )
    }

    private static func sequentialComparableOutputCount(
        bonus: Int,
        draftTokenIds: [Int],
        baselineTokenIds: [Int],
        generated: Int
    ) -> Int {
        let inputIds = [bonus] + draftTokenIds
        let expectedInputIds = Array(
            baselineTokenIds.dropFirst(Swift.max(0, generated - 1)).prefix(inputIds.count))
        let count = Swift.min(inputIds.count, expectedInputIds.count)
        for index in 0 ..< count where inputIds[index] != expectedInputIds[index] {
            return index
        }
        return count
    }

    private static func targetSequentialLogitsRow(
        target: any DFlashTargetModel,
        prompt: MLXArray,
        baselineTokenIds: [Int],
        generatedOffset: Int
    ) -> MLXArray {
        var generationParameters = GenerateParameters(temperature: 0)
        generationParameters.maxTokens = Swift.max(generatedOffset + 1, 1)
        generationParameters.temperature = 0

        let cache = target.newCache(parameters: generationParameters)
        var logits = target.callAsFunction(prompt, cache: cache)
        if generatedOffset == 0 {
            return logits[0, -1, 0...]
        }

        for token in baselineTokenIds.prefix(generatedOffset) {
            let input = MLXArray([Int32(token)])[.newAxis, .ellipsis]
            logits = target.callAsFunction(input, cache: cache)
        }
        return logits[0, -1, 0...]
    }

    private static func targetSequentialHiddenRowsForDFlash(
        target: any DFlashTargetModel,
        prompt: MLXArray,
        baselineTokenIds: [Int],
        generatedOffset: Int,
        targetLayerIds: [Int]
    ) throws -> [MLXArray] {
        var generationParameters = GenerateParameters(temperature: 0)
        generationParameters.maxTokens = Swift.max(generatedOffset + 1, 1)
        generationParameters.temperature = 0

        let cache = target.newCache(parameters: generationParameters)
        if generatedOffset == 0 {
            let forward = try target.forwardForDFlash(
                prompt,
                cache: cache,
                targetLayerIds: targetLayerIds
            )
            return forward.hiddenStates.map { $0[0, -1, 0...] }
        }

        _ = target.callAsFunction(prompt, cache: cache)
        for token in baselineTokenIds.prefix(Swift.max(0, generatedOffset - 1)) {
            let input = MLXArray([Int32(token)])[.newAxis, .ellipsis]
            _ = target.callAsFunction(input, cache: cache)
        }

        let finalInput = tokenRow([baselineTokenIds[generatedOffset - 1]])
        let forward = try target.forwardForDFlash(
            finalInput,
            cache: cache,
            targetLayerIds: targetLayerIds
        )
        return forward.hiddenStates.map { $0[0, 0, 0...] }
    }

    private static func tokenRow(_ ids: [Int]) -> MLXArray {
        MLXArray(ids.map { Int32($0) })[.newAxis, .ellipsis]
    }

    private static func formatTopK(_ row: MLXArray, k: Int) -> String {
        let limit = Swift.max(1, Swift.min(k, row.dim(-1)))
        let indices = argPartition(-row, kth: limit - 1, axis: -1)[0 ..< limit]
        let scores = row[indices]
        eval(indices, scores)
        let ids = indices.asArray(Int32.self).map { Int($0) }
        let values = scores.asArray(Float.self)
        let pairs = zip(ids, values).sorted { lhs, rhs in lhs.1 > rhs.1 }
        return pairs
            .map { "\($0.0):\(String(format: "%.4f", $0.1))" }
            .joined(separator: ",")
    }

    private static func formatScore(_ row: MLXArray, token: Int) -> String {
        guard token >= 0, token < row.dim(-1) else {
            return "nan"
        }
        let score = row[token].item(Float.self)
        return String(format: "%.4f", score)
    }
}

private struct DFlashParityResult {
    let passed: Bool
}

private struct DFlashBaselineBlockScanResult {
    let rounds: Int
    let failures: Int
}

private struct DFlashLayerDriftMetric {
    let layer: Int
    let maxAbs: Float
    let meanAbs: Float
}

private struct DFlashPhaseTotals {
    var rounds = 0
    var cacheSnapshotSeconds = 0.0
    var draftLaunchSeconds = 0.0
    var draftCacheTrimSeconds = 0.0
    var verifyAndWaitSeconds = 0.0
    var targetTrunkSeconds = 0.0
    var targetHiddenConcatSeconds = 0.0
    var targetLMHeadSeconds = 0.0
    var targetSoftcapArgmaxSeconds = 0.0
    var targetTrunkEmbeddingSeconds = 0.0
    var targetTrunkPLESeconds = 0.0
    var targetTrunkMaskSeconds = 0.0
    var targetTrunkAttentionSeconds = 0.0
    var targetTrunkDenseMLPSeconds = 0.0
    var targetTrunkRouterSeconds = 0.0
    var targetTrunkExpertsSeconds = 0.0
    var targetTrunkPLEGateSeconds = 0.0
    var targetTrunkFinalNormSeconds = 0.0
    var acceptWalkSeconds = 0.0
    var cacheRollbackSeconds = 0.0
    var roundSeconds = 0.0

    mutating func add(_ phases: DFlashBenchmarkPhaseTimings) {
        rounds += phases.rounds
        cacheSnapshotSeconds += phases.cacheSnapshotSeconds
        draftLaunchSeconds += phases.draftLaunchSeconds
        draftCacheTrimSeconds += phases.draftCacheTrimSeconds
        verifyAndWaitSeconds += phases.verifyAndWaitSeconds
        targetTrunkSeconds += phases.targetTrunkSeconds
        targetHiddenConcatSeconds += phases.targetHiddenConcatSeconds
        targetLMHeadSeconds += phases.targetLMHeadSeconds
        targetSoftcapArgmaxSeconds += phases.targetSoftcapArgmaxSeconds
        targetTrunkEmbeddingSeconds += phases.targetTrunkEmbeddingSeconds
        targetTrunkPLESeconds += phases.targetTrunkPLESeconds
        targetTrunkMaskSeconds += phases.targetTrunkMaskSeconds
        targetTrunkAttentionSeconds += phases.targetTrunkAttentionSeconds
        targetTrunkDenseMLPSeconds += phases.targetTrunkDenseMLPSeconds
        targetTrunkRouterSeconds += phases.targetTrunkRouterSeconds
        targetTrunkExpertsSeconds += phases.targetTrunkExpertsSeconds
        targetTrunkPLEGateSeconds += phases.targetTrunkPLEGateSeconds
        targetTrunkFinalNormSeconds += phases.targetTrunkFinalNormSeconds
        acceptWalkSeconds += phases.acceptWalkSeconds
        cacheRollbackSeconds += phases.cacheRollbackSeconds
        roundSeconds += phases.roundSeconds
    }

    func summary(generationSeconds: Double) -> String {
        "phase ms/round: "
            + "snapshot=\(msPerRound(cacheSnapshotSeconds)) "
            + "draft_launch=\(msPerRound(draftLaunchSeconds)) "
            + "draft_trim=\(msPerRound(draftCacheTrimSeconds)) "
            + "verify_wait=\(msPerRound(verifyAndWaitSeconds)) "
            + "accept=\(msPerRound(acceptWalkSeconds)) "
            + "rollback=\(msPerRound(cacheRollbackSeconds)) "
            + "round=\(msPerRound(roundSeconds)); "
            + "%gen verify_wait=\(percent(verifyAndWaitSeconds, of: generationSeconds)) "
            + "accept=\(percent(acceptWalkSeconds, of: generationSeconds)) "
            + "rollback=\(percent(cacheRollbackSeconds, of: generationSeconds))"
            + verifySubphaseSummary()
    }

    private func verifySubphaseSummary() -> String {
        let total = targetTrunkSeconds + targetHiddenConcatSeconds + targetLMHeadSeconds
            + targetSoftcapArgmaxSeconds
        guard total > 0 else { return "" }
        return "; target_verify ms/round: "
            + "trunk=\(msPerRound(targetTrunkSeconds)) "
            + "hidden_concat=\(msPerRound(targetHiddenConcatSeconds)) "
            + "lm_head=\(msPerRound(targetLMHeadSeconds)) "
            + "softcap_argmax=\(msPerRound(targetSoftcapArgmaxSeconds))"
            + trunkBreakdownSummary()
    }

    private func trunkBreakdownSummary() -> String {
        let total = targetTrunkEmbeddingSeconds + targetTrunkPLESeconds + targetTrunkMaskSeconds
            + targetTrunkAttentionSeconds + targetTrunkDenseMLPSeconds
            + targetTrunkRouterSeconds + targetTrunkExpertsSeconds + targetTrunkPLEGateSeconds
            + targetTrunkFinalNormSeconds
        guard total > 0 else { return "" }
        return "; trunk_breakdown ms/round: "
            + "embed=\(msPerRound(targetTrunkEmbeddingSeconds)) "
            + "ple=\(msPerRound(targetTrunkPLESeconds)) "
            + "mask=\(msPerRound(targetTrunkMaskSeconds)) "
            + "attn=\(msPerRound(targetTrunkAttentionSeconds)) "
            + "dense_mlp=\(msPerRound(targetTrunkDenseMLPSeconds)) "
            + "router=\(msPerRound(targetTrunkRouterSeconds)) "
            + "experts=\(msPerRound(targetTrunkExpertsSeconds)) "
            + "ple_gate=\(msPerRound(targetTrunkPLEGateSeconds)) "
            + "final_norm=\(msPerRound(targetTrunkFinalNormSeconds))"
    }

    private func msPerRound(_ seconds: Double) -> String {
        String(format: "%.2f", seconds * 1000 / Double(max(rounds, 1)))
    }

    private func percent(_ seconds: Double, of totalSeconds: Double) -> String {
        guard totalSeconds > 0 else { return "0.0%" }
        return String(format: "%.1f%%", seconds * 100 / totalSeconds)
    }
}

