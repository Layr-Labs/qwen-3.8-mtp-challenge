// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon
@testable import MLXLMServer
import Testing

/// CI-safe wiring tests for ``MLXBatchedEngineServerEngine`` (no Metal).
/// The streaming path is covered by the gated MLXLMTests integration suite.
struct MLXBatchedEngineServerEngineTests {
    @Test("BatchedEngineServerConfiguration round-trips into ContinuousBatchingConfig")
    func configurationRoundTripIntoContinuousBatchingConfig() {
        let configuration = BatchedEngineServerConfiguration(
            maxNumSeqs: 12,
            maxNumBatchedTokens: 4096,
            prefillStepSize: 256,
            streamInterval: 4,
            maxKVCacheTokens: 1024,
            stepInterval: 0.002,
            prefixCacheConfig: nil,
            mtpEnabled: true
        )

        let runtime = configuration.makeContinuousBatchingConfig()

        #expect(runtime.schedulerConfig.maxNumSeqs == 12)
        #expect(runtime.schedulerConfig.maxNumBatchedTokens == 4096)
        #expect(runtime.schedulerConfig.prefillStepSize == 256)
        #expect(runtime.schedulerConfig.streamInterval == 4)
        #expect(runtime.schedulerConfig.maxKVCacheTokens == 1024)
        #expect(runtime.stepInterval == 0.002)
        #expect(runtime.prefixCacheConfig == nil)
        #expect(runtime.mtpEnabled == true)
    }

    @Test("BatchedEngineServerConfiguration defaults match documented values")
    func configurationDefaults() {
        let configuration = BatchedEngineServerConfiguration()

        #expect(configuration.maxNumSeqs == 24)
        #expect(configuration.maxNumBatchedTokens == 8192)
        #expect(configuration.prefillStepSize == 512)
        #expect(configuration.streamInterval == 1)
        #expect(configuration.maxKVCacheTokens == 0)
        #expect(configuration.stepInterval == 0.001)
        #expect(configuration.prefixCacheConfig == nil)
        #expect(configuration.mtpEnabled == false)
    }

    @Test("default MLXServerConfiguration prefers the batched engine")
    func defaultServerConfigurationPrefersBatchedEngine() {
        let configuration = MLXServerConfiguration(model: "local")
        #expect(configuration.engineKind == .batched)
    }

    @Test("CLI accepts --engine-kind and both spellings of single-request")
    func cliParsesEngineKindFlagAndEnvironment() throws {
        let batched = try parsedRunConfiguration(
            arguments: ["mlx-server", "--engine-kind", "batched", "--model", "m"]
        )
        #expect(batched.engineKind == .batched)

        let singleHyphen = try parsedRunConfiguration(
            arguments: ["mlx-server", "--engine-kind", "single-request", "--model", "m"]
        )
        #expect(singleHyphen.engineKind == .singleRequest)

        let singleUnderscore = try parsedRunConfiguration(
            arguments: ["mlx-server", "--engine-kind", "single_request", "--model", "m"]
        )
        #expect(singleUnderscore.engineKind == .singleRequest)

        let viaEnv = try parsedRunConfiguration(
            arguments: ["mlx-server", "--model", "m"],
            environment: ["MLX_SERVER_ENGINE_KIND": "single-request"]
        )
        #expect(viaEnv.engineKind == .singleRequest)
    }

    @Test("CLI rejects unknown engine kinds")
    func cliRejectsUnknownEngineKinds() {
        #expect(throws: MLXServerCLIError.self) {
            _ = try MLXServerCLI.parse(
                arguments: ["mlx-server", "--engine-kind", "speculative", "--model", "m"],
                environment: [:]
            )
        }
    }

    @Test("Equatable on MLXServerConfiguration accounts for batchedEngineConfiguration")
    func serverConfigurationEqualityIncludesBatchedEngineConfiguration() {
        let baseline = MLXServerConfiguration(
            model: "m",
            batchedEngineConfiguration: BatchedEngineServerConfiguration(maxNumSeqs: 8)
        )
        let bumped = MLXServerConfiguration(
            model: "m",
            batchedEngineConfiguration: BatchedEngineServerConfiguration(maxNumSeqs: 16)
        )
        let echo = MLXServerConfiguration(
            model: "m",
            batchedEngineConfiguration: BatchedEngineServerConfiguration(maxNumSeqs: 8)
        )

        #expect(baseline == echo)
        #expect(baseline != bumped)
    }

    @Test("Per-request tool_call_parser override matching the pinned format is accepted")
    func toolCallParserOverrideMatchingPinnedFormatIsNoOp() throws {
        try MLXBatchedEngineServerEngine.validateToolCallParserOverride(
            requested: "harmony",
            pinned: .harmony,
            modelType: nil
        )
    }

    @Test("Per-request tool_call_parser override differing from pinned format throws")
    func toolCallParserOverrideDifferingFromPinnedFormatThrows() {
        #expect(throws: MLXBatchedEngineServerEngineError.self) {
            try MLXBatchedEngineServerEngine.validateToolCallParserOverride(
                requested: "harmony",
                pinned: .json,
                modelType: nil
            )
        }
        do {
            try MLXBatchedEngineServerEngine.validateToolCallParserOverride(
                requested: "harmony",
                pinned: .json,
                modelType: nil
            )
            Issue.record("expected unsupportedToolCallParser to throw")
        } catch let error as MLXBatchedEngineServerEngineError {
            #expect(error == .unsupportedToolCallParser(pinned: .json, requested: .harmony))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    @Test("Per-request tool_call_parser override of nil is accepted")
    func toolCallParserOverrideNilIsNoOp() throws {
        try MLXBatchedEngineServerEngine.validateToolCallParserOverride(
            requested: nil,
            pinned: .harmony,
            modelType: nil
        )
    }

    @Test("Per-request tool_call_parser override of \"auto\" is accepted against any pinned format")
    func toolCallParserOverrideAutoIsNoOp() throws {
        // "auto" with nil modelType resolves to .json; without the carve-out
        // it would spuriously reject every non-JSON pinned engine.
        for pinned in ToolCallFormat.allCases where pinned != .json {
            try MLXBatchedEngineServerEngine.validateToolCallParserOverride(
                requested: "auto",
                pinned: pinned,
                modelType: nil
            )
        }
        // Case-insensitive.
        try MLXBatchedEngineServerEngine.validateToolCallParserOverride(
            requested: "AUTO",
            pinned: .harmony,
            modelType: nil
        )
        try MLXBatchedEngineServerEngine.validateToolCallParserOverride(
            requested: "Auto",
            pinned: .gemma,
            modelType: "gpt-oss"
        )
    }

    @Test("batchedSamplingParams maps an omitted max_tokens to the uncapped sentinel")
    func batchedSamplingParamsKeepsMaxTokensUncappedWhenRequestOmitsIt() {
        let request = OpenAIChatCompletionRequest(
            model: "m",
            messages: [.init(role: .user, content: .text("hi"))]
            // maxTokens omitted
        )

        // Int.max is the "uncapped" sentinel matching MLXModelContainerEngine,
        // which forwards a nil cap. A finite default would silently truncate.
        #expect(request.batchedSamplingParams.maxTokens == Int.max)
    }

    @Test("batchedSamplingParams forwards an explicit max_tokens unchanged")
    func batchedSamplingParamsForwardsExplicitMaxTokens() {
        let request = OpenAIChatCompletionRequest(
            model: "m",
            messages: [.init(role: .user, content: .text("hi"))],
            maxTokens: 64
        )

        #expect(request.batchedSamplingParams.maxTokens == 64)
    }

    @Test("batchedSamplingParams coerces repetition_penalty=0 to the disabled sentinel (1.0)")
    func batchedSamplingParamsCoercesZeroRepetitionPenaltyToOne() {
        let request = OpenAIChatCompletionRequest(
            model: "m",
            messages: [.init(role: .user, content: .text("hi"))],
            repetitionPenalty: 0
        )

        // The batched scheduler treats anything != 1.0 as active and divides
        // positive logits by the penalty (Scheduler.applyRepetitionPenalty),
        // so a literal 0 from the wire would divide-by-zero. The legacy
        // GenerateParameters path treats 0 as "disabled"; preserve that
        // semantic by coercing to the batched-engine disabled sentinel.
        #expect(request.batchedSamplingParams.repetitionPenalty == 1.0)
    }

    @Test("batchedSamplingParams forwards an explicit repetition_penalty unchanged")
    func batchedSamplingParamsForwardsExplicitRepetitionPenalty() {
        let request = OpenAIChatCompletionRequest(
            model: "m",
            messages: [.init(role: .user, content: .text("hi"))],
            repetitionPenalty: 1.2
        )

        #expect(request.batchedSamplingParams.repetitionPenalty == 1.2)
    }

    private func parsedRunConfiguration(
        arguments: [String],
        environment: [String: String] = [:]
    ) throws -> MLXServerConfiguration {
        switch try MLXServerCLI.parse(arguments: arguments, environment: environment) {
        case .run(let configuration):
            return configuration
        case .help, .listRoutes:
            Issue.record("Expected .run command from arguments \(arguments)")
            throw MLXServerCLIError.unknownOption("expected .run")
        }
    }
}
