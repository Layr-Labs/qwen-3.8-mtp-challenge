// Copyright © 2026 Eigen Labs Inc.

import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

/// Single-request server engine backed by ``ModelContainer``'s serial actor.
/// Concurrent ``streamChatCompletion(request:)`` calls serialise; for
/// concurrent production traffic use ``MLXBatchedEngineServerEngine``.
public struct MLXModelContainerEngine: MLXServerEngine {
    private let modelID: String
    private let model: ModelContainer
    private let modelType: String?
    private let contextTokens: Int
    private let defaultToolCallParser: String?
    private let generationDriver: (any MLXServerGenerationDriver)?

    public init(
        modelID: String,
        model: ModelContainer,
        modelType: String? = nil,
        contextTokens: Int = 32768,
        defaultToolCallParser: String? = nil,
        generationDriver: (any MLXServerGenerationDriver)? = nil
    ) {
        self.modelID = modelID
        self.model = model
        self.modelType = modelType
        self.contextTokens = contextTokens
        self.defaultToolCallParser = defaultToolCallParser
        self.generationDriver = generationDriver
    }

    public func availableModels() async throws -> [MLXServerModel] {
        [.init(id: modelID)]
    }

    public func streamChatCompletion(
        request: OpenAIChatCompletionRequest
    ) async throws -> AsyncThrowingStream<MLXServerGenerationEvent, Error> {
        // This text-only engine passes template-ready message dictionaries,
        // which retain tool history but discard `image_url`/`video_url` parts. Rather
        // than accept media and silently ignore it, fail loud — real media
        // serving lives in the downstream VLM path, not this text-only engine.
        // Checked before any model state is mutated so a rejected request is a
        // no-op.
        if request.messages.contains(where: { $0.content.hasMedia }) {
            throw MLXModelContainerEngineError.mediaUnsupported
        }

        let requestToolFormat: ToolCallFormat?
        if let generationDriver {
            try generationDriver.validate(request: request)
            requestToolFormat = try await toolParserFormat(for: request)
        } else {
            try await configureToolParser(for: request)
            requestToolFormat = nil
        }

        let userInput = request.templateInput
        let tools = userInput.tools
        let input = try await model.prepare(input: userInput)
        var parameters = request.generationParameters
        parameters.maxTokens = try ContextTokenBudget.outputLimit(
            promptTokens: input.text.tokens.size,
            requested: request.maxTokens,
            contextTokens: contextTokens
        )
        // Match ModelContainer's single-consumption transfer for non-Sendable
        // MLX arrays; only this producer task can consume the prepared input.
        let preparedInput = SendableBox(input)
        let (stream, continuation) = AsyncThrowingStream<MLXServerGenerationEvent, Error>.makeStream()
        let task = Task {
            do {
                try Task.checkCancellation()
                let emit: @Sendable (Generation) -> Void = { item in
                    switch item {
                    case .chunk(let text):
                        continuation.yield(.content(text))
                    case .toolCall(let toolCall):
                        continuation.yield(.toolCall(toolCall))
                    case .info(let info):
                        continuation.yield(.info(.init(info)))
                    }
                }
                if let generationDriver, let requestToolFormat {
                    let generated = try await generationDriver.generate(
                        model: model, input: preparedInput.consume(), parameters: parameters,
                        request: request, tools: tools, toolCallFormat: requestToolFormat)
                    for try await item in generated {
                        try Task.checkCancellation()
                        emit(item)
                    }
                } else {
                    let generated = try await model.generate(
                        input: preparedInput.consume(), parameters: parameters, tools: tools)
                    for await item in generated {
                        try Task.checkCancellation()
                        emit(item)
                    }
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in
            task.cancel()
        }
        return stream
    }

    public func tokenize(_ request: TokenizeRequest) async throws -> TokenizeResponse {
        let tokenizer = await model.tokenizer
        return .init(
            tokens: tokenizer.encode(
                text: request.prompt,
                addSpecialTokens: request.addSpecialTokens ?? true
            )
        )
    }

    public func detokenize(_ request: DetokenizeRequest) async throws -> DetokenizeResponse {
        let tokenizer = await model.tokenizer
        return .init(
            text: tokenizer.decode(
                tokenIds: request.tokens,
                skipSpecialTokens: request.skipSpecialTokens ?? false
            )
        )
    }

    public func applyTemplate(_ request: ApplyTemplateRequest) async throws -> TokenizeResponse {
        let messages = request.messages.map { $0.templateMessage() }
        let tools = request.tools?.map { $0.toolSpec() }
        let tokens = try await model.perform { context in
            try context.tokenizer.applyChatTemplate(
                messages: messages,
                tools: tools,
                additionalContext: nil
            )
        }
        return .init(tokens: tokens)
    }

    private func configureToolParser(for request: OpenAIChatCompletionRequest) async throws {
        let format = try await toolParserFormat(for: request)
        await model.update { context in
            context.configuration.toolCallFormat = format
        }
    }

    private func toolParserFormat(for request: OpenAIChatCompletionRequest) async throws -> ToolCallFormat {
        let configuration = await model.configuration
        let format: ToolCallFormat
        let requested = request.toolCallParser ?? defaultToolCallParser
        if requested == nil, let existing = configuration.toolCallFormat {
            format = existing
        } else {
            format = try ServerToolParser.resolve(
                requested: requested,
                modelType: modelType
            )
        }

        return format
    }
}

public enum MLXServerModelLoader {
    public static func load(
        configuration: ModelConfiguration
    ) async throws -> ModelContainer {
        try await #huggingFaceLoadModelContainer(configuration: configuration)
    }

    /// Load a model and return the raw ``ModelContext`` (bypasses the
    /// serial-access container so ``MLXBatchedEngineServerEngine`` can drive
    /// ``BatchedEngine`` directly).
    public static func loadContext(
        configuration: ModelConfiguration
    ) async throws -> sending ModelContext {
        try await #huggingFaceLoadModel(configuration: configuration)
    }
}
