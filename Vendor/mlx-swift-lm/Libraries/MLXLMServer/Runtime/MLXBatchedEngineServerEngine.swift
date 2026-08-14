// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon

/// Continuous-batching server engine that drives ``MLXLMCommon.BatchedEngine``.
///
/// Tool-call format is pinned at construction; per-request
/// `tool_call_parser` overrides that differ from the pinned format throw
/// ``MLXBatchedEngineServerEngineError/unsupportedToolCallParser(pinned:requested:)``
/// because mutating ``ModelContext`` state across concurrent requests is
/// unsafe under continuous batching.
///
/// The implementation is split across files for readability:
/// - `MLXBatchedEngineServerEngine.swift` (this file): class declaration,
///   construction, lifecycle, and the non-streaming
///   ``MLXServerEngine`` conformance methods.
/// - `MLXBatchedEngineServerEngine+Streaming.swift`:
///   ``streamChatCompletion(request:)`` plus cancellation/abort handling.
/// - `MLXBatchedEngineServerEngine+Translation.swift`: OpenAI-request → engine
///   sampling-parameter mapping and chat-template message shaping.
/// - `MLXBatchedEngineServerEngine+ToolCallParser.swift`: per-request
///   `tool_call_parser` override validation and the streaming tool-call
///   handler wrapper.
/// - `MLXBatchedEngineServerEngineError.swift`: error enum and
///   ``LocalizedError`` conformance.
public final class MLXBatchedEngineServerEngine: MLXServerEngine, @unchecked Sendable {
    public let modelID: String
    public let modelType: String?
    /// Tool-call format pinned for every request served by this engine.
    public let toolCallFormat: ToolCallFormat
    public let engine: BatchedEngine

    private let tokenizer: any Tokenizer

    public init(
        modelID: String,
        modelContext: consuming sending ModelContext,
        modelType: String? = nil,
        configuration: BatchedEngineServerConfiguration = .init(),
        defaultToolCallParser: String? = nil
    ) throws {
        self.modelID = modelID
        self.modelType = modelType

        let resolvedFormat: ToolCallFormat
        if defaultToolCallParser == nil,
            let existing = modelContext.configuration.toolCallFormat
        {
            resolvedFormat = existing
        } else {
            resolvedFormat = try ServerToolParser.resolve(
                requested: defaultToolCallParser,
                modelType: modelType
            )
        }
        self.toolCallFormat = resolvedFormat
        self.tokenizer = modelContext.tokenizer

        var context = modelContext
        context.configuration.toolCallFormat = resolvedFormat
        self.engine = BatchedEngine(
            context: context,
            config: configuration.makeContinuousBatchingConfig()
        )
    }

    public init(
        modelID: String,
        engine: BatchedEngine,
        tokenizer: any Tokenizer,
        toolCallFormat: ToolCallFormat,
        modelType: String? = nil
    ) {
        self.modelID = modelID
        self.engine = engine
        self.tokenizer = tokenizer
        self.toolCallFormat = toolCallFormat
        self.modelType = modelType
    }

    /// Start the underlying engine loop. Idempotent.
    public func start() async {
        await engine.start()
    }

    /// Stop the underlying engine loop. Idempotent; safe from a shutdown handler.
    public func stop() async {
        await engine.stop()
    }

    // MARK: - MLXServerEngine

    public func availableModels() async throws -> [MLXServerModel] {
        [.init(id: modelID)]
    }

    // `streamChatCompletion(request:)` lives in
    // `MLXBatchedEngineServerEngine+Streaming.swift`.

    public func tokenize(_ request: TokenizeRequest) async throws -> TokenizeResponse {
        .init(
            tokens: tokenizer.encode(
                text: request.prompt,
                addSpecialTokens: request.addSpecialTokens ?? true
            )
        )
    }

    public func detokenize(_ request: DetokenizeRequest) async throws -> DetokenizeResponse {
        .init(
            text: tokenizer.decode(
                tokenIds: request.tokens,
                skipSpecialTokens: request.skipSpecialTokens ?? false
            )
        )
    }

    public func applyTemplate(_ request: ApplyTemplateRequest) async throws -> TokenizeResponse {
        let messages = request.messages.map { $0.templateMessage() }
        let tools = request.tools?.map { $0.toolSpec() }
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: tools,
            additionalContext: nil
        )
        return .init(tokens: tokens)
    }
}
