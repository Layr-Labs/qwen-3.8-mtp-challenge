// Copyright © 2026 Eigen Labs.
//
// Port of omlx/omlx/engine/batched.py — High-level batched engine.
// https://github.com/jundot/omlx/blob/main/omlx/engine/batched.py
//
// Provides the user-facing generate/stream API.
// The EngineCore runs the scheduler loop; BatchedEngine handles
// prompt tokenization, chat templates, and output decoding.

import Foundation
import MLX

/// High-level continuous-batching engine.
///
/// Wraps EngineCore + Scheduler with the tokenization, chat template,
/// and streaming APIs that the server layer needs.
///
/// Usage:
/// ```swift
/// let engine = try BatchedEngine(model: model, tokenizer: tokenizer)
/// await engine.start()
///
/// // Non-streaming
/// let result = await engine.generate(prompt: "Hello", maxTokens: 100)
///
/// // Streaming
/// for await output in engine.streamGenerate(prompt: "Hi", maxTokens: 50) {
///     print(output.text, terminator: "")
/// }
/// ```
public final class BatchedEngine: @unchecked Sendable {
    public let core: EngineCore

    private let tokenizer: any Tokenizer
    public let modelName: String
    /// Chat template loaded from `chat_template.jinja` in the model directory,
    /// used as fallback when the tokenizer config lacks a `chat_template` field.
    private let externalChatTemplate: String?

    /// Create a batched engine from a loaded model context.
    public convenience init(
        context: ModelContext,
        config: ContinuousBatchingConfig = ContinuousBatchingConfig()
    ) {
        let prefixCache = config.prefixCacheConfig.map {
            PrefixCache(config: $0, modelName: context.configuration.name)
        }
        let scheduler = Scheduler(
            model: context.model,
            tokenizer: context.tokenizer,
            config: config.schedulerConfig,
            eosTokenIds: context.configuration.eosTokenIds,
            prefixCache: prefixCache
        )
        // Load chat_template.jinja if the tokenizer lacks an embedded chat template.
        let externalTemplate: String? = (try? context.configuration.modelDirectory).flatMap {
            try? String(contentsOf: $0.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        }
        self.init(
            scheduler: scheduler,
            tokenizer: context.tokenizer,
            modelName: context.configuration.name,
            config: config,
            externalChatTemplate: externalTemplate
        )
    }

    /// Create a batched engine with explicit scheduler and tokenizer.
    public init(
        scheduler: Scheduler,
        tokenizer: any Tokenizer,
        modelName: String = "",
        config: ContinuousBatchingConfig = ContinuousBatchingConfig(),
        externalChatTemplate: String? = nil
    ) {
        self.core = EngineCore(scheduler: scheduler, config: config)
        self.tokenizer = tokenizer
        self.modelName = modelName
        self.externalChatTemplate = externalChatTemplate
    }

    // MARK: - Lifecycle

    /// Start the engine (loads nothing — model is already loaded).
    public func start() async {
        core.start()
    }

    /// Stop the engine and cleanup resources.
    public func stop() async {
        core.stop()
    }

    /// Adjust the engine's runtime concurrency cap. Forwarded to
    /// `EngineCore.setMaxNumSeqs(_:)` which serialises on the engine queue.
    public func setMaxNumSeqs(_ value: Int) {
        core.setMaxNumSeqs(value)
    }

    // MARK: - Non-streaming Generation

    /// Generate a complete response (non-streaming).
    public func generate(prompt: String, samplingParams: SamplingParams = SamplingParams()) async throws -> String {
        try await generateWithResult(prompt: prompt, samplingParams: samplingParams).outputText
    }

    /// Generate with structured result (includes token counts).
    public func generateWithResult(prompt: String, samplingParams: SamplingParams = SamplingParams()) async throws -> RequestOutput {
        try await core.generate(prompt: prompt, samplingParams: samplingParams)
    }

    // MARK: - Streaming Generation

    /// Stream outputs (RequestOutput per step) with full SamplingParams control.
    public func streamOutputs(prompt: String, samplingParams: SamplingParams = SamplingParams()) -> AsyncStream<RequestOutput> {
        let rid = UUID().uuidString
        let request = Request(requestId: rid, prompt: prompt, samplingParams: samplingParams)
        return AsyncStream { continuation in
            Task {
                let _ = await core.addRequest(request)
                for await output in core.streamOutputs(requestId: rid) {
                    continuation.yield(output)
                    if output.finished || output.error != nil { break }
                }
                continuation.finish()
            }
        }
    }

    /// Stream generation token by token (returns text chunks).
    public func streamGenerate(prompt: String, samplingParams: SamplingParams = SamplingParams()) -> AsyncStream<String> {
        AsyncStream { continuation in
            Task {
                for await output in streamOutputs(prompt: prompt, samplingParams: samplingParams) {
                    if !output.newText.isEmpty { continuation.yield(output.newText) }
                    if output.finished || output.error != nil { break }
                }
                continuation.finish()
            }
        }
    }

    // MARK: - Chat Completion

    /// Apply chat template to messages and return the prompt string.
    ///
    /// Tries in order:
    /// 1. The tokenizer's inline `chat_template` (from `tokenizer_config.json`).
    /// 2. The external `chat_template.jinja` loaded from the model directory.
    /// 3. A per-family hand-crafted fallback when swift-jinja can't parse the
    ///    official template (currently Gemma 4: its template uses macros and
    ///    whitespace-control constructs that the swift-jinja lexer rejects
    ///    with `syntax("Unexpected token: multiplicativeBinaryOperator")`).
    /// 4. A generic `role: content` fallback that logs a loud warning — most
    ///    chat-tuned models won't recognize it and will produce degenerate
    ///    output, so the warning is the user's signal to add a per-family
    ///    fallback or upgrade swift-jinja.
    public func buildPrompt(messages: [[String: String]]) -> String {
        buildPrompt(messages: messages.map { message in
            message.mapValues { $0 as any Sendable }
        })
    }

    public func buildPrompt(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) -> String {
        if let ids = try? tokenizer.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: additionalContext)
        {
            return tokenizer.decode(tokenIds: ids)
        }
        if let template = externalChatTemplate,
           let ids = try? tokenizer.applyChatTemplate(
               messages: messages,
               chatTemplate: template,
               tools: tools,
               additionalContext: additionalContext)
        {
            return tokenizer.decode(tokenIds: ids)
        }
        if isGemma4Template(externalChatTemplate) || modelName.lowercased().contains("gemma-4") {
            return Gemma4PromptRenderer.render(
                messages: messages,
                tools: tools,
                additionalContext: additionalContext,
                bosToken: tokenizer.bosToken)
        }
        // Generic plain-text fallback. Most chat-tuned models won't recognize
        // this and will produce degenerate output — emit a loud stderr
        // warning so the user knows the chat template path failed.
        let warning = "[BatchedEngine.buildPrompt] WARNING: chat template unavailable; falling back to plain 'role: content' format. Output may be degenerate. Provide a swift-jinja-compatible chat template or add a per-family fallback in buildPrompt.\n"
        FileHandle.standardError.write(Data(warning.utf8))
        return messages.map { message in
            let role = (message["role"] as? String) ?? "user"
            let content = Gemma4PromptRenderer.contentString(message["content"])
            return "\(role): \(content)"
        }
            .joined(separator: "\n") + "\nassistant:"
    }

    /// Chat completion (non-streaming). Applies chat template.
    public func chat(messages: [[String: String]], samplingParams: SamplingParams = SamplingParams()) async throws -> String {
        let prompt = buildPrompt(messages: messages)
        return try await generate(prompt: prompt, samplingParams: samplingParams)
    }

    /// Stream chat completion.
    public func streamChat(messages: [[String: String]], samplingParams: SamplingParams = SamplingParams()) -> AsyncStream<String> {
        streamGenerate(prompt: buildPrompt(messages: messages), samplingParams: samplingParams)
    }

    // MARK: - Status

    public var hasActiveRequests: Bool {
        core.scheduler.hasRequests()
    }

    public func getStats() -> [String: Any] {
        var stats = core.getStats()
        stats["engine_type"] = "batched"
        stats["model_name"] = modelName
        return stats
    }

    private func isGemma4Template(_ template: String?) -> Bool {
        guard let template else { return false }
        return template.contains("<|turn>") || template.contains("<|channel>")
    }
}

private enum Gemma4PromptRenderer {
    static func render(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?,
        bosToken: String?
    ) -> String {
        var output = bosToken ?? ""
        var loopMessages = messages
        var previousMessageType: String?
        let enableThinking = (additionalContext?["enable_thinking"] as? Bool) ?? false

        if enableThinking || !(tools ?? []).isEmpty || isSystemRole(loopMessages.first) {
            output += "<|turn>system\n"
            if enableThinking {
                output += "<|think|>\n"
                previousMessageType = "think"
            }
            if let first = loopMessages.first, isSystemRole(first) {
                output += contentString(first["content"]).trimmingCharacters(in: .whitespacesAndNewlines)
                loopMessages.removeFirst()
            }
            for tool in tools ?? [] {
                output += "<|tool>"
                output += formatToolDeclaration(tool).trimmingCharacters(in: .whitespacesAndNewlines)
                output += "<tool|>"
                previousMessageType = "tool"
            }
            output += "<turn|>\n"
        }

        for (index, message) in loopMessages.enumerated() {
            guard (message["role"] as? String) != "tool" else { continue }

            previousMessageType = nil
            let originalRole = (message["role"] as? String) ?? "user"
            let role = originalRole == "assistant" ? "model" : originalRole
            output += "<|turn>\(role)\n"

            let toolCalls = dictionaryArray(message["tool_calls"])
            for toolCall in toolCalls {
                output += formatToolCall(toolCall)
                previousMessageType = "tool_call"
            }

            var renderedToolResponse = false
            let responses = dictionaryArray(message["tool_responses"])
            if !responses.isEmpty {
                for response in responses {
                    output += formatToolResponse(
                        name: (response["name"] as? String) ?? "unknown",
                        response: response["response"])
                    renderedToolResponse = true
                    previousMessageType = "tool_response"
                }
            } else if !toolCalls.isEmpty {
                for follow in loopMessages.dropFirst(index + 1) {
                    guard (follow["role"] as? String) == "tool" else { break }
                    let name = toolName(for: follow["tool_call_id"] as? String, in: toolCalls)
                        ?? (follow["name"] as? String)
                        ?? "unknown"
                    output += formatToolResponse(name: name, response: follow["content"])
                    renderedToolResponse = true
                    previousMessageType = "tool_response"
                }
            }

            let content = contentString(message["content"])
            if role == "model" {
                output += stripThinking(content)
            } else {
                output += content.trimmingCharacters(in: .whitespacesAndNewlines)
            }

            if previousMessageType == "tool_call", !renderedToolResponse {
                output += "<|tool_response>"
            } else if !(renderedToolResponse && content.isEmpty) {
                output += "<turn|>\n"
            }
        }

        if previousMessageType != "tool_response", previousMessageType != "tool_call" {
            output += "<|turn>model\n"
        }
        if !enableThinking {
            output += "<|channel>thought\n<channel|>"
        }
        return output
    }

    static func contentString(_ value: (any Sendable)?) -> String {
        if let string = value as? String { return string }
        if let parts = value as? [[String: any Sendable]] {
            return parts.map { part in
                switch part["type"] as? String {
                case "text":
                    return (part["text"] as? String) ?? ""
                case "image":
                    return "<|image|>"
                case "audio":
                    return "<|audio|>"
                case "video":
                    return "<|video|>"
                default:
                    return ""
                }
            }.joined()
        }
        if let parts = value as? [any Sendable] {
            return parts.compactMap { $0 as? [String: any Sendable] }.map { part in
                switch part["type"] as? String {
                case "text":
                    return (part["text"] as? String) ?? ""
                case "image":
                    return "<|image|>"
                case "audio":
                    return "<|audio|>"
                case "video":
                    return "<|video|>"
                default:
                    return ""
                }
            }.joined()
        }
        return value.map { String(describing: $0) } ?? ""
    }

    private static func isSystemRole(_ message: [String: any Sendable]?) -> Bool {
        let role = message?["role"] as? String
        return role == "system" || role == "developer"
    }

    private static func formatToolDeclaration(_ tool: [String: any Sendable]) -> String {
        guard let function = tool["function"] as? [String: any Sendable] else {
            return formatArgument(tool)
        }
        let name = (function["name"] as? String) ?? "unknown"
        let description = (function["description"] as? String) ?? ""
        var result = "declaration:\(name){description:<|\"|>\(description)<|\"|>"
        if let parameters = function["parameters"] as? [String: any Sendable] {
            result += ",parameters:{"
            if let properties = parameters["properties"] as? [String: any Sendable] {
                result += "properties:{ \(formatParameters(properties)) },"
            }
            if let required = parameters["required"] as? [String] {
                result += "required:["
                result += required.map { "<|\"|>\($0)<|\"|>" }.joined(separator: ",")
                result += "],"
            }
            if let type = parameters["type"] as? String {
                result += "type:<|\"|>\(type.uppercased())<|\"|>"
            }
            result += "}"
        }
        result += "}"
        return result
    }

    private static func dictionaryArray(_ value: (any Sendable)?) -> [[String: any Sendable]] {
        if let dictionaries = value as? [[String: any Sendable]] {
            return dictionaries
        }
        if let array = value as? [any Sendable] {
            return array.compactMap { $0 as? [String: any Sendable] }
        }
        return []
    }

    private static func formatParameters(_ properties: [String: any Sendable]) -> String {
        properties.keys.sorted().map { key in
            guard let value = properties[key] as? [String: any Sendable] else {
                return "\(key):{type:<|\"|>STRING<|\"|>}"
            }
            var chunks: [String] = []
            if let description = value["description"] as? String {
                chunks.append("description:<|\"|>\(description)<|\"|>")
            }
            if let enumValues = value["enum"] {
                chunks.append("enum:\(formatArgument(enumValues))")
            }
            if let nested = value["properties"] as? [String: any Sendable] {
                chunks.append("properties:{\(formatParameters(nested))}")
            }
            if let required = value["required"] as? [String] {
                chunks.append("required:[\(required.map { "<|\"|>\($0)<|\"|>" }.joined(separator: ","))]")
            }
            if let type = value["type"] as? String {
                chunks.append("type:<|\"|>\(type.uppercased())<|\"|>")
            }
            return "\(key):{\(chunks.joined(separator: ","))}"
        }.joined(separator: ",")
    }

    private static func formatToolCall(_ toolCall: [String: any Sendable]) -> String {
        guard let function = toolCall["function"] as? [String: any Sendable] else { return "" }
        let name = (function["name"] as? String) ?? "unknown"
        var result = "<|tool_call>call:\(name){"
        if let arguments = function["arguments"] as? [String: any Sendable] {
            result += arguments.keys.sorted().map { key in
                "\(key):\(formatArgument(arguments[key], escapeKeys: false))"
            }.joined(separator: ",")
        } else if let arguments = function["arguments"] as? String {
            if let decoded = decodeArgumentObject(arguments) {
                result += decoded.keys.sorted().map { key in
                    "\(key):\(formatArgument(decoded[key], escapeKeys: false))"
                }.joined(separator: ",")
            } else {
                result += arguments
            }
        }
        result += "}<tool_call|>"
        return result
    }

    private static func decodeArgumentObject(_ arguments: String) -> [String: any Sendable]? {
        guard let data = arguments.data(using: .utf8),
              let decoded = deserializeJSON(data) as? [String: any Sendable]
        else {
            return nil
        }
        return decoded
    }

    private static func formatToolResponse(name: String, response: (any Sendable)?) -> String {
        if let object = response as? [String: any Sendable] {
            let body = object.keys.sorted().map { key in
                "\(key):\(formatArgument(object[key], escapeKeys: false))"
            }.joined(separator: ",")
            return "<|tool_response>response:\(name){\(body)}<tool_response|>"
        }
        return "<|tool_response>response:\(name){value:\(formatArgument(response, escapeKeys: false))}<tool_response|>"
    }

    private static func formatArgument(_ value: (any Sendable)?, escapeKeys: Bool = true) -> String {
        switch value {
        case let string as String:
            return "<|\"|>\(string)<|\"|>"
        case let bool as Bool:
            return bool ? "true" : "false"
        case let int as Int:
            return String(int)
        case let int as Int64:
            return String(int)
        case let double as Double:
            return String(double)
        case let float as Float:
            return String(float)
        case let array as [any Sendable]:
            return "[" + array.map { formatArgument($0, escapeKeys: escapeKeys) }.joined(separator: ",") + "]"
        case let dictionary as [String: any Sendable]:
            let body = dictionary.keys.sorted().map { key in
                let renderedKey = escapeKeys ? "<|\"|>\(key)<|\"|>" : key
                return "\(renderedKey):\(formatArgument(dictionary[key], escapeKeys: escapeKeys))"
            }.joined(separator: ",")
            return "{\(body)}"
        case is NSNull:
            return "null"
        case .none:
            return "null"
        default:
            return String(describing: value!)
        }
    }

    private static func toolName(
        for id: String?,
        in toolCalls: [[String: any Sendable]]
    ) -> String? {
        guard let id else { return nil }
        for call in toolCalls where call["id"] as? String == id {
            let function = call["function"] as? [String: any Sendable]
            return function?["name"] as? String
        }
        return nil
    }

    private static func stripThinking(_ text: String) -> String {
        var result = ""
        var remainder = text[...]
        while let start = remainder.range(of: "<|channel>"),
              let end = remainder.range(of: "<channel|>", range: start.upperBound..<remainder.endIndex)
        {
            result += remainder[..<start.lowerBound]
            remainder = remainder[end.upperBound...]
        }
        result += remainder
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
