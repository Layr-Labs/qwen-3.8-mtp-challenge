// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon
@testable import MLXLMServer
import Testing

/// Regression tests for issue #249 at the MLXLMServer translation boundary.
///
/// `templateMessage()` must hand the Gemma chat template a tool-call
/// `arguments` *mapping* (which renders the `is mapping` branch → single brace
/// `command:<|"|>ls -la<|"|>`) rather than the raw OpenAI JSON *string* (which
/// renders the `is string` branch → corrupting double brace
/// `{{"command":"ls -la"}}`). The OpenAI wire contract is unchanged; only the
/// template-input shape is normalized.
struct ToolCallTemplateArgumentsTests {
    private func assistantWithToolCall(arguments: String) -> OpenAIChatMessage {
        OpenAIChatMessage(
            role: .assistant,
            content: .text(""),
            toolCalls: [
                OpenAIToolCall(
                    id: "call_1",
                    function: .init(name: "run_terminal", arguments: arguments)
                )
            ]
        )
    }

    private func renderedArguments(_ message: OpenAIChatMessage) throws -> any Sendable {
        let dict = message.templateMessage()
        let toolCalls = try #require(dict["tool_calls"] as? [[String: any Sendable]])
        let function = try #require(toolCalls.first?["function"] as? [String: any Sendable])
        return try #require(function["arguments"])
    }

    @Test("templateMessage decodes JSON-string arguments into a mapping")
    func decodesArgumentsIntoMapping() throws {
        let arguments = try renderedArguments(
            assistantWithToolCall(arguments: #"{"command":"ls -la"}"#))
        let mapping = try #require(arguments as? [String: any Sendable])
        #expect(mapping["command"] as? String == "ls -la")
        // Guards the regression: a String here re-enables the double-brace bug.
        #expect(!(arguments is String))
    }

    @Test("templateMessage keeps non-JSON arguments as a string")
    func keepsNonJSONArgumentsAsString() throws {
        let arguments = try renderedArguments(
            assistantWithToolCall(arguments: "not json"))
        #expect(arguments as? String == "not json")
    }
}
