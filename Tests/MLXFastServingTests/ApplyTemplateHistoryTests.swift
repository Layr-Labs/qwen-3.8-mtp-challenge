import Foundation
import MLXLMCommon
@testable import MLXLMServer
import Testing

struct ApplyTemplateHistoryTests {
    @Test("Chat template messages keep tool calls, tool results, and reasoning")
    func templateMessageKeepsToolHistory() throws {
        let assistant = OpenAIChatMessage(
            role: .assistant,
            content: .null,
            toolCalls: [
                OpenAIToolCall(
                    id: "call_1",
                    function: .init(name: "read", arguments: #"{"path":"notes.txt"}"#)
                )
            ],
            reasoningContent: "look"
        )
        let assistantEntry = assistant.templateMessage()
        #expect(assistantEntry["role"] as? String == "assistant")
        #expect(assistantEntry["reasoning_content"] as? String == "look")
        let calls = try #require(assistantEntry["tool_calls"] as? [[String: any Sendable]])
        let function = try #require(calls.first?["function"] as? [String: any Sendable])
        #expect(function["name"] as? String == "read")
        let arguments = try #require(function["arguments"] as? [String: any Sendable])
        #expect(arguments["path"] as? String == "notes.txt")

        let tool = OpenAIChatMessage(
            role: .tool,
            content: .text("line-one"),
            toolCallID: "call_1"
        )
        let toolEntry = tool.templateMessage()
        #expect(toolEntry["tool_call_id"] as? String == "call_1")
        #expect(toolEntry["content"] as? String == "line-one")
    }

    @Test("applyTemplate builds messages with templateMessage")
    func applyTemplateUsesTemplateMessage() throws {
        let source = try String(
            contentsOf: URL(filePath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appending(path: "Vendor/mlx-swift-lm/Libraries/MLXLMServer/Runtime/MLXModelContainerEngine.swift"),
            encoding: .utf8
        )
        let start = try #require(source.range(of: "func applyTemplate"))
        let end = try #require(
            source.range(of: "private func configureToolParser", range: start.upperBound..<source.endIndex)
        )
        let body = source[start.lowerBound..<end.lowerBound]
        #expect(body.contains("templateMessage()"))
        #expect(body.contains("message.textContent") == false)
    }
}
