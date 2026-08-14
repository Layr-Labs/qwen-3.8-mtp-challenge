// Tests for BatchedEngine — §12 of ContinuousBatchingTestPlan.md

import Foundation
import XCTest

@testable import MLXLMCommon

final class CBBatchedEngineTests: XCTestCase {

    private func makeEngine() -> BatchedEngine {
        BatchedEngine(
            scheduler: makeTestScheduler(eosTokenIds: [5]),
            tokenizer: IdentityTokenizer(),
            modelName: "test-model"
        )
    }

    func testBatchedEngineGenerateReturnsString() async throws {
        let engine = makeEngine()
        await engine.start()

        // "3" encodes to [3] via IdentityTokenizer → model generates 4, then 5 (EOS)
        // outputTokenIds = [4], decoded = "4" (non-empty)
        let result = try await engine.generate(prompt: "3", samplingParams: SamplingParams(maxTokens: 10))

        XCTAssertFalse(result.isEmpty, "generate must return a non-empty string")
        await engine.stop()
    }

    func testBatchedEngineStreamGenerateYieldsChunks() async throws {
        let engine = makeEngine()
        await engine.start()

        var chunks: [String] = []
        for await chunk in engine.streamGenerate(prompt: "3", samplingParams: SamplingParams(maxTokens: 10)) {
            chunks.append(chunk)
        }

        XCTAssertFalse(chunks.isEmpty, "stream must yield at least one chunk")
        await engine.stop()
    }

    func testBatchedEngineChatAppliesTemplate() async throws {
        let engine = makeEngine()
        await engine.start()

        // applyChatTemplate on IdentityTokenizer wraps messages as "[role]:content"
        // and encodes them. The resulting prompt should produce valid output.
        let messages: [[String: String]] = [
            ["role": "user", "content": "hi"],
        ]
        let result = try await engine.chat(messages: messages, samplingParams: SamplingParams(maxTokens: 5))

        // Just verify it completes without throwing.
        XCTAssertNotNil(result)
        await engine.stop()
    }

    func testGemma4FallbackPreservesToolsAndToolHistory() {
        let engine = BatchedEngine(
            scheduler: makeTestScheduler(eosTokenIds: [5]),
            tokenizer: MissingTemplateTokenizer(),
            modelName: "gemma-4-test",
            externalChatTemplate: "<bos>\n<|turn>system\n<|channel>thought\n<channel|>"
        )
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "Use tools when needed."],
            ["role": "user", "content": "Weather in Paris?"],
            [
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    [
                        "id": "call_1",
                        "type": "function",
                        "function": [
                            "name": "get_weather",
                            "arguments": "{\"location\":\"Paris\"}",
                        ] as [String: any Sendable],
                    ] as [String: any Sendable]
                ] as [[String: any Sendable]],
            ],
            [
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "{\"temperature\": \"18C\"}",
            ],
            ["role": "user", "content": "Summarize that."],
        ]
        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get the weather.",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "location": [
                                "type": "string",
                                "description": "City name.",
                            ] as [String: any Sendable]
                        ] as [String: any Sendable],
                        "required": ["location"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]

        let prompt = engine.buildPrompt(
            messages: messages,
            tools: tools,
            additionalContext: ["enable_thinking": false]
        )

        XCTAssertTrue(prompt.hasPrefix("<bos>"))
        XCTAssertTrue(prompt.contains("<|tool>declaration:get_weather"))
        XCTAssertTrue(prompt.contains("<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>"))
        XCTAssertTrue(prompt.contains("<|tool_response>response:get_weather{value:<|\"|>{\"temperature\": \"18C\"}<|\"|>}<tool_response|>"))
        XCTAssertTrue(prompt.contains("<|turn>user\nSummarize that.<turn|>"))
        XCTAssertTrue(prompt.hasSuffix("<|turn>model\n<|channel>thought\n<channel|>"))
    }

    func testExternalTemplateReceivesToolsAndAdditionalContext() {
        let engine = BatchedEngine(
            scheduler: makeTestScheduler(eosTokenIds: [5]),
            tokenizer: ExternalTemplateTokenizer(),
            modelName: "test-model",
            externalChatTemplate: "external-template"
        )

        let prompt = engine.buildPrompt(
            messages: [["role": "user", "content": "hi"]],
            tools: [["type": "function"]],
            additionalContext: ["enable_thinking": false]
        )

        XCTAssertEqual(prompt, "external-template-with-tools")
    }
}

private struct MissingTemplateTokenizer: Tokenizer {
    var bosToken: String? { "<bos>" }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }
}

private struct ExternalTemplateTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds == [7] ? "external-template-with-tools" : "unexpected"
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        chatTemplate: String,
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        guard chatTemplate == "external-template",
              tools?.isEmpty == false,
              additionalContext?["enable_thinking"] as? Bool == false
        else {
            throw TokenizerError.missingChatTemplate
        }
        return [7]
    }
}
