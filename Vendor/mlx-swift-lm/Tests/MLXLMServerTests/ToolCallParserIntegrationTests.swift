// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon
@testable import MLXLMServer
import Testing

// MARK: - BatchedToolStreamHandler Integration Tests

/// Integration tests for the tool call parsing pipeline:
/// ToolCallFormat.infer -> createParser -> BatchedToolStreamHandler -> ToolCall extraction.
///
/// These tests exercise the ``BatchedToolStreamHandler`` wrapper that is used
/// in the real streaming-completion path, verifying that chunked delivery of
/// model output correctly produces tool calls.
struct ToolCallParserIntegrationTests {

    // MARK: - BatchedToolStreamHandler: Gemma Format

    @Test("BatchedToolStreamHandler extracts Gemma tool call from single chunk")
    func batchedHandlerGemmaSingleChunk() throws {
        let handler = BatchedToolStreamHandler(format: .gemma, tools: nil)

        let result = handler.processChunk(
            "<|tool_call>call:get_weather{city:Paris,units:metric}<tool_call|>"
        )
        // Tool call markup should be consumed (nil)
        #expect(result == nil)

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)

        let tc = try #require(toolCalls.first)
        #expect(tc.function.name == "get_weather")
        #expect(tc.function.arguments["city"] == .string("Paris"))
        #expect(tc.function.arguments["units"] == .string("metric"))
    }

    @Test("BatchedToolStreamHandler extracts Gemma tool call from multiple chunks")
    func batchedHandlerGemmaMultipleChunks() throws {
        let handler = BatchedToolStreamHandler(format: .gemma, tools: nil)

        let chunks = [
            "<|tool_call>",
            "call:get_weather",
            "{city:Paris}",
            "<tool_call|>",
        ]

        for chunk in chunks {
            _ = handler.processChunk(chunk)
        }

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)

        let tc = try #require(toolCalls.first)
        #expect(tc.function.name == "get_weather")
        #expect(tc.function.arguments["city"] == .string("Paris"))
    }

    @Test("BatchedToolStreamHandler separates visible text from Gemma tool call")
    func batchedHandlerGemmaWithLeadingText() throws {
        let handler = BatchedToolStreamHandler(format: .gemma, tools: nil)

        // First chunk is regular text
        let visible1 = handler.processChunk("Let me check the weather.")
        #expect(visible1 == "Let me check the weather.")

        // Then tool call arrives
        let visible2 = handler.processChunk(
            "<|tool_call>call:get_weather{city:Tokyo}<tool_call|>"
        )
        #expect(visible2 == nil)

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "get_weather")
        #expect(toolCalls[0].function.arguments["city"] == .string("Tokyo"))
    }

    @Test("BatchedToolStreamHandler handles Gemma escaped string arguments")
    func batchedHandlerGemmaEscapedStrings() throws {
        let handler = BatchedToolStreamHandler(format: .gemma, tools: nil)

        _ = handler.processChunk(
            "<|tool_call>call:search{query:<|\"|>hello world<|\"|>}<tool_call|>"
        )

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)

        let tc = try #require(toolCalls.first)
        #expect(tc.function.name == "search")
        #expect(tc.function.arguments["query"] == .string("hello world"))
    }

    @Test("BatchedToolStreamHandler handles Gemma legacy tags")
    func batchedHandlerGemmaLegacyTags() throws {
        let handler = BatchedToolStreamHandler(format: .gemma, tools: nil)

        _ = handler.processChunk(
            "<start_function_call>call:get_weather{location:<escape>San Francisco<escape>}<end_function_call>"
        )

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "get_weather")
        #expect(toolCalls[0].function.arguments["location"] == .string("San Francisco"))
    }

    // MARK: - BatchedToolStreamHandler: Harmony Format

    @Test("BatchedToolStreamHandler extracts Harmony tool call from single chunk")
    func batchedHandlerHarmonySingleChunk() throws {
        let handler = BatchedToolStreamHandler(format: .harmony, tools: nil)

        _ = handler.processChunk(
            "<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{\"location\":\"Paris\",\"unit\":\"celsius\"}<|call|>"
        )

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)

        let tc = try #require(toolCalls.first)
        #expect(tc.function.name == "get_weather")
        #expect(tc.function.arguments["location"] == .string("Paris"))
        #expect(tc.function.arguments["unit"] == .string("celsius"))
    }

    @Test("BatchedToolStreamHandler extracts Harmony tool call from streamed chunks")
    func batchedHandlerHarmonyStreamedChunks() throws {
        let handler = BatchedToolStreamHandler(format: .harmony, tools: nil)

        let chunks = [
            "<|start|>assistant to=functions.get_weather",
            "<|channel|>commentary json<|message|>",
            "{\"location\":\"Tokyo\"}",
            "<|call|>",
        ]

        var visibleText = ""
        for chunk in chunks {
            if let text = handler.processChunk(chunk) {
                visibleText += text
            }
        }

        // No visible text should leak from tool call markup
        #expect(visibleText.isEmpty)

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "get_weather")
        #expect(toolCalls[0].function.arguments["location"] == .string("Tokyo"))
    }

    @Test("BatchedToolStreamHandler handles Harmony commentary-channel variant")
    func batchedHandlerHarmonyCommentaryChannel() throws {
        let handler = BatchedToolStreamHandler(format: .harmony, tools: nil)

        _ = handler.processChunk(
            "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"Berlin\"}"
        )

        // No end tag - use processEOS via finish()
        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "get_weather")
        #expect(toolCalls[0].function.arguments["location"] == .string("Berlin"))
    }

    @Test("BatchedToolStreamHandler separates Harmony analysis channel from tool call")
    func batchedHandlerHarmonyWithAnalysis() throws {
        let handler = BatchedToolStreamHandler(format: .harmony, tools: nil)

        // Analysis block is not a tool call - should pass through
        let visible1 = handler.processChunk(
            "<|channel|>analysis<|message|>Need weather data.<|end|>"
        )
        #expect(visible1 != nil)

        // Tool call follows
        _ = handler.processChunk(
            "<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{\"location\":\"Paris\"}<|call|>"
        )

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "get_weather")
    }

    @Test("BatchedToolStreamHandler handles Harmony EOS without end tag")
    func batchedHandlerHarmonyEOS() throws {
        let handler = BatchedToolStreamHandler(format: .harmony, tools: nil)

        // Harmony call without closing <|call|> - processEOS should extract it
        _ = handler.processChunk(
            "<|start|>assistant to=functions.search<|channel|>commentary json<|message|>{\"query\":\"mlx swift\"}"
        )

        // Before finish(), no tool calls yet
        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "search")
        #expect(toolCalls[0].function.arguments["query"] == .string("mlx swift"))
    }

    // MARK: - BatchedToolStreamHandler: Mistral Format

    @Test("BatchedToolStreamHandler extracts Mistral tool call via processEOS")
    func batchedHandlerMistralEOS() throws {
        let handler = BatchedToolStreamHandler(format: .mistral, tools: nil)

        let chunks = [
            "[TOOL_CALLS]",
            "get_weather",
            " [ARGS]",
            "{\"location\": \"Paris\"}",
        ]

        for chunk in chunks {
            _ = handler.processChunk(chunk)
        }

        // Mistral has no end tag - tool call only extracted on finish()
        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)

        let tc = try #require(toolCalls.first)
        #expect(tc.function.name == "get_weather")
        #expect(tc.function.arguments["location"] == .string("Paris"))
    }

    @Test("BatchedToolStreamHandler extracts multiple Mistral tool calls via processEOS")
    func batchedHandlerMistralMultipleToolCalls() throws {
        let handler = BatchedToolStreamHandler(format: .mistral, tools: nil)

        let chunks = [
            "[TOOL_CALLS]get_weather[ARGS]{\"location\": \"Paris\"}",
            "[TOOL_CALLS]get_time[ARGS]{\"timezone\": \"CET\"}",
        ]

        for chunk in chunks {
            _ = handler.processChunk(chunk)
        }

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 2)
        #expect(toolCalls[0].function.name == "get_weather")
        #expect(toolCalls[0].function.arguments["location"] == .string("Paris"))
        #expect(toolCalls[1].function.name == "get_time")
        #expect(toolCalls[1].function.arguments["timezone"] == .string("CET"))
    }

    // MARK: - BatchedToolStreamHandler: XML Function Format (Qwen3.5)

    @Test("BatchedToolStreamHandler extracts XML Function tool call with streaming")
    func batchedHandlerXMLFunctionStreaming() throws {
        let handler = BatchedToolStreamHandler(format: .xmlFunction, tools: nil)

        let chunks = [
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=location>Tokyo</parameter>\n",
            "<parameter=unit>celsius</parameter>\n",
            "</function>\n</tool_call>",
        ]

        for chunk in chunks {
            _ = handler.processChunk(chunk)
        }

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)

        let tc = try #require(toolCalls.first)
        #expect(tc.function.name == "get_weather")
        #expect(tc.function.arguments["location"] == .string("Tokyo"))
        #expect(tc.function.arguments["unit"] == .string("celsius"))
    }

    @Test("BatchedToolStreamHandler extracts XML Function with no arguments")
    func batchedHandlerXMLFunctionNoArgs() throws {
        let handler = BatchedToolStreamHandler(format: .xmlFunction, tools: nil)

        _ = handler.processChunk(
            "<tool_call>\n<function=get_current_datetime>\n</function>\n</tool_call>"
        )

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "get_current_datetime")
        #expect(toolCalls[0].function.arguments.isEmpty)
    }

    // MARK: - BatchedToolStreamHandler: JSON Format

    @Test("BatchedToolStreamHandler extracts JSON tool call with fine-grained chunks")
    func batchedHandlerJSONFineGrainedChunks() throws {
        let handler = BatchedToolStreamHandler(format: .json, tools: nil)

        let chunks = [
            "<tool_call>", "{\"name\":", " \"get_weather\",",
            " \"arguments\": {\"location\":", " \"Paris\"}}",
            "</tool_call>",
        ]

        for chunk in chunks {
            _ = handler.processChunk(chunk)
        }

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)

        let tc = try #require(toolCalls.first)
        #expect(tc.function.name == "get_weather")
        #expect(tc.function.arguments["location"] == .string("Paris"))
    }

    // MARK: - BatchedToolStreamHandler: Declared-Tool Filter (JSON)

    /// A realistic OpenAI-style nested tool spec: `{"type":"function",
    /// "function":{"name": ...}}`. The JSON parser's declared-tool filter reads
    /// `tool["function"]["name"]` (upstream 1335fb5).
    private func functionTool(named name: String) -> [String: any Sendable] {
        [
            "type": "function",
            "function": [
                "name": name,
                "description": "test tool \(name)",
            ] as [String: any Sendable],
        ]
    }

    @Test("Declared JSON tool call is kept when its name matches a declared tool")
    func batchedHandlerJSONDeclaredToolKept() throws {
        let handler = BatchedToolStreamHandler(
            format: .json, tools: [functionTool(named: "get_weather")])

        let result = handler.processChunk(
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}</tool_call>"
        )
        #expect(result == nil)

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)

        let tc = try #require(toolCalls.first)
        #expect(tc.function.name == "get_weather")
        #expect(tc.function.arguments["location"] == .string("Paris"))
    }

    @Test("Undeclared JSON tool call is dropped when its name is not a declared tool")
    func batchedHandlerJSONUndeclaredToolDropped() throws {
        // Only `get_weather` is declared; the model emits a call to a tool that
        // was never offered. Bare JSON that merely looks like a tool call must
        // not be misparsed into one.
        let handler = BatchedToolStreamHandler(
            format: .json, tools: [functionTool(named: "get_weather")])

        _ = handler.processChunk(
            "<tool_call>{\"name\": \"transfer_funds\", \"arguments\": {\"amount\": 1000}}</tool_call>"
        )

        let toolCalls = handler.finish()
        #expect(toolCalls.isEmpty)
    }

    @Test("Declared JSON tool call is kept when multiple tools are declared")
    func batchedHandlerJSONDeclaredAmongMultipleKept() throws {
        let handler = BatchedToolStreamHandler(
            format: .json,
            tools: [
                functionTool(named: "get_weather"),
                functionTool(named: "get_time"),
            ])

        _ = handler.processChunk(
            "<tool_call>{\"name\": \"get_time\", \"arguments\": {\"timezone\": \"CET\"}}</tool_call>"
        )

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)
        #expect(toolCalls.first?.function.name == "get_time")
        #expect(toolCalls.first?.function.arguments["timezone"] == .string("CET"))
    }

    // MARK: - BatchedToolStreamHandler: LFM2 Format

    @Test("BatchedToolStreamHandler extracts LFM2 Pythonic tool call")
    func batchedHandlerLFM2() throws {
        let handler = BatchedToolStreamHandler(format: .lfm2, tools: nil)

        _ = handler.processChunk(
            "<|tool_call_start|>[calculator(expression='2+2')]<|tool_call_end|>"
        )

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "calculator")
        #expect(toolCalls[0].function.arguments["expression"] == .string("2+2"))
    }

    // MARK: - BatchedToolStreamHandler: GLM4 Format

    @Test("BatchedToolStreamHandler extracts GLM4 tool call")
    func batchedHandlerGLM4() throws {
        let handler = BatchedToolStreamHandler(format: .glm4, tools: nil)

        _ = handler.processChunk(
            "<tool_call>search<arg_key>query</arg_key><arg_value>machine learning</arg_value></tool_call>"
        )

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "search")
        #expect(toolCalls[0].function.arguments["query"] == .string("machine learning"))
    }

    // MARK: - BatchedToolStreamHandler: Kimi K2 Format

    @Test("BatchedToolStreamHandler extracts Kimi K2 tool call")
    func batchedHandlerKimiK2() throws {
        let handler = BatchedToolStreamHandler(format: .kimiK2, tools: nil)

        _ = handler.processChunk(
            "<|tool_calls_section_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\": \"London\"}<|tool_calls_section_end|>"
        )

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "get_weather")
        #expect(toolCalls[0].function.arguments["location"] == .string("London"))
    }

    // MARK: - BatchedToolStreamHandler: MiniMax M2 Format

    @Test("BatchedToolStreamHandler extracts MiniMax M2 tool call")
    func batchedHandlerMiniMaxM2() throws {
        let handler = BatchedToolStreamHandler(format: .minimaxM2, tools: nil)

        _ = handler.processChunk(
            "<minimax:tool_call><invoke name=\"get_weather\"><parameter name=\"location\">Sydney</parameter></invoke></minimax:tool_call>"
        )

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "get_weather")
        #expect(toolCalls[0].function.arguments["location"] == .string("Sydney"))
    }

    // MARK: - BatchedToolStreamHandler: Llama 3 Format

    @Test("BatchedToolStreamHandler extracts Llama 3 JSON tool call via processEOS")
    func batchedHandlerLlama3EOS() throws {
        let handler = BatchedToolStreamHandler(format: .llama3, tools: nil)

        _ = handler.processChunk(
            "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"location\": \"NYC\"}}"
        )

        // Llama 3 uses inline format - EOS triggers extraction
        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "get_weather")
        #expect(toolCalls[0].function.arguments["location"] == .string("NYC"))
    }

    // MARK: - BatchedToolStreamHandler: finish() idempotence

    @Test("BatchedToolStreamHandler finish() returns empty when no tool calls present")
    func batchedHandlerFinishNoToolCalls() {
        let handler = BatchedToolStreamHandler(format: .gemma, tools: nil)

        // Regular text only
        let visible = handler.processChunk("The weather in Paris is sunny.")
        #expect(visible == "The weather in Paris is sunny.")

        let toolCalls = handler.finish()
        #expect(toolCalls.isEmpty)
    }

    // MARK: - Full Pipeline: Format Inference -> BatchedToolStreamHandler

    @Test("Full pipeline: gemma2 model type -> Gemma parser -> streaming extraction")
    func fullPipelineGemma() throws {
        let format = try #require(ToolCallFormat.infer(from: "gemma2"))
        #expect(format == .gemma)

        let handler = BatchedToolStreamHandler(format: format, tools: nil)
        _ = handler.processChunk(
            "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>"
        )

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "get_weather")
        #expect(toolCalls[0].function.arguments["city"] == .string("Paris"))
    }

    @Test("Full pipeline: gpt_oss model type -> Harmony parser -> streaming extraction")
    func fullPipelineHarmony() throws {
        let format = try #require(ToolCallFormat.infer(from: "gpt_oss"))
        #expect(format == .harmony)

        let handler = BatchedToolStreamHandler(format: format, tools: nil)
        _ = handler.processChunk(
            "<|start|>assistant to=functions.search<|channel|>commentary json<|message|>{\"query\":\"test\"}<|call|>"
        )

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "search")
        #expect(toolCalls[0].function.arguments["query"] == .string("test"))
    }

    @Test("Full pipeline: mistral3 model type -> Mistral parser -> EOS extraction")
    func fullPipelineMistral() throws {
        let format = try #require(ToolCallFormat.infer(from: "mistral3"))
        #expect(format == .mistral)

        let handler = BatchedToolStreamHandler(format: format, tools: nil)
        _ = handler.processChunk("[TOOL_CALLS]get_weather [ARGS]{\"location\": \"Berlin\"}")

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "get_weather")
        #expect(toolCalls[0].function.arguments["location"] == .string("Berlin"))
    }

    @Test("Full pipeline: qwen3_5 model type -> XML Function parser -> streaming extraction")
    func fullPipelineQwen35() throws {
        let format = try #require(ToolCallFormat.infer(from: "qwen3_5"))
        #expect(format == .xmlFunction)

        let handler = BatchedToolStreamHandler(format: format, tools: nil)
        _ = handler.processChunk(
            "<tool_call>\n<function=get_weather>\n<parameter=location>Tokyo</parameter>\n</function>\n</tool_call>"
        )

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "get_weather")
        #expect(toolCalls[0].function.arguments["location"] == .string("Tokyo"))
    }

    // MARK: - ServerToolParser.resolve Integration

    @Test("ServerToolParser.resolve maps named formats correctly")
    func serverToolParserResolveNamedFormats() throws {
        // Direct format names
        #expect(try ServerToolParser.resolve(requested: "json", modelType: nil) == .json)
        #expect(try ServerToolParser.resolve(requested: "default", modelType: nil) == .json)
        #expect(try ServerToolParser.resolve(requested: "gemma", modelType: nil) == .gemma)
        #expect(try ServerToolParser.resolve(requested: "gemma4", modelType: nil) == .gemma)
        #expect(try ServerToolParser.resolve(requested: "gemma_4", modelType: nil) == .gemma)
        #expect(try ServerToolParser.resolve(requested: "harmony", modelType: nil) == .harmony)
        #expect(try ServerToolParser.resolve(requested: "gpt_oss", modelType: nil) == .harmony)
        #expect(try ServerToolParser.resolve(requested: "openai_harmony", modelType: nil) == .harmony)
        #expect(try ServerToolParser.resolve(requested: "mistral", modelType: nil) == .mistral)
        #expect(try ServerToolParser.resolve(requested: "mistral_v11", modelType: nil) == .mistral)
        #expect(try ServerToolParser.resolve(requested: "llama3", modelType: nil) == .llama3)
        #expect(try ServerToolParser.resolve(requested: "llama3_json", modelType: nil) == .llama3)
        #expect(try ServerToolParser.resolve(requested: "llama_3", modelType: nil) == .llama3)
        #expect(try ServerToolParser.resolve(requested: "lfm2", modelType: nil) == .lfm2)
        #expect(try ServerToolParser.resolve(requested: "lfm2_5", modelType: nil) == .lfm2)
        #expect(try ServerToolParser.resolve(requested: "lfm25", modelType: nil) == .lfm2)
        #expect(try ServerToolParser.resolve(requested: "xml", modelType: nil) == .xmlFunction)
        #expect(try ServerToolParser.resolve(requested: "xml_function", modelType: nil) == .xmlFunction)
        #expect(try ServerToolParser.resolve(requested: "qwen_xml", modelType: nil) == .xmlFunction)
        #expect(try ServerToolParser.resolve(requested: "hermes", modelType: nil) == .xmlFunction)
        #expect(try ServerToolParser.resolve(requested: "nemotron", modelType: nil) == .xmlFunction)
        #expect(try ServerToolParser.resolve(requested: "glm4", modelType: nil) == .glm4)
        #expect(try ServerToolParser.resolve(requested: "glm_4", modelType: nil) == .glm4)
        #expect(try ServerToolParser.resolve(requested: "kimi_k2", modelType: nil) == .kimiK2)
        #expect(try ServerToolParser.resolve(requested: "kimi", modelType: nil) == .kimiK2)
        #expect(try ServerToolParser.resolve(requested: "minimax_m2", modelType: nil) == .minimaxM2)
        #expect(try ServerToolParser.resolve(requested: "minimax", modelType: nil) == .minimaxM2)
    }

    @Test("ServerToolParser.resolve auto-detects from model type")
    func serverToolParserResolveAutoDetects() throws {
        #expect(try ServerToolParser.resolve(requested: nil, modelType: "gemma2") == .gemma)
        #expect(try ServerToolParser.resolve(requested: nil, modelType: "gemma4") == .gemma)
        #expect(try ServerToolParser.resolve(requested: nil, modelType: "gpt_oss") == .harmony)
        #expect(try ServerToolParser.resolve(requested: nil, modelType: "mistral3") == .mistral)
        #expect(try ServerToolParser.resolve(requested: nil, modelType: "qwen3_5") == .xmlFunction)
        #expect(try ServerToolParser.resolve(requested: nil, modelType: "qwen3_next") == .xmlFunction)
        #expect(try ServerToolParser.resolve(requested: nil, modelType: "nemotron_h") == .xmlFunction)
        #expect(try ServerToolParser.resolve(requested: nil, modelType: "lfm2") == .lfm2)
        #expect(try ServerToolParser.resolve(requested: nil, modelType: "glm4") == .glm4)
        #expect(try ServerToolParser.resolve(requested: "auto", modelType: "qwen3_5") == .xmlFunction)
        #expect(try ServerToolParser.resolve(requested: "auto", modelType: "gemma4") == .gemma)
    }

    @Test("ServerToolParser.resolve falls back to JSON for unknown model types")
    func serverToolParserResolveFallsBackToJSON() throws {
        // No requested, unknown model type -> .json default
        #expect(try ServerToolParser.resolve(requested: nil, modelType: "qwen2") == .json)
        #expect(try ServerToolParser.resolve(requested: nil, modelType: "phi4") == .json)
        #expect(try ServerToolParser.resolve(requested: nil, modelType: nil) == .json)
        #expect(try ServerToolParser.resolve(requested: "auto", modelType: nil) == .json)
    }

    @Test("ServerToolParser.resolve throws on unsupported format name")
    func serverToolParserResolveThrowsOnUnsupported() {
        #expect(throws: ServerToolParserError.self) {
            _ = try ServerToolParser.resolve(requested: "nonexistent_format", modelType: nil)
        }
    }

    @Test("ServerToolParser.resolve normalizes hyphens to underscores")
    func serverToolParserResolveNormalizesHyphens() throws {
        #expect(try ServerToolParser.resolve(requested: "kimi-k2", modelType: nil) == .kimiK2)
        #expect(try ServerToolParser.resolve(requested: "minimax-m2", modelType: nil) == .minimaxM2)
        #expect(try ServerToolParser.resolve(requested: "xml-function", modelType: nil) == .xmlFunction)
        #expect(try ServerToolParser.resolve(requested: "llama3-json", modelType: nil) == .llama3)
    }

    @Test("ServerToolParser.resolve is case-insensitive")
    func serverToolParserResolveCaseInsensitive() throws {
        #expect(try ServerToolParser.resolve(requested: "GEMMA", modelType: nil) == .gemma)
        #expect(try ServerToolParser.resolve(requested: "Harmony", modelType: nil) == .harmony)
        #expect(try ServerToolParser.resolve(requested: "MISTRAL", modelType: nil) == .mistral)
        #expect(try ServerToolParser.resolve(requested: "JSON", modelType: nil) == .json)
    }

    // MARK: - ToolCallFormat.infer Comprehensive Coverage

    @Test("ToolCallFormat.infer returns correct format for all known model families")
    func toolCallFormatInferAllFamilies() {
        // Gemma family
        #expect(ToolCallFormat.infer(from: "gemma") == .gemma)
        #expect(ToolCallFormat.infer(from: "gemma2") == .gemma)
        #expect(ToolCallFormat.infer(from: "gemma4") == .gemma)
        #expect(ToolCallFormat.infer(from: "gemma4_text") == .gemma)
        #expect(ToolCallFormat.infer(from: "GEMMA") == .gemma)

        // GPT-OSS / Harmony
        #expect(ToolCallFormat.infer(from: "gpt_oss") == .harmony)

        // Mistral3
        #expect(ToolCallFormat.infer(from: "mistral3") == .mistral)
        #expect(ToolCallFormat.infer(from: "mistral3_text") == .mistral)
        #expect(ToolCallFormat.infer(from: "MISTRAL3") == .mistral)

        // Qwen3.5 / Qwen3-Next / Nemotron -> xmlFunction
        #expect(ToolCallFormat.infer(from: "qwen3_5") == .xmlFunction)
        #expect(ToolCallFormat.infer(from: "qwen3_5_moe") == .xmlFunction)
        #expect(ToolCallFormat.infer(from: "qwen3_next") == .xmlFunction)
        #expect(ToolCallFormat.infer(from: "nemotron_h") == .xmlFunction)

        // LFM2 family
        #expect(ToolCallFormat.infer(from: "lfm2") == .lfm2)
        #expect(ToolCallFormat.infer(from: "lfm2_moe") == .lfm2)
        #expect(ToolCallFormat.infer(from: "lfm2_5") == .lfm2)

        // GLM4 family
        #expect(ToolCallFormat.infer(from: "glm4") == .glm4)
        #expect(ToolCallFormat.infer(from: "glm4_moe") == .glm4)
    }

    @Test("ToolCallFormat.infer returns nil for models without specialized format")
    func toolCallFormatInferReturnsNilForUnknown() {
        #expect(ToolCallFormat.infer(from: "qwen2") == nil)
        #expect(ToolCallFormat.infer(from: "phi4") == nil)
        #expect(ToolCallFormat.infer(from: "mistral") == nil)  // Not mistral3
        #expect(ToolCallFormat.infer(from: "llama") == nil)  // Needs configData
        #expect(ToolCallFormat.infer(from: "starcoder") == nil)
    }

    @Test("ToolCallFormat.infer detects Llama 3 via vocab_size signal")
    func toolCallFormatInferLlama3VocabSize() {
        let config = """
            {"model_type": "llama", "vocab_size": 128256}
            """.data(using: .utf8)!
        #expect(ToolCallFormat.infer(from: "llama", configData: config) == .llama3)

        // Below threshold
        let smallConfig = """
            {"model_type": "llama", "vocab_size": 32000}
            """.data(using: .utf8)!
        #expect(ToolCallFormat.infer(from: "llama", configData: smallConfig) == nil)
    }

    @Test("ToolCallFormat.infer detects Llama 3 via rope_scaling signal")
    func toolCallFormatInferLlama3RopeScaling() {
        let config = """
            {"model_type": "llama", "rope_scaling": {"rope_type": "llama3"}}
            """.data(using: .utf8)!
        #expect(ToolCallFormat.infer(from: "llama", configData: config) == .llama3)
    }

    // MARK: - ToolCallFormat.createParser round-trip

    @Test("Every ToolCallFormat case creates a valid parser with expected tags")
    func toolCallFormatCreateParserAllCases() {
        for format in ToolCallFormat.allCases {
            let parser = format.createParser()

            switch format {
            case .json:
                #expect(parser.startTag == "<tool_call>")
                #expect(parser.endTag == "</tool_call>")
            case .lfm2:
                #expect(parser.startTag == "<|tool_call_start|>")
                #expect(parser.endTag == "<|tool_call_end|>")
            case .gemma:
                #expect(parser.startTag == "<|tool_call>")
                #expect(parser.endTag == "<tool_call|>")
            case .harmony:
                #expect(parser.startTag == "<|start|>assistant to=functions.")
                #expect(parser.endTag == "<|call|>")
            case .mistral:
                #expect(parser.startTag == "[TOOL_CALLS]")
                #expect(parser.endTag == nil)  // Mistral uses EOS, no end tag
            case .xmlFunction:
                #expect(parser.startTag == "<tool_call>")
                #expect(parser.endTag == "</tool_call>")
            case .glm4:
                #expect(parser.startTag == "<tool_call>")
                #expect(parser.endTag == "</tool_call>")
            case .kimiK2:
                #expect(parser.startTag == "<|tool_calls_section_begin|>")
                #expect(parser.endTag == "<|tool_calls_section_end|>")
            case .minimaxM2:
                #expect(parser.startTag == "<minimax:tool_call>")
                #expect(parser.endTag == "</minimax:tool_call>")
            case .llama3:
                #expect(parser.startTag == nil)  // Llama 3 uses inline format
                #expect(parser.endTag == nil)
            }
        }
    }

    // MARK: - Edge Cases: Streaming Partial Tags

    @Test("BatchedToolStreamHandler handles tag split across chunk boundaries")
    func batchedHandlerPartialTagBoundary() throws {
        let handler = BatchedToolStreamHandler(format: .gemma, tools: nil)

        // "<|tool_call>" split across two chunks
        _ = handler.processChunk("<|tool")
        _ = handler.processChunk("_call>call:weather{city:London}<tool_call|>")

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "weather")
        #expect(toolCalls[0].function.arguments["city"] == .string("London"))
    }

    @Test("BatchedToolStreamHandler passes through text that looks like but is not a tag")
    func batchedHandlerFalsePositiveTag() {
        let handler = BatchedToolStreamHandler(format: .gemma, tools: nil)

        // Text that starts with "<" but is not a tool call tag
        let visible = handler.processChunk("<hello> this is not a tool call")
        #expect(visible == "<hello> this is not a tool call")

        let toolCalls = handler.finish()
        #expect(toolCalls.isEmpty)
    }

    // MARK: - Edge Cases: Complex Arguments

    @Test("BatchedToolStreamHandler handles Gemma with nested JSON arguments")
    func batchedHandlerGemmaNestedJSON() throws {
        let handler = BatchedToolStreamHandler(format: .gemma, tools: nil)

        _ = handler.processChunk(
            "<|tool_call>call:search{query:<|\"|>swift, mlx<|\"|>,filters:{\"limit\":5,\"active\":true}}<tool_call|>"
        )

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)

        let tc = try #require(toolCalls.first)
        #expect(tc.function.name == "search")
        #expect(tc.function.arguments["query"] == .string("swift, mlx"))
        #expect(tc.function.arguments["filters"] == .object([
            "limit": .int(5),
            "active": .bool(true),
        ]))
    }

    @Test("BatchedToolStreamHandler handles Harmony with nested JSON arguments")
    func batchedHandlerHarmonyNestedJSON() throws {
        let handler = BatchedToolStreamHandler(format: .harmony, tools: nil)

        _ = handler.processChunk(
            "<|start|>assistant to=functions.create_event<|channel|>commentary json<|message|>{\"title\":\"Meeting\",\"attendees\":[\"Alice\",\"Bob\"],\"metadata\":{\"priority\":\"high\"}}<|call|>"
        )

        let toolCalls = handler.finish()
        #expect(toolCalls.count == 1)

        let tc = try #require(toolCalls.first)
        #expect(tc.function.name == "create_event")
        #expect(tc.function.arguments["title"] == .string("Meeting"))
        #expect(tc.function.arguments["attendees"] == .array([.string("Alice"), .string("Bob")]))
        #expect(tc.function.arguments["metadata"] == .object(["priority": .string("high")]))
    }

    // MARK: - validateToolCallParserOverride Integration

    @Test("validateToolCallParserOverride accepts matching formats")
    func validateOverrideMatchingFormats() throws {
        try MLXBatchedEngineServerEngine.validateToolCallParserOverride(
            requested: "gemma4",
            pinned: .gemma,
            modelType: nil
        )
        try MLXBatchedEngineServerEngine.validateToolCallParserOverride(
            requested: "gpt_oss",
            pinned: .harmony,
            modelType: nil
        )
        try MLXBatchedEngineServerEngine.validateToolCallParserOverride(
            requested: "mistral",
            pinned: .mistral,
            modelType: nil
        )
    }

    @Test("validateToolCallParserOverride rejects mismatched formats")
    func validateOverrideMismatchedFormats() {
        #expect(throws: MLXBatchedEngineServerEngineError.self) {
            try MLXBatchedEngineServerEngine.validateToolCallParserOverride(
                requested: "gemma",
                pinned: .harmony,
                modelType: nil
            )
        }
        #expect(throws: MLXBatchedEngineServerEngineError.self) {
            try MLXBatchedEngineServerEngine.validateToolCallParserOverride(
                requested: "mistral",
                pinned: .gemma,
                modelType: nil
            )
        }
    }

    @Test("validateToolCallParserOverride accepts nil and auto for any pinned format")
    func validateOverrideNilAndAuto() throws {
        for format in ToolCallFormat.allCases {
            try MLXBatchedEngineServerEngine.validateToolCallParserOverride(
                requested: nil,
                pinned: format,
                modelType: nil
            )
            try MLXBatchedEngineServerEngine.validateToolCallParserOverride(
                requested: "auto",
                pinned: format,
                modelType: nil
            )
        }
    }
}
