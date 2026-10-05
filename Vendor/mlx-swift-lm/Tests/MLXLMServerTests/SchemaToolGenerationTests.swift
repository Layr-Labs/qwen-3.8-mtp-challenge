// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXLMServer
import MLXNN
import Testing

struct SchemaToolGenerationTests {
    @Test("Request schemas reach actual single-request generation and chat tool arguments", arguments: [false, true])
    func requestSchemasReachChatTools(streaming: Bool) async throws {
        let engine = schemaEngine(script: readXML)
        let service = MLXOpenAIService(engine: engine)
        let request = schemaRequest(tools: [readSchema()], streaming: streaming)
        let call: OpenAIToolCall
        if streaming {
            var calls: [OpenAIToolCall] = []
            var finishReason: String?
            let frames = try await service.streamChatCompletionFrames(request: request)
            for try await frame in frames {
                if frame == ServerSentEventEncoder.done { continue }
                let payload = String(frame.dropFirst("data: ".count))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let chunk = try JSONDecoder().decode(OpenAIChatCompletionChunk.self, from: Data(payload.utf8))
                calls += chunk.choices.flatMap { $0.delta.toolCalls ?? [] }
                finishReason = chunk.choices.first?.finishReason ?? finishReason
            }
            #expect(finishReason == "tool_calls")
            call = try #require(calls.first)
            #expect(calls.count == 1)
        } else {
            let response = try await service.createChatCompletion(request: request)
            #expect(response.choices.first?.finishReason == "tool_calls")
            call = try #require(response.choices.first?.message.toolCalls?.first)
        }
        let arguments = try JSONDecoder().decode([String: JSONValue].self, from: Data(call.function.arguments.utf8))
        #expect(arguments == ["path": .string("sample.txt"), "offset": .int(1), "limit": .int(3)])
    }

    @Test("Actual generation preserves boolean, object, array, and declared string types")
    func generationPreservesSchemaTypes() async throws {
        let script = "<tool_call><function=configure><parameter=enabled>true</parameter><parameter=meta>{\"key\":2,\"ok\":false}</parameter><parameter=items>[1,\"2\",true]</parameter><parameter=id>001</parameter></function></tool_call>"
        let tool = schemaTool(name: "configure", properties: [
            "enabled": "boolean", "meta": "object", "items": "array", "id": "string",
        ])
        let call = try await generatedCall(engine: schemaEngine(script: script), tools: [tool])
        #expect(call.function.arguments == [
            "enabled": .bool(true), "meta": .object(["key": .int(2), "ok": .bool(false)]),
            "items": .array([.int(1), .string("2"), .bool(true)]), "id": .string("001"),
        ])
    }

    @Test("Tool schemas belong to one request and do not persist into another")
    func schemasAreRequestScoped() async throws {
        let engine = schemaEngine(script: readXML)
        let first = try await generatedCall(engine: engine, tools: [readSchema()])
        #expect(first.function.arguments["offset"] == .int(1))
        let second = try await generatedCall(engine: engine, tools: [readSchema(numericType: "string")])
        #expect(second.function.arguments["offset"] == .string("1"))
        let third = try await generatedCall(engine: engine, tools: nil)
        #expect(third.function.arguments["offset"] == .string("1"))
    }

    @Test("Omitted schemas preserve the original XML text behavior")
    func omittedSchemasPreserveText() async throws {
        let call = try await generatedCall(engine: schemaEngine(script: readXML), tools: nil)
        #expect(call.function.arguments == ["path": .string("sample.txt"), "offset": .string("1"), "limit": .string("3")])
    }

    @Test("JSON tool argument strings retain their wire types despite integer schemas")
    func jsonArgumentsAreNotCoerced() async throws {
        let script = #"<tool_call>{"name":"read","arguments":{"path":"sample.txt","offset":"1","limit":"3"}}</tool_call>"#
        let call = try await generatedCall(engine: schemaEngine(script: script), tools: [readSchema()], parser: "json")
        #expect(call.function.arguments["offset"] == .string("1"))
        #expect(call.function.arguments["limit"] == .string("3"))
    }

    @Test("Unknown parameters and invalid integer text remain visible to strict clients")
    func invalidAndUnknownValuesRemainVisible() async throws {
        let script = "<tool_call><function=read><parameter=offset>one</parameter><parameter=undeclared>3</parameter></function></tool_call>"
        let call = try await generatedCall(engine: schemaEngine(script: script), tools: [readSchema()])
        #expect(call.function.arguments["offset"] == .string("one"))
        #expect(call.function.arguments["undeclared"] == .string("3"))
    }

    @Test("Invalid boolean XML text stays invalid instead of inventing false")
    func invalidBooleanRemainsInvalidString() throws {
        let arguments = try parsedArguments(value: "banana", type: "boolean")
        #expect(arguments["value"] == .string("banana"))
    }

    @Test("Documented XML boolean aliases retain their values")
    func validBooleanAliasesRemainSupported() throws {
        for value in ["true", "1", "yes", "on"] {
            #expect(try parsedArguments(value: value, type: "boolean")["value"] == .bool(true))
        }
        for value in ["false", "0", "no", "off"] {
            #expect(try parsedArguments(value: value, type: "boolean")["value"] == .bool(false))
        }
    }

    // The red run selects each numeric method in a separate SwiftPM test
    // subprocess, containing the old unchecked Int(Double) trap. The green
    // run includes them in the ordinary full suite.
    @Test("Non-finite XML numbers remain invalid strings without crashing", arguments: ["nan", "inf", "1e999"])
    func nonFiniteNumberRemainsInvalidString(value: String) throws {
        let arguments = try parsedArguments(value: value, type: "number")
        #expect(arguments["value"] == .string(value))
    }

    @Test("A finite XML number outside Int range remains a number without crashing")
    func largeFiniteNumberRemainsNumber() throws {
        let arguments = try parsedArguments(value: "1e100", type: "number")
        #expect(arguments["value"] == .double(1e100))
    }
}

private let readXML = "<tool_call><function=read><parameter=path>sample.txt</parameter><parameter=offset>1</parameter><parameter=limit>3</parameter></function></tool_call>"

private func schemaTool(name: String, properties: [String: String]) -> OpenAITool {
    .init(function: .init(name: name, parameters: .object([
        "type": .string("object"),
        "properties": .object(properties.mapValues { .object(["type": .string($0)]) }),
        "additionalProperties": .bool(false),
    ])))
}

private func readSchema(numericType: String = "integer") -> OpenAITool {
    schemaTool(name: "read", properties: ["path": "string", "offset": numericType, "limit": numericType])
}

private func schemaRequest(tools: [OpenAITool]?, streaming: Bool = false, parser: String? = nil)
    -> OpenAIChatCompletionRequest {
    .init(model: "default_model", messages: [.init(role: .user, content: .text("Use the tool"))],
          tools: tools, toolCallParser: parser, stream: streaming, temperature: 0, maxTokens: 512)
}

private func generatedCall(engine: MLXModelContainerEngine, tools: [OpenAITool]?, parser: String? = nil)
    async throws -> ToolCall {
    let stream = try await engine.streamChatCompletion(request: schemaRequest(tools: tools, parser: parser))
    var calls: [ToolCall] = []
    var info: ServerGenerationInfo?
    for try await event in stream {
        if case .toolCall(let call) = event { calls.append(call) }
        if case .info(let generatedInfo) = event { info = generatedInfo }
    }
    #expect(info?.stopReason == "stop")
    #expect(calls.count == 1)
    return try #require(calls.first)
}

private func parsedArguments(value: String, type: String) throws -> [String: JSONValue] {
    let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
    let call = try #require(parser.parse(
        content: "<function=test><parameter=value>\(value)</parameter></function>",
        tools: [schemaTool(name: "test", properties: ["value": type]).toolSpec()]
    ))
    return call.function.arguments
}

/// Every request executes the actual ModelContainer, TokenIterator, generation
/// loop, and XML parser. Only the model/tokenizer boundary is deterministic;
/// the fixture loads no weights and does not synthesize tool events or usage.
private func schemaEngine(script: String) -> MLXModelContainerEngine {
    let tokenizer = SchemaScriptTokenizer(script: script)
    var configuration = ModelConfiguration(id: "schema-script", toolCallFormat: .xmlFunction)
    configuration.eosTokenIds = [tokenizer.stopToken]
    let container = ModelContainer(context: .init(
        configuration: configuration, model: SchemaScriptModel(stopToken: tokenizer.stopToken),
        processor: SchemaScriptProcessor(), tokenizer: tokenizer
    ))
    return MLXModelContainerEngine(modelID: "default_model", model: container, modelType: "qwen3_5", contextTokens: 4096)
}

private struct SchemaScriptProcessor: UserInputProcessor {
    func prepare(input: UserInput) throws -> LMInput { LMInput(tokens: MLXArray([Int32(0)])) }
}

private struct SchemaScriptTokenizer: Tokenizer {
    let bytes: [UInt8]
    var stopToken: Int { bytes.count + 1 }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    init(script: String) { bytes = Array(script.utf8) }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [0] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.compactMap { id in
            (1...bytes.count).contains(id) ? bytes[id - 1] : nil
        }, as: UTF8.self)
    }
    func convertTokenToId(_ token: String) -> Int? { Int(token) }
    func convertIdToToken(_ id: Int) -> String? { String(id) }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                           additionalContext: [String: any Sendable]?) throws -> [Int] { [0] }
}

private final class SchemaScriptModel: Module, LanguageModel {
    let stopToken: Int
    init(stopToken: Int) {
        self.stopToken = stopToken
        super.init()
    }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult { .tokens(input.text) }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let current = inputs.item(Int.self)
        let next = min(current + 1, stopToken)
        var logits = Array(repeating: Float(-1000), count: stopToken + 1)
        logits[next] = 0
        return MLXArray(logits).reshaped([1, 1, stopToken + 1])
    }
}
