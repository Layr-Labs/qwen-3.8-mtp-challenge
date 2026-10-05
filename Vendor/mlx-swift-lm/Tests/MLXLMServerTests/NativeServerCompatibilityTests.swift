import Foundation
import Hummingbird
import MLXLMCommon
import Testing

@testable import MLXLMServer

struct NativeServerCompatibilityTests {
    @Test("CLI bounds the context limit and honors explicit launch arguments")
    func contextCLI() throws {
        let parsed = try MLXServerCLI.parse(
            arguments: ["mlx-server", "--model", "/tmp/pinned-snapshot", "--context-tokens", "65536"],
            environment: [:]
        )
        guard case .run(let configuration) = parsed else {
            Issue.record("Expected serving configuration")
            return
        }
        #expect(configuration.contextTokens == 65536)
        #expect(configuration.model == "/tmp/pinned-snapshot")
        for invalid in ["0", "-1", "262145", "not-a-number", "9223372036854775807"] {
            #expect(throws: MLXServerCLIError.self) {
                try MLXServerCLI.parse(
                    arguments: ["mlx-server", "--context-tokens", invalid], environment: [:]
                )
            }
        }
        let viaEnvironment = try MLXServerCLI.parse(
            arguments: ["mlx-server"], environment: ["MLX_SERVER_CONTEXT_TOKENS": "4096"]
        )
        guard case .run(let environmentConfiguration) = viaEnvironment else { return }
        #expect(environmentConfiguration.contextTokens == 4096)
    }

    @Test("Prepared token budget accepts the exact boundary and caps omitted output")
    func tokenBudgetBoundaries() throws {
        #expect(try ContextTokenBudget.outputLimit(promptTokens: 11, requested: 5, contextTokens: 16) == 5)
        #expect(try ContextTokenBudget.outputLimit(promptTokens: 11, requested: nil, contextTokens: 16) == 5)
        #expect(try ContextTokenBudget.outputLimit(promptTokens: 15, requested: 1, contextTokens: 16) == 1)
        for requested in [0, -1, 6, Int.max] {
            #expect(throws: HTTPError.self) {
                try ContextTokenBudget.outputLimit(promptTokens: 11, requested: requested, contextTokens: 16)
            }
        }
        for prompt in [-1, 16, 17, Int.max] {
            #expect(throws: HTTPError.self) {
                try ContextTokenBudget.outputLimit(promptTokens: prompt, requested: 1, contextTokens: 16)
            }
        }
    }

    @Test("Template kwargs survive decode and disable thinking without changing the output cap")
    func thinkingKwargs() throws {
        let raw = """
        {"model":"default_model","messages":[{"role":"user","content":"Hello"}],
         "chat_template_kwargs":{"enable_thinking":false},"max_tokens":128}
        """
        let request = try JSONDecoder().decode(OpenAIChatCompletionRequest.self, from: Data(raw.utf8))
        #expect(request.chatTemplateKwargs?["enable_thinking"] == .bool(false))
        #expect(request.templateContext?["enable_thinking"] as? Bool == false)
        #expect(request.templateInput.additionalContext?["enable_thinking"] as? Bool == false)
        #expect(request.generationParameters.maxTokens == 128)
        let encoded = try JSONEncoder().encode(request)
        let again = try JSONDecoder().decode(OpenAIChatCompletionRequest.self, from: encoded)
        #expect(again == request)
    }

    @Test("Single-request prepared input preserves assistant tool arguments and tool-result linkage")
    func toolHistory() {
        let request = OpenAIChatCompletionRequest(
            model: "default_model",
            messages: [
                .init(role: .assistant, content: .null,
                      toolCalls: [.init(id: "call_1", function: .init(name: "read", arguments: "{\"path\":\"go.mod\"}"))],
                      reasoningContent: "Need the module name."),
                .init(role: .tool, content: .text("module example"), toolCallID: "call_1"),
            ]
        )
        guard case .messages(let messages) = request.templateInput.prompt else {
            Issue.record("Expected template-ready messages")
            return
        }
        let calls = messages[0]["tool_calls"] as? [[String: any Sendable]]
        #expect(calls?.first?["id"] as? String == "call_1")
        let function = calls?.first?["function"] as? [String: any Sendable]
        let arguments = function?["arguments"] as? [String: any Sendable]
        #expect(arguments?["path"] as? String == "go.mod")
        #expect(messages[0]["reasoning_content"] as? String == "Need the module name.")
        #expect(messages[1]["tool_call_id"] as? String == "call_1")
    }

    @Test("FloCode assistant tool history decodes omitted content precisely")
    func omittedAssistantToolContent() throws {
        let raw = """
        {"model":"default_model","messages":[
          {"role":"user","content":"Read the module name"},
          {"role":"assistant","tool_calls":[{"id":"call_1","type":"function",
            "function":{"name":"read","arguments":"{\\"path\\":\\"go.mod\\"}"}}]},
          {"role":"tool","tool_call_id":"call_1","content":"module example"}
        ],"max_tokens":128}
        """
        let request = try JSONDecoder().decode(OpenAIChatCompletionRequest.self, from: Data(raw.utf8))
        #expect(request.messages[1].content == .null)
        #expect(request.messages[1].toolCalls?.first?.id == "call_1")
        #expect(request.messages[2].toolCallID == "call_1")
        guard case .messages(let messages) = request.templateInput.prompt else {
            Issue.record("Expected template-ready messages")
            return
        }
        #expect((messages[1]["tool_calls"] as? [[String: any Sendable]])?.first?["id"] as? String == "call_1")
        #expect(messages[2]["tool_call_id"] as? String == "call_1")
        for invalid in [
            "{\"role\":\"user\"}",
            "{\"role\":\"assistant\"}",
            "{\"role\":\"assistant\",\"tool_calls\":[]}",
            "{\"role\":\"tool\",\"tool_call_id\":\"call_1\"}",
        ] {
            #expect(throws: DecodingError.self) {
                try JSONDecoder().decode(OpenAIChatMessage.self, from: Data(invalid.utf8))
            }
        }
    }
}
