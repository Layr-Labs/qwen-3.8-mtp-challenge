import Hummingbird
import MLXLMCommon

/// Optional request-local generation implementation. The engine retains its
/// model processor, token budget, routes, and output mapping.
public protocol MLXServerGenerationDriver: Sendable {
    func validate(request: OpenAIChatCompletionRequest) throws
    func generate(
        model: ModelContainer, input: consuming sending LMInput,
        parameters: GenerateParameters, request: OpenAIChatCompletionRequest,
        tools: [ToolSpec]?, toolCallFormat: ToolCallFormat
    ) async throws -> AsyncThrowingStream<Generation, Error>
}

public enum MLXServerGreedyPolicy {
    public static func validate(_ request: OpenAIChatCompletionRequest) throws {
        guard request.temperature == 0,
            request.repetitionPenalty == nil || request.repetitionPenalty == 1,
            request.presencePenalty == nil || request.presencePenalty == 0,
            request.frequencyPenalty == nil || request.frequencyPenalty == 0
        else {
            throw HTTPError(.badRequest, message:
                "native_mtp requires temperature=0 and disabled repetition, presence, and frequency penalties.")
        }
        if let stops = request.stop,
            stops.count > 4 || stops.contains(where: { $0.isEmpty || $0.utf8.count > 256 })
        {
            throw HTTPError(.badRequest,
                message: "native_mtp supports at most four nonempty stop strings, each at most 256 UTF-8 bytes.")
        }
        switch request.toolChoice {
        case nil, .mode(.auto):
            break
        case .mode(.none), .mode(.required), .function:
            throw HTTPError(.badRequest, message:
                "native_mtp accepts only tool_choice auto.")
        }
        if request.parallelToolCalls != nil {
            throw HTTPError(.badRequest, message:
                "native_mtp does not accept parallel_tool_calls.")
        }
        if request.tools?.contains(where: { $0.function.strict == true }) == true {
            throw HTTPError(.badRequest, message:
                "native_mtp does not accept strict tool schemas.")
        }
        if request.responseFormat != nil || request.toolCallParser != nil
            || request.reasoningParser != nil
        {
            throw HTTPError(.badRequest, message:
                "native_mtp does not accept response_format or a per-request parser override.")
        }
        if let arguments = request.chatTemplateKwargs {
            for (key, value) in arguments {
                guard key == "enable_thinking", case .bool = value else {
                    throw HTTPError(.badRequest, message:
                        "native_mtp accepts only a boolean enable_thinking chat template argument.")
                }
            }
        }
    }

    public static func busy() -> HTTPError {
        HTTPError(.tooManyRequests, message: "native_mtp is serving another request.")
    }
}
