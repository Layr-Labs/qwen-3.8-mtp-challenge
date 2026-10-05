import Foundation
import Testing

@testable import MLXLMServer

struct RequestAwareReasoningTests {
    @Test("Disabled Qwen thinking streams the first plain chunk immediately")
    func disabledThinkingStreamsImmediately() {
        let policy = RequestReasoningPolicy(
            format: .qwen3, enableThinking: false, qwenPromptStartsInReasoning: true
        )
        var parser = policy.makeStreamingParser()
        #expect(policy.format == .none)
        #expect(parser.parse("Hello") == [.init(content: "Hello", reasoningContent: nil)])
        #expect(parser.parse(" world") == [.init(content: " world", reasoningContent: nil)])
        #expect(parser.finish().isEmpty)
    }

    @Test("An implicit Qwen thinking prefix streams reasoning before its closing tag")
    func implicitThinkingStreamsAndTruncatesAsReasoning() {
        for enabled in [nil, true] as [Bool?] {
            let policy = RequestReasoningPolicy(
                format: .qwen3, enableThinking: enabled, qwenPromptStartsInReasoning: true
            )
            var parser = policy.makeStreamingParser()
            #expect(parser.parse("Check assumptions") == [
                .init(content: "", reasoningContent: "Check assumptions")
            ])
            let final = parser.parse(" carefully</thi") + parser.finish()
            #expect(final.map(\.content).joined().isEmpty)
            #expect(final.compactMap(\.reasoningContent).joined() == " carefully</thi")
            #expect(policy.parse("Check assumptions").content.isEmpty)
            #expect(policy.parse("Check assumptions").reasoningContent == "Check assumptions")
        }
    }

    @Test("Every split closing-tag boundary preserves reasoning and answer text")
    func splitClosingTag() {
        let policy = RequestReasoningPolicy(
            format: .qwen3, enableThinking: true, qwenPromptStartsInReasoning: true
        )
        let closing = "</think>"
        for offset in 1..<closing.count {
            var parser = policy.makeStreamingParser()
            let split = closing.index(closing.startIndex, offsetBy: offset)
            let events = parser.parse("分析" + closing[..<split])
                + parser.parse(String(closing[split...]) + "Answer") + parser.finish()
            #expect(events.compactMap(\.reasoningContent).joined() == "分析")
            #expect(events.map(\.content).joined() == "Answer")
        }
    }

    @Test("An explicitly repeated opening tag is stripped across every split boundary")
    func splitExplicitOpeningTag() {
        let policy = RequestReasoningPolicy(
            format: .qwen3, enableThinking: true, qwenPromptStartsInReasoning: true
        )
        let opening = "<think>"
        for offset in 1..<opening.count {
            var parser = policy.makeStreamingParser()
            let split = opening.index(opening.startIndex, offsetBy: offset)
            let first = parser.parse("\n " + opening[..<split])
            #expect(first.isEmpty)
            let events = first + parser.parse(String(opening[split...]) + "Reason</think>Answer")
                + parser.finish()
            #expect(events.compactMap(\.reasoningContent).joined() == "Reason")
            #expect(events.map(\.content).joined() == "Answer")
        }
        #expect(policy.parse("<think>Reason</think>Answer") ==
            .init(content: "Answer", reasoningContent: "Reason"))
    }

    @Test("Generic parsing retains tag detection unless the request explicitly selects the Qwen seed")
    func genericBehaviorIsPreserved() {
        var generic = StreamingReasoningParser(format: .qwen3)
        #expect(generic.parse("Plain").isEmpty)
        #expect(generic.finish() == [.init(content: "Plain", reasoningContent: nil)])
        let policy = RequestReasoningPolicy(format: .qwen3, enableThinking: nil)
        #expect(!policy.startsInReasoning)
        #expect(policy.parse("Plain").content == "Plain")
        for format in [ReasoningParserFormat.deepseekR1, .harmony, .gemma4, .none] {
            let untouched = RequestReasoningPolicy(
                format: format, enableThinking: false, qwenPromptStartsInReasoning: true
            )
            #expect(untouched.format == format)
            #expect(!untouched.startsInReasoning)
        }
    }

    @Test("Service forwards no-thinking content before generation can finish")
    func serviceNoThinkingHasImmediateCadence() async throws {
        let gate = ReasoningGenerationGate()
        let engine = GatedReasoningEngine(first: "Hello", gate: gate)
        let service = MLXOpenAIService(engine: engine, defaultReasoningParser: .qwen3)
        let request = OpenAIChatCompletionRequest(
            model: "default_model", messages: [.init(role: .user, content: .text("Hi"))],
            stream: true, chatTemplateKwargs: ["enable_thinking": .bool(false)]
        )
        let timeout = Task {
            try? await Task.sleep(for: .seconds(1))
            await gate.release()
        }
        defer { timeout.cancel() }
        let frames = try await service.streamChatCompletionFrames(request: request)
        var firstContentSeen = false
        var finished = false
        for try await frame in frames {
            if frame == ServerSentEventEncoder.done { finished = true; continue }
            let raw = String(frame.dropFirst("data: ".count)).trimmingCharacters(in: .whitespacesAndNewlines)
            let chunk = try JSONDecoder().decode(OpenAIChatCompletionChunk.self, from: Data(raw.utf8))
            if chunk.choices.contains(where: { $0.delta.content == "Hello" }) {
                #expect(await gate.isReleased == false)
                firstContentSeen = true
                await gate.release()
            }
        }
        #expect(firstContentSeen)
        #expect(finished)
    }

    @Test("Service keeps truncated implicit thinking separate in both streaming and nonstreaming responses")
    func serviceImplicitThinkingClassification() async throws {
        let engine = GatedReasoningEngine(first: "Check assumptions", gate: nil)
        let service = MLXOpenAIService(engine: engine, defaultReasoningParser: .qwen3)
        let request = OpenAIChatCompletionRequest(
            model: "default_model", messages: [.init(role: .user, content: .text("Hi"))]
        )
        let response = try await service.createChatCompletion(request: request)
        #expect(response.choices.first?.message.textContent == "")
        #expect(response.choices.first?.message.reasoningContent == "Check assumptions")
        var reasoning = ""
        var content = ""
        let frames = try await service.streamChatCompletionFrames(request: request)
        for try await frame in frames {
            if frame == ServerSentEventEncoder.done { continue }
            let raw = String(frame.dropFirst("data: ".count)).trimmingCharacters(in: .whitespacesAndNewlines)
            let chunk = try JSONDecoder().decode(OpenAIChatCompletionChunk.self, from: Data(raw.utf8))
            for choice in chunk.choices {
                content += choice.delta.content ?? ""
                reasoning += choice.delta.reasoningContent ?? ""
            }
        }
        #expect(content.isEmpty)
        #expect(reasoning == "Check assumptions")
    }
}

private actor ReasoningGenerationGate {
    private(set) var isReleased = false
    private var waiter: CheckedContinuation<Void, Never>?

    func wait() async {
        if isReleased { return }
        await withCheckedContinuation { waiter = $0 }
    }

    func release() {
        isReleased = true
        waiter?.resume()
        waiter = nil
    }
}

private struct GatedReasoningEngine: MLXServerEngine {
    let first: String
    let gate: ReasoningGenerationGate?

    func availableModels() async throws -> [MLXServerModel] { [.init(id: "default_model")] }
    func streamChatCompletion(request: OpenAIChatCompletionRequest) async throws
        -> AsyncThrowingStream<MLXServerGenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                continuation.yield(.content(first))
                await gate?.wait()
                continuation.yield(.info(.init(
                    promptTokens: 1, completionTokens: 1, promptTime: 0.01,
                    generationTime: 0.01, stopReason: "length"
                )))
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
    func tokenize(_ request: TokenizeRequest) async throws -> TokenizeResponse { .init(tokens: [1]) }
    func detokenize(_ request: DetokenizeRequest) async throws -> DetokenizeResponse { .init(text: first) }
    func applyTemplate(_ request: ApplyTemplateRequest) async throws -> TokenizeResponse { .init(tokens: [1]) }
}
