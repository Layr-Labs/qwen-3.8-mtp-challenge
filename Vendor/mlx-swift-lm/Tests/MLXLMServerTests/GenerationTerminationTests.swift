// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon
import MLXLMServer
import Testing

struct GenerationTerminationTests {
    @Test("A value iterator reaching its output cap reports length")
    func cappedGenerationReportsLength() async throws {
        let output = await collectGeneration(tokens: [97, 98, 99, 100], maxTokens: 3)
        let info = try #require(output.info)
        #expect(output.content == "abc")
        #expect(info.generationTokenCount == 3)
        #expect(info.stopReason == .length)
    }

    @Test("Chat preserves the actual generation length reason and capped usage", arguments: [false, true])
    func chatPreservesLengthReason(streaming: Bool) async throws {
        let service = MLXOpenAIService(engine: TerminationLoopEngine())
        let request = OpenAIChatCompletionRequest(
            model: "default_model",
            messages: [.init(role: .user, content: .text("Continue the sequence"))],
            stream: streaming,
            maxTokens: 3,
            streamOptions: .init(includeUsage: true, continuousUsageStats: nil)
        )

        if streaming {
            var content = ""
            var terminal: OpenAIChatCompletionChunk?
            var receivedDone = false
            for try await frame in try await service.streamChatCompletionFrames(request: request) {
                if frame == ServerSentEventEncoder.done {
                    receivedDone = true
                    continue
                }
                let payload = String(frame.dropFirst("data: ".count))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let chunk = try JSONDecoder().decode(OpenAIChatCompletionChunk.self, from: Data(payload.utf8))
                content += chunk.choices.compactMap(\.delta.content).joined()
                if chunk.choices.contains(where: { $0.finishReason != nil }) {
                    terminal = chunk
                }
            }
            #expect(content == "abc")
            #expect(terminal?.usage?.completionTokens == 3)
            #expect(terminal?.choices.first?.finishReason == "length")
            #expect(receivedDone)
        } else {
            let response = try await service.createChatCompletion(request: request)
            #expect(response.choices.first?.message.content == .text("abc"))
            #expect(response.usage.completionTokens == 3)
            #expect(response.choices.first?.finishReason == "length")
        }
    }

    @Test("EOS before the output cap reports stop and excludes the stop token")
    func eosBeforeCapReportsStop() async throws {
        let output = await collectGeneration(tokens: [97, 999, 98], maxTokens: 3)
        let info = try #require(output.info)
        #expect(output.content == "a")
        #expect(info.generationTokenCount == 1)
        #expect(info.stopReason == .stop)
    }

    @Test("EOS on the final permitted token reports stop rather than length")
    func eosAtCapReportsStop() async throws {
        let output = await collectGeneration(tokens: [97, 98, 999, 99], maxTokens: 3)
        let info = try #require(output.info)
        #expect(output.content == "ab")
        #expect(info.generationTokenCount == 2)
        #expect(info.stopReason == .stop)
    }

    @Test("Explicit cancellation before the first token remains cancellation")
    func cancellationBeforeFirstTokenRemainsCancellation() async throws {
        let output = await collectGeneration(tokens: [97, 98, 99, 100], maxTokens: 3, cancelBeforeToken: 0)
        let info = try #require(output.info)
        #expect(output.content.isEmpty)
        #expect(info.generationTokenCount == 0)
        #expect(info.stopReason == .cancelled)
    }

    @Test("Explicit cancellation at the output cap takes precedence over length")
    func cancellationAtCapRemainsCancellation() async throws {
        let output = await collectGeneration(tokens: [97, 98, 99, 100], maxTokens: 3, cancelBeforeToken: 3)
        let info = try #require(output.info)
        #expect(output.content == "abc")
        #expect(info.generationTokenCount == 3)
        #expect(info.stopReason == .cancelled)
    }
}

private struct TerminationOutput {
    var content = ""
    var info: GenerateCompletionInfo?
}

private func collectGeneration(
    tokens: [Int], maxTokens: Int, cancelBeforeToken: Int? = nil
) async -> TerminationOutput {
    let (stream, task) = generationStream(tokens: tokens, maxTokens: maxTokens, cancelBeforeToken: cancelBeforeToken)
    var output = TerminationOutput()
    for await event in stream {
        switch event {
        case .chunk(let text): output.content += text
        case .info(let info): output.info = info
        case .toolCall: Issue.record("Plain text fixture unexpectedly produced a tool call")
        }
    }
    await task.value
    return output
}

private func generationStream(
    tokens: [Int], maxTokens: Int, cancelBeforeToken: Int? = nil
) -> (AsyncStream<Generation>, Task<Void, Never>) {
    var configuration = ModelConfiguration(id: "generation-termination-test")
    configuration.eosTokenIds = [999]
    return MLXLMCommon.generateTask(
        promptTokenCount: 1,
        modelConfiguration: configuration,
        tokenizer: TerminationTokenizer(),
        iterator: TerminationIterator(tokens: tokens, maxTokens: maxTokens, cancelBeforeToken: cancelBeforeToken)
    )
}

/// A real value-type iterator at the public generation boundary. It supplies
/// deterministic tokens without loading model weights or invoking a model.
private struct TerminationIterator: TokenIteratorProtocol {
    let tokens: [Int]
    let maxTokens: Int?
    let cancelBeforeToken: Int?
    var tokenCount = 0
    let promptPrefillTime: TimeInterval = 0

    mutating func next() -> Int? {
        if tokenCount == cancelBeforeToken {
            withUnsafeCurrentTask { $0?.cancel() }
        }
        guard tokenCount < (maxTokens ?? Int.max), tokenCount < tokens.count else { return nil }
        let token = tokens[tokenCount]
        tokenCount += 1
        return token
    }
}

private struct TerminationTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map { UInt8($0) }, as: UTF8.self)
    }
    func convertTokenToId(_ token: String) -> Int? { Int(token) }
    func convertIdToToken(_ id: Int) -> String? { String(id) }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [97] }
}

/// The service receives genuine shared-loop events, including its computed
/// stop reason; this adapter never invents terminal metadata.
private struct TerminationLoopEngine: MLXServerEngine {
    func availableModels() async throws -> [MLXServerModel] { [.init(id: "default_model")] }

    func streamChatCompletion(request: OpenAIChatCompletionRequest) async throws
        -> AsyncThrowingStream<MLXServerGenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                let (stream, generationTask) = generationStream(tokens: [97, 98, 99, 100], maxTokens: request.maxTokens ?? 3)
                for await event in stream {
                    switch event {
                    case .chunk(let text): continuation.yield(.content(text))
                    case .toolCall(let call): continuation.yield(.toolCall(call))
                    case .info(let info): continuation.yield(.info(.init(info)))
                    }
                }
                await generationTask.value
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    func tokenize(_ request: TokenizeRequest) async throws -> TokenizeResponse { .init(tokens: [97]) }
    func detokenize(_ request: DetokenizeRequest) async throws -> DetokenizeResponse { .init(text: "a") }
    func applyTemplate(_ request: ApplyTemplateRequest) async throws -> TokenizeResponse { .init(tokens: [97]) }
}
