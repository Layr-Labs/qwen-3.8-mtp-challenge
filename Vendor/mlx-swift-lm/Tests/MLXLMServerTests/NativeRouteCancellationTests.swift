import Foundation
import Hummingbird
import HummingbirdTesting
@testable import MLXLMServer
import Testing

struct NativeRouteCancellationTests {
    @Test("Closing HTTP before the first content event cancels native prefill")
    func disconnectBeforeFirstContentCancelsPrefill() async throws {
        let engine = RouteCancellationProbeEngine()
        let app = MLXServerApplication.buildApplication(
            service: MLXOpenAIService(engine: engine), host: "127.0.0.1", port: 0
        )
        try await app.test(.live) { client in
            let port = try #require(client.port)
            let url = try #require(URL(string: "http://localhost:\(port)/v1/chat/completions"))
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = Data(
                #"{"model":"local-model","messages":[{"role":"user","content":"slow"}],"stream":true}"#.utf8
            )
            let session = URLSession(configuration: .ephemeral)
            defer { session.invalidateAndCancel() }
            let connection = session.dataTask(with: request) { _, _, _ in }
            defer { connection.cancel() }
            connection.resume()
            try #require(await observesEvent(engine.prefillStarted), "HTTP request did not reach fake prefill")
            connection.cancel()
            #expect(
                await observesEvent(engine.prefillCancelled),
                "Disconnected client kept prefill running past the cancellation deadline"
            )
        }
    }

    @Test("Native SSE remains readable through the final DONE event")
    func ordinaryStreamFinishesWithoutPrematureSocketClose() async throws {
        let app = MLXServerApplication.buildApplication(
            service: MLXOpenAIService(engine: RouteCancellationProbeEngine()),
            host: "127.0.0.1", port: 0
        )
        try await app.test(.live) { client in
            try await client.execute(
                uri: "/v1/chat/completions", method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"model":"local-model","messages":[{"role":"user","content":"quick"}],"stream":true,"stream_options":{"include_usage":true}}"#)
            ) { response in
                let text = String(buffer: response.body)
                #expect(response.status == .ok)
                #expect(text.contains("route ok"))
                #expect(text.contains("completion_tokens"))
                #expect(text.contains("[DONE]"))
            }
        }
    }

    @Test("Native nonstream chat still returns a complete JSON response")
    func ordinaryNonstreamResponseIsUnchanged() async throws {
        let app = MLXServerApplication.buildApplication(
            service: MLXOpenAIService(engine: RouteCancellationProbeEngine()),
            host: "127.0.0.1", port: 0
        )
        try await app.test(.live) { client in
            try await client.execute(
                uri: "/v1/chat/completions", method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: #"{"model":"local-model","messages":[{"role":"user","content":"quick"}]}"#)
            ) { response in
                #expect(response.status == .ok)
                #expect(String(buffer: response.body).contains("route ok"))
                #expect(!String(buffer: response.body).contains("[DONE]"))
            }
        }
    }

    @Test("Collected large chat bodies do not stall the SSE close watcher")
    func largeCollectedRequestStillStreamsThroughDone() async throws {
        let app = MLXServerApplication.buildApplication(
            service: MLXOpenAIService(engine: RouteCancellationProbeEngine()),
            host: "127.0.0.1", port: 0
        )
        let content = String(repeating: "x", count: 1024 * 1024)
        let body = ByteBuffer(string:
            "{\"model\":\"local-model\",\"messages\":[{\"role\":\"user\",\"content\":\"\(content)\"}],\"stream\":true}"
        )
        try await app.test(.live) { client in
            try await client.execute(
                uri: "/v1/chat/completions", method: .post,
                headers: [.contentType: "application/json"], body: body
            ) { response in
                #expect(response.status == .ok)
                #expect(String(buffer: response.body).contains("[DONE]"))
            }
        }
    }
}

private func observesEvent(_ events: AsyncStream<Void>) async -> Bool {
    await withTaskGroup(of: Bool.self) { group in
        group.addTask {
            var iterator = events.makeAsyncIterator()
            return await iterator.next() != nil
        }
        group.addTask {
            do {
                try await Task.sleep(for: .seconds(2))
            } catch {}
            return false
        }
        defer { group.cancelAll() }
        return await group.next() ?? false
    }
}

/// A native generation stream whose first event waits for synthetic prefill.
/// Cancellation must reach that work even when no response text can be written.
private actor RouteCancellationProbeEngine: MLXServerEngine {
    nonisolated let prefillStarted: AsyncStream<Void>
    nonisolated let prefillCancelled: AsyncStream<Void>
    private let started: AsyncStream<Void>.Continuation
    private let cancelled: AsyncStream<Void>.Continuation

    init() {
        (prefillStarted, started) = AsyncStream<Void>.makeStream()
        (prefillCancelled, cancelled) = AsyncStream<Void>.makeStream()
    }

    func availableModels() async throws -> [MLXServerModel] {
        [.init(id: "local-model")]
    }

    func streamChatCompletion(
        request: OpenAIChatCompletionRequest
    ) async throws -> AsyncThrowingStream<MLXServerGenerationEvent, Error> {
        let slow = request.messages.contains { $0.textContent == "slow" }
        let started = self.started
        let cancelled = self.cancelled
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    if slow {
                        started.yield(())
                        try await Task.sleep(for: .seconds(6))
                    }
                    continuation.yield(.content("route ok"))
                    continuation.yield(.info(.init(
                        promptTokens: 1, completionTokens: 2,
                        promptTime: 0.001, generationTime: 0.001, stopReason: "stop"
                    )))
                    continuation.finish()
                } catch {
                    if error is CancellationError {
                        cancelled.yield(())
                    }
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    func tokenize(_ request: TokenizeRequest) async throws -> TokenizeResponse {
        .init(tokens: [1])
    }

    func detokenize(_ request: DetokenizeRequest) async throws -> DetokenizeResponse {
        .init(text: "route ok")
    }

    func applyTemplate(_ request: ApplyTemplateRequest) async throws -> TokenizeResponse {
        .init(tokens: [1])
    }
}
