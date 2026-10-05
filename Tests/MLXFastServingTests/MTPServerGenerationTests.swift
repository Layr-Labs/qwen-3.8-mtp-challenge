import Foundation
import Hummingbird
import HummingbirdTesting
import MLX
import MLXFastServing
import MLXLMCommon
import MLXLMServer
import MLXNN
import Testing

struct MTPServerGenerationTests {
    @Test("Each request reports native work separately from capped API usage")
    func requestDiagnosticsAreIsolated() async throws {
        let recorder = DiagnosticRecorder()
        let driver = MTPGenerationDriver(diagnostics: recorder.record) { _, _ in DiagnosticSession() }
        let engine = servingEngine(script: "abc", blocks: [], driverOverride: driver)
        #expect(try await collect(engine, maxTokens: 8).text == "abc")
        #expect(try await collect(engine, maxTokens: 2).text == "ab")
        let events = recorder.events
        try #require(events.count == 2)
        #expect(events[0].requestID != events[1].requestID)
        #expect(events.allSatisfy { $0.outcome == .completed })
        #expect(events.allSatisfy { $0.statistics == .init(cycles: 1, acceptedDrafts: 3,
            rejectedDrafts: 2, committedTokens: 4) })
        #expect(events.map(\.completionTokens) == [3, 2])
        #expect(events[0].stderrLine.contains("cycles=1 accepted_drafts=3 rejected_drafts=2 committed_tokens=4"))
        #expect(events[0].stderrLine.contains("completion_tokens=3"))
        #expect(events[0].stderrLine.split(separator: "\n").count == 1)
    }

    @Test("Injected MTP blocks use the standard detokenizer, EOS, cap, and usage")
    func blockOutputUsesStandardGeneration() async throws {
        let engine = servingEngine(script: "abc", blocks: [[1, 2, 3, 4]])
        let result = try await collect(engine, maxTokens: 8)
        #expect(result.text == "abc")
        #expect(result.info?.completionTokens == 3)
        #expect(result.info?.promptTokens == 3)
        #expect(result.info?.stopReason == "stop")
        let capped = try await collect(engine, maxTokens: 2)
        #expect(capped.text == "ab")
        #expect(capped.info?.completionTokens == 2)
        #expect(capped.info?.stopReason == "length")
    }

    @Test("Fast mode rejects non-greedy requests before preparing their prompt")
    func ineligibleRequestIsRejected() async throws {
        let preparation = PreparationRecorder()
        let engine = servingEngine(script: "abc", blocks: [[1, 4]], recorder: preparation)
        do {
            _ = try await engine.streamChatCompletion(request: request(maxTokens: 2, temperature: 0.7))
            Issue.record("Non-greedy request was accepted")
        } catch {
            #expect(String(describing: error).contains("temperature=0"))
        }
        #expect(preparation.count == 0)
    }

    @Test("Literal stop sequences span chunks and exclude stop text and its tail")
    func stopSequenceSpansChunks() async throws {
        let engine = servingEngine(script: "abENDtail", blocks: [Array(1...10)])
        let stream = try await engine.streamChatCompletion(request: .init(model: "test",
            messages: [.init(role: .user, content: .text("hello"))],
            temperature: 0, maxTokens: 16, stop: ["END"]))
        var text = ""
        var info: ServerGenerationInfo?
        for try await event in stream {
            if case .content(let chunk) = event { text += chunk }
            if case .info(let value) = event { info = value }
        }
        #expect(text == "ab")
        #expect(info?.stopReason == "stop")
        #expect(info?.completionTokens == 5)
    }

    @Test("XML tools retain request schemas and complete tool history")
    func toolsAndHistoryAreRequestScoped() async throws {
        let script = "<tool_call><function=read><parameter=offset>1</parameter></function></tool_call>"
        let recorder = PreparationRecorder()
        let ids = Array(1...(script.utf8.count + 1))
        let blocks = stride(from: 0, to: ids.count, by: 8).map {
            Array(ids[$0..<min($0 + 8, ids.count)])
        }
        let engine = servingEngine(script: script, blocks: blocks, recorder: recorder)
        let history: [OpenAIChatMessage] = [
            .init(role: .user, content: .text("read")),
            .init(role: .assistant, content: .null, toolCalls: [
                .init(id: "call1", function: .init(name: "read", arguments: "{\"offset\":0}"))]),
            .init(role: .tool, content: .text("previous result"), toolCallID: "call1"),
        ]
        for type in ["integer", "string"] {
            let tool = OpenAITool(function: .init(name: "read", parameters: .object([
                "type": .string("object"), "properties": .object([
                    "offset": .object(["type": .string(type)])])
            ])))
            let stream = try await engine.streamChatCompletion(request: .init(model: "test",
                messages: history, tools: [tool], temperature: 0, maxTokens: 256))
            var calls: [ToolCall] = []
            var info: ServerGenerationInfo?
            for try await event in stream {
                if case .toolCall(let call) = event { calls.append(call) }
                if case .info(let value) = event { info = value }
            }
            #expect(calls.count == 1)
            #expect(calls.first?.function.arguments["offset"] == (type == "integer" ? .int(1) : .string("1")))
            #expect(info?.completionTokens == script.utf8.count)
            #expect(info?.stopReason == "stop")
        }
        #expect(recorder.roles == ["user", "assistant", "tool"])
        #expect(recorder.toolCallID == "call1")
        #expect(recorder.assistantFunction == "read")
    }

    @Test("Streamed native tool calls keep distinct indexes")
    func streamedToolCallsUseDistinctIndexes() async throws {
        let script = "<tool_call><function=read><parameter=path>a</parameter></function></tool_call><tool_call><function=read><parameter=path>b</parameter></function></tool_call>"
        let ids = Array(1...(script.utf8.count + 1))
        let blocks = stride(from: 0, to: ids.count, by: 8).map {
            Array(ids[$0..<min($0 + 8, ids.count)])
        }
        let service = MLXOpenAIService(engine: servingEngine(script: script, blocks: blocks))
        let frames = try await service.streamChatCompletionFrames(request: .init(
            model: "test",
            messages: [.init(role: .user, content: .text("read both"))],
            tools: [.init(function: .init(name: "read", parameters: .object([
                "type": .string("object"),
                "properties": .object(["path": .object(["type": .string("string")])]),
            ])))],
            stream: true,
            temperature: 0,
            maxTokens: 256
        ))
        var indexes: [Int] = []
        for try await frame in frames {
            if frame == ServerSentEventEncoder.done { continue }
            let payload = String(frame.dropFirst("data: ".count))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let chunk = try JSONDecoder().decode(OpenAIChatCompletionChunk.self, from: Data(payload.utf8))
            indexes += chunk.choices.flatMap { $0.delta.toolCalls ?? [] }.compactMap { $0.index }
        }
        #expect(indexes == [0, 1])
    }

    @Test("Round errors reach the API without a false success info event")
    func errorIsPropagated() async throws {
        let recorder = DiagnosticRecorder()
        let driver = MTPGenerationDriver(diagnostics: recorder.record) { _, _ in
            BlockSession(blocks: [], failRound: true)
        }
        let engine = servingEngine(script: "abc", blocks: [], driverOverride: driver)
        let stream = try await engine.streamChatCompletion(request: request(maxTokens: 8))
        var sawInfo = false
        do {
            for try await event in stream { if case .info = event { sawInfo = true } }
            Issue.record("A failed MTP round was reported as success")
        } catch { #expect(error is ServingProbeError) }
        #expect(!sawInfo)
        #expect(recorder.events.count == 1)
        #expect(recorder.events.first?.outcome == .error)
        #expect(recorder.events.first?.completionTokens == nil)
    }

    @Test("A short request waits for canceled prefill cleanup without overlapping caches")
    func cancelThenShortRequest() async throws {
        let probe = ServingCancellationProbe()
        defer { probe.releaseCleanup() }
        let diagnostics = DiagnosticRecorder()
        let driver = MTPGenerationDriver(diagnostics: diagnostics.record) { _, _ in probe.makeSession() }
        let engine = servingEngine(script: "abc", blocks: [], driverOverride: driver)
        let first = Task { try await collect(engine, maxTokens: 8) }
        #expect(await observe(probe.started))
        first.cancel()
        #expect(await observe(probe.cancelling))
        let next = Task { try await collect(engine, maxTokens: 8) }
        try await Task.sleep(for: .milliseconds(30))
        probe.releaseCleanup()
        _ = try? await first.value
        let result = try await next.value
        #expect(result.text == "abc")
        #expect(!probe.overlapped)
        #expect(probe.created == 2)
        #expect(diagnostics.events.map(\.outcome) == [.cancelled, .completed])
        #expect(diagnostics.events.first?.completionTokens == nil)
    }

    @Test("Actual HTTP disconnect cancels injected MTP prefill and SSE reaches DONE")
    func actualHTTPDisconnect() async throws {
        let probe = ServingCancellationProbe()
        defer { probe.releaseCleanup() }
        let driver = MTPGenerationDriver { _, _ in probe.makeSession() }
        let engine = servingEngine(script: "abc", blocks: [], driverOverride: driver)
        let app = MLXServerApplication.buildApplication(service: MLXOpenAIService(engine: engine),
                                                        host: "127.0.0.1", port: 0)
        try await app.test(.live) { client in
            let port = try #require(client.port)
            var httpRequest = URLRequest(url: URL(string: "http://localhost:\(port)/v1/chat/completions")!)
            httpRequest.httpMethod = "POST"
            httpRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
            httpRequest.httpBody = Data(#"{"model":"test","messages":[{"role":"user","content":"hello"}],"temperature":0,"max_tokens":8,"stream":true}"#.utf8)
            let session = URLSession(configuration: .ephemeral)
            defer { session.invalidateAndCancel() }
            let connection = session.dataTask(with: httpRequest) { _, _, _ in }
            connection.resume()
            try #require(await observe(probe.started))
            connection.cancel()
            #expect(await observe(probe.cancelling))
            probe.releaseCleanup()
            try await client.execute(uri: "/v1/chat/completions", method: .post,
                headers: [.contentType: "application/json"], body: ByteBuffer(data: httpRequest.httpBody!)) { response in
                let body = String(buffer: response.body)
                #expect(response.status == .ok)
                #expect(body.contains("[DONE]"))
                let text = try body.split(separator: "\n").filter {
                    $0.hasPrefix("data: {")
                }.flatMap { line in
                    try JSONDecoder().decode(OpenAIChatCompletionChunk.self,
                        from: Data(line.dropFirst(6).utf8)).choices.compactMap { $0.delta.content }
                }.joined()
                #expect(text == "abc")
            }
            #expect(!probe.overlapped)
        }
    }

    @Test("Invalid stop options are rejected before prompt work", arguments: [[""], [String(repeating: "é", count: 129)], ["a", "b", "c", "d", "e"]])
    func invalidStops(stops: [String]) async throws {
        let recorder = PreparationRecorder()
        let engine = servingEngine(script: "abc", blocks: [[1, 4]], recorder: recorder)
        do {
            _ = try await engine.streamChatCompletion(request: .init(model: "test",
                messages: [.init(role: .user, content: .text("hello"))], temperature: 0,
                maxTokens: 8, stop: stops))
            Issue.record("Invalid stop request was accepted")
        } catch { #expect(String(describing: error).contains("256 UTF-8 bytes")) }
        #expect(recorder.count == 0)
    }
}

private final class DiagnosticRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [MTPRequestDiagnostic] = []
    var events: [MTPRequestDiagnostic] { lock.withLock { values } }
    func record(_ diagnostic: MTPRequestDiagnostic) { lock.withLock { values.append(diagnostic) } }
}

private final class DiagnosticSession: MTPServingSession {
    private var emitted = false
    var statistics: MTPServingStatistics? {
        .init(cycles: emitted ? 1 : 0, acceptedDrafts: emitted ? 3 : 0,
              rejectedDrafts: emitted ? 2 : 0, committedTokens: emitted ? 4 : 0)
    }
    func begin(seedTokens: [Int]) throws {}
    func round(depth: Int) throws -> [Int] {
        guard !emitted else { return [] }
        emitted = true
        return [1, 2, 3, 4]
    }
}

private func collect(_ engine: MLXModelContainerEngine, maxTokens: Int) async throws
    -> (text: String, info: ServerGenerationInfo?) {
    let stream = try await engine.streamChatCompletion(request: request(maxTokens: maxTokens))
    var text = ""
    var info: ServerGenerationInfo?
    for try await item in stream {
        if case .content(let value) = item { text += value }
        if case .info(let value) = item { info = value }
    }
    return (text, info)
}

private func request(maxTokens: Int, temperature: Float = 0) -> OpenAIChatCompletionRequest {
    .init(model: "test", messages: [.init(role: .user, content: .text("hello"))],
          temperature: temperature, maxTokens: maxTokens)
}

private func servingEngine(script: String, blocks: [[Int]], recorder: PreparationRecorder = .init(), failRound: Bool = false,
                           driverOverride: MTPGenerationDriver? = nil)
    -> MLXModelContainerEngine {
    let tokenizer = ServingTokenizer(script: script)
    var configuration = ModelConfiguration(id: "test", toolCallFormat: .xmlFunction)
    configuration.eosTokenIds = [tokenizer.stopToken]
    let model = ModelContainer(context: .init(configuration: configuration,
        model: ImmediateEOSModel(stopToken: tokenizer.stopToken),
        processor: ServingProcessor(recorder: recorder), tokenizer: tokenizer))
    let driver = driverOverride ?? MTPGenerationDriver { _, _ in BlockSession(blocks: blocks, failRound: failRound) }
    return MLXModelContainerEngine(modelID: "test", model: model, modelType: "qwen3_5",
                                  contextTokens: 1024, generationDriver: driver)
}

private func observe(_ stream: AsyncStream<Void>) async -> Bool {
    await withTaskGroup(of: Bool.self) { group in
        group.addTask { var iterator = stream.makeAsyncIterator(); return await iterator.next() != nil }
        group.addTask { try? await Task.sleep(for: .seconds(2)); return false }
        defer { group.cancelAll() }
        return await group.next() ?? false
    }
}

private final class ServingCancellationProbe: @unchecked Sendable {
    let started: AsyncStream<Void>
    let cancelling: AsyncStream<Void>
    private let startedContinuation: AsyncStream<Void>.Continuation
    private let cancellingContinuation: AsyncStream<Void>.Continuation
    private let condition = NSCondition()
    private var cleanupReleased = false
    private var cleaning = false
    private var didOverlap = false
    private var count = 0
    var overlapped: Bool { condition.withLock { didOverlap } }
    var created: Int { condition.withLock { count } }
    init() {
        (started, startedContinuation) = AsyncStream<Void>.makeStream()
        (cancelling, cancellingContinuation) = AsyncStream<Void>.makeStream()
    }
    func makeSession() -> any MTPServingSession {
        condition.withLock {
            count += 1
            if cleaning { didOverlap = true }
            return CancellationSession(probe: self, slow: count == 1)
        }
    }
    func releaseCleanup() { condition.withLock { cleanupReleased = true; condition.broadcast() } }
    func beginSlow() throws {
        startedContinuation.yield(())
        let deadline = Date().addingTimeInterval(2)
        while !Task.isCancelled, Date() < deadline { Thread.sleep(forTimeInterval: 0.001) }
        guard Task.isCancelled else { throw ServingProbeError.round }
        condition.lock()
        cleaning = true
        cancellingContinuation.yield(())
        let cleanupDeadline = Date().addingTimeInterval(2)
        while !cleanupReleased, Date() < cleanupDeadline { _ = condition.wait(until: cleanupDeadline) }
        cleaning = false
        condition.unlock()
        throw CancellationError()
    }
}

private final class CancellationSession: MTPServingSession {
    let probe: ServingCancellationProbe
    let slow: Bool
    private var emitted = false
    init(probe: ServingCancellationProbe, slow: Bool) { self.probe = probe; self.slow = slow }
    func begin(seedTokens: [Int]) throws { if slow { try probe.beginSlow() } }
    func round(depth: Int) throws -> [Int] {
        guard !emitted else { return [] }
        emitted = true
        return [1, 2, 3, 4]
    }
}

private final class BlockSession: MTPServingSession {
    private var blocks: [[Int]]
    private let failRound: Bool
    init(blocks: [[Int]], failRound: Bool = false) { self.blocks = blocks; self.failRound = failRound }
    func begin(seedTokens: [Int]) throws {}
    func round(depth: Int) throws -> [Int] {
        if failRound { throw ServingProbeError.round }
        return blocks.isEmpty ? [] : blocks.removeFirst()
    }
}

private enum ServingProbeError: Error { case round }

private final class PreparationRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0
    private var lastRoles: [String] = []
    private var lastToolCallID: String?
    private var lastAssistantFunction: String?
    var count: Int { lock.withLock { value } }
    var roles: [String] { lock.withLock { lastRoles } }
    var toolCallID: String? { lock.withLock { lastToolCallID } }
    var assistantFunction: String? { lock.withLock { lastAssistantFunction } }
    func record(_ input: UserInput) {
        lock.withLock {
            value += 1
            if case .messages(let messages) = input.prompt {
                lastRoles = messages.compactMap { $0["role"] as? String }
                lastToolCallID = messages.last?["tool_call_id"] as? String
                let calls = messages.dropLast().last?["tool_calls"] as? [[String: any Sendable]]
                let function = calls?.first?["function"] as? [String: any Sendable]
                lastAssistantFunction = function?["name"] as? String
            }
        }
    }
}

private struct ServingProcessor: UserInputProcessor {
    let recorder: PreparationRecorder
    func prepare(input: UserInput) throws -> LMInput {
        recorder.record(input)
        return Device.withDefaultDevice(.cpu) { .init(tokens: MLXArray([Int32(7), 8, 9])) }
    }
}

private struct ServingTokenizer: Tokenizer {
    let bytes: [UInt8]
    var stopToken: Int { bytes.count + 1 }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    init(script: String) { bytes = Array(script.utf8) }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [7, 8, 9] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.compactMap { (1...bytes.count).contains($0) ? bytes[$0 - 1] : nil }, as: UTF8.self)
    }
    func convertTokenToId(_ token: String) -> Int? { Int(token) }
    func convertIdToToken(_ id: Int) -> String? { String(id) }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                           additionalContext: [String: any Sendable]?) throws -> [Int] { [7, 8, 9] }
}

private final class ImmediateEOSModel: Module, LanguageModel {
    let stopToken: Int
    init(stopToken: Int) { self.stopToken = stopToken; super.init() }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult { .tokens(input.text) }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var logits = Array(repeating: Float(-1000), count: stopToken + 1)
        logits[stopToken] = 0
        return Device.withDefaultDevice(.cpu) { MLXArray(logits).reshaped([1, 1, stopToken + 1]) }
    }
}
