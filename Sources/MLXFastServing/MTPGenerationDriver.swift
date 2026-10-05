import Foundation
import MLX
import MLXFastModel
import MLXLMCommon
import MLXLMServer

public struct MTPGenerationDriver: MLXServerGenerationDriver {
    public typealias SessionFactory = @Sendable (ModelContext, Set<Int>) throws -> any MTPServingSession
    private let maxDepth: Int
    private let makeSession: SessionFactory
    private let diagnostics: @Sendable (MTPRequestDiagnostic) -> Void
    private let gate = MTPAdmission()

    public init(maxDepth: Int = 8,
                diagnostics: @escaping @Sendable (MTPRequestDiagnostic) -> Void = MTPRequestDiagnostic.writeStandardError,
                makeSession: @escaping SessionFactory) {
        self.maxDepth = maxDepth
        self.makeSession = makeSession
        self.diagnostics = diagnostics
    }

    public init(maxDepth: Int = 8) {
        self.init(maxDepth: maxDepth) { context, stops in
            guard let target = context.model as? any Qwen36MTPTarget else {
                throw MTPServingError.invalidModel
            }
            return WinningMTPServingSession(session:
                try Qwen36MTPBlockSession(model: target, stopTokens: stops))
        }
    }

    public func validate(request: OpenAIChatCompletionRequest) throws {
        try MLXServerGreedyPolicy.validate(request)
    }

    public func generate(
        model: ModelContainer, input: consuming sending LMInput,
        parameters: GenerateParameters, request: OpenAIChatCompletionRequest,
        tools: [ToolSpec]?, toolCallFormat: ToolCallFormat
    ) async throws -> AsyncThrowingStream<Generation, Error> {
        try await gate.acquire()
        let started = DispatchTime.now().uptimeNanoseconds
        let requestID = UUID().uuidString
        let statistics = MTPRequestStatisticsStore()
        @Sendable func report(_ outcome: MTPRequestDiagnostic.Outcome, completionTokens: Int?) {
            diagnostics(.init(requestID: requestID, outcome: outcome, maxDepth: maxDepth,
                statistics: statistics.snapshot, completionTokens: completionTokens,
                generationElapsedMilliseconds: Int((DispatchTime.now().uptimeNanoseconds - started) / 1_000_000)))
        }
        let handle: MTPStreamHandle
        var created: MTPStreamHandle?
        do {
            try Task.checkCancellation()
            handle = try await model.perform(nonSendable: input) { context, input in
                try Task.checkCancellation()
                var configuration = context.configuration
                configuration.toolCallFormat = toolCallFormat
                var stops = configuration.eosTokenIds
                if let eos = context.tokenizer.eosTokenId { stops.insert(eos) }
                for token in configuration.extraEOSTokens {
                    if let id = context.tokenizer.convertTokenToId(token) { stops.insert(id) }
                }
                if let unknown = context.tokenizer.unknownTokenId { stops.insert(unknown) }
                let session = ObservedMTPServingSession(session: try makeSession(context, stops),
                                                        store: statistics)
                let failure = MTPGenerationFailure()
                let iterator = MTPTokenIterator(session: session,
                    seedTokens: input.text.tokens.asArray(Int.self),
                    maxTokens: parameters.maxTokens ?? 1, maxDepth: maxDepth, failure: failure)
                let (stream, task) = generateTask(promptTokenCount: input.text.tokens.size,
                    modelConfiguration: configuration, tokenizer: context.tokenizer,
                    iterator: iterator, tools: tools, stopSequences: request.stop)
                return MTPStreamHandle(stream: stream, task: task, failure: failure)
            }
            created = handle
            try Task.checkCancellation()
        } catch {
            if let created {
                created.task.cancel()
                await created.task.value
            }
            report(error is CancellationError ? .cancelled : .error, completionTokens: nil)
            await gate.release()
            throw error
        }
        let (stream, continuation) = AsyncThrowingStream<Generation, Error>.makeStream()
        let producer = Task {
            var completionTokens: Int?
            do {
                for await item in handle.stream {
                    try Task.checkCancellation()
                    if case .info = item, let error = handle.failure.error { throw error }
                    if case .info(let info) = item { completionTokens = info.generationTokenCount }
                    continuation.yield(item)
                }
                await handle.task.value
                if let error = handle.failure.error { throw error }
                try Task.checkCancellation()
                report(.completed, completionTokens: completionTokens)
                await gate.release()
                continuation.finish()
            } catch {
                handle.task.cancel()
                await handle.task.value
                report(error is CancellationError ? .cancelled : .error, completionTokens: completionTokens)
                await gate.release()
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { termination in
            if case .cancelled = termination {
                handle.task.cancel()
                producer.cancel()
            }
        }
        return stream
    }
}

public enum MTPServingError: Error, CustomStringConvertible {
    case invalidModel
    public var description: String { "native_mtp requires the pinned MTP-capable Qwen3.8 27B model." }
}

private final class WinningMTPServingSession: MTPServingSession {
    private let session: Qwen36MTPBlockSession
    init(session: Qwen36MTPBlockSession) { self.session = session }
    func begin(seedTokens: [Int]) throws { try session.beginForServing(seedTokens: seedTokens) }
    func round(depth: Int) throws -> [Int] { try session.generateRound(depth: depth).tokens }
    var statistics: MTPServingStatistics? {
        .init(cycles: session.roundCount, acceptedDrafts: session.acceptedDraftTotal,
              rejectedDrafts: session.rejectedDraftTotal, committedTokens: session.committedTokenCount)
    }
}

private struct MTPStreamHandle: Sendable {
    let stream: AsyncStream<Generation>
    let task: Task<Void, Never>
    let failure: MTPGenerationFailure
}

private actor MTPAdmission {
    private var active = false
    private var waiting: (UUID, CheckedContinuation<Void, any Error>)?
    func acquire() async throws {
        try Task.checkCancellation()
        if !active { active = true; return }
        guard waiting == nil else { throw MLXServerGreedyPolicy.busy() }
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                if Task.isCancelled { continuation.resume(throwing: CancellationError()) }
                else { waiting = (id, continuation) }
            }
        } onCancel: {
            Task { await self.cancelWaiting(id) }
        }
    }
    private func cancelWaiting(_ id: UUID) {
        guard let waiter = waiting, waiter.0 == id else { return }
        waiting = nil
        waiter.1.resume(throwing: CancellationError())
    }
    func release() {
        if let waiter = waiting {
            waiting = nil
            waiter.1.resume()
        } else { active = false }
    }
}
