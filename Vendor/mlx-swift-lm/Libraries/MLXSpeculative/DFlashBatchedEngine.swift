// Copyright © 2026 Apple Inc.

import Foundation
import MLXLLM
import MLXLMCommon

/// Single-instance DFlash engine for local serving.
///
/// This mirrors the public surface of `BatchedEngine` closely enough for the
/// HTTP server to swap it in when a DFlash drafter is configured. The engine is
/// intentionally greedy-only; non-greedy requests are rejected instead of being
/// silently sampled with the wrong algorithm.
public final class DFlashBatchedEngine: @unchecked Sendable {
    private let context: ModelContext
    private let target: any DFlashTargetModel
    private let drafter: DFlashDraftModel
    private let tokenizer: any Tokenizer
    private let eosTokenIds: Set<Int>
    private let blockSize: Int?
    private let stepInterval: TimeInterval

    public let modelName: String

    private let engineQueue = DispatchQueue(
        label: "com.eigen.dflash.engine",
        qos: .userInitiated
    )
    private var running = false
    private var task: Task<Void, Never>?
    private var waiting: [DFlashEnginePendingRequest] = []
    private var collectors: [String: RequestOutputCollector] = [:]
    private var activeByUID: [Int: DFlashEngineActiveRequest] = [:]
    private var requestIdToUID: [String: Int] = [:]
    private var nextUID = 0
    private var generator: DFlashBatchedTokenGenerator
    private var stepsExecuted = 0

    public init(
        context: ModelContext,
        drafter: DFlashDraftModel,
        blockSize: Int? = nil,
        stepInterval: TimeInterval = 0.001
    ) throws {
        guard let target = context.model as? any DFlashTargetModel else {
            throw DFlashError.unsupportedTarget(String(describing: type(of: context.model)))
        }
        self.context = context
        self.target = target
        self.drafter = drafter
        self.tokenizer = context.tokenizer
        self.eosTokenIds = Self.makeBuiltInStopTokenIds(context: context)
        self.blockSize = blockSize
        self.stepInterval = stepInterval
        self.modelName = context.configuration.name
        self.generator = try DFlashBatchedTokenGenerator(
            target: target,
            drafter: drafter,
            blockSize: blockSize,
            parameters: GenerateParameters(temperature: 0))
    }

    public func start() async {
        guard !running else { return }
        running = true
        task = Task { [weak self] in await self?.engineLoop() }
    }

    public func stop() async {
        running = false
        task?.cancel()
        task = nil
    }

    public func generate(
        prompt: String,
        samplingParams: SamplingParams = SamplingParams()
    ) async throws -> String {
        try await generateWithResult(prompt: prompt, samplingParams: samplingParams).outputText
    }

    public func generateWithResult(
        prompt: String,
        samplingParams: SamplingParams = SamplingParams()
    ) async throws -> RequestOutput {
        for await output in streamOutputs(prompt: prompt, samplingParams: samplingParams) {
            if output.finished || output.error != nil {
                if let error = output.error {
                    throw EngineError.generationFailed(error)
                }
                return output
            }
        }
        throw EngineError.missingOutput
    }

    public func streamOutputs(
        prompt: String,
        samplingParams: SamplingParams = SamplingParams()
    ) -> AsyncStream<RequestOutput> {
        let rid = UUID().uuidString
        let collector = RequestOutputCollector(aggregate: true)
        let validationError = Self.validateGreedyRequest(samplingParams)
        let terminationState = DFlashStreamTerminationState()

        engineQueue.async { [weak self] in
            guard let self else { return }
            collectors[rid] = collector

            if let validationError {
                collector.put(RequestOutput(
                    requestId: rid,
                    finished: true,
                    finishReason: "error",
                    error: validationError))
                return
            }

            let promptTokens = tokenizer.encode(text: prompt).map(Int32.init)
            guard !promptTokens.isEmpty else {
                collector.put(RequestOutput(
                    requestId: rid,
                    finished: true,
                    finishReason: "error",
                    error: "Prompt tokenization produced no tokens."))
                return
            }

            let uid = nextUID
            nextUID += 1
            requestIdToUID[rid] = uid
            waiting.append(DFlashEnginePendingRequest(
                requestId: rid,
                uid: uid,
                promptTokens: promptTokens,
                samplingParams: samplingParams,
                machine: makeStateMachine(for: samplingParams)))
        }

        return AsyncStream { continuation in
            let streamTask = Task {
                while true {
                    let output: RequestOutput
                    if let available = collector.getNowait() {
                        output = available
                    } else {
                        output = await collector.get()
                    }
                    continuation.yield(output)
                    if output.finished || output.error != nil {
                        break
                    }
                }
                terminationState.markCompleted()
                cleanupRequest(rid)
                continuation.finish()
            }
            continuation.onTermination = { _ in
                streamTask.cancel()
                if !terminationState.isCompleted {
                    self.abortRequest(rid)
                }
            }
        }
    }

    public func streamGenerate(
        prompt: String,
        samplingParams: SamplingParams = SamplingParams()
    ) -> AsyncStream<String> {
        AsyncStream { continuation in
            Task {
                for await output in streamOutputs(prompt: prompt, samplingParams: samplingParams) {
                    if !output.newText.isEmpty {
                        continuation.yield(output.newText)
                    }
                    if output.finished || output.error != nil {
                        break
                    }
                }
                continuation.finish()
            }
        }
    }

    public func buildPrompt(messages: [[String: String]]) -> String {
        do {
            let tokenIds = try tokenizer.applyChatTemplate(messages: messages)
            return tokenizer.decode(tokenIds: tokenIds)
        } catch {
            return messages.map { "\($0["role"] ?? "user"): \($0["content"] ?? "")" }
                .joined(separator: "\n") + "\nassistant:"
        }
    }

    @discardableResult
    public func abortRequest(_ requestId: String) -> Bool {
        engineQueue.async { [weak self] in
            guard let self else { return }
            waiting.removeAll { $0.requestId == requestId }
            if let uid = requestIdToUID.removeValue(forKey: requestId) {
                generator.cancel(uid: uid)
                activeByUID.removeValue(forKey: uid)
            }
            collectors[requestId]?.put(RequestOutput(
                requestId: requestId,
                finished: true,
                finishReason: "abort",
                error: "Request aborted"))
        }
        return true
    }

    public func getStats() -> [String: Any] {
        var stats: [String: Any] = [:]
        engineQueue.sync {
            stats = [
                "engine_type": "dflash_batched",
                "model_name": modelName,
                "running": running,
                "steps_executed": stepsExecuted,
                "waiting_requests": waiting.count,
                "active_requests": activeByUID.count,
                "block_size": blockSize ?? drafter.config.recommendedBlockSize,
            ]
        }
        return stats
    }

    private func engineLoop() async {
        while running {
            let didWork = await withCheckedContinuation { continuation in
                engineQueue.async { [weak self] in
                    let didWork = self?.stepLocked() ?? false
                    continuation.resume(returning: didWork)
                }
            }
            if !didWork {
                try? await Task.sleep(nanoseconds: UInt64(stepInterval * 1_000_000_000))
            }
            if Task.isCancelled { break }
        }
    }

    @discardableResult
    private func stepLocked() -> Bool {
        var didWork = false
        if !waiting.isEmpty {
            didWork = true
            admitWaitingLocked()
        }
        guard !generator.isEmpty else {
            return didWork
        }

        do {
            let responses = try generator.next()
            stepsExecuted += 1
            if !responses.isEmpty {
                didWork = true
                for response in responses {
                    processResponseLocked(response)
                }
            }
        } catch {
            failActiveLocked(error: error)
            didWork = true
        }

        return didWork
    }

    private func admitWaitingLocked() {
        let pending = waiting
        waiting.removeAll(keepingCapacity: true)

        do {
            try generator.insert(
                prompts: pending.map(\.promptTokens),
                uids: pending.map(\.uid),
                maxTokens: pending.map { $0.samplingParams.maxTokens })
            for request in pending {
                activeByUID[request.uid] = DFlashEngineActiveRequest(
                    requestId: request.requestId,
                    promptTokenCount: request.promptTokens.count,
                    outputTokenIds: [],
                    outputText: "",
                    detokenizer: NaiveStreamingDetokenizer(tokenizer: tokenizer),
                    machine: request.machine,
                    matcherState: request.machine.makeState())
            }
        } catch {
            for request in pending {
                collectors[request.requestId]?.put(RequestOutput(
                    requestId: request.requestId,
                    finished: true,
                    finishReason: "error",
                    error: String(describing: error)))
                requestIdToUID.removeValue(forKey: request.requestId)
            }
        }
    }

    private func processResponseLocked(_ response: DFlashBatchedTokenResponse) {
        guard var active = activeByUID[response.uid] else { return }

        let (nextMatcher, matchedSequence, currentState) =
            active.machine.match(active.matcherState, response.token)
        active.matcherState = nextMatcher
        let hitStop = matchedSequence != nil && currentState == nil

        var newText = ""
        if !hitStop {
            active.outputTokenIds.append(response.token)
            active.detokenizer.append(token: response.token)
            newText = active.detokenizer.next() ?? ""
            active.outputText += newText
        }

        let finishReason: String?
        if hitStop {
            finishReason = "stop"
        } else {
            finishReason = response.finishReason
        }

        if let finishReason {
            if hitStop {
                generator.cancel(uid: response.uid)
            }
            let finalText = tokenizer.decode(tokenIds: active.outputTokenIds)
            collectors[active.requestId]?.put(RequestOutput(
                requestId: active.requestId,
                newTokenIds: hitStop ? [] : [response.token],
                newText: newText,
                outputTokenIds: active.outputTokenIds,
                outputText: finalText,
                finished: true,
                finishReason: finishReason,
                promptTokens: active.promptTokenCount,
                completionTokens: active.outputTokenIds.count,
                cachedTokens: 0))
            activeByUID.removeValue(forKey: response.uid)
            requestIdToUID.removeValue(forKey: active.requestId)
        } else {
            activeByUID[response.uid] = active
            collectors[active.requestId]?.put(RequestOutput(
                requestId: active.requestId,
                newTokenIds: [response.token],
                newText: newText,
                outputTokenIds: active.outputTokenIds,
                outputText: active.outputText,
                finished: false,
                promptTokens: active.promptTokenCount,
                completionTokens: active.outputTokenIds.count,
                cachedTokens: 0))
        }
    }

    private func failActiveLocked(error: Error) {
        let message = String(describing: error)
        for active in activeByUID.values {
            collectors[active.requestId]?.put(RequestOutput(
                requestId: active.requestId,
                finished: true,
                finishReason: "error",
                promptTokens: active.promptTokenCount,
                completionTokens: active.outputTokenIds.count,
                cachedTokens: 0,
                error: message))
        }
        activeByUID.removeAll()
        requestIdToUID.removeAll()
        waiting.removeAll()
    }

    private func cleanupRequest(_ requestId: String) {
        engineQueue.async { [weak self] in
            self?.collectors.removeValue(forKey: requestId)
        }
    }

    private func makeStateMachine(for samplingParams: SamplingParams) -> SequenceStateMachine {
        var sequences: [(sequence: [Int], next: String?)] = eosTokenIds.map {
            (sequence: [$0], next: nil)
        }
        for id in samplingParams.stopTokenIds where !eosTokenIds.contains(id) {
            sequences.append((sequence: [id], next: nil))
        }
        for stop in samplingParams.stop where !stop.isEmpty {
            let tokens = tokenizer.encode(text: stop)
            if !tokens.isEmpty {
                sequences.append((sequence: tokens, next: nil))
            }
        }
        guard !sequences.isEmpty else {
            return SequenceStateMachine()
        }
        return SequenceStateMachine(states: ["normal": sequences])
    }

    private static func validateGreedyRequest(_ params: SamplingParams) -> String? {
        if params.temperature != 0 {
            return "DFlash serving currently supports greedy decoding only; set temperature to 0."
        }
        if params.repetitionPenalty != 1.0
            || params.presencePenalty != 0.0
            || params.frequencyPenalty != 0.0
        {
            return "DFlash serving does not yet support repetition, presence, or frequency penalties."
        }
        return nil
    }

    private static func makeBuiltInStopTokenIds(context: ModelContext) -> Set<Int> {
        var ids = context.configuration.eosTokenIds
        if let tokenizerEOS = context.tokenizer.eosTokenId {
            ids.insert(tokenizerEOS)
        }
        for token in context.configuration.extraEOSTokens {
            if let id = context.tokenizer.convertTokenToId(token) {
                ids.insert(id)
            }
        }
        return ids
    }
}

private struct DFlashEnginePendingRequest {
    let requestId: String
    let uid: Int
    let promptTokens: [Int32]
    let samplingParams: SamplingParams
    let machine: SequenceStateMachine
}

private struct DFlashEngineActiveRequest {
    let requestId: String
    let promptTokenCount: Int
    var outputTokenIds: [Int]
    var outputText: String
    var detokenizer: NaiveStreamingDetokenizer
    let machine: SequenceStateMachine
    var matcherState: SequenceStateMachineState
}

private final class DFlashStreamTerminationState: @unchecked Sendable {
    private let lock = NSLock()
    private var completed = false

    var isCompleted: Bool {
        lock.lock()
        defer { lock.unlock() }
        return completed
    }

    func markCompleted() {
        lock.lock()
        completed = true
        lock.unlock()
    }
}
