// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Generate text from a DFlash-capable target using a DFlash drafter.
///
/// The current implementation is the Qwen-first greedy path:
/// `parameters.temperature` must be `0`.
public func generateDFlash(
    input: LMInput,
    cache: [KVCache]? = nil,
    draftCache: [KVCache]? = nil,
    parameters: GenerateParameters,
    target: ModelContext,
    drafter: DFlashDraftModel,
    blockSize: Int? = nil,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<Generation> {
    guard let targetModel = target.model as? any DFlashTargetModel else {
        throw DFlashError.unsupportedTarget(String(describing: type(of: target.model)))
    }

    let iterator = try DFlashTokenIterator(
        input: input,
        target: targetModel,
        drafter: drafter,
        targetCache: cache,
        draftCache: draftCache,
        parameters: parameters,
        blockSize: blockSize
    )

    let (stream, _) = generateTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: target.configuration,
        tokenizer: target.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket
    )
    return stream
}

/// Generate raw token IDs from a DFlash-capable target using a DFlash drafter.
public func generateDFlashTokens(
    input: LMInput,
    cache: [KVCache]? = nil,
    draftCache: [KVCache]? = nil,
    parameters: GenerateParameters,
    target: ModelContext,
    drafter: DFlashDraftModel,
    blockSize: Int? = nil,
    includeStopToken: Bool = false,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<TokenGeneration> {
    guard let targetModel = target.model as? any DFlashTargetModel else {
        throw DFlashError.unsupportedTarget(String(describing: type(of: target.model)))
    }

    let iterator = try DFlashTokenIterator(
        input: input,
        target: targetModel,
        drafter: drafter,
        targetCache: cache,
        draftCache: draftCache,
        parameters: parameters,
        blockSize: blockSize
    )
    _ = wiredMemoryTicket

    var builtStopTokenIds = target.configuration.eosTokenIds
    if let tokenizerEOS = target.tokenizer.eosTokenId {
        builtStopTokenIds.insert(tokenizerEOS)
    }
    for token in target.configuration.extraEOSTokens {
        if let id = target.tokenizer.convertTokenToId(token) {
            builtStopTokenIds.insert(id)
        }
    }

    let stopTokenIds = builtStopTokenIds
    let promptTokenCount = input.text.tokens.size
    let boxed = DFlashIteratorBox(iterator: iterator)
    return AsyncStream<TokenGeneration> { continuation in
        let task = Task {
            let generateStart = Date()
            var tokenCount = 0
            var stopReason: GenerateStopReason = .length
            while let token = boxed.next() {
                tokenCount += 1
                if stopTokenIds.contains(token) {
                    stopReason = .stop
                    if includeStopToken {
                        continuation.yield(.token(token))
                    }
                    break
                }
                continuation.yield(.token(token))
                if Task.isCancelled {
                    stopReason = .cancelled
                    break
                }
            }
            let info = GenerateCompletionInfo(
                promptTokenCount: promptTokenCount,
                generationTokenCount: tokenCount,
                promptTime: boxed.promptPrefillTime,
                generationTime: Date().timeIntervalSince(generateStart),
                stopReason: stopReason
            )
            continuation.yield(.info(info))
            continuation.finish()
        }
        continuation.onTermination = { _ in
            task.cancel()
        }
    }
}

private final class DFlashIteratorBox: @unchecked Sendable {
    private var iterator: DFlashTokenIterator

    var promptPrefillTime: TimeInterval {
        iterator.promptPrefillTime
    }

    init(iterator: DFlashTokenIterator) {
        self.iterator = iterator
    }

    func next() -> Int? {
        iterator.next()
    }
}
