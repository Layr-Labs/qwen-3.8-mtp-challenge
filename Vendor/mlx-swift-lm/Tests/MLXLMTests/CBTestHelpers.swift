// Shared helpers for continuous-batching tests.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

// MARK: - IncrTestLanguageModel
// Deterministic language model: always outputs (inputToken + 1) % 16.

final class IncrTestLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
    let vocabularySize = 16
    var kvHeads: [Int] { [1] }

    func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)

        if let cache {
            let keys = MLXArray.ones([B, 1, S, 1], dtype: .float32)
            let vals = keys * 2
            for c in cache { _ = c.update(keys: keys, values: vals) }
        }

        let tok = inputs.asArray(UInt32.self).map { Int($0) }
        var logits = Array(repeating: Float(-1_000), count: B * S * vocabularySize)
        for b in 0 ..< B {
            for t in 0 ..< S {
                let next = (tok[b * S + t] + 1) % vocabularySize
                logits[(b * S + t) * vocabularySize + next] = 0
            }
        }
        return MLXArray(logits).reshaped([B, S, vocabularySize])
    }
}

// MARK: - IdentityTokenizer
// Encodes prompt strings by splitting on spaces and parsing ints.
// Falls back to a fixed sequence. Decodes as space-joined numbers.

struct IdentityTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    let encodingMap: [String: [Int]]

    init(encodingMap: [String: [Int]] = [:]) {
        self.encodingMap = encodingMap
    }

    func encode(text: String, addSpecialTokens: Bool = false) -> [Int] {
        if let mapped = encodingMap[text] { return mapped }
        // Try parsing "1 2 3" → [1, 2, 3]
        let parsed = text.split(separator: " ").compactMap { Int($0) }
        return parsed.isEmpty ? [1, 2] : parsed
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool = false) -> String {
        tokenIds.map { String($0) }.joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? { Int(token) }
    func convertIdToToken(_ id: Int) -> String? { String(id) }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) throws -> [Int] {
        // Include role markers so the chat template test can verify them.
        let parts = messages.compactMap { m -> String? in
            guard let role = m["role"] as? String,
                  let content = m["content"] as? String else { return nil }
            return "[\(role)]:\(content)"
        }
        return encode(text: parts.joined(separator: " "))
    }
}

// MARK: - Factory helpers

func makeTestScheduler(
    eosTokenIds: Set<Int> = [5],
    encodingMap: [String: [Int]] = [:]
) -> Scheduler {
    Scheduler(
        model: IncrTestLanguageModel(),
        tokenizer: IdentityTokenizer(encodingMap: encodingMap),
        config: SchedulerConfig(
            maxNumSeqs: 4,
            maxNumBatchedTokens: 128,
            prefillStepSize: 16,
            streamInterval: 1
        ),
        eosTokenIds: eosTokenIds
    )
}

func makeTestEngine(eosTokenIds: Set<Int> = [5]) -> EngineCore {
    EngineCore(
        scheduler: makeTestScheduler(eosTokenIds: eosTokenIds),
        config: ContinuousBatchingConfig(stepInterval: 0.0005, yieldInterval: 1)
    )
}

/// Request with an [Int] prompt (bypasses tokenization).
func makeIntRequest(
    id: String = UUID().uuidString,
    tokens: [Int] = [3],
    maxTokens: Int = 10
) -> Request {
    Request(
        requestId: id,
        prompt: tokens as AnyHashable,
        samplingParams: SamplingParams(maxTokens: maxTokens)
    )
}
