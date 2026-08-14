import Foundation
import MLXFastCore
import Tokenizers

// QwenRuntime is split across QwenRuntime*.swift for auditability.
// Generated split; behavior identical to the original single file.

extension QwenRuntime {
    struct SemanticGPQAAnswerDocument: Encodable {
        let version: Int
        let cases: [SemanticGPQAAnswerCase]
    }

    struct SemanticGPQAAnswerCase: Encodable {
        let id: String
        let domain: String?
        let subdomain: String?
        let prompt: String
        let answerKey: String?
        let referenceAnswer: String
        let candidateAnswer: String
        let candidateTokens: [Int]
        let maxNewTokens: Int

        enum CodingKeys: String, CodingKey {
            case id
            case domain
            case subdomain
            case prompt
            case answerKey = "answer_key"
            case referenceAnswer = "reference_answer"
            case candidateAnswer = "candidate_answer"
            case candidateTokens = "candidate_tokens"
            case maxNewTokens = "max_new_tokens"
        }
    }

    static func semanticAnswerCase(
        behavior: GoldenBehaviorCase,
        generatedTokens: [Int],
        tokenizer: any Tokenizer,
        maxNewTokens: Int
    ) throws -> SemanticGPQAAnswerCase? {
        guard let prompt = trimmedNonEmpty(behavior.semanticPrompt),
              let referenceAnswer = trimmedNonEmpty(behavior.semanticReferenceAnswer)
        else {
            return nil
        }
        // Truncate at the first end-of-turn BEFORE decoding. The behavior loop
        // is fixed-length by contract (checked_steps is predicted statically
        // from the golden and cross-checked against the run), so it keeps
        // generating after the model has ended its turn; past `<|im_end|>` that
        // tail measured as degenerate repetition, and skipSpecialTokens would
        // splice it onto an otherwise correct answer for the judge to read.
        let candidateTokens = QwenChatTemplate.truncatedAtFirstEndOfTurn(
            Array(generatedTokens.prefix(maxNewTokens)),
            eosTokenId: tokenizer.eosTokenId
        )
        guard !candidateTokens.isEmpty else {
            throw MLXFastError.invalidInput("\(behavior.name) semantic GPQA candidate token list is empty")
        }
        let candidateAnswer = tokenizer.decode(tokens: candidateTokens, skipSpecialTokens: true)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return SemanticGPQAAnswerCase(
            id: behavior.name,
            domain: trimmedNonEmpty(behavior.semanticDomain),
            subdomain: trimmedNonEmpty(behavior.semanticSubdomain),
            prompt: prompt,
            answerKey: trimmedNonEmpty(behavior.semanticAnswerKey),
            referenceAnswer: referenceAnswer,
            candidateAnswer: candidateAnswer,
            candidateTokens: candidateTokens,
            maxNewTokens: maxNewTokens
        )
    }

    static func writeSemanticGPQAAnswers(
        _ answers: [SemanticGPQAAnswerCase],
        to path: String
    ) throws {
        let document = SemanticGPQAAnswerDocument(version: 1, cases: answers)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let outputURL = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try encoder.encode(document).write(to: outputURL, options: [.atomic])
    }

}
