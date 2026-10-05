import Hummingbird

/// Called only after the model tokenizer and chat template produced the actual
/// prompt. Subtraction avoids integer overflow on untrusted max_tokens values.
enum ContextTokenBudget {
    static func outputLimit(promptTokens: Int, requested: Int?, contextTokens: Int) throws -> Int {
        guard (1...262144).contains(contextTokens), promptTokens >= 0,
            promptTokens < contextTokens
        else {
            throw HTTPError(.badRequest, message: "Prompt exceeds the configured context token limit.")
        }
        let remaining = contextTokens - promptTokens
        guard let requested else { return remaining }
        guard requested > 0, requested <= remaining else {
            throw HTTPError(
                .badRequest,
                message: "Prompt plus max_tokens exceeds the configured context token limit."
            )
        }
        return requested
    }
}
