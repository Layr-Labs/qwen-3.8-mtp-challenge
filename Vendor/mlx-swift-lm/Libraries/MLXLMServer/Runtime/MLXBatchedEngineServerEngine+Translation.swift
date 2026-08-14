// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon

extension OpenAIChatCompletionRequest {
    /// OpenAI sampling fields mapped to ``MLXLMCommon.SamplingParams``.
    ///
    /// Omitted `max_tokens` becomes `Int.max` (uncapped sentinel) to match
    /// ``MLXModelContainerEngine``, which forwards a nil cap meaning "stop
    /// at EOS / stop tokens / KV exhaustion".
    ///
    /// `repetition_penalty` is coerced to `1.0` (the disabled sentinel)
    /// when the wire value is non-positive. The legacy
    /// ``GenerateParameters`` path treated `0` as "disabled", but the
    /// batched scheduler treats anything `!= 1.0` as active and divides
    /// positive logits by the penalty (see
    /// `Scheduler.applyRepetitionPenalty`), so a literal `0` from the wire
    /// would divide-by-zero on positive logits. `presence_penalty` and
    /// `frequency_penalty` are forwarded as-is because `0.0` is already
    /// the documented "disabled" value for both fields in
    /// ``SamplingParams``.
    var batchedSamplingParams: SamplingParams {
        let requestedRepetitionPenalty = repetitionPenalty ?? 1.0
        let safeRepetitionPenalty: Float =
            requestedRepetitionPenalty <= 0 ? 1.0 : requestedRepetitionPenalty
        return SamplingParams(
            maxTokens: maxTokens ?? Int.max,
            temperature: temperature ?? 0.6,
            topP: topP ?? 1.0,
            topK: topK ?? 0,
            minP: minP ?? 0,
            repetitionPenalty: safeRepetitionPenalty,
            presencePenalty: presencePenalty ?? 0.0,
            frequencyPenalty: frequencyPenalty ?? 0.0,
            stop: stop ?? []
        )
    }
}

extension OpenAIChatMessage {
    /// Chat-template-ready dictionary in the shape expected by
    /// ``BatchedEngine/buildPrompt(messages:tools:additionalContext:)``.
    /// Gemma4's renderer reads `tool_calls` from assistant history.
    func templateMessage() -> [String: any Sendable] {
        var entry: [String: any Sendable] = [
            "role": role.rawValue,
            "content": textContent,
        ]
        if let name {
            entry["name"] = name
        }
        if let toolCallID {
            entry["tool_call_id"] = toolCallID
        }
        if let toolCalls, !toolCalls.isEmpty {
            entry["tool_calls"] = toolCalls.map { call -> [String: any Sendable] in
                [
                    "id": call.id,
                    "type": call.type,
                    "function": [
                        "name": call.function.name,
                        // Decode the JSON-string arguments into an object so the
                        // chat template renders the `is mapping` branch. See
                        // `decodeToolCallArguments` for why a raw string corrupts
                        // Gemma multi-turn tool calls (double-brace rendering).
                        "arguments": decodeToolCallArguments(call.function.arguments),
                    ] as [String: any Sendable],
                ]
            }
        }
        if let reasoningContent {
            entry["reasoning_content"] = reasoningContent
        }
        return entry
    }
}
