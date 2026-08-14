import Foundation

// ChatML framing for the hidden GPQA capture path.
//
// WHY THIS EXISTS. The Qwen 3.6 checkpoint is ChatML instruction-tuned
// (eos_token `<|im_end|>`) and ships its chat template as a
// `chat_template.jinja` sidecar -- transformers v5 moved templates out of
// tokenizer_config.json, so `tokenizer_config.json` here has no inline
// `chat_template` and no `bos_token` at all. The capture path encoded the raw
// question with `addSpecialTokens: true`, which for this tokenizer adds
// nothing, so the model received a bare question with no turn framing. Measured
// on m5-max-128gb-3 (2026-08-13, 9 hidden cases, 512-token budget): the model
// never committed to an answer in ANY case, unique-token ratios fell to 0.148
// and three captures collapsed into an exact 14-token loop. That is what
// produced the semantic gate's 1/9 -- a prompting defect, not a model or
// budget limit.
//
// The serial/Laguna reference answered letter-first under raw encoding, which
// is why the un-framed path went unnoticed until this track.
//
// THINKING IS DISABLED ON PURPOSE. The sidecar's `add_generation_prompt` branch
// opens a `<think>` block unless `enable_thinking` is false, in which case it
// emits a pre-closed `<think>\n\n</think>\n\n`. With thinking open the model
// reasons past any budget the gate can afford (0/9 committed within 512 tokens,
// measured). With it pre-closed the model answers immediately: all 9 hidden
// cases emitted a single bare option letter followed by `<|im_end|>`, and 8 of
// 9 matched the answer key. The semantic gate is a short-answer check with a
// fixed token budget, so the direct-answer mode is the one it can actually
// score -- and it restores the letter-first shape the 128-token budget was
// calibrated around.
public enum QwenChatTemplate {
    // Mirrors the no-tools, single-user-message path of the checkpoint's
    // chat_template.jinja with add_generation_prompt=true and
    // enable_thinking=false. The sidecar emits a system block ONLY when the
    // caller supplies a system message, so there is no default system prompt
    // to reproduce here.
    public static func userTurnDisablingThinking(_ prompt: String) -> String {
        "<|im_start|>user\n"
            + prompt
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    // The behavior-gate generation loop is fixed-length by contract: its step
    // count is what verify-correctness-golden.sh predicts statically from the
    // golden (sum of behavior[].max_new_tokens) and what
    // validate-benchmark-artifacts.sh cross-checks against the reported
    // checked_steps. So generation must NOT stop early. Instead the capture
    // truncates at the first end-of-turn token before decoding: past
    // `<|im_end|>` the model is generating into a finished turn, which measured
    // as degenerate repetition (unique-token ratio as low as 0.02), and
    // `skipSpecialTokens` would silently splice that tail onto an otherwise
    // correct answer for the judge to read.
    public static func truncatedAtFirstEndOfTurn(_ tokens: [Int], eosTokenId: Int?) -> [Int] {
        guard let eosTokenId, let index = tokens.firstIndex(of: eosTokenId) else {
            return tokens
        }
        // An immediate end-of-turn means the model answered with nothing. Keep
        // that as a genuinely empty answer for the judge rather than inventing
        // content, but never hand back an empty token list to callers that
        // require at least one token -- they treat empty as a capture fault,
        // which is a different (and wrong) diagnosis.
        guard index > 0 else {
            return Array(tokens.prefix(1))
        }
        return Array(tokens.prefix(index))
    }
}
