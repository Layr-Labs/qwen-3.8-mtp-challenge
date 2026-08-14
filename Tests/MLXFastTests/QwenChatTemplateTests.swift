import Foundation
import Testing
@testable import MLXFastCore

// The rendered wrap is pinned against the checkpoint's chat_template.jinja
// (no-tools, single user message, add_generation_prompt=true,
// enable_thinking=false). Read off the sidecar shipped in the transformed
// weights on m5-max-128gb-3, 2026-08-13. If a re-pin ships a different
// template, this fixture must be re-derived from the new sidecar -- a silent
// drift here reintroduces the un-framed prompting that scored the semantic
// gate 1/9.
private let expectedRenderedTurn = """
<|im_start|>user
QUESTION<|im_end|>
<|im_start|>assistant
<think>

</think>


"""

@Test
func qwenChatTemplateMatchesTheCheckpointSidecarPlainPath() {
    #expect(QwenChatTemplate.userTurnDisablingThinking("QUESTION") == expectedRenderedTurn)
}

@Test
func qwenChatTemplateEmitsNoDefaultSystemBlock() {
    // The sidecar renders a system block ONLY when the caller supplies a
    // system message; inventing one here would change what the model is asked.
    let rendered = QwenChatTemplate.userTurnDisablingThinking("Q")
    #expect(!rendered.contains("<|im_start|>system"))
    #expect(rendered.hasPrefix("<|im_start|>user\n"))
}

@Test
func qwenChatTemplatePreClosesTheThinkingBlock() {
    // Thinking left OPEN is what made the model reason past any affordable
    // budget (0/9 committed within 512 tokens, measured 2026-08-13).
    let rendered = QwenChatTemplate.userTurnDisablingThinking("Q")
    #expect(rendered.contains("<think>\n\n</think>"))
    #expect(rendered.hasSuffix("<think>\n\n</think>\n\n"))
}

@Test
func qwenChatTemplateEmbedsThePromptVerbatim() {
    let prompt = "Line one\nLine two with <angle> chars & symbols"
    #expect(QwenChatTemplate.userTurnDisablingThinking(prompt).contains(prompt))
}

@Test
func endOfTurnTruncationCutsTheFinishedTurnTail() {
    // The behavior loop keeps generating after the turn ends; everything at or
    // after the eos is a finished-turn tail and must not reach the judge.
    let tokens = [11, 12, 13, 99, 44, 44, 44]
    #expect(QwenChatTemplate.truncatedAtFirstEndOfTurn(tokens, eosTokenId: 99) == [11, 12, 13])
}

@Test
func endOfTurnTruncationCutsAtTheFirstEndOfTurnOnly() {
    let tokens = [7, 99, 8, 99, 9]
    #expect(QwenChatTemplate.truncatedAtFirstEndOfTurn(tokens, eosTokenId: 99) == [7])
}

@Test
func endOfTurnTruncationIsAnIdentityWhenTheTurnNeverEnds() {
    let tokens = [1, 2, 3]
    #expect(QwenChatTemplate.truncatedAtFirstEndOfTurn(tokens, eosTokenId: 99) == tokens)
    // An unknown eos id must not silently empty the capture.
    #expect(QwenChatTemplate.truncatedAtFirstEndOfTurn(tokens, eosTokenId: nil) == tokens)
}

@Test
func endOfTurnTruncationNeverReturnsAnEmptyCapture() {
    // An immediate end-of-turn is an empty ANSWER, not a capture fault: the
    // callers throw on an empty token list, which would misreport the
    // diagnosis.
    #expect(QwenChatTemplate.truncatedAtFirstEndOfTurn([99, 1, 2], eosTokenId: 99) == [99])
    #expect(!QwenChatTemplate.truncatedAtFirstEndOfTurn([99], eosTokenId: 99).isEmpty)
}
