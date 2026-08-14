// Copyright © 2026 Eigen Labs Inc.

import Foundation
@testable import MLXLMServer
import Testing

// MARK: - Helper

/// Feeds chunks to a StreamingReasoningParser and returns aggregated content + reasoning.
private func collectStreaming(
    format: ReasoningParserFormat,
    chunks: [String]
) -> (content: String, reasoning: String?) {
    var parser = StreamingReasoningParser(format: format)
    var allContent: [String] = []
    var allReasoning: [String] = []

    for chunk in chunks {
        for r in parser.parse(chunk) {
            if !r.content.isEmpty { allContent.append(r.content) }
            if let reasoning = r.reasoningContent, !reasoning.isEmpty {
                allReasoning.append(reasoning)
            }
        }
    }
    for r in parser.finish() {
        if !r.content.isEmpty { allContent.append(r.content) }
        if let reasoning = r.reasoningContent, !reasoning.isEmpty {
            allReasoning.append(reasoning)
        }
    }

    return (
        allContent.joined(),
        allReasoning.isEmpty ? nil : allReasoning.joined()
    )
}

// MARK: - ReasoningParserFormat Enum

struct ReasoningParserFormatTests {
    @Test("gemma4 alias decodes to .gemma4")
    func gemma4Alias() throws {
        let decoded = try JSONDecoder().decode(
            ReasoningParserFormat.self, from: Data("\"gemma4\"".utf8))
        #expect(decoded == .gemma4)
    }

    @Test("gemma_4 alias decodes to .gemma4")
    func gemma4UnderscoreAlias() throws {
        let decoded = try JSONDecoder().decode(
            ReasoningParserFormat.self, from: Data("\"gemma_4\"".utf8))
        #expect(decoded == .gemma4)
    }

    @Test("gemma alias decodes to .gemma4")
    func gemmaAlias() throws {
        let decoded = try JSONDecoder().decode(
            ReasoningParserFormat.self, from: Data("\"gemma\"".utf8))
        #expect(decoded == .gemma4)
    }

    @Test("harmony alias decodes to .harmony")
    func harmonyAlias() throws {
        let decoded = try JSONDecoder().decode(
            ReasoningParserFormat.self, from: Data("\"harmony\"".utf8))
        #expect(decoded == .harmony)
    }

    @Test("gpt_oss alias decodes to .harmony")
    func gptOssAlias() throws {
        let decoded = try JSONDecoder().decode(
            ReasoningParserFormat.self, from: Data("\"gpt_oss\"".utf8))
        #expect(decoded == .harmony)
    }

    @Test("hyphenated gpt-oss normalises to .harmony")
    func gptOssHyphenated() throws {
        let decoded = try JSONDecoder().decode(
            ReasoningParserFormat.self, from: Data("\"gpt-oss\"".utf8))
        #expect(decoded == .harmony)
    }

    @Test("deepseek_r1 alias decodes to .deepseekR1")
    func deepseekAlias() throws {
        let decoded = try JSONDecoder().decode(
            ReasoningParserFormat.self, from: Data("\"deepseek_r1\"".utf8))
        #expect(decoded == .deepseekR1)
    }

    @Test("none alias decodes to .none")
    func noneAlias() throws {
        let decoded = try JSONDecoder().decode(
            ReasoningParserFormat.self, from: Data("\"none\"".utf8))
        #expect(decoded == .none)
    }

    @Test("unsupported alias throws DecodingError")
    func unsupportedAliasThrows() {
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(
                ReasoningParserFormat.self, from: Data("\"banana\"".utf8))
        }
    }
}

// MARK: - Gemma 4 Non-Streaming

struct Gemma4NonStreamingTests {
    let parser = ReasoningParser(format: .gemma4)

    @Test("full thinking block with channel markers")
    func fullThinkingBlock() {
        let input =
            "<|channel>thought\nThe user is asking about weather. Let me think about this carefully.\n<channel|>The weather in Paris is sunny today with a high of 25°C."
        let result = parser.parse(input)
        #expect(
            result.content == "The weather in Paris is sunny today with a high of 25°C.")
        #expect(
            result.reasoningContent
                == "The user is asking about weather. Let me think about this carefully.")
    }

    @Test("trailing <turn|> is stripped from content")
    func trailingTurnMarkerStripped() {
        let input =
            "<|channel>thought\nI need to analyze this.\n<channel|>Here is my answer.<turn|>"
        let result = parser.parse(input)
        #expect(result.content == "Here is my answer.")
        #expect(result.reasoningContent == "I need to analyze this.")
    }

    @Test("trailing <eos> is stripped from content")
    func trailingEosMarkerStripped() {
        let input =
            "<|channel>thought\nThinking...\n<channel|>Answer here.<eos>"
        let result = parser.parse(input)
        #expect(result.content == "Answer here.")
        #expect(result.reasoningContent == "Thinking...")
    }

    @Test("bare thought prefix without channel markers is stripped")
    func bareThoughtPrefixStripped() {
        let input = "thought\nDirect response without channel markers"
        let result = parser.parse(input)
        #expect(result.content == "Direct response without channel markers")
        #expect(result.reasoningContent == nil)
    }

    @Test("clean output with no reasoning markers")
    func cleanOutput() {
        let input = "Clean output with no reasoning"
        let result = parser.parse(input)
        #expect(result.content == "Clean output with no reasoning")
        #expect(result.reasoningContent == nil)
    }

    @Test("unclosed channel block treats everything as reasoning")
    func unclosedChannelBlock() {
        let input = "<|channel>thought\nStill thinking about this..."
        let result = parser.parse(input)
        #expect(result.content == "")
        #expect(result.reasoningContent == "Still thinking about this...")
    }

    @Test("empty reasoning after stripping yields nil")
    func emptyReasoningYieldsNil() {
        let input = "<|channel>thought\n\n<channel|>Just content."
        let result = parser.parse(input)
        #expect(result.content == "Just content.")
        #expect(result.reasoningContent == nil)
    }

    @Test("both trailing markers stripped simultaneously")
    func bothTrailingMarkersStripped() {
        let input =
            "<|channel>thought\nThink\n<channel|>Answer<turn|> more<eos>"
        let result = parser.parse(input)
        #expect(result.content == "Answer more")
        #expect(result.reasoningContent == "Think")
    }
}

// MARK: - Gemma 4 Streaming

struct Gemma4StreamingTests {
    @Test("basic chunked delivery produces correct reasoning and content")
    func basicChunkedDelivery() {
        let chunks = [
            "<|chan", "nel>tho", "ught\nI am thi", "nking about this.<ch",
            "annel|>Here is", " my answer.",
        ]
        let result = collectStreaming(format: .gemma4, chunks: chunks)
        #expect(result.content == "Here is my answer.")
        #expect(result.reasoning == "I am thinking about this.")
    }

    @Test("open tag split across chunk boundary")
    func openTagSplitAcrossChunks() {
        let chunks = [
            "<|", "channel>thought\nReasoning here.<channel|>Content here.",
        ]
        let result = collectStreaming(format: .gemma4, chunks: chunks)
        #expect(result.content == "Content here.")
        #expect(result.reasoning == "Reasoning here.")
    }

    @Test("thought prefix split across chunks")
    func thoughtPrefixSplitAcrossChunks() {
        let chunks = [
            "<|channel>thou",
            "ght\nActual reasoning.<channel|>Final content.",
        ]
        let result = collectStreaming(format: .gemma4, chunks: chunks)
        #expect(result.content == "Final content.")
        #expect(result.reasoning == "Actual reasoning.")
    }

    @Test("close tag split across chunks")
    func closeTagSplitAcrossChunks() {
        let chunks = [
            "<|channel>thought\nDeep thinking.<chan",
            "nel|>The answer is 42.",
        ]
        let result = collectStreaming(format: .gemma4, chunks: chunks)
        #expect(result.content == "The answer is 42.")
        #expect(result.reasoning == "Deep thinking.")
    }

    @Test("no channel markers emits content only")
    func noChannelMarkers() {
        let chunks = ["Hello, ", "world!"]
        let result = collectStreaming(format: .gemma4, chunks: chunks)
        #expect(result.content == "Hello, world!")
        #expect(result.reasoning == nil)
    }

    // Fixed: consumeSafePrefix now checks up to marker.count (not
    // marker.count - 1), so a complete trailing marker at the end of a
    // non-final chunk is held back and stripped on finalization.
    @Test("trailing <turn|> is stripped even when arriving in a separate non-final chunk")
    func trailingTurnMarkerStripped() {
        let chunks = [
            "<|channel>thought\nThinking<channel|>Done.", "<turn|>",
        ]
        let result = collectStreaming(format: .gemma4, chunks: chunks)
        #expect(result.content == "Done.")
        #expect(result.reasoning == "Thinking")
    }

    @Test("single chunk with full output")
    func singleChunkFull() {
        let chunks = [
            "<|channel>thought\nShort thought<channel|>Short answer."
        ]
        let result = collectStreaming(format: .gemma4, chunks: chunks)
        #expect(result.content == "Short answer.")
        #expect(result.reasoning == "Short thought")
    }

    @Test("character-at-a-time delivery")
    func characterAtATime() {
        let full =
            "<|channel>thought\nHello<channel|>World"
        let chunks = full.map { String($0) }
        let result = collectStreaming(format: .gemma4, chunks: chunks)
        #expect(result.content == "World")
        #expect(result.reasoning == "Hello")
    }
}

// MARK: - Harmony (GPT-OSS) Non-Streaming

struct HarmonyNonStreamingTests {
    let parser = ReasoningParser(format: .harmony)

    @Test("analysis and final channels parsed correctly")
    func analysisAndFinalChannels() {
        let input =
            "<|channel|>analysis<|message|>User says yoooo. Likely wants a casual response.<|end|><|start|>assistant<|channel|>final<|message|>Hey there! How can I help you today?"
        let result = parser.parse(input)
        #expect(result.content == "Hey there! How can I help you today?")
        #expect(
            result.reasoningContent
                == "User says yoooo. Likely wants a casual response.")
    }

    @Test("no harmony markers returns content as-is")
    func noHarmonyMarkers() {
        let input = "Just a plain response"
        let result = parser.parse(input)
        #expect(result.content == "Just a plain response")
        #expect(result.reasoningContent == nil)
    }

    @Test("role markers are stripped from final channel content")
    func roleMarkersStripped() {
        let input =
            "<|channel|>final<|message|><|start|>assistant Here is my response"
        let result = parser.parse(input)
        #expect(result.content == "Here is my response")
        #expect(result.reasoningContent == nil)
    }

    @Test("multiple analysis blocks are joined with newline")
    func multipleAnalysisBlocksJoined() {
        let input =
            "<|channel|>analysis<|message|>First thought.<|end|><|channel|>analysis<|message|>Second thought.<|end|><|channel|>final<|message|>Final answer."
        let result = parser.parse(input)
        #expect(result.content == "Final answer.")
        #expect(result.reasoningContent == "First thought.\nSecond thought.")
    }

    @Test("<|return|> works as end marker")
    func returnEndMarker() {
        let input =
            "<|channel|>analysis<|message|>Thought.<|return|><|channel|>final<|message|>Done."
        let result = parser.parse(input)
        #expect(result.content == "Done.")
        #expect(result.reasoningContent == "Thought.")
    }

    @Test("analysis-only output has empty content")
    func analysisOnlyOutput() {
        let input =
            "<|channel|>analysis<|message|>Just reasoning, no final."
        let result = parser.parse(input)
        #expect(result.content == "")
        #expect(result.reasoningContent == "Just reasoning, no final.")
    }
}

// MARK: - Harmony Streaming

struct HarmonyStreamingTests {
    @Test("real model output parsed correctly in chunks")
    func realModelOutputParsed() {
        let chunks = [
            "<|channel|>", "analysis", "<|message|>",
            "User says \"yoooo\".",
            " Likely wants a casual response. We should respond friendly.",
            "<|end|>", "<|start|>assistant", "<|channel|>", "final",
            "<|message|>",
            "Hey there!", " How can I help you today?",
        ]
        let result = collectStreaming(format: .harmony, chunks: chunks)
        #expect(result.content == "Hey there! How can I help you today?")
        #expect(
            result.reasoning
                == "User says \"yoooo\". Likely wants a casual response. We should respond friendly."
        )
    }

    @Test("channel marker split across chunks")
    func channelMarkerSplitAcrossChunks() {
        let chunks = [
            "<|chan",
            "nel|>analysis<|message|>Thinking...<|end|><|channel|>final<|message|>Answer.",
        ]
        let result = collectStreaming(format: .harmony, chunks: chunks)
        #expect(result.content == "Answer.")
        #expect(result.reasoning == "Thinking...")
    }

    @Test("message marker split across chunks")
    func messageMarkerSplitAcrossChunks() {
        let chunks = [
            "<|channel|>analysis<|mes",
            "sage|>Analysis content.<|end|><|channel|>final<|message|>Response.",
        ]
        let result = collectStreaming(format: .harmony, chunks: chunks)
        #expect(result.content == "Response.")
        #expect(result.reasoning == "Analysis content.")
    }

    @Test("end marker split across chunks")
    func endMarkerSplitAcrossChunks() {
        let chunks = [
            "<|channel|>analysis<|message|>Reasoning.<|en",
            "d|><|channel|>final<|message|>Content.",
        ]
        let result = collectStreaming(format: .harmony, chunks: chunks)
        #expect(result.content == "Content.")
        #expect(result.reasoning == "Reasoning.")
    }

    @Test("no markers in streaming returns content")
    func noMarkersReturnsContent() {
        let chunks = ["Just ", "plain ", "text."]
        let result = collectStreaming(format: .harmony, chunks: chunks)
        #expect(result.content == "Just plain text.")
        #expect(result.reasoning == nil)
    }

    @Test("character-at-a-time delivery")
    func characterAtATime() {
        let full =
            "<|channel|>analysis<|message|>Think<|end|><|channel|>final<|message|>OK"
        let chunks = full.map { String($0) }
        let result = collectStreaming(format: .harmony, chunks: chunks)
        #expect(result.content == "OK")
        #expect(result.reasoning == "Think")
    }
}

// MARK: - StreamingReasoningParser Dispatcher

struct StreamingReasoningParserDispatcherTests {
    @Test(".none format passes content through unchanged")
    func noneFormatPassthrough() {
        let result = collectStreaming(format: .none, chunks: ["Hello ", "world"])
        #expect(result.content == "Hello world")
        #expect(result.reasoning == nil)
    }

    @Test("empty chunks produce no output")
    func emptyChunksProduceNoOutput() {
        let result = collectStreaming(format: .gemma4, chunks: ["", "", ""])
        #expect(result.content == "")
        #expect(result.reasoning == nil)
    }
}
