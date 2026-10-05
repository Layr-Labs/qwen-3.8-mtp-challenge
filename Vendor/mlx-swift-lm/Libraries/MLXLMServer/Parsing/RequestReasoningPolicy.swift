// Copyright © 2026 Eigen Labs Inc.

/// The Qwen launch recipe knows whether its template has already opened the
/// thinking block. Generic parser callers keep their existing tag detection.
struct RequestReasoningPolicy: Sendable {
    let format: ReasoningParserFormat
    let startsInReasoning: Bool

    init(
        format: ReasoningParserFormat,
        enableThinking: Bool?,
        qwenPromptStartsInReasoning: Bool = false
    ) {
        self.format = format == .qwen3 && enableThinking == false ? .none : format
        startsInReasoning = self.format == .qwen3 && qwenPromptStartsInReasoning
    }

    func makeStreamingParser() -> StreamingReasoningParser {
        .init(format: format, startsInReasoning: startsInReasoning)
    }

    func parse(_ text: String) -> ParsedReasoning {
        guard startsInReasoning else { return ReasoningParser(format: format).parse(text) }
        var parser = makeStreamingParser()
        let events = parser.parse(text) + parser.finish()
        let content = events.map(\.content).joined().trimmedForReasoning
        let reasoning = events.compactMap(\.reasoningContent).joined().trimmedForReasoning
        return .init(content: content, reasoningContent: reasoning.isEmpty ? nil : reasoning)
    }
}
