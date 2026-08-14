// Copyright © 2026 Eigen Labs Inc.

public struct StreamingReasoningParser: Sendable {
    public var format: ReasoningParserFormat

    private var thinkParser = StreamingThinkReasoningParser()
    private var harmonyParser = StreamingHarmonyReasoningParser()
    private var gemma4Parser = StreamingGemma4ReasoningParser()

    public init(format: ReasoningParserFormat) {
        self.format = format
    }

    public mutating func parse(_ chunk: String) -> [ParsedReasoning] {
        guard !chunk.isEmpty else { return [] }
        switch format {
        case .none:
            return [.init(content: chunk, reasoningContent: nil)]
        case .deepseekR1, .qwen3:
            return thinkParser.parse(chunk)
        case .harmony:
            return harmonyParser.parse(chunk)
        case .gemma4:
            return gemma4Parser.parse(chunk)
        }
    }

    public mutating func finish() -> [ParsedReasoning] {
        switch format {
        case .none:
            return []
        case .deepseekR1, .qwen3:
            return thinkParser.finish()
        case .harmony:
            return harmonyParser.finish()
        case .gemma4:
            return gemma4Parser.finish()
        }
    }
}
