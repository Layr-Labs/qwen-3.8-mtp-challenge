// Copyright © 2026 Eigen Labs Inc.

struct StreamingThinkReasoningParser: Sendable {
    private enum State: Sendable {
        case undecided
        case reasoningStart
        case reasoning
        case content
    }

    private var state: State = .undecided
    private var buffer = ""

    init(startsInReasoning: Bool = false) {
        state = startsInReasoning ? .reasoningStart : .undecided
    }

    mutating func parse(_ chunk: String) -> [ParsedReasoning] {
        buffer += chunk
        return drain(final: false)
    }

    mutating func finish() -> [ParsedReasoning] {
        drain(final: true)
    }

    private mutating func drain(final: Bool) -> [ParsedReasoning] {
        let opening = "<think>"
        let closing = "</think>"
        var output: [ParsedReasoning] = []
        var shouldContinue = true

        while shouldContinue {
            switch state {
            case .reasoningStart:
                // The pinned Qwen template already opened the thinking block.
                // Some model outputs repeat that opening tag; hold only a
                // possible initial tag prefix, rather than buffering reasoning.
                let prefix = buffer.prefix { $0.isWhitespace }
                let remaining = buffer.dropFirst(prefix.count)
                if remaining.hasPrefix(opening) {
                    buffer.removeFirst(prefix.count + opening.count)
                    state = .reasoning
                } else if !final && (remaining.isEmpty || opening.hasPrefix(remaining)) {
                    shouldContinue = false
                } else {
                    state = .reasoning
                }
            case .undecided:
                if let open = buffer.range(of: opening) {
                    appendContent(String(buffer[..<open.lowerBound]), to: &output)
                    buffer.removeSubrange(buffer.startIndex..<open.upperBound)
                    state = .reasoning
                } else if let close = buffer.range(of: closing) {
                    appendReasoning(String(buffer[..<close.lowerBound]), to: &output)
                    buffer.removeSubrange(buffer.startIndex..<close.upperBound)
                    state = .content
                } else if final {
                    appendContent(buffer, to: &output)
                    buffer.removeAll(keepingCapacity: true)
                    state = .content
                } else {
                    shouldContinue = false
                }
            case .reasoning:
                if let close = buffer.range(of: closing) {
                    appendReasoning(String(buffer[..<close.lowerBound]), to: &output)
                    buffer.removeSubrange(buffer.startIndex..<close.upperBound)
                    state = .content
                } else {
                    let reasoning = consumeSafePrefix(
                        from: &buffer,
                        preservingPotentialPrefixOf: closing,
                        final: final
                    )
                    appendReasoning(reasoning, to: &output)
                    shouldContinue = false
                }
            case .content:
                if let open = buffer.range(of: opening) {
                    appendContent(String(buffer[..<open.lowerBound]), to: &output)
                    buffer.removeSubrange(buffer.startIndex..<open.upperBound)
                    state = .reasoning
                } else {
                    let content = consumeSafePrefix(
                        from: &buffer,
                        preservingPotentialPrefixOf: opening,
                        final: final
                    )
                    appendContent(content, to: &output)
                    shouldContinue = false
                }
            }
        }

        return output
    }
}
