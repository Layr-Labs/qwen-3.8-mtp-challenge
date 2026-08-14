// Copyright © 2026 Eigen Labs Inc.

struct StreamingHarmonyReasoningParser: Sendable {
    private enum State: Sendable {
        case outside
        case channel
        case message
    }

    private let channelMarker = "<|channel|>"
    private let messageMarker = "<|message|>"
    private let roleMarker = "<|start|>assistant"
    private let endMarkers = ["<|end|>", "<|return|>", "<|call|>"]

    private var buffer = ""
    private var channel = ""
    private var state: State = .outside

    private var outsideControlMarkers: [String] {
        [channelMarker, roleMarker]
    }

    private var messageControlMarkers: [String] {
        endMarkers + [roleMarker]
    }

    mutating func parse(_ chunk: String) -> [ParsedReasoning] {
        buffer += chunk
        return drain(final: false)
    }

    mutating func finish() -> [ParsedReasoning] {
        drain(final: true)
    }

    private mutating func drain(final: Bool) -> [ParsedReasoning] {
        var output: [ParsedReasoning] = []
        var shouldContinue = true

        while shouldContinue {
            switch state {
            case .outside:
                drainOutside(final: final, output: &output, shouldContinue: &shouldContinue)
            case .channel:
                drainChannel(final: final, shouldContinue: &shouldContinue)
            case .message:
                drainMessage(final: final, output: &output, shouldContinue: &shouldContinue)
            }
        }

        return output
    }

    private mutating func drainOutside(
        final: Bool,
        output: inout [ParsedReasoning],
        shouldContinue: inout Bool
    ) {
        if let marker = firstRange(
            ofAny: outsideControlMarkers,
            in: buffer,
            range: buffer.startIndex..<buffer.endIndex
        ) {
            appendContent(String(buffer[..<marker.lowerBound]), to: &output)
            let matchedMarker = String(buffer[marker])
            buffer.removeSubrange(buffer.startIndex..<marker.upperBound)
            if matchedMarker == channelMarker {
                channel.removeAll(keepingCapacity: true)
                state = .channel
            }
            return
        }

        let content = consumeSafePrefix(
            from: &buffer,
            preservingPotentialPrefixOfAny: outsideControlMarkers,
            final: final
        )
        appendContent(content, to: &output)
        shouldContinue = false
    }

    private mutating func drainChannel(final: Bool, shouldContinue: inout Bool) {
        if let messageStart = buffer.range(of: messageMarker) {
            channel += String(buffer[..<messageStart.lowerBound])
            buffer.removeSubrange(buffer.startIndex..<messageStart.upperBound)
            state = .message
            return
        }

        channel += consumeSafePrefix(
            from: &buffer,
            preservingPotentialPrefixOf: messageMarker,
            final: final
        )
        if final {
            channel.removeAll(keepingCapacity: true)
            state = .outside
        }
        shouldContinue = false
    }

    private mutating func drainMessage(
        final: Bool,
        output: inout [ParsedReasoning],
        shouldContinue: inout Bool
    ) {
        if let marker = firstRange(
            ofAny: messageControlMarkers,
            in: buffer,
            range: buffer.startIndex..<buffer.endIndex
        ) {
            appendMessage(String(buffer[..<marker.lowerBound]), to: &output)
            let matchedMarker = String(buffer[marker])
            buffer.removeSubrange(buffer.startIndex..<marker.upperBound)
            if matchedMarker != roleMarker {
                channel.removeAll(keepingCapacity: true)
                state = .outside
            }
            return
        }

        let content = consumeSafePrefix(
            from: &buffer,
            preservingPotentialPrefixOfAny: messageControlMarkers,
            final: final
        )
        appendMessage(content, to: &output)
        if final {
            channel.removeAll(keepingCapacity: true)
            state = .outside
        }
        shouldContinue = false
    }

    private func appendMessage(_ text: String, to output: inout [ParsedReasoning]) {
        switch channel.trimmedForReasoning {
        case "analysis":
            appendReasoning(text.removingHarmonyRoleMarkers, to: &output)
        case "final":
            appendContent(text.removingHarmonyRoleMarkers, to: &output)
        default:
            break
        }
    }
}
