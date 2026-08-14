// Copyright © 2026 Eigen Labs Inc.

func firstRange(
    ofAny markers: [String],
    in text: String,
    range: Range<String.Index>
) -> Range<String.Index>? {
    var first: Range<String.Index>?
    for marker in markers {
        guard let candidate = text.range(of: marker, range: range) else { continue }
        if first == nil || candidate.lowerBound < first!.lowerBound {
            first = candidate
        }
    }
    return first
}

func consumeSafePrefix(
    from buffer: inout String,
    preservingPotentialPrefixOf marker: String,
    final: Bool
) -> String {
    if final {
        let output = buffer
        buffer.removeAll(keepingCapacity: true)
        return output
    }

    let suffixLength = potentialMarkerPrefixSuffixLength(in: buffer, marker: marker)
    guard suffixLength > 0 else {
        let output = buffer
        buffer.removeAll(keepingCapacity: true)
        return output
    }

    let suffixStart = buffer.index(buffer.endIndex, offsetBy: -suffixLength)
    let output = String(buffer[..<suffixStart])
    buffer = String(buffer[suffixStart...])
    return output
}

func consumeSafePrefix(
    from buffer: inout String,
    preservingPotentialPrefixOfAny markers: [String],
    final: Bool
) -> String {
    if final {
        let output = buffer
        buffer.removeAll(keepingCapacity: true)
        return output
    }

    let suffixLength = markers
        .map { potentialMarkerPrefixSuffixLength(in: buffer, marker: $0) }
        .max() ?? 0
    guard suffixLength > 0 else {
        let output = buffer
        buffer.removeAll(keepingCapacity: true)
        return output
    }

    let suffixStart = buffer.index(buffer.endIndex, offsetBy: -suffixLength)
    let output = String(buffer[..<suffixStart])
    buffer = String(buffer[suffixStart...])
    return output
}

func potentialMarkerPrefixSuffixLength(in text: String, marker: String) -> Int {
    // Use marker.count (not marker.count - 1) so that a COMPLETE marker
    // at the end of the buffer is also held back. This is critical for
    // drainContent in the Gemma4 streaming parser where trailing markers
    // (<turn|>, <eos>) must not leak through non-final chunks.
    let maxLength = min(text.count, marker.count)
    guard maxLength > 0 else { return 0 }

    for length in stride(from: maxLength, through: 1, by: -1) {
        let suffix = text.suffix(length)
        if marker.hasPrefix(suffix) {
            return length
        }
    }
    return 0
}

func appendContent(_ content: String, to output: inout [ParsedReasoning]) {
    guard !content.isEmpty else { return }
    output.append(.init(content: content, reasoningContent: nil))
}

func appendReasoning(_ reasoning: String, to output: inout [ParsedReasoning]) {
    guard !reasoning.isEmpty else { return }
    output.append(.init(content: "", reasoningContent: reasoning))
}

extension String {
    var trimmedForReasoning: String {
        trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var removingHarmonyRoleMarkers: String {
        replacingOccurrences(of: "<|start|>assistant", with: "")
    }
}
