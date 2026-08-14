// Copyright © 2026 Eigen Labs Inc.

import Foundation

public enum ReasoningParserFormat: String, Codable, Sendable, CaseIterable {
    case none
    case deepseekR1 = "deepseek_r1"
    case qwen3
    case harmony
    case gemma4

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let raw = try container.decode(String.self)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")

        switch raw {
        case "none", "off", "disabled":
            self = .none
        case "deepseek_r1", "deepseek", "r1", "think":
            self = .deepseekR1
        case "qwen3", "qwen":
            self = .qwen3
        case "harmony", "gpt_oss", "openai_harmony":
            self = .harmony
        case "gemma4", "gemma_4", "gemma":
            self = .gemma4
        default:
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported reasoning parser '\(raw)'"
            )
        }
    }
}

public struct ParsedReasoning: Sendable, Equatable {
    public var content: String
    public var reasoningContent: String?
}

public struct ReasoningParser: Sendable {
    public var format: ReasoningParserFormat

    public init(format: ReasoningParserFormat) {
        self.format = format
    }

    public func parse(_ text: String) -> ParsedReasoning {
        switch format {
        case .none:
            return .init(content: text, reasoningContent: nil)
        case .deepseekR1, .qwen3:
            return parseThinkTags(text)
        case .harmony:
            return parseHarmony(text)
        case .gemma4:
            return parseGemma4(text)
        }
    }

    private func parseThinkTags(_ text: String) -> ParsedReasoning {
        var remaining = text
        var reasoning: [String] = []

        while let start = remaining.range(of: "<think>"),
            let end = remaining.range(of: "</think>", range: start.upperBound..<remaining.endIndex)
        {
            reasoning.append(String(remaining[start.upperBound..<end.lowerBound]))
            remaining.removeSubrange(start.lowerBound..<end.upperBound)
        }

        if reasoning.isEmpty, let end = remaining.range(of: "</think>") {
            reasoning.append(String(remaining[..<end.lowerBound]))
            remaining.removeSubrange(remaining.startIndex..<end.upperBound)
        }

        let reasoningText = reasoning.joined(separator: "\n").trimmedForReasoning
        return .init(
            content: remaining.trimmedForReasoning,
            reasoningContent: reasoningText.isEmpty ? nil : reasoningText
        )
    }

    private func parseGemma4(_ text: String) -> ParsedReasoning {
        let channelOpen = "<|channel>"
        let channelClose = "<channel|>"
        let thoughtPrefix = "thought\n"
        let trailingMarkers = ["<turn|>", "<eos>"]

        guard let closeRange = text.range(of: channelClose) else {
            // No close marker -- check for an unclosed thinking block
            var remaining = text
            if let openRange = remaining.range(of: channelOpen) {
                var thinking = String(remaining[openRange.upperBound...])
                if thinking.hasPrefix(thoughtPrefix) {
                    thinking = String(thinking.dropFirst(thoughtPrefix.count))
                }
                let r = thinking.trimmedForReasoning
                return .init(content: "", reasoningContent: r.isEmpty ? nil : r)
            }
            // Strip bare thought\n prefix
            if remaining.hasPrefix(thoughtPrefix) {
                remaining = String(remaining.dropFirst(thoughtPrefix.count))
            }
            return .init(content: remaining.trimmedForReasoning, reasoningContent: nil)
        }

        var thinkingBlock = String(text[..<closeRange.lowerBound])
        var answer = String(text[closeRange.upperBound...])

        if let openRange = thinkingBlock.range(of: channelOpen) {
            thinkingBlock = String(thinkingBlock[openRange.upperBound...])
        }
        if thinkingBlock.hasPrefix(thoughtPrefix) {
            thinkingBlock = String(thinkingBlock.dropFirst(thoughtPrefix.count))
        }

        for marker in trailingMarkers {
            answer = answer.replacingOccurrences(of: marker, with: "")
        }

        let r = thinkingBlock.trimmedForReasoning
        return .init(
            content: answer.trimmedForReasoning,
            reasoningContent: r.isEmpty ? nil : r
        )
    }

    private func parseHarmony(_ text: String) -> ParsedReasoning {
        let channelMarker = "<|channel|>"
        let messageMarker = "<|message|>"
        let endMarkers = ["<|end|>", "<|return|>", "<|call|>"]
        var final: [String] = []
        var analysis: [String] = []
        var cursor = text.startIndex

        while let channelStart = text.range(of: channelMarker, range: cursor..<text.endIndex) {
            let channelNameStart = channelStart.upperBound
            guard let messageStart = text.range(
                of: messageMarker,
                range: channelNameStart..<text.endIndex
            ) else { break }

            let channel = String(text[channelNameStart..<messageStart.lowerBound])
            let contentStart = messageStart.upperBound
            let contentEnd = firstRange(ofAny: endMarkers, in: text, range: contentStart..<text.endIndex)
            let messageEnd = contentEnd?.lowerBound ?? text.endIndex
            let content = String(text[contentStart..<messageEnd]).removingHarmonyRoleMarkers

            switch channel {
            case "analysis":
                analysis.append(content)
            case "final":
                final.append(content)
            default:
                break
            }

            cursor = contentEnd?.upperBound ?? text.endIndex
        }

        if final.isEmpty && analysis.isEmpty {
            return .init(content: text, reasoningContent: nil)
        }

        let reasoningText = analysis.joined(separator: "\n").trimmedForReasoning
        return .init(
            content: final.joined(separator: "\n").trimmedForReasoning,
            reasoningContent: reasoningText.isEmpty ? nil : reasoningText
        )
    }
}
