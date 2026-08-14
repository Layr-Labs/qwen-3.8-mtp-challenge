// Copyright © 2026 Eigen Labs Inc.

import Foundation

/// Parser for OpenAI Harmony tool calls emitted by GPT-OSS-style models.
///
/// Example:
/// `<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{"city":"Paris"}<|call|>`
public struct HarmonyToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<|start|>assistant to=functions."
    public let endTag: String? = "<|call|>"
    public let alternateStartTags: [String] = [
        "<|start|>assistant<|channel|>commentary to=functions."
    ]
    public let alternateEndTags: [String] = [
        "<|end|>",
        "<|return|>",
    ]

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var text = content

        if let startRange = firstRange(ofAny: acceptedStartTags, in: text) {
            text = String(text[startRange.upperBound...])
        } else if let functionsRange = text.range(of: "to=functions.") ?? text.range(of: "functions.") {
            text = String(text[functionsRange.upperBound...])
        }

        guard let nameEnd = firstRange(
            ofAny: ["<|channel|>commentary", "<|constrain|>", "<|message|>", " "],
            in: text
        ) else {
            return nil
        }

        let rawName = text[..<nameEnd.lowerBound]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !rawName.isEmpty else { return nil }

        guard let messageRange = text.range(of: "<|message|>") else {
            return nil
        }

        var argumentsText = String(text[messageRange.upperBound...])
        for marker in acceptedEndTags {
            if let markerRange = argumentsText.range(of: marker) {
                argumentsText = String(argumentsText[..<markerRange.lowerBound])
            }
        }

        let json = argumentsText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = json.data(using: .utf8),
            let arguments = try? JSONDecoder().decode([String: JSONValue].self, from: data)
        else {
            return nil
        }

        return ToolCall(function: .init(name: rawName, arguments: arguments))
    }

    private func firstRange(ofAny markers: [String], in text: String) -> Range<String.Index>? {
        var first: Range<String.Index>?
        for marker in markers {
            guard let candidate = text.range(of: marker) else { continue }
            if first == nil || candidate.lowerBound < first!.lowerBound {
                first = candidate
            }
        }
        return first
    }
}
