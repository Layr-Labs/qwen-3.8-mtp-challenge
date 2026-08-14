// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for Gemma format: call:name{key:value,k:<escape>str<escape>}
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/function_gemma.py
public struct GemmaFunctionParser: ToolCallParser, Sendable {
    public let startTag: String? = "<|tool_call>"
    public let endTag: String? = "<tool_call|>"
    public let alternateStartTags: [String] = ["<start_function_call>"]
    public let alternateEndTags: [String] = ["<end_function_call>"]

    private let quoteMarkers = ["<|\"|>", "<escape>"]
    private let wrapperTags = [
        "<|tool_call>",
        "<tool_call|>",
        "<start_function_call>",
        "<end_function_call>",
    ]

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        // Strip tags if present. Gemma 4 emits the newer `<|tool_call>` form,
        // while older Gemma-family templates use `<start_function_call>`.
        var text = content.trimmingCharacters(in: .whitespacesAndNewlines)
        for tag in wrapperTags {
            text = text.replacingOccurrences(of: tag, with: "")
        }

        // Pattern: call:(\w+)\{(.*?)\}
        // Find "call:" followed by function name and arguments in braces
        guard let callRange = text.range(of: "call:") else { return nil }

        let remaining = String(text[callRange.upperBound...])

        // Extract function name (word characters until {)
        guard let braceStart = remaining.firstIndex(of: "{") else { return nil }
        let funcName = String(remaining[..<braceStart])

        guard !funcName.isEmpty else { return nil }

        // Extract arguments string (everything between { and })
        guard let braceEnd = remaining.lastIndex(of: "}") else { return nil }
        let argsStr = String(remaining[remaining.index(after: braceStart) ..< braceEnd])
        let arguments = parseArguments(argsStr)

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    private func parseArguments(_ text: String) -> [String: any Sendable] {
        var arguments: [String: any Sendable] = [:]
        for pair in splitTopLevel(text, separator: ",") {
            guard let colon = pair.firstIndex(of: ":") else { continue }
            let key = pair[..<colon].trimmingCharacters(in: .whitespacesAndNewlines)
            let rawValue = pair[pair.index(after: colon)...].trimmingCharacters(in: .whitespacesAndNewlines)
            guard !key.isEmpty else { continue }
            arguments[key] = parseValue(rawValue)
        }
        return arguments
    }

    private func parseValue(_ rawValue: String) -> any Sendable {
        for marker in quoteMarkers where rawValue.hasPrefix(marker) && rawValue.hasSuffix(marker) {
            let start = rawValue.index(rawValue.startIndex, offsetBy: marker.count)
            let end = rawValue.index(rawValue.endIndex, offsetBy: -marker.count)
            return String(rawValue[start..<end])
        }
        if let data = rawValue.data(using: .utf8),
           let json = deserializeJSON(data)
        {
            return json
        }
        return rawValue
    }

    private func splitTopLevel(_ text: String, separator: Character) -> [String] {
        var result: [String] = []
        var start = text.startIndex
        var index = text.startIndex
        var depth = 0

        while index < text.endIndex {
            if let marker = quoteMarkers.first(where: { text[index...].hasPrefix($0) }) {
                let afterStart = text.index(index, offsetBy: marker.count)
                if let end = text.range(of: marker, range: afterStart..<text.endIndex) {
                    index = end.upperBound
                    continue
                }
            }

            switch text[index] {
            case "{", "[":
                depth += 1
            case "}", "]":
                depth = max(0, depth - 1)
            case separator where depth == 0:
                result.append(String(text[start..<index]))
                start = text.index(after: index)
            default:
                break
            }
            index = text.index(after: index)
        }

        result.append(String(text[start..<text.endIndex]))
        return result
    }
}
