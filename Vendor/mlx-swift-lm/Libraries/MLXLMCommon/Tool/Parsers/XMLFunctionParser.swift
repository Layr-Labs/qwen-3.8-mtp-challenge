// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for XML function format: <function=name><parameter=key>value</parameter></function>
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/qwen3_coder.py
public struct XMLFunctionParser: ToolCallParser, Sendable {
    public let startTag: String?
    public let endTag: String?

    public init(startTag: String, endTag: String) {
        self.startTag = startTag
        self.endTag = endTag
    }

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        // Pattern: <function=(content)</function> — [\s\S] matches newlines
        guard
            let funcMatch = content.range(
                of: #"<function=([\s\S]*?)</function>"#, options: .regularExpression)
        else { return nil }

        let funcContent = String(content[funcMatch])

        // Extract function name (everything between <function= and first >)
        guard let nameStart = funcContent.range(of: "<function="),
            let nameEnd = funcContent.range(
                of: ">", range: nameStart.upperBound ..< funcContent.endIndex)
        else { return nil }

        let funcName = String(funcContent[nameStart.upperBound ..< nameEnd.lowerBound])
        let paramSection = String(funcContent[nameEnd.upperBound...])

        var arguments: [String: any Sendable] = [:]

        // Find all parameter tags
        var searchRange = paramSection.startIndex ..< paramSection.endIndex
        while let paramStart = paramSection.range(of: "<parameter=", range: searchRange) {
            // Find the parameter name (between = and >)
            guard
                let nameEnd = paramSection.range(
                    of: ">", range: paramStart.upperBound ..< paramSection.endIndex)
            else { break }

            let paramName = String(paramSection[paramStart.upperBound ..< nameEnd.lowerBound])

            // Close at the </parameter> that is followed by the next parameter
            // or </function>. An earlier copy inside the value stays in the text.
            guard
                let paramEnd = structuralParameterClose(
                    in: paramSection, from: nameEnd.upperBound)
            else { break }

            var paramValue = String(paramSection[nameEnd.upperBound ..< paramEnd.lowerBound])

            // Trim leading/trailing newlines (matching Python behavior)
            if paramValue.hasPrefix("\n") {
                paramValue = String(paramValue.dropFirst())
            }
            if paramValue.hasSuffix("\n") {
                paramValue = String(paramValue.dropLast())
            }

            // Convert value based on schema type
            arguments[paramName] = convertParameterValue(
                paramValue, paramName: paramName, funcName: funcName, tools: tools)

            searchRange = paramEnd.upperBound ..< paramSection.endIndex
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    /// A parameter closer is structural when the next tag is a sibling parameter
    /// or the function end. A `</parameter>` buried in a file body is not.
    private func structuralParameterClose(
        in text: String, from start: String.Index
    ) -> Range<String.Index>? {
        let marker = "</parameter>"
        var search = start ..< text.endIndex
        while let candidate = text.range(of: marker, range: search) {
            let rest = text[candidate.upperBound...].drop(while: { $0.isWhitespace })
            if rest.isEmpty || rest.hasPrefix("<parameter=") || rest.hasPrefix("</function>") {
                return candidate
            }
            search = candidate.upperBound ..< text.endIndex
        }
        return nil
    }
}
