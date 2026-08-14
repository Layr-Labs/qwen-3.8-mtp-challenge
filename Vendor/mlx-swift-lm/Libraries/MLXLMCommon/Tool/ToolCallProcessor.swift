// Copyright © 2025 Apple Inc.

import Foundation

/// Processes generated text to detect and extract tool calls during streaming generation.
///
/// `ToolCallProcessor` handles the streaming detection of tool calls in model output,
/// buffering partial content and extracting complete tool calls when detected.
///
/// Example:
/// ```swift
/// let processor = ToolCallProcessor(format: .lfm2)
/// for chunk in generatedChunks {
///     if let text = processor.processChunk(chunk) {
///         // Regular text to display
///         print(text)
///     }
/// }
/// // After generation completes:
/// for toolCall in processor.toolCalls {
///     // Handle extracted tool calls
///     print(toolCall.function.name)
/// }
/// ```
public class ToolCallProcessor {

    // MARK: - Properties

    private let parser: any ToolCallParser
    private let tools: [[String: any Sendable]]?
    private var state = State.normal
    private var toolCallBuffer = ""

    /// The tool calls extracted during processing.
    public var toolCalls: [ToolCall] = []

    // MARK: - State Enum

    private enum State {
        case normal
        case potentialToolCall
        case collectingToolCall
    }

    // MARK: - Initialization

    /// Initialize with a specific tool call format.
    /// - Parameters:
    ///   - format: The tool call format to use (defaults to `.json` for standard JSON format)
    ///   - tools: Optional tool schemas for type-aware parsing
    public init(format: ToolCallFormat = .json, tools: [[String: any Sendable]]? = nil) {
        self.parser = format.createParser()
        self.tools = tools
    }

    // MARK: - Computed Properties

    /// Whether this processor uses inline format (no start tag).
    private var isInlineFormat: Bool {
        parser.acceptedStartTags.isEmpty
    }

    /// The first characters of accepted start tags for quick detection.
    private var startTagFirstChars: [Character] {
        Array(Set(parser.acceptedStartTags.compactMap(\.first)))
    }

    // MARK: - Public Methods

    /// Process a generated text chunk and extract any tool call content.
    /// - Parameter chunk: The text chunk to process
    /// - Returns: Regular text that should be displayed (non-tool call content), or `nil` if buffering
    public func processChunk(_ chunk: String) -> String? {
        if isInlineFormat {
            return processInlineChunk(chunk)
        }
        return processTaggedChunk(chunk)
    }

    /// Drain all extracted tool calls in parse (FIFO) order, clearing the queue.
    ///
    /// Prefer this over reading/`popLast()`-ing `toolCalls` directly: it emits
    /// every complete tool call from a multi-call turn exactly once and in
    /// order. Upstream 1335fb5.
    public func drainToolCalls() -> [ToolCall] {
        let drained = toolCalls
        toolCalls = []
        return drained
    }

    /// Process end-of-sequence, parsing any buffered content as tool call(s).
    ///
    /// Call this when generation ends (e.g., on EOS token) to handle formats
    /// whose end tag is never delivered as text (e.g., Mistral where `</s>`
    /// is intercepted at the token ID level).
    ///
    /// For formats with end tags that appear in the text stream, the buffer
    /// will already be empty at generation end, making this a no-op.
    ///
    /// - Parameter returnBufferedText: when `true`, if the residual buffer did
    ///   not parse into any tool call it is returned as ordinary text instead of
    ///   being discarded (it was withheld only while probing for a tool call).
    /// - Returns: residual non-tool text when `returnBufferedText` is `true`,
    ///   otherwise `nil`.
    @discardableResult
    public func processEOS(returnBufferedText: Bool = false) -> String? {
        guard state == .collectingToolCall || state == .potentialToolCall else { return nil }
        guard !toolCallBuffer.isEmpty else {
            state = .normal
            return nil
        }

        let parsed = parser.parseEOS(toolCallBuffer, tools: tools)
        toolCalls.append(contentsOf: parsed)

        let buffered = toolCallBuffer
        toolCallBuffer = ""
        state = .normal

        if returnBufferedText && parsed.isEmpty {
            return buffered
        }
        return nil
    }

    // MARK: - Private Methods

    /// Process chunk for inline formats (no wrapper tags).
    ///
    /// Uses brace counting to detect when output looks like a JSON tool call.
    /// While braces are unbalanced the content is buffered (returns `nil`)
    /// so partial JSON is never leaked to the UI.
    private func processInlineChunk(_ chunk: String) -> String? {
        switch state {
        case .normal:
            // Check if this chunk starts what looks like a JSON tool call
            if let braceIndex = chunk.firstIndex(of: "{") {
                let leading = String(chunk[..<braceIndex])
                let jsonPart = String(chunk[braceIndex...])
                toolCallBuffer = jsonPart
                state = .collectingToolCall

                if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
                    toolCalls.append(toolCall)
                    toolCallBuffer = ""
                    state = .normal
                    return leading.isEmpty ? nil : leading
                }

                // Still collecting — check if braces are balanced (would mean parse
                // failed on complete JSON, so it's not a tool call)
                if jsonBracesBalanced(toolCallBuffer) {
                    state = .normal
                    let buffer = toolCallBuffer
                    toolCallBuffer = ""
                    return leading + buffer
                }

                return leading.isEmpty ? nil : leading
            }

            // No brace seen — pass through as regular text
            return chunk

        case .potentialToolCall, .collectingToolCall:
            toolCallBuffer += chunk

            if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
                toolCalls.append(toolCall)
                toolCallBuffer = ""
                state = .normal
                return nil
            }

            // If braces are balanced but parse failed, this isn't a tool call — flush
            if jsonBracesBalanced(toolCallBuffer) {
                state = .normal
                let buffer = toolCallBuffer
                toolCallBuffer = ""
                return buffer
            }

            // Still collecting
            return nil
        }
    }

    /// Check whether open/close braces are balanced in the string.
    ///
    /// String-aware: braces inside a JSON string value (and escaped quotes) are
    /// ignored, so streaming content like `{"path": "a}b"` is not treated as a
    /// complete object the moment the in-string `}` arrives. A naive counter
    /// would prematurely flush partial JSON and corrupt the tool call. The
    /// result also requires the scanner to not end mid-string. (Upstream 1335fb5
    /// inString/isEscaped scanner, re-derived for our multi-tag processor.)
    private func jsonBracesBalanced(_ text: String) -> Bool {
        var depth = 0
        var inString = false
        var isEscaped = false
        for ch in text {
            if isEscaped {
                isEscaped = false
                continue
            }
            if ch == "\\" {
                if inString { isEscaped = true }
                continue
            }
            if ch == "\"" {
                inString.toggle()
                continue
            }
            if inString { continue }
            if ch == "{" {
                depth += 1
            } else if ch == "}" {
                depth -= 1
            }
        }
        return depth == 0 && !inString
    }

    /// Process chunk for tagged formats.
    private func processTaggedChunk(_ chunk: String) -> String? {
        let startTags = parser.acceptedStartTags
        let endTags = parser.acceptedEndTags

        guard !startTags.isEmpty else {
            return chunk
        }

        guard
            (state == .normal && startTagFirstChars.contains(where: { chunk.contains($0) }))
                || state != .normal
        else {
            return chunk
        }

        toolCallBuffer += chunk
        var leadingToken: String?

        switch state {
        case .normal:
            // Change state to potential tool call
            state = .potentialToolCall

            leadingToken = separateToken(
                from: &toolCallBuffer, separators: startTags, returnLeading: true)

            fallthrough
        case .potentialToolCall:
            if let startTag = partialMatch(buffer: toolCallBuffer, tags: startTags) {
                if toolCallBuffer.starts(with: startTag) {
                    state = .collectingToolCall
                    fallthrough
                } else {
                    return nil
                }
            } else {
                // Otherwise, return the collected text and reset the state
                state = .normal
                let buffer = toolCallBuffer
                toolCallBuffer = ""
                return (leadingToken ?? "") + buffer
            }
        case .collectingToolCall:
            guard !endTags.isEmpty else {
                return nil
            }

            if endTags.contains(where: { toolCallBuffer.contains($0) }) {
                // Separate the trailing token
                let trailingToken = separateToken(
                    from: &toolCallBuffer, separators: endTags, returnLeading: false)

                // Parse the tool call using the parser
                if let toolCall = parser.parse(content: toolCallBuffer, tools: tools) {
                    toolCalls.append(toolCall)
                }

                state = .normal
                toolCallBuffer = ""

                // If the token contains the start character, there may be more tool calls to come
                if let trailingToken,
                    startTagFirstChars.contains(where: { trailingToken.contains($0) })
                {
                    return processChunk(trailingToken)
                } else {
                    // Otherwise, return the collected token, or nil if it's empty
                    return trailingToken?.isEmpty ?? true ? nil : trailingToken
                }
            } else {
                return nil
            }
        }
    }

    private func separateToken(
        from buffer: inout String, separators: [String], returnLeading: Bool
    ) -> String? {
        guard let separator = firstSeparator(in: buffer, separators: separators) else {
            return nil
        }
        return separateToken(from: &buffer, separator: separator, returnLeading: returnLeading)
    }

    /// Separates a token from a string buffer based on a separator
    /// - Parameters:
    ///   - buffer: The string buffer to modify
    ///   - separator: The separator string to search for
    ///   - returnLeading: If true, returns text before separator; if false, returns text after
    /// - Returns: The separated token, or nil if separator not found
    private func separateToken(from buffer: inout String, separator: String, returnLeading: Bool)
        -> String?
    {
        guard let range = buffer.range(of: separator) else { return nil }

        let token: String
        if returnLeading {
            token = String(buffer[..<range.lowerBound])
            buffer = String(buffer[range.lowerBound...])
        } else {
            token = String(buffer[range.upperBound...])
            buffer = String(buffer[..<range.upperBound])
        }

        return token
    }

    private func partialMatch(buffer: String, tag: String) -> Bool {
        for (tagIndex, bufferIndex) in zip(tag.indices, buffer.indices) {
            if buffer[bufferIndex] != tag[tagIndex] {
                return false
            }
        }

        return true
    }

    private func partialMatch(buffer: String, tags: [String]) -> String? {
        tags.first { partialMatch(buffer: buffer, tag: $0) }
    }

    private func firstSeparator(in buffer: String, separators: [String]) -> String? {
        var match: (range: Range<String.Index>, separator: String)?
        for separator in separators {
            guard let range = buffer.range(of: separator) else { continue }
            if match == nil || range.lowerBound < match!.range.lowerBound {
                match = (range, separator)
            }
        }
        return match?.separator
    }
}
