// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon

extension MLXBatchedEngineServerEngine {
    /// Accepts `nil`, `"auto"` (case-insensitive), or any value resolving
    /// to `pinned`. Throws on a resolved-format mismatch.
    static func validateToolCallParserOverride(
        requested: String?,
        pinned: ToolCallFormat,
        modelType: String?
    ) throws {
        guard let requested else { return }
        let normalized = requested.lowercased()
        // "auto" with nil modelType resolves to .json, which would spuriously
        // reject every non-JSON pinned engine. Treat it as "trust the server".
        if normalized == "auto" { return }
        let resolved = try ServerToolParser.resolve(
            requested: requested,
            modelType: modelType
        )
        if resolved != pinned {
            throw MLXBatchedEngineServerEngineError.unsupportedToolCallParser(
                pinned: pinned,
                requested: resolved
            )
        }
    }
}

/// Sendable wrapper around the non-Sendable ``ToolCallProcessor`` for
/// capture in the streaming-completion Task closure. Only touched from
/// that single Task.
public final class BatchedToolStreamHandler: @unchecked Sendable {
    private let processor: ToolCallProcessor

    public init(format: ToolCallFormat, tools: [[String: any Sendable]]?) {
        self.processor = ToolCallProcessor(format: format, tools: tools)
    }

    public func processChunk(_ chunk: String) -> String? {
        processor.processChunk(chunk)
    }

    public func finish() -> [ToolCall] {
        processor.processEOS()
        return processor.toolCalls
    }
}
