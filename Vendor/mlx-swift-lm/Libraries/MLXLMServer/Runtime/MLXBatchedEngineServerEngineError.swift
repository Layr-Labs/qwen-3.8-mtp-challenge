// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon

/// Errors thrown by ``MLXBatchedEngineServerEngine`` during streaming.
public enum MLXBatchedEngineServerEngineError: Error, LocalizedError, Equatable {
    case generationFailed(String)
    /// Per-request `tool_call_parser` override resolved to a format other than
    /// the one pinned at engine construction.
    case unsupportedToolCallParser(pinned: ToolCallFormat, requested: ToolCallFormat)

    public var errorDescription: String? {
        switch self {
        case .generationFailed(let message):
            return "Batched engine generation failed: \(message)"
        case .unsupportedToolCallParser(let pinned, let requested):
            return
                "per-request tool_call_parser override is not supported with the batched engine; pinned format is \(pinned.rawValue), requested \(requested.rawValue)"
        }
    }
}
