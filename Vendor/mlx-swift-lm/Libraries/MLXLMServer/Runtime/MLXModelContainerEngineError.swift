// Copyright © 2026 Eigen Labs Inc.

import Foundation

/// Errors thrown by ``MLXModelContainerEngine``.
public enum MLXModelContainerEngineError: Error, LocalizedError, Equatable {
    /// A request carried `image_url` or `video_url` content. This engine
    /// flattens chat content to text (see `OpenAIChatMessage.chatMessage()`),
    /// so media cannot be served and is rejected instead of silently dropped.
    case mediaUnsupported

    public var errorDescription: String? {
        switch self {
        case .mediaUnsupported:
            return
                "This model server is text-only and cannot process image_url or video_url content. Send text-only messages."
        }
    }
}
