// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXLMServer

/// CI-safe tests for ``MLXModelContainerEngine``'s media-rejection guard.
///
/// The engine flattens chat content to text via `OpenAIChatMessage.chatMessage()`,
/// which drops `image_url`/`video_url` parts. Rather than silently ignore media,
/// `streamChatCompletion` rejects it with ``MLXModelContainerEngineError/mediaUnsupported``.
/// These tests cover the `hasMedia` classifier and the request-level guard
/// predicate directly (driving the full engine needs a loaded MLX model, which is
/// not available in CI).
struct MLXModelContainerEngineMediaTests {
    @Test("hasMedia is false for a plain string content")
    func hasMediaFalseForStringContent() {
        #expect(OpenAIMessageContent.text("hello").hasMedia == false)
    }

    @Test("hasMedia is false for null content")
    func hasMediaFalseForNullContent() {
        #expect(OpenAIMessageContent.null.hasMedia == false)
    }

    @Test("hasMedia is false for text-only content parts")
    func hasMediaFalseForTextOnlyParts() {
        let content = OpenAIMessageContent.parts([.text("a"), .text("b")])
        #expect(content.hasMedia == false)
    }

    @Test("hasMedia is false for an unsupported (non-media) content part")
    func hasMediaFalseForUnsupportedPart() {
        let content = OpenAIMessageContent.parts([.text("hi"), .unsupported(type: "input_audio")])
        #expect(content.hasMedia == false)
    }

    @Test("hasMedia is true when an image_url part is present")
    func hasMediaTrueForImageURL() {
        let content = OpenAIMessageContent.parts([
            .text("describe this"),
            .imageURL("data:image/png;base64,AAAA"),
        ])
        #expect(content.hasMedia == true)
    }

    @Test("hasMedia is true when a video_url part is present")
    func hasMediaTrueForVideoURL() {
        let content = OpenAIMessageContent.parts([
            .text("summarize"),
            .videoURL("https://example.com/clip.mp4"),
        ])
        #expect(content.hasMedia == true)
    }

    @Test("a request carrying image_url trips the engine's media guard")
    func requestWithImageTripsGuard() {
        let request = OpenAIChatCompletionRequest(
            model: "m",
            messages: [
                .init(role: .system, content: .text("you are helpful")),
                .init(
                    role: .user,
                    content: .parts([
                        .text("what is in this image?"),
                        .imageURL("data:image/png;base64,AAAA"),
                    ])),
            ]
        )

        // Mirrors the predicate in MLXModelContainerEngine.streamChatCompletion.
        #expect(request.messages.contains(where: { $0.content.hasMedia }))
    }

    @Test("a request carrying video_url trips the engine's media guard")
    func requestWithVideoTripsGuard() {
        let request = OpenAIChatCompletionRequest(
            model: "m",
            messages: [
                .init(
                    role: .user,
                    content: .parts([
                        .text("summarize this video"),
                        .videoURL("https://example.com/clip.mp4"),
                    ]))
            ]
        )

        #expect(request.messages.contains(where: { $0.content.hasMedia }))
    }

    @Test("a text-only request does not trip the engine's media guard")
    func textOnlyRequestDoesNotTripGuard() {
        let request = OpenAIChatCompletionRequest(
            model: "m",
            messages: [
                .init(role: .system, content: .text("you are helpful")),
                .init(role: .user, content: .text("hello")),
                .init(role: .user, content: .parts([.text("still text")])),
            ]
        )

        #expect(!request.messages.contains(where: { $0.content.hasMedia }))
    }

    @Test("mediaUnsupported carries an actionable description")
    func mediaUnsupportedDescription() {
        let message = MLXModelContainerEngineError.mediaUnsupported.errorDescription ?? ""
        #expect(message.contains("text-only"))
        #expect(message.contains("image_url"))
        #expect(message.contains("video_url"))
    }
}
