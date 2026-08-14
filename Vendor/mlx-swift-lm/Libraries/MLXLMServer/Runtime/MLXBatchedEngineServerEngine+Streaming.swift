// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon

extension MLXBatchedEngineServerEngine {
    public func streamChatCompletion(
        request: OpenAIChatCompletionRequest
    ) async throws -> AsyncThrowingStream<MLXServerGenerationEvent, Error> {
        // TODO: route per-request tool_call_parser overrides through
        // ToolCallProcessor at the stream boundary so they don't require
        // mutating shared ModelContext state.
        try Self.validateToolCallParserOverride(
            requested: request.toolCallParser,
            pinned: toolCallFormat,
            modelType: modelType
        )

        let prompt = engine.buildPrompt(
            messages: request.messages.map { $0.templateMessage() },
            tools: request.tools?.map { $0.toolSpec() },
            additionalContext: nil
        )

        let samplingParams = request.batchedSamplingParams
        let requestId = UUID().uuidString
        let batchedRequest = Request(
            requestId: requestId,
            prompt: prompt,
            samplingParams: samplingParams
        )

        let toolHandler = BatchedToolStreamHandler(
            format: toolCallFormat,
            tools: request.tools?.map { $0.toolSpec() }
        )
        let core = engine.core
        let started = Date()

        return AsyncThrowingStream { continuation in
            let task = Task {
                // Early-cancel: onTermination races addRequest, so the
                // scheduler may never see this id. Bail with the same
                // terminal .info shape the in-flight abort branch emits
                // so consumers always see exactly one terminal event.
                if Task.isCancelled {
                    Self.emitCancelledTerminal(continuation, since: started)
                    return
                }

                // Re-check immediately before enqueue. Cancellation can
                // land in the (very small) window between the first check
                // above and the addRequest call below — catching it here
                // avoids spending a scheduler slot and prefill cycle on a
                // request the caller has already abandoned. There remains
                // a vanishingly narrow window between this check and
                // addRequest returning, but it contains no awaits.
                if Task.isCancelled {
                    Self.emitCancelledTerminal(continuation, since: started)
                    return
                }

                _ = await core.addRequest(batchedRequest)

                var firstTokenAt: Date?
                var promptTokens = 0
                var completionTokens = 0
                var stopReason: GenerateStopReason = .stop

                for await output in core.streamOutputs(requestId: requestId) {
                    if Task.isCancelled {
                        _ = core.abortRequest(requestId)
                        continuation.finish()
                        return
                    }

                    // Abort first: EngineCore.abortRequest sets both
                    // finishReason="abort" and error="Request aborted", so
                    // cancellation must surface as .info(.cancelled) rather
                    // than a thrown error.
                    if output.finished, output.finishReason == "abort" {
                        for toolCall in toolHandler.finish() {
                            continuation.yield(.toolCall(toolCall))
                        }
                        promptTokens = output.promptTokens
                        completionTokens = output.completionTokens
                        stopReason = .cancelled
                        break
                    }

                    if let error = output.error {
                        continuation.finish(
                            throwing: MLXBatchedEngineServerEngineError.generationFailed(error)
                        )
                        return
                    }

                    if firstTokenAt == nil, !output.newText.isEmpty {
                        firstTokenAt = Date()
                    }
                    promptTokens = output.promptTokens
                    completionTokens = output.completionTokens

                    if !output.newText.isEmpty,
                        let visible = toolHandler.processChunk(output.newText),
                        !visible.isEmpty
                    {
                        continuation.yield(.content(visible))
                    }

                    if output.finished {
                        for toolCall in toolHandler.finish() {
                            continuation.yield(.toolCall(toolCall))
                        }
                        switch output.finishReason {
                        case "length":
                            stopReason = .length
                        default:
                            stopReason = .stop
                        }
                        break
                    }
                }

                let now = Date()
                let promptTime = (firstTokenAt ?? now).timeIntervalSince(started)
                let generateTime = now.timeIntervalSince(firstTokenAt ?? started)
                continuation.yield(
                    .info(
                        ServerGenerationInfo(
                            promptTokens: promptTokens,
                            completionTokens: completionTokens,
                            promptTime: max(0, promptTime),
                            generationTime: max(0, generateTime),
                            stopReason: stopReason.openAIFinishReason
                        )
                    )
                )
                continuation.finish()
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
                _ = core.abortRequest(requestId)
            }
        }
    }

    /// Emit the single terminal `.info(cancelled)` event used by both
    /// pre-`addRequest` cancellation checks, then close the stream.
    /// Kept identical to the in-flight abort branch so consumers always
    /// see exactly one terminal event regardless of when cancellation
    /// lands.
    private static func emitCancelledTerminal(
        _ continuation: AsyncThrowingStream<MLXServerGenerationEvent, Error>.Continuation,
        since started: Date
    ) {
        let now = Date()
        let promptTime = now.timeIntervalSince(started)
        continuation.yield(
            .info(
                ServerGenerationInfo(
                    promptTokens: 0,
                    completionTokens: 0,
                    promptTime: max(0, promptTime),
                    generationTime: 0,
                    stopReason: GenerateStopReason.cancelled.openAIFinishReason
                )
            )
        )
        continuation.finish()
    }
}
