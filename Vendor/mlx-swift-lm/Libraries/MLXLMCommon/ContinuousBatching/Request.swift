// Copyright © 2026 Eigen Labs.
//
// Port of omlx/omlx/request.py — Request management for continuous batching.
// https://github.com/jundot/omlx/blob/main/omlx/request.py

import Foundation
import MLX

/// Status of a request in the continuous-batching scheduling system.
public enum RequestStatus: Int, Comparable, Sendable {
    /// Request is waiting to be scheduled.
    case waiting = 0
    /// Request is currently being processed (generating tokens).
    case running = 1
    /// Request was preempted and needs to be resumed.
    case preempted = 2
    /// Request finished successfully (hit stop token).
    case finishedStopped = 3
    /// Request finished due to max_tokens limit.
    case finishedLengthCapped = 4
    /// Request was aborted by user.
    case finishedAborted = 5

    public var isFinished: Bool { self.rawValue > RequestStatus.preempted.rawValue }

    public var finishReason: String? {
        switch self {
        case .finishedStopped: "stop"
        case .finishedLengthCapped: "length"
        case .finishedAborted: "abort"
        default: nil
        }
    }

    public static func < (lhs: RequestStatus, rhs: RequestStatus) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// Sampling parameters for text generation.
/// Mirrors omlx's SamplingParams with all supported sampling knobs.
public struct SamplingParams: Sendable, Codable {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var topK: Int            // 0 = disabled
    public var minP: Float          // 0.0 = disabled
    public var repetitionPenalty: Float   // 1.0 = disabled
    public var presencePenalty: Float     // 0.0 = disabled
    public var frequencyPenalty: Float    // 0.0 = disabled
    public var stop: [String]
    public var stopTokenIds: Set<Int>

    // Logprobs (memory optimization: disabled by default)
    public var logprobs: Bool
    public var topLogprobs: Int?   // 1-20

    // Seed for reproducible generation (best-effort)
    public var seed: UInt64?

    public init(
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        topK: Int = 0,
        minP: Float = 0.0,
        repetitionPenalty: Float = 1.0,
        presencePenalty: Float = 0.0,
        frequencyPenalty: Float = 0.0,
        stop: [String] = [],
        stopTokenIds: Set<Int> = [],
        logprobs: Bool = false,
        topLogprobs: Int? = nil,
        seed: UInt64? = nil
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.stop = stop
        self.stopTokenIds = stopTokenIds
        self.logprobs = logprobs
        self.topLogprobs = topLogprobs
        self.seed = seed
    }
}

/// Represents a single inference request in the continuous-batching system.
/// Mirrors omlx's Request with simplifications for Swift/MLX.
public final class Request: @unchecked Sendable {
    public let requestId: String
    public let prompt: AnyHashable  // String or [Int] token IDs
    public var samplingParams: SamplingParams
    public let arrivalTime: TimeInterval
    public var priority: Int  // Lower = higher priority

    // Set after tokenization
    public var promptTokenIds: [Int]?
    public var numPromptTokens: Int

    // Generation state
    public var status: RequestStatus
    public var numComputedTokens: Int
    public var outputTokenIds: [Int]
    public var outputText: String

    // For BatchGenerator integration
    public var batchUid: Int?

    /// Prefix cache fields
    public var promptCache: [KVCache]?
    public var cachedTokens: Int
    public var remainingTokens: [Int]?

    /// Externally-restored checkpoint cache (hybrid sliding-window models):
    /// the per-layer single-stream caches covering the first `tokenCount`
    /// prompt tokens, plus that count. Set by the provider after a
    /// `PrefixCacheManager` hit BEFORE the request is enqueued. When non-nil
    /// the scheduler routes the request through the dedicated checkpoint
    /// restore-admit path (B==1) instead of cold prefill. nil for every
    /// request that didn't hit the checkpoint cache (the default).
    public var restoredCheckpoint: (caches: [any KVCache], tokenCount: Int)?

    // Finish reason
    public var finishReason: String?

    // Cache corruption recovery
    public var cacheCorruptionRetries: Int

    // Detokenizer state (NaiveStreamingDetokenizer)
    public var detokenizer: NaiveStreamingDetokenizer?

    public init(
        requestId: String,
        prompt: AnyHashable,
        samplingParams: SamplingParams = SamplingParams(),
        priority: Int = 0
    ) {
        self.requestId = requestId
        self.prompt = prompt
        self.samplingParams = samplingParams
        self.arrivalTime = Date.timeIntervalSinceReferenceDate
        self.priority = priority
        self.promptTokenIds = nil
        self.numPromptTokens = 0
        self.status = .waiting
        self.numComputedTokens = 0
        self.outputTokenIds = []
        self.outputText = ""
        self.batchUid = nil
        self.promptCache = nil
        self.cachedTokens = 0
        self.remainingTokens = nil
        self.restoredCheckpoint = nil
        self.finishReason = nil
        self.cacheCorruptionRetries = 0
        self.detokenizer = nil
    }

    public var numOutputTokens: Int { outputTokenIds.count }
    public var numTotalTokens: Int { numPromptTokens + numOutputTokens }
    public var maxTokens: Int { samplingParams.maxTokens }

    public var isFinished: Bool { status.isFinished }

    public func appendOutputToken(_ tokenId: Int) {
        outputTokenIds.append(tokenId)
        numComputedTokens += 1
    }

    public func setFinished(_ status: RequestStatus, reason: String? = nil) {
        self.status = status
        self.finishReason = reason ?? status.finishReason
    }
}

/// Output for a single request after a generation step.
/// Returned by the engine to communicate results back to the API layer.
public struct RequestOutput: Sendable {
    public let requestId: String
    /// New tokens generated in this step.
    public var newTokenIds: [Int]
    public var newText: String
    /// Cumulative output.
    public var outputTokenIds: [Int]
    public var outputText: String
    /// Status.
    public var finished: Bool
    public var finishReason: String?
    /// Timing.
    public var promptTokens: Int
    public var completionTokens: Int
    /// Prefix cache stats.
    public var cachedTokens: Int
    /// Error message.
    public var error: String?

    public init(
        requestId: String,
        newTokenIds: [Int] = [],
        newText: String = "",
        outputTokenIds: [Int] = [],
        outputText: String = "",
        finished: Bool = false,
        finishReason: String? = nil,
        promptTokens: Int = 0,
        completionTokens: Int = 0,
        cachedTokens: Int = 0,
        error: String? = nil
    ) {
        self.requestId = requestId
        self.newTokenIds = newTokenIds
        self.newText = newText
        self.outputTokenIds = outputTokenIds
        self.outputText = outputText
        self.finished = finished
        self.finishReason = finishReason
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.cachedTokens = cachedTokens
        self.error = error
    }

    public var usage: (promptTokens: Int, completionTokens: Int, totalTokens: Int) {
        (promptTokens, completionTokens, promptTokens + completionTokens)
    }
}