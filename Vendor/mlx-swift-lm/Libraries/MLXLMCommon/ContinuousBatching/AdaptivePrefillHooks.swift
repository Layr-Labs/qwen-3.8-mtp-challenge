import Foundation

public struct PrefillChunkContext: Sendable, Equatable {
    public let configuredStepSize: Int
    public let maxNumBatchedTokens: Int
    public let decodeBatchSize: Int
    public let maxRemaining: Int
    public let defaultChunkSize: Int

    public init(
        configuredStepSize: Int,
        maxNumBatchedTokens: Int,
        decodeBatchSize: Int,
        maxRemaining: Int,
        defaultChunkSize: Int
    ) {
        self.configuredStepSize = configuredStepSize
        self.maxNumBatchedTokens = maxNumBatchedTokens
        self.decodeBatchSize = decodeBatchSize
        self.maxRemaining = maxRemaining
        self.defaultChunkSize = defaultChunkSize
    }
}

public struct ColdPrefillChunkSample: Sendable, Equatable {
    public let requestedChunkSize: Int
    public let actualChunkSize: Int
    public let batchSize: Int
    public let totalTokens: Int
    /// KV-cache length (in tokens) BEFORE this chunk was prefilled, i.e. the
    /// position at which this chunk starts. 0 marks the first chunk of a cold
    /// prefill; any value > 0 means earlier chunks of the same prompt already
    /// populated the cache, so the per-chunk cost carries the O(N²) attention
    /// growth of those earlier positions. Consumers that calibrate chunk size
    /// on measured ms/token MUST gate on `positionOffset == 0` so the
    /// position-dependent attention cost can never masquerade as a chunk-size
    /// effect. For batch > 1 this is the maximum row length before the chunk.
    public let positionOffset: Int
    public let durationSeconds: Double
    public let decodeBatchSize: Int
    public let cappedByBudget: Bool
    public let cappedByCheckpoint: Bool
    public let cappedByRemaining: Bool

    public init(
        requestedChunkSize: Int,
        actualChunkSize: Int,
        batchSize: Int,
        totalTokens: Int,
        positionOffset: Int = 0,
        durationSeconds: Double,
        decodeBatchSize: Int,
        cappedByBudget: Bool,
        cappedByCheckpoint: Bool,
        cappedByRemaining: Bool
    ) {
        self.requestedChunkSize = requestedChunkSize
        self.actualChunkSize = actualChunkSize
        self.batchSize = batchSize
        self.totalTokens = totalTokens
        self.positionOffset = positionOffset
        self.durationSeconds = durationSeconds
        self.decodeBatchSize = decodeBatchSize
        self.cappedByBudget = cappedByBudget
        self.cappedByCheckpoint = cappedByCheckpoint
        self.cappedByRemaining = cappedByRemaining
    }
}
