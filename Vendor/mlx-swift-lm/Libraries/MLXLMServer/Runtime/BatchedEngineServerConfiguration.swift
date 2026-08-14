// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon

/// Server-facing knobs for ``MLXBatchedEngineServerEngine``. Re-exposes the
/// ``SchedulerConfig`` + ``ContinuousBatchingConfig`` fields worth tuning
/// from the server layer.
public struct BatchedEngineServerConfiguration: Sendable, Equatable {
    /// Maximum number of concurrently running sequences. Default: 24.
    public var maxNumSeqs: Int

    /// Maximum total tokens (prefill + decode) processed per engine step. Default: 8192.
    public var maxNumBatchedTokens: Int

    /// Tokens processed per prefill chunk. Default: 512.
    public var prefillStepSize: Int

    /// Streaming-update interval, in tokens. Default: 1.
    public var streamInterval: Int

    /// Maximum total KV-cache tokens across all running requests; `0` is unlimited. Default: 0.
    public var maxKVCacheTokens: Int

    /// Seconds the engine sleeps between idle iterations. Default: 0.001.
    public var stepInterval: TimeInterval

    /// When non-nil, enables block-level KV reuse across requests. Default: nil.
    public var prefixCacheConfig: PrefixCacheConfig?

    /// Enable Multi-Token Prediction speculative decoding. Default: false.
    public var mtpEnabled: Bool

    public init(
        maxNumSeqs: Int = 24,
        maxNumBatchedTokens: Int = 8192,
        prefillStepSize: Int = 512,
        streamInterval: Int = 1,
        maxKVCacheTokens: Int = 0,
        stepInterval: TimeInterval = 0.001,
        prefixCacheConfig: PrefixCacheConfig? = nil,
        mtpEnabled: Bool = false
    ) {
        self.maxNumSeqs = maxNumSeqs
        self.maxNumBatchedTokens = maxNumBatchedTokens
        self.prefillStepSize = prefillStepSize
        self.streamInterval = streamInterval
        self.maxKVCacheTokens = maxKVCacheTokens
        self.stepInterval = stepInterval
        self.prefixCacheConfig = prefixCacheConfig
        self.mtpEnabled = mtpEnabled
    }

    /// Build a ``ContinuousBatchingConfig`` matching this configuration.
    public func makeContinuousBatchingConfig() -> ContinuousBatchingConfig {
        ContinuousBatchingConfig(
            schedulerConfig: SchedulerConfig(
                maxNumSeqs: maxNumSeqs,
                maxNumBatchedTokens: maxNumBatchedTokens,
                prefillStepSize: prefillStepSize,
                streamInterval: streamInterval,
                maxKVCacheTokens: maxKVCacheTokens
            ),
            stepInterval: stepInterval,
            prefixCacheConfig: prefixCacheConfig,
            mtpEnabled: mtpEnabled
        )
    }
}
