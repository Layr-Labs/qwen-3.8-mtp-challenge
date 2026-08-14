// Copyright © 2026 Eigen Labs.
//
// Port of omlx/omlx/output_collector.py — vLLM-style streaming output collector.
// https://github.com/jundot/omlx/blob/main/omlx/output_collector.py

import Foundation

/// Per-request output collector with smart buffering.
///
/// Implements the vLLM pattern for efficient streaming:
/// - Non-blocking `getNowait()` to avoid unnecessary task switches
/// - Output aggregation when producer is faster than consumer
/// - Event-based signaling for efficient waiting
///
/// ### Usage
/// ```swift
/// let collector = RequestOutputCollector()
///
/// // Producer side (engine loop):
/// collector.put(output)
///
/// // Consumer side (streaming generator):
/// if let output = collector.getNowait() {
///     // use it
/// } else {
///     // await collector.get()
/// }
/// ```
public final class RequestOutputCollector: @unchecked Sendable {
    private var _output: RequestOutput?
    private let _lock = NSLock()
    private var _waitingContinuations: [CheckedContinuation<RequestOutput, Never>] = []
    private let aggregate: Bool

    public init(aggregate: Bool = true) {
        self.aggregate = aggregate
    }

    /// Put an output into the collector (non-blocking).
    /// If aggregation is enabled and an output already exists,
    /// the new output is merged with the existing one.
    public func put(_ output: RequestOutput) {
        _lock.lock()

        // Fast path: deliver directly to waiting consumers without buffering.
        // Mirrors omlx's asyncio.Event pattern where get() calls get_nowait()
        // (which clears self.output) immediately after the event fires.
        // If we buffered AND delivered, the next getNowait() call would
        // return a duplicate.
        if !_waitingContinuations.isEmpty {
            let continuations = _waitingContinuations
            _waitingContinuations.removeAll()
            _lock.unlock()
            for cont in continuations {
                cont.resume(returning: output)
            }
            return
        }

        if _output == nil {
            _output = output
        } else if aggregate {
            _output = mergeOutputs(_output!, output)
        } else {
            _output = output
        }
        _lock.unlock()
    }

    /// Get output without blocking.
    /// Returns the output if available, nil otherwise.
    public func getNowait() -> RequestOutput? {
        _lock.lock()
        defer { _lock.unlock() }
        let output = _output
        _output = nil
        return output
    }

    /// Get output, blocking only if none available.
    /// For low-latency streaming, prefer:
    /// ```swift
    /// let output = collector.getNowait() ?? await collector.get()
    /// ```
    public func get() async -> RequestOutput {
        await withCheckedContinuation { continuation in
            _lock.lock()
            if let output = _output {
                _output = nil
                _lock.unlock()
                continuation.resume(returning: output)
            } else {
                _waitingContinuations.append(continuation)
                _lock.unlock()
            }
        }
    }

    /// Clear any pending output.
    public func clear() {
        _lock.lock()
        _output = nil
        _waitingContinuations.removeAll()
        _lock.unlock()
    }

    // MARK: - Merge

    private func mergeOutputs(_ existing: RequestOutput, _ new: RequestOutput) -> RequestOutput {
        RequestOutput(
            requestId: new.requestId,
            newTokenIds: existing.newTokenIds + new.newTokenIds,
            newText: existing.newText + new.newText,
            outputTokenIds: new.outputTokenIds,
            outputText: new.outputText,
            finished: new.finished,
            finishReason: new.finishReason,
            promptTokens: new.promptTokens,
            completionTokens: new.completionTokens,
            cachedTokens: new.cachedTokens,
            error: new.error ?? existing.error
        )
    }
}

/// Tracks streaming state for a request.
/// Used to implement stream_interval batching,
/// allowing tokens to be accumulated before sending.
public struct RequestStreamState: Sendable {
    public let streamInterval: Int
    public private(set) var sentTokens: Int

    public init(streamInterval: Int = 1) {
        self.streamInterval = streamInterval
        self.sentTokens = 0
    }

    /// Determine if output should be sent based on streamInterval.
    /// Always sends on finish and first token (for low TTFT).
    public func shouldSend(totalTokens: Int, finished: Bool) -> Bool {
        if finished { return true }
        if sentTokens == 0 { return true }
        return (totalTokens - sentTokens) >= streamInterval
    }

    public mutating func markSent(totalTokens: Int) {
        sentTokens = totalTokens
    }
}
