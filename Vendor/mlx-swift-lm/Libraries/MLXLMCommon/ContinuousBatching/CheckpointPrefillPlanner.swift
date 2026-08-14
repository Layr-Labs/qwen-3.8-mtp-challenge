import Foundation

/// Pure planner for boundary-aligned prefill chunking used by the
/// checkpoint KV-cache capture path (hybrid sliding-window models).
///
/// Why: a checkpoint snapshot must be taken when the prefilled token count
/// lands EXACTLY on a checkpoint boundary (256, 512, …) — that is the only
/// length a future request's exact-checkpoint lookup will probe, and a
/// sliding-window (`RotatingKVCache`) layer can't be sliced back to a
/// boundary after the fact (it discards/rotates). So instead of capturing
/// at end-of-prefill, the scheduler caps each prefill chunk so the running
/// prefilled count steps onto each boundary, captures there, and continues.
///
/// This is a no-op for the default path: it is consulted ONLY when a
/// capture hook is installed AND the cold batch is a single row. When the
/// next boundary is farther than the default chunk, it returns the default
/// chunk unchanged (capture happens on a later step as the count nears the
/// boundary) — so it never *grows* a chunk and never overshoots a boundary.
public enum CheckpointPrefillPlanner {

    public struct Step: Equatable, Sendable {
        /// Number of tokens to prefill this step (≥1).
        public let chunk: Int
        /// Boundary length to capture at AFTER this chunk, or nil. Non-nil
        /// iff `prefilled + chunk` equals a checkpoint boundary.
        public let captureAt: Int?
    }

    /// - prefilled: tokens already prefilled into the cache (= the cache's
    ///   covered prefix length so far).
    /// - defaultChunk: the chunk size the scheduler would otherwise use
    ///   (min of prefillStepSize, budget, remaining) — already ≥1.
    /// - remaining: prefill tokens still to process this prompt.
    /// - boundaries: checkpoint boundaries (e.g. from
    ///   `PrefixDigest.checkpoints(forSlidingWindow:)`), need not be sorted.
    public static func plan(
        prefilled: Int, defaultChunk: Int, remaining: Int, boundaries: [Int]
    ) -> Step {
        let safeRemaining = max(0, remaining)
        let safeDefault = max(1, defaultChunk)
        // The smallest boundary strictly ahead of us that we can still reach
        // within this prompt's remaining prefill.
        let next = boundaries
            .filter { $0 > prefilled && $0 <= prefilled + safeRemaining }
            .min()
        guard let boundary = next else {
            return Step(chunk: max(1, min(safeDefault, safeRemaining == 0 ? safeDefault : safeRemaining)),
                        captureAt: nil)
        }
        let toBoundary = boundary - prefilled
        let chunk = max(1, min(safeDefault, toBoundary, safeRemaining))
        let captureAt = (prefilled + chunk == boundary) ? boundary : nil
        return Step(chunk: chunk, captureAt: captureAt)
    }
}
