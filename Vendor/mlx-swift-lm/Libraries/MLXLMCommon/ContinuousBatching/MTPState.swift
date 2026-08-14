// Copyright © 2026 Eigen Labs.
//
// Port of omlx commit 696d90a:
//   patches/mlx_lm_mtp/batch_generator.py _MtpStats / _MtpState

import Foundation
import MLX

// MARK: - MTPStats

/// Per-sequence MTP timing and accept/reject counters.
/// Logged at sequence finish so operators can evaluate draft+verify productivity.
/// omlx: batch_generator.py _MtpStats
struct MTPStats {
    var cycles: Int = 0        // number of verify cycles run
    var accepts: Int = 0       // cycles where the draft was accepted
    var rejects: Int = 0       // cycles where the draft was rejected
    var initEmits: Int = 0     // tokens emitted from the post-init queue (always 2)
    var draftEmits: Int = 0    // tokens emitted as accepted drafts
    var bonusEmits: Int = 0    // tokens emitted as bonus on accept path
    var verifyEmits: Int = 0   // corrected tokens emitted on reject path
    // Component timings
    var backboneMs: Double = 0  // cumulative time inside 2-token verify forward
    var mtpHeadMs: Double = 0   // cumulative time inside MTP-head forwards
    var sampleMs: Double = 0    // cumulative time in sampling + acceptance check
    var cacheOpsMs: Double = 0  // cumulative time in trim / rollback restore

    var acceptRate: Double {
        cycles > 0 ? Double(accepts) / Double(cycles) : 0
    }
    var totalEmits: Int {
        initEmits + draftEmits + bonusEmits + verifyEmits
    }
}

// MARK: - MTPQueueItem

/// One token buffered for emission from the MTP dispatch path.
/// omlx: batch_generator.py _MtpState.queue element
struct MTPQueueItem {
    let tokenId: Int
    let source: String  // "init" | "draft" | "bonus" | "verify"
}

// MARK: - MTPState

/// Per-batch MTP state stashed on GenerationBatch when the MTP path is active.
/// omlx: batch_generator.py _MtpState
final class MTPState {
    /// Pending tokens to emit in upcoming next() calls.
    /// Max 2 items at any time (one draft + one bonus on accept).
    var queue: [MTPQueueItem] = []

    /// First input token for the next verify forward. Shape: (1,) uint32.
    /// omlx: state.next_main
    var nextMain: MLXArray?

    /// Current draft token. Shape: (1,) uint32.
    /// omlx: state.draft_tok
    var draftTok: MLXArray?

    /// Log-probabilities of the draft distribution. Shape: (vocab,).
    /// Used for stochastic acceptance (Leviathan & Chen 2023).
    /// omlx: state.draft_lp
    var draftLp: MLXArray?

    /// Host-side int copy of draftTok, cached at draft creation time.
    /// Avoids a GPU→CPU sync on every verify cycle's accept/reject check.
    /// omlx: state.draft_id
    var draftId: Int = -1

    /// KV caches for the MTP transformer layer(s). Created once at init and
    /// accumulated across cycles — the MTP head is auto-regressive and was
    /// trained with a growing KV context, matching the backbone's behaviour.
    /// PR #990: `mtp_cache = model.make_mtp_cache()` (created once, reused).
    /// omlx uses fresh-per-cycle; we now match PR #990 for correctness.
    var mtpCache: [any KVCache]?

    /// Stats to be logged on sequence finish.
    var stats: MTPStats = MTPStats()
}
