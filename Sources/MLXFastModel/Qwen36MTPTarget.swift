import Foundation
import MLX
import MLXLLM
import MLXLMCommon

// The model surface the native-MTP session needs, as a protocol.
//
// WHY THIS EXISTS. The first migration threaded the CONCRETE `Qwen35TextModel`
// through the session, the reference session and the worker. That was a
// deviation from the validated driver, which ran on `Qwen35Model`, and it made
// the worker unloadable in both directions: the transformed `weights/` tree
// declares `qwen3_5_text` and therefore builds a `Qwen35TextModel`, while the
// raw pinned reference declares `qwen3_5` and builds a `Qwen35Model` that the
// concrete-type guard then rejected.
//
// WHY NOT WIDEN `MTPCapable` INSTEAD. `MTPCapable` is vendored and
// `DeepseekV4Model` also conforms to it; adding `applyFinalNorm` and
// `mtpForwardWithHidden` as requirements would break that conformance for a
// model this track never runs. A retroactive conformance declared HERE costs
// the vendored surface nothing.
//
// WHY BOTH CONFORMANCES ARE BEHAVIOURALLY IDENTICAL, which is the load-bearing
// claim: `Qwen35Model` is a pure pass-through wrapper. Every member below is a
// one-line delegation to the same inner `Qwen35TextModel`
// (Vendor/.../Qwen35.swift, `extension Qwen35Model: MTPCapable`) --
// `callWithHidden` forwards `nConfirmed` unchanged, `applyFinalNorm` forwards to
// the inner `model.norm`, `makeMTPCache` and `mtpForwardWithHidden` likewise.
// There is no wrapper-level arithmetic, no second norm and no second cache
// policy. So running the session on either type is the same computation, and
// the driver's proven configuration is reproducible on this code by pointing it
// at a checkpoint that builds the outer class.
/// `AnyObject`-constrained: both conformers are `Module` subclasses, and the
/// session holds one for the whole decode, so reference semantics are the
/// contract -- a value-type conformer would silently copy the model away from
/// its caches.
public protocol Qwen36MTPTarget: AnyObject {
    /// True when the MTP head is attached and operational.
    var hasMTPHead: Bool { get }

    /// The hybrid cache stack: `MambaCache` on the gated-delta layers,
    /// `KVCacheSimple` on the full-attention layers.
    func newCache(parameters: GenerateParameters?) -> [KVCache]

    /// Backbone forward returning `(logits, PRE-norm hidden)`.
    ///
    /// INVARIANT #7 LIVES ON THIS CALL. `nConfirmed` must be 0 on every forward
    /// this track issues: a non-zero value installs the vendored depth-1-only
    /// rollback (`ArraysCache.rollbackState`, written by `Qwen35GatedDeltaNet`)
    /// AND changes the gated-delta chunk geometry, both of which fight the
    /// snapshot/rollback this track uses instead. The requirement is identical
    /// on both conformers because both reach the same `Qwen35TextModel` method.
    func callWithHidden(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray)

    /// SEED-PREFILL FORWARD returning `(last logit row, last PRE-norm hidden
    /// row)`, both shaped `[1, 1, ...]`.
    ///
    /// Same backbone forward as `callWithHidden` -- same tokens, same cache, same
    /// `nConfirmed`, so every cache ends in the same state at the same offset --
    /// with the final norm and the vocabulary projection narrowed to the last row.
    /// `Qwen36MTPBlockSession.begin` is the ONLY caller: it is the only site that
    /// bulk-forwards a long block and then keeps just the tail row, so it is the
    /// only site where the other `S - 1` vocabulary rows are pure waste.
    ///
    /// DELIBERATELY A REQUIREMENT AND NOT A DEFAULTED PROTOCOL EXTENSION. A
    /// default that fell back to `callWithHidden` would make a signature drift on
    /// a conformer compile clean and silently un-optimize the seed prefill, which
    /// is precisely the failure mode nobody would notice. Both conformers below
    /// implement it in the editable `Vendor/.../Qwen35.swift`, and a mismatch
    /// there is a build error.
    func callWithLastTokenHidden(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray)

    /// K=1 VERIFY FORWARD: same `(logits, PRE-norm hidden)` contract as
    /// `callWithHidden` with `nConfirmed: 0` -- same rows, same vocabulary
    /// projection over ALL of them in one call -- plus a lazy post-row-0
    /// recurrent boundary stash on each gated-delta cache.
    ///
    /// Invariant #7 above says a non-zero `nConfirmed` changes the gated-delta
    /// chunk geometry AND installs the vendored rollback. This call is the
    /// nConfirmed-ZERO way to get the second half of that without the first: the
    /// verify rows run as one fused chunk (the geometry this track used before
    /// the checkpoint split landed), and the boundary the split used to write
    /// eagerly is reconstructed on demand by ``recomputeFusedPrimaryBoundary``
    /// only on the rounds that reject.
    ///
    /// A REQUIREMENT, NOT A DEFAULTED EXTENSION, for the same reason as
    /// `callWithLastTokenHidden`: a default that silently fell back to the split
    /// path would un-optimize the hot path with no build error.
    func callWithHiddenStashingPrimaryBoundary(
        input: LMInput.Text, cache: [any KVCache]
    ) -> (MLXArray, MLXArray)

    /// Rebuild every gated-delta layer's post-row-0 recurrent boundary from the
    /// stash left by `callWithHiddenStashingPrimaryBoundary`, writing it into
    /// `cache`. Fail-closed: `false` means no layer was mutated and the caller
    /// must use the generic snapshot-and-repair path.
    func recomputeFusedPrimaryBoundary(cache: [any KVCache]) -> Bool

    /// MTP head forward returning `(logits, head post-`mtp.norm` hidden)`.
    func mtpForwardWithHidden(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> (MLXArray, MLXArray)

    /// Fresh KV caches for the MTP head layers, one per draft round.
    func makeMTPCache() -> [any KVCache]

    /// INVARIANT #2. The backbone's final `model.norm`, applied to a hidden row.
    /// The MTP head's `pre_fc_norm_hidden` weights were trained on POST-norm
    /// input -- the vendored `MTPCapable.mtpForward` documentation says so
    /// outright -- while `callWithHidden` returns PRE-norm hidden by design, so
    /// this is the step that reconciles them. On `Qwen35Model` it forwards to the
    /// inner text model's `model.norm`, i.e. exactly the norm the inner
    /// `callAsFunction` applies; there is no outer norm to pick by mistake.
    func applyFinalNorm(_ x: MLXArray) -> MLXArray
}

// Both Qwen 3.6 model classes already implement every member; these
// conformances add no code and change no behaviour.
//
// `MLXLLM.` IS REQUIRED ON THE SECOND ONE. This module declares its own
// `public enum Qwen35Model` (Qwen35Model.swift — the scored serial forward's
// namespace), which SHADOWS the vendored class inside MLXFastModel. Written
// unqualified, the conformance silently binds to the local enum and fails with
// "non-class type 'Qwen35Model' cannot conform" — a diagnostic that points at
// the protocol rather than at the shadowing, which is why this note exists.
// `QwenMTPBackboneLayoutTests` pins that the qualified spelling stays.
extension Qwen35TextModel: Qwen36MTPTarget {}
extension MLXLLM.Qwen35Model: Qwen36MTPTarget {}

/// `MLXFAST_QWEN_MTP_SEED_TAIL_PROJECTION` (DEFAULT ON; set "0" to disable):
/// route `Qwen36MTPBlockSession.begin`'s seed prefill through
/// `callWithLastTokenHidden` instead of `callWithHidden`, so the untied
/// 248,320-way vocabulary head is applied to the one seed row the session keeps
/// rather than to all 512.
///
/// Read ONCE, at first use, from the process environment -- not per call -- so
/// the guarded and unguarded paths differ by a branch on a `let`, never by an
/// environment lookup inside the timed window. Flipping it to "0" restores the
/// exact pre-change call, which is the ablation this submission's claim rests
/// on. The flag is a build-time-equivalent switch over an ADDITIVE seam: the
/// old `callWithHidden` path is fully intact and still serves every other
/// caller unconditionally.
let qwenMTPSeedTailProjectionEnabled =
    ProcessInfo.processInfo.environment["MLXFAST_QWEN_MTP_SEED_TAIL_PROJECTION"]
        != "0"

/// `MLXFAST_QWEN_MTP_FUSED_ACCEPT_VERIFY` (DEFAULT ON; set "0" to disable):
/// run the K=1 verify's gated-delta stack as ONE fused chunk with a lazy
/// post-primary boundary stash, instead of splitting it at `nConfirmed: 1` on
/// every round.
///
/// The split exists purely to serve the ~1/3 of rounds that reject; the other
/// ~2/3 pay for a checkpoint they never read. Setting this to "0" restores the
/// exact `nConfirmed: fastK1 ? 1 : 0` call and the checkpoint restore that
/// consumes it -- both are fully intact -- which is the ablation this
/// submission's claim rests on. Same read-once discipline as the flag above: no
/// environment lookup ever happens inside the timed window.
let qwenMTPFusedAcceptVerifyEnabled =
    ProcessInfo.processInfo.environment["MLXFAST_QWEN_MTP_FUSED_ACCEPT_VERIFY"]
        != "0"
