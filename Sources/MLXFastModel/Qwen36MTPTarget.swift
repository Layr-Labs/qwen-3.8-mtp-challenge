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

    /// Tape-capable variant used by the multi-draft speculative verify: with
    /// `recordTape` the GDN layers split per row and record every
    /// accepted-prefix boundary state into `ArraysCache.boundaryTape`.
    /// `logitsLimit` restricts the LM-head projection to the first `limit`
    /// rows (all-row hidden is still returned) for the lazy bonus-row path.
    /// `stashPrimaryInputs` runs the GDN batched and stashes row-0 projections
    /// for reject-time boundary reconstruction.
    func callWithHidden(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int,
        recordTape: Bool, logitsLimit: Int?, stashPrimaryInputs: Bool
    ) -> (MLXArray, MLXArray)

    /// LM-head logits for one row of a pre-norm hidden `[1, S, H]` — the lazy
    /// counterpart of a logits-limited verify. Bit-exact with the batched
    /// projection (RMSNorm is row-wise; matmul rows are independent).
    func logitsForRow(_ hidden: MLXArray, row: Int) -> MLXArray

    /// Reconstruct every GDN layer's post-primary boundary state from the
    /// stashed row-0 projections and the pre-verify snapshot. Fail-closed
    /// preflight; `false` leaves the cache untouched.
    func recomputePrimaryBoundary(
        cache: [any KVCache], snapshot: [Int: [MLXArray?]]
    ) -> Bool

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

/// The tape-capable forward is a requirement of this track's multi-draft
/// verify loop (see `Qwen36MTPBlockSession`); both vendored conformers
/// implement it directly. A conformer that has not adopted the tape yet still
/// satisfies the protocol through this fallback, which simply never records.
extension Qwen36MTPTarget {
    public func callWithHidden(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int,
        recordTape: Bool, logitsLimit: Int?, stashPrimaryInputs: Bool
    ) -> (MLXArray, MLXArray) {
        _ = logitsLimit
        _ = recordTape
        _ = stashPrimaryInputs
        return callWithHidden(input: input, cache: cache, nConfirmed: nConfirmed)
    }

    public func logitsForRow(_ hidden: MLXArray, row: Int) -> MLXArray {
        _ = hidden
        _ = row
        preconditionFailure(
            "logitsForRow requires a conformer with the lazy-row head")
    }

    public func recomputePrimaryBoundary(
        cache: [any KVCache], snapshot: [Int: [MLXArray?]]
    ) -> Bool {
        _ = cache
        _ = snapshot
        return false
    }
}
