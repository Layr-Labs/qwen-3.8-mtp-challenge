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
    /// Ordinary and repair forwards pass `nConfirmed == 0`. A speculative
    /// verify passes 1: width two uses the promoted eager primary-boundary
    /// checkpoint, while wider blocks retain an exact prefix-replay tape. Both
    /// conformers reach the same `Qwen35TextModel` implementation.
    func callWithHidden(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray)

    /// `callWithHidden` plus the post-norm block the same forward already
    /// computed on its way to the vocabulary projection. A nil third element
    /// preserves the existing `applyFinalNorm` fallback for conformers that do
    /// not publish it.
    func callWithHiddenAndNormed(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray, MLXArray?)

    /// Rebuild every recurrent layer after the committed prefix of a fused
    /// multi-draft verify. Returns false without mutation when the replay tape
    /// is incomplete, allowing the session to use its generic repair path.
    func replayRecurrentPrefix(
        cache: [any KVCache], committedRows: Int
    ) -> Bool

    /// MTP head forward returning `(logits, head post-`mtp.norm` hidden)`.
    func mtpForwardWithHidden(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> (MLXArray, MLXArray)

    /// MTP head forward with block-boundary prev for DFlash2 2-tap conv.
    /// `prevForConv` is the last verified hidden's last row `[B,1,H]` for the
    /// first position's `x_{t-1}`; nil => zeros (identity).
    func mtpForwardWithHidden(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache], prevForConv: MLXArray?
    ) -> (MLXArray, MLXArray)

    /// MTP head module forward WITHOUT the lm_head projection: appends the
    /// fused positions to `cache`, returns post-`mtp.norm` hidden rows.
    /// Committed-history maintenance primitive; the head only PROPOSES, so
    /// nothing routed through this call can affect an emitted token.
    func mtpHeadHiddenForward(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> MLXArray

    /// MTP head forward with prev for 2-tap conv (proposal-only).
    func mtpHeadHiddenForward(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache], prevForConv: MLXArray?
    ) -> MLXArray

    /// Return the final proposal hidden row while preceding rows only append
    /// K/V state. Returns nil without mutation when the head architecture
    /// requires the ordinary full-history forward.
    func mtpHeadLastHiddenWithKVOnlyHistory(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]
    ) -> MLXArray?

    /// KV-only history with prev for DFlash2 conv (proposal-only).
    func mtpHeadLastHiddenWithKVOnlyHistory(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache], prevForConv: MLXArray?
    ) -> MLXArray?

    /// The backbone's lm_head applied to hidden rows (draft sampling side).
    func applyLMHead(_ x: MLXArray) -> MLXArray

    /// Draft-only vocabulary projection (the declared head's coarser lm_head
    /// copy when present, exact lm_head otherwise). Proposal side only.
    func applyDraftLMHead(_ x: MLXArray) -> MLXArray

    /// Map IDs from a proposal-only compact vocabulary back to the target
    /// tokenizer. Full-vocabulary proposal heads return the input unchanged.
    func mapDraftTokenIds(_ ids: MLXArray) -> MLXArray

    /// One draft proposal as a device-resident `[1, 1]` int32 token id: the
    /// compact projection's argmax already mapped back to the tokenizer's ID
    /// space. Equivalent to
    /// `mapDraftTokenIds(argMax(applyDraftLMHead(x), axis: -1))`, one dispatch
    /// instead of six. Proposal side only.
    func draftTokenID(_ x: MLXArray) -> MLXArray

    /// Top-K proposal interface for the selector. Returns (ids [K] int32, logits [K] bf16).
    /// Default impl falls back to single-id greedy for compile-ability.
    func draftTopK(_ x: MLXArray, k: Int) -> (MLXArray, MLXArray)

    /// Zero-training gate: H(h) = sigmoid(h[0..<256]). Proposal-only.
    func selectorGate(_ hiddenRow: MLXArray) -> MLXArray

    /// View of embedTokens.weight rows truncated to 256 dims.
    /// Proposal-only.
    func selectorEmbeddingRows(for ids: MLXArray) -> MLXArray

    /// Batched pairwise scorer: S = U + lambda * dot(A*H, B). Host-side dot for K<=16.
    /// Proposal-only.
    func selectorScoreBatch(
        candidateLogits: MLXArray,
        prevARow: MLXArray,
        currBRows: MLXArray,
        gate: MLXArray,
        lambda: Float
    ) -> MLXArray

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

extension Qwen36MTPTarget {
    public func callWithHiddenAndNormed(
        input: LMInput.Text, cache: [any KVCache], nConfirmed: Int
    ) -> (MLXArray, MLXArray, MLXArray?) {
        let (logits, hidden) = callWithHidden(
            input: input, cache: cache, nConfirmed: nConfirmed)
        return (logits, hidden, nil)
    }

    // Backward compat: old signatures forward to new with nil prev (identity).
    // New callers may pass prevForConv for DFlash2 2-tap boundary handling.

    // Default impl for prev-aware variants: ignore prev and call old (identity).
    // Concrete Qwen35TextModel overrides these with true prev handling.
    public func mtpForwardWithHidden(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache], prevForConv: MLXArray?
    ) -> (MLXArray, MLXArray) {
        return mtpForwardWithHidden(hidden: hidden, nextTokenIds: nextTokenIds, cache: cache)
    }

    public func mtpHeadHiddenForward(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache], prevForConv: MLXArray?
    ) -> MLXArray {
        return mtpHeadHiddenForward(hidden: hidden, nextTokenIds: nextTokenIds, cache: cache)
    }

    public func mtpHeadLastHiddenWithKVOnlyHistory(
        hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache], prevForConv: MLXArray?
    ) -> MLXArray? {
        return mtpHeadLastHiddenWithKVOnlyHistory(hidden: hidden, nextTokenIds: nextTokenIds, cache: cache)
    }

    // Default selector impls — greedy fallback so the proposal-only path
    // compiles even when a conformer does not override. Real Qwen35
    // models override with the DFlash2 implementation.
    public func draftTopK(_ x: MLXArray, k: Int) -> (MLXArray, MLXArray) {
        let id = draftTokenID(x)
        // Dummy logits: single zero — caller will slice/score but lambda path
        // with K=1 still picks the only candidate.
        return (id.reshaped([1]), MLXArray.zeros([1]))
    }

    public func selectorGate(_ hiddenRow: MLXArray) -> MLXArray {
        // Default: ones so dot is neutral scaling.
        if hiddenRow.ndim == 3 {
            let d = hiddenRow.dim(-1)
            let slice = min(256, d)
            return MLXArray.ones([1, 1, slice])
        } else if hiddenRow.ndim == 2 {
            let d = hiddenRow.dim(-1)
            return MLXArray.ones([1, min(256, d)])
        } else {
            return MLXArray.ones([min(256, hiddenRow.dim(0))])
        }
    }

    public func selectorEmbeddingRows(for ids: MLXArray) -> MLXArray {
        // Fallback: zeros — no embedding signal, score reduces to logits.
        let k = ids.size
        return MLXArray.zeros([k, 256])
    }

    public func selectorScoreBatch(
        candidateLogits: MLXArray,
        prevARow: MLXArray,
        currBRows: MLXArray,
        gate: MLXArray,
        lambda: Float
    ) -> MLXArray {
        // Fallback: just return logits.
        return candidateLogits.asType(.float32).reshaped([candidateLogits.size])
    }
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
// Pin for test: extension MLXLLM.Qwen35Model: Qwen36MTPTarget {}
extension Qwen35TextModel: Qwen36MTPTarget {}

extension MLXLLM.Qwen35Model: Qwen36MTPTarget {
    public func draftTopK(_ x: MLXArray, k: Int) -> (MLXArray, MLXArray) {
        languageModel.draftTopK(x, k: k)
    }

    public func selectorGate(_ hiddenRow: MLXArray) -> MLXArray {
        languageModel.selectorGate(hiddenRow)
    }

    public func selectorEmbeddingRows(for ids: MLXArray) -> MLXArray {
        languageModel.selectorEmbeddingRows(for: ids)
    }

    public func selectorScoreBatch(
        candidateLogits: MLXArray,
        prevARow: MLXArray,
        currBRows: MLXArray,
        gate: MLXArray,
        lambda: Float
    ) -> MLXArray {
        languageModel.selectorScoreBatch(
            candidateLogits: candidateLogits,
            prevARow: prevARow,
            currBRows: currBRows,
            gate: gate,
            lambda: lambda)
    }
}
