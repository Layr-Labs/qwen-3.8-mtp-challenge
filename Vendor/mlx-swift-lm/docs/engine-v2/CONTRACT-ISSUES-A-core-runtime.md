# Contract issues — WS-A (core runtime)

Places where `CBv2Contracts.swift` (frozen) was ambiguous or insufficient.
In each case we picked the closest conforming shape and continued; no
contract types were changed.

## 1. Windowed `update` return length vs `retainedCount`

`CBv2SequenceKV.update` documents the returned views as
`[1, kvHeads, retainedAfterUpdate, headDim]`, and `retainedCount` as
"physically retained (≤ absoluteOffset for windowed caches)". For a
MULTI-TOKEN update (prefill chunk of `n`) on a windowed layer these two
cannot both hold: exact sliding-window semantics require the FIRST chunk
token to attend to the `window - 1` tokens before it, which are evicted from
the ring by the chunk's own writes. Returning only the post-eviction ring
(`min(offset+n, window)` entries) would under-attend catastrophically (e.g.
window=512, second 512-token chunk: the chunk's first token would see only
itself).

**Resolution:** `CBv2WindowedSequenceKV.update` returns
`min(retainedBefore, window-1)` history entries plus the `n` new tokens
(i.e. up to `window - 1 + n`, matching `RotatingKVCache`'s temporary growth
and vLLM's "free blocks behind `computed - window + 1`" semantics). We read
the contract's "views suitable for attention" as the binding clause.
`retainedCount` still reports the physically retained post-eviction count
(≤ window). For `n == 1` (decode) the two definitions coincide exactly.

## 2. Prefill mask on windowed layers

The WS-A spec sketch says B==1 prefill uses `.causal`. A plain causal mask
over the return shape of issue 1 would let LATE chunk tokens over-attend
past their window. `CBv2AttentionV1.maskMode` therefore returns a
causal∧window ARRAY mask exactly when `returnedKVLength > window`, else
`.causal`. This stays numerically pinned: the mode is a pure function of
`(L, kL, window)` — request-local lengths — so it is batch-composition
invariant and cannot drift for a given request's step sequence
(report 10 §4 invariant 5).

## 3. `makeSequenceState(adopting:)` leaves windowed rows' start offset implicit

The contract says windowed layers always receive `nil` prefix snapshots and
"are recomputed by the scheduler", but does not say what `absoluteOffset` a
fresh windowed row should start at. If they started at 0, absolute RoPE
positions would disagree with the adopted full-attention layers.

**Resolution:** windowed rows are created with
`initialOffset = max(0, matched - window)` (where `matched` is the uniform
offset of the donated full-attention snapshots), so the scheduler can roll
back full layers by `min(matched, window)` tokens and re-run the trailing
tokens through ALL layers: full layers re-append back to `matched`, windowed
layers fill to `matched`, and every layer's absolute positions line up.
`CBv2WindowedSequenceKV.init(window:kvHeads:headDim:initialOffset:)` exposes
this.

## 4. Rollback past a wrapped window destroys the oldest in-window entries

`rollback(n)` on a wrapped ring cannot restore the `n` oldest in-window
entries — the speculative tokens' writes destroyed them (slot aliasing at
distance `window`). The contract only demands the un-confirmed tail be
unreachable (it is, structurally). We additionally shrink `retainedCount` to
`window - n` (monotone `oldestValidPosition`) so destroyed history is never
re-exposed as garbage. Callers doing deep speculative rollback on windowed
layers should expect transient window shrinkage, identical to the
`zeroTailPerRow` semantics of the legacy engine but per-request.

## 5. `CBv2KVBackend` admission needs a byte capacity and dtype the contract does not carry

`bytesCapacity` is a protocol requirement but its provenance is unspecified,
and initial-allocation estimates need an element size before the first
`update` reveals the dtype. `CBv2ContiguousBackendConfig(bytesCapacity:
quantization:kvDType:)` carries both; `bytesInUse` remains truthful (sums
actual allocated buffer bytes of live rows, which start at 0 until the first
append because allocation is lazy).
