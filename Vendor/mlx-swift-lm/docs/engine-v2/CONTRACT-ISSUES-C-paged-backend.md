# Contract issues — WS-C (paged backend)

Places where `CBv2Contracts.swift` (frozen) was insufficient for the paged
backend, the conforming shape chosen, and what integration should decide.

## 1. Windowed-layer prefix adoption has no way to place absolute positions

`CBv2KVBackend.makeSequenceState(adopting:layerKinds:maxLength:)` documents
that windowed layers receive `nil` snapshots and "are recomputed by the
scheduler". But a freshly created `CBv2SequenceKV` starts at
`absoluteOffset == 0`, and the contract exposes no way to advance the
counter without writing storage. The recomputed trailing-window tokens for
a windowed layer must land at their TRUE absolute positions (RoPE offsets
and window clamping are keyed to absolute positions), so writing them at
offset 0 would be silently wrong whenever full-attention layers adopt a
nonzero prefix.

**Chosen shape:** `PagedSequenceKV.fastForward(to:)` — a public method
OUTSIDE the contract, valid only on a fresh windowed state, that sets
`absoluteOffset = baseOffset = offset` without allocating pages. The
scheduler (WS-B) must downcast (or the integration branch adds
`fastForward` to `CBv2SequenceKV`) and call it with the adopted prefix
length before replaying the trailing window tokens.

**Integration ask:** add `fastForward(to:)` (or an
`adoptionOffset:` parameter on the adopting factory) to the contract.

## 2. Attention-logit softcap is not representable in `updateAndAttend`

Gemma-2-style attn-logit softcapping changes the attention math
(`cap * tanh(qk / cap)` before softmax), but
`CBv2AttendingLayerCache.updateAndAttend` only carries
`(queries, keys, values, scale, sinks)`. Sinks got a parameter; softcap did
not.

**Chosen shape:** softcap is per-layer-cache CONFIGURATION —
`PagedLayerCache.init(..., attentionSoftcap:)`, plumbed from model config
by whoever builds the caches (`PagedKVBackend.makeLayerCaches(
attentionSoftcap:)`). This is workable because softcap is a static model
property, never per-request. Prefill for softcap configs uses the composed
reference path (SDPA cannot express softcapping); decode folds it into the
kernel (`HAS_SOFTCAP`).

**Integration ask:** none required; document the construction-time plumbing
in the model-adaptation spec (WS-F) so both backends receive it the same
way.

## 3. No contract surface for binding rows to a layer cache

`CBv2AttendingLayerCache` exposes `rows: [CBv2SequenceKV] { get }` but no
setter — the contract does not say how the engine loop tells a layer cache
which sequences (in which order) form the current batch.

**Chosen shape:** `PagedLayerCache.setRows(_:)` (O(B); join = append row
object, leave = drop). Rows must be `PagedSequenceKV` from the same
backend; foreign rows trap. WS-A presumably needs the same method on its
layer cache.

**Integration ask:** add `setRows(_:)` (or an initializer taking rows) to
`CBv2AttendingLayerCache`.

## 4. `CBv2SequenceKV.update` cannot throw, but paged allocation can fail

Page allocation is fallible in general, but the contract makes `update`
non-throwing (correctly — capacity failure mid-decode is unrecoverable).

**Chosen shape:** reservation-based admission. `makeSequenceState` reserves
the WORST-CASE page count for the request's `maxLength` (windowed layers
capped at their ring size) and throws `capacityExhausted` up front; pages
materialize lazily so `bytesInUse` stays truthful, and `update` can then
never fail by construction. Consequence: admission is by `bytesReserved`
(exposed on `PagedKVBackend`), not `bytesInUse`; the contract's
`CBv2KVBackend` only exposes `bytesInUse`/`bytesCapacity`, so a scheduler
admitting on those alone would over-admit.

**Integration ask:** consider adding `bytesReserved` to `CBv2KVBackend`
(harmless for the contiguous backend: reserved == in use).

## 5. KV-shared layers and `positionOffsets`

A KV-shared layer cache owns no rows (`setRows` demands empty), so its
`positionOffsets` is an empty array; models must read `positionOffsets`
from the SOURCE layer's cache when computing RoPE for a shared layer. This
is implied but not stated by the contract.

**Integration ask:** document it; no signature change needed.
