# WS-A: Core runtime — per-sequence KV + v1 attention backend

You own the heart of v2: per-request KV storage and the v1 attention dispatch.

## Deliverables (`Libraries/MLXLMCommon/ContinuousBatchingV2/`)

1. `SequenceKV/FullSequenceKV.swift` — `CBv2SequenceKV` for full attention.
   Contiguous `[1, kvHeads, capacity, headDim]` buffer per layer, grown by
   doubling (initial capacity = promptLen + 256, cap at maxLength), appended
   via `array[..., offset..<offset+n, :] = k` slice-assignment (donation makes
   this O(n)). `update` returns temporal-order views `[..., 0..<retained, :]`
   (zero-copy strided slices — MLX SDPA accepts strided K/V).
2. `SequenceKV/WindowedSequenceKV.swift` — sliding window. Ring buffer of
   exactly `window` slots; eviction keyed to ABSOLUTE position; `update`
   returns the retained window **in temporal order** (unwind the ring with at
   most one concat of two slices; for the common decode case where the ring
   hasn't wrapped, it's one slice). Keep the RECENT end. `rollback` must
   handle un-wrapping. Absolute offset keeps counting past the window.
3. `SequenceKV/QuantizedSequenceKV.swift` — wraps Full with mlx quantize
   (group 64, 4/8-bit) storing quantized triples; `update` returns
   dequantized views for v1 attention (per-request, so the dequant cost is
   O(that sequence's retained KV), not O(whole batch)). Follow the existing
   `QuantizedKVCache` scheme for compatibility.
4. `SequenceKV/ContiguousKVBackend.swift` — `CBv2KVBackend` producing the
   above; truthful `bytesInUse` accounting; `release` drops references;
   `makeSequenceState(adopting:)` copies donated prefix arrays into fresh
   sequence state (bounded one-time copy; paged backend later makes it free).
5. `LayerCacheV2.swift` — `CBv2AttendingLayerCache` + conforms to the legacy
   `KVCache` protocol (offset = max row offset; `update(keys:values:)` traps —
   v2 models must call `updateAndAttend`). Holds ordered `rows:
   [CBv2SequenceKV]`, maintains `positionOffsets` as a cached `MLXArray [B]`
   updated ONLY on membership change (never `.item()` in the step loop).
6. `AttentionV1.swift` — the v1 dispatch used by `updateAndAttend`:
   - B==1: single `MLXFast.scaledDotProductAttention` (mask: `.none` for L==1,
     `.causal` for L>1), sinks passed through when `kind.hasSinks`.
   - B>1 (decode, L==1): split queries per row (`queries[i..<i+1]`), per-row
     update+SDPA against that row's KV, `concatenated(axis: 0)`. No masks —
     each row sees exactly its own KV. NaN by construction impossible.
   - `attendBorrowing` for KV-shared layers: attend against `source`'s rows
     WITHOUT updating (Gemma-4 shared layers project Q only).
   - Numerical pinning: the mask mode for a given (phase, B) is a pure
     function of (L, retained) — never data-dependent.
7. EXCLUSIVE: add the v2 hook at the TOP of `attentionWithCacheUpdate` in
   `Libraries/MLXLMCommon/AttentionUtils.swift`:
   `if let v2 = cache as? CBv2AttendingLayerCache { return v2.updateAndAttend(...) }`
   (sinks: nil on this generic path). Nothing else in that file changes.

## Tests (`Tests/MLXLMTests/CBv2CoreTests.swift`)
- Full/Windowed/Quantized: append/rollback/snapshot round-trips; windowed
  eviction at exact boundaries (window, window±1, wrap, multi-chunk cross);
  temporal-order guarantee after wrap.
- AttentionV1 parity: B=3 mixed lengths {7, 130, 950} per-row output ==
  running each row alone (rtol 1e-4 fp16), with and without window, with
  sinks (synthesize per-head sink values), GQA (kvHeads < queryHeads).
- positionOffsets correctness under join/leave churn.
- No-`.item()`-in-decode: instrument that a 10-step decode performs zero
  host syncs from LayerCacheV2 (e.g., assert via a counter you add around
  your own sync points; do not modify MLX).

## References
Report 10 §4 invariants 1–6, 9 (`/Users/gaj/Documents/Builds/d-inference/docs/research/batching-engine-2026-07/10-gemma4-gptoss-model-study.md`), report 04 (MLX primitives: donation, strided SDPA views, RoPE array offsets).
