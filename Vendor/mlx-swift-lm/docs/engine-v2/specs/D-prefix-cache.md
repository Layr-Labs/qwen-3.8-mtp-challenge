# WS-D: Prefix cache v2 (zero-copy donation, block-hash keyed)

## Deliverables (`Libraries/MLXLMCommon/ContinuousBatchingV2/`)

1. `BlockHasher.swift` — SHA-256 chain hashing at 256-token blocks:
   `h_i = SHA256(h_{i-1} || tokens[block_i])`, matching the EXISTING
   checkpoint tier scheme (find it in the legacy `ContinuousBatching/
   PrefixCache.swift` and provider `CheckpointCapturePipeline.swift`; keep
   byte-format compatibility so the SSD tier can interop later). Include a
   `cacheSalt` namespace prefix in h_0 (tenant scoping, TB-007).
2. `PrefixCacheV2.swift` — implements `CBv2PrefixCache`:
   - In-memory map: chainHash → entry {per-layer full-attention snapshots
     (windowed layers: nil), tokenCount, lastAccess, bytes, refInUse}.
   - `donate(...)`: zero-copy move of a finished request's
     `CBv2SequenceKV.snapshot()` arrays (called off the engine thread by B's
     loop; you just store). Store the LONGEST whole-block prefix; drop the
     partial tail.
   - `lookup(...)`: longest chain-hash match in whole blocks, capped at
     `tokens.count - 1` (last token always recomputed for logits — vLLM
     rule). Bump LRU; return per-layer snapshots.
   - `evict(toFit:)` LRU by lastAccess; never evict entries currently
     being adopted (simple in-use refcount).
   - Windowed layers policy (Gemma/GPT-OSS): full-attention layers are
     cached; the engine recomputes the trailing `window` tokens for
     windowed layers on adoption. Expose `requiredRecompute(layerKinds:,
     matched:) -> Int` = max window among windowed layers (so B's loop
     knows to re-prefill that suffix), 0 for all-full models.
3. Thread-safety: an actor or lock — donate/lookup/evict race-free;
   `bytesInUse` accurate.

## Tests (`Tests/MLXLMTests/CBv2PrefixCacheTests.swift`)
- Chain-hash vectors (fixed tokens → fixed hex) incl. salt separation.
- Donate → lookup roundtrip: matched count in whole blocks; last-token cap;
  longest-match wins among nested prefixes; miss on divergence mid-block.
- LRU eviction respects in-use entries and byte budget.
- Windowed policy: requiredRecompute correct for Gemma-4-like (window 512)
  and GPT-OSS-like (window 128) layerKind sets; all-full model → 0.
- Mock `CBv2SequenceKV` (tiny arrays) — do NOT depend on WS-A internals.

## References
Report 08 §2 (vLLM block hashing/eviction rules), report 09 §2 (SGLang
radix — we deliberately use fixed blocks, not a radix tree), plan §7
Phase 4, report 10 invariant 6 (windowed layers must not enter full-history
prefix reuse). Corpus: `/Users/gaj/Documents/Builds/d-inference/docs/research/batching-engine-2026-07/`.
