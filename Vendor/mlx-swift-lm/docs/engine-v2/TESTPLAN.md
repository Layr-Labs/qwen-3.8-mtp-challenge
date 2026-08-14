# ContinuousBatchingV2 — test plan (WS-G)

Maps every engine invariant to the executable test that pins it. Sources:
report 10 §4 (the 11 invariants), report 12 (critique items 4, 5, 18), and
the WS-G spec (`docs/engine-v2/specs/G-harness.md`). Corpus:
`/Users/gaj/Documents/Builds/d-inference/docs/research/batching-engine-2026-07/`.

All suites live in `Tests/MLXLMTests/CBv2*.swift` and run on
`TinyTestModel` (2-layer: full + sliding-window(16), optional GPT-OSS-style
sinks, seeded random weights — `CBv2Fixtures.swift`). **No model downloads.**

## How to run

```bash
# The whole WS-G gate (mocks/fixtures only):
swift test -j 4 --filter 'CBv2(Invariance|Parity|InvariantSuite)Tests'

# Strict legacy-ENGINE parity (pins the engine to its eager decode path —
# see "Known legacy discrepancy" below):
DARKBLOOM_COMPILED_DECODE=0 swift test -j 4 --filter CBv2ParityTests

# Benchmark (CI-runnable tiny mode; markdown report to stdout / --out):
swift run -c release CBv2Benchmark [--steps N] [--out benchmarks/cbv2-<date>.md]
# Real weights (manual, legacy-engine baseline until EngineV2 merges):
swift run -c release CBv2Benchmark --model /path/to/model --batch 1,2,4 --prompts 64,2048
# Note: executables need mlx.metallib colocated with the binary
# (d-inference: scripts/fetch-metallib.sh; or copy the one inside
# .build/<config>/mlx-swift-lmPackageTests.xctest/Contents/MacOS/).
```

## Batch-composition invariance (report 12 item 5 — THE core contract)

A request's greedy output is token-exact identical regardless of batchmates.

| Scenario | Test |
|---|---|
| Solo == batched with 1 neighbor | `CBv2InvarianceTests.testSoloEqualsBatchWithOneNeighbor` |
| Solo == batched with 2 neighbors | `CBv2InvarianceTests.testSoloEqualsBatchWithTwoNeighbors` |
| Solo == batched with 3 neighbors | `CBv2InvarianceTests.testSoloEqualsBatchWithThreeNeighbors` |
| Join mid-stream; neighbors finish/join/leave around the subject | `CBv2InvarianceTests.testJoinMidStreamWithNeighborsFinishingAround` |
| Sliding window crossed mid-decode under batching | `CBv2InvarianceTests.testWindowCrossingUnderBatching` |
| Attention sinks (GPT-OSS shape) under batching | `CBv2InvarianceTests.testInvarianceWithAttentionSinks` |
| Per-request seeds on stochastic sampling | `CBv2InvarianceTests.testPerRequestSeedInvarianceUnderBatching` — `XCTSkip` pending integration: WS-E SamplerV2 |
| 50-event join/leave churn storm, every request vs its solo run | `CBv2InvarianceTests.testChurnStormEveryRequestMatchesSolo` |

All comparisons are token-exact over ≥64 decode steps (churn storm: over
however many tokens each request generated before finish/cancel).

## Report 10 §4 invariants 1–11

| # | Invariant | Test(s) |
|---|---|---|
| 1 | Position invariance under batch composition; offsets snapshotted pre-update; KV-shared layers reuse the source's captured offset | `CBv2InvariantSuiteTests.testInvariant1_PositionCountersUnaffectedByMembership`, `testInvariant1_SharedLayerBorrowsSourceOffsetAndKV`; whole-pipeline: `CBv2ParityTests.testAbsoluteOffsetParityAfterDecode`, `testPositionOffsetsMirrorRowCounters` |
| 2 | Mask/KV congruence per layer type | N/A by construction in v2 (decode is mask-free, per-row); asserted structurally by `CBv2InvariantSuiteTests.testInvariant2_DecodeIsMaskFreeAndPerRow` |
| 3 | Fixed update-order convention (post-update retention; window-1 pre-append on chunks) | `CBv2InvariantSuiteTests.testInvariant3_PostUpdateRetentionConvention` |
| 4 | Padding provably inert / rollback scrubs unconfirmed tail KV | No padding exists in v2; rollback scrubbing (incl. NaN-poisoned tails): `CBv2InvariantSuiteTests.testInvariant4_RollbackScrubsTailState` |
| 5 | One numerically-pinned attention path per (model, phase); mask representation never drifts across steps | `CBv2InvariantSuiteTests.testInvariant5_AttentionPathPinnedPerPhase` (mask-mode log: decode always `.none`, prefill always `.array`) |
| 6 | Sliding window == storage eviction keyed to absolute positions, recent end kept | `CBv2InvariantSuiteTests.testInvariant6_WindowEvictionBoundaries` (window, window±1, multi-wrap); whole-pipeline: `CBv2InvarianceTests.testWindowCrossingUnderBatching` |
| 7 | Graph/metadata hygiene: no host syncs from the KV/attention layer in the decode hot path | `CBv2InvariantSuiteTests.testInvariant7_NoHostSyncsInDecodeHotPath` (exactly one boundary sync per step — the sampled-token readback) |
| 8 | Explicit state sync points; snapshots in temporal order; windowed caches never coerced into full-history prefixes | `CBv2InvariantSuiteTests.testInvariant8_SnapshotTemporalOrderAndWindowedPrefixExclusion` |
| 9 | Sinks are a first-class attention capability: static backend eligibility + real numerical effect | `CBv2InvariantSuiteTests.testInvariant9_SinklessBackendStaticallyIneligible`, `testInvariant9_SinksAffectAttentionOutput`; whole-pipeline: `CBv2InvarianceTests.testInvarianceWithAttentionSinks`, `CBv2ParityTests.testGreedyParityWithSinks` |
| 10 | B=1 / batched / compiled semantic parity as a test contract | `CBv2ParityTests` (all greedy parity cases below); cross-ref guard: `CBv2InvariantSuiteTests.testInvariant10_ParitySuitesExist` |
| 11 | Model structure is config-driven data (`CBv2LayerKind`), validated by the cache/attention layer | Encoded in the fixture (`TinyTestModel.layerKinds`) and consumed by every mock; validation preconditions exercised throughout the suites (sinks declaration, shared-layer storage exclusion in `HarnessKVBackend`) |

## v2 vs legacy parity at B=1 (report 12 item 4 — token-exact, no fp tolerance)

| Case | Test |
|---|---|
| Prompt < window (window crossed mid-decode) | `CBv2ParityTests.testGreedyParityShortPrompt` |
| Prompt > window (windowed multi-chunk prefill) | `CBv2ParityTests.testGreedyParityPromptLongerThanWindow` |
| Multi-chunk, multi-window prompt | `CBv2ParityTests.testGreedyParityMultiChunkPrompt` |
| Attention sinks | `CBv2ParityTests.testGreedyParityWithSinks` |
| RoPE scalar vs array-offset overload parity | `CBv2ParityTests.testRoPEScalarVsArrayOffsetParity` |
| Absolute offsets == legacy positions after prefill + decode | `CBv2ParityTests.testAbsoluteOffsetParityAfterDecode` |

Each greedy case asserts, in order: (1) v2 == legacy EAGER reference
(the Scheduler's exact fp16 cache substitution, driven directly) —
**unconditional**, this is the WS-G gate; (2) v2 == full Scheduler/EngineCore
output — strict whenever the engine is on its eager path.

### Known legacy discrepancy (pre-existing, found by this suite)

The legacy engine's compiled B=1 decode path (on by default) diverges from
the legacy engine's OWN eager path when the prompt already straddles a
sliding-window boundary at compile time. v2 matches eager token-exactly in
every case. When the compiled path is active and diverges, the engine-level
assertion is skipped with a pointer to
`docs/engine-v2/CONTRACT-ISSUES-G-harness.md` item 2; run with
`DARKBLOOM_COMPILED_DECODE=0` for strict engine parity (7/7).

## Benchmarks (report 12 item 18 — the missing perf harness)

`benchmarks/CBv2Benchmark.swift` (executable target `CBv2Benchmark`):

- Matrix: B ∈ {1, 2, 4} × prompt ∈ {64, 2048}; greedy decode, default 128 steps.
- Reported per cell (markdown table): TPS/request, aggregate TPS, TTFT p50,
  ITL p50/p99, decode-step CPU μs vs GPU μs split at the asyncEval boundary
  (tiny mode).
- Tiny mode (default): v2-style step loop (per-request KV, per-row SDPA,
  [B,1] decode) on the seeded tiny model — CI-runnable, no downloads.
- `--model <path>`: real weights through the LEGACY engine as the baseline;
  integration adds the `CBv2Engine` binding once WS-B merges.

## Integration re-pointing checklist

The suites run against contract-conforming mocks in `CBv2Fixtures.swift`.
At integration:

1. Re-point `CBv2HarnessSteppableModel` conformance at WS-B's real
   `CBv2SteppableModel` (CONTRACT-ISSUES item 1) and retire the mirror.
2. Swap `HarnessKVBackend` / `HarnessLayerCache` /
   `Harness{Full,Windowed}SequenceKV` for WS-A's `ContiguousKVBackend` /
   `LayerCacheV2` / `SequenceKV` implementations (and WS-C's paged backend
   behind the same protocols) in the invariance + invariant suites.
3. Replace `CBv2HarnessEngine` scenarios with `EngineV2.submit()` driving,
   and activate `testPerRequestSeedInvarianceUnderBatching` against WS-E's
   SamplerV2 keyed RNG.
4. Point `CBv2Benchmark` at `CBv2Engine` (add `--engine v2`) and re-baseline.
