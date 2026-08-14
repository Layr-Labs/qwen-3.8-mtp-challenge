# WS-G: Correctness harness + benchmarks (the v2 test contract)

You own the cross-cutting suites that gate integration. Build everything
against the CONTRACT (mocks for anything unmerged); integration re-points
your suites at real implementations.

## Deliverables

1. `Tests/MLXLMTests/CBv2Fixtures.swift` — `TinyTestModel`: a real 2-layer
   transformer (hidden 64, 4 heads, vocab 128, random seeded weights) with
   one full + one sliding-window(16) layer, optional sinks, implementing
   both the legacy path (KVCache) and the v2 path (`CBv2SteppableModel`
   from WS-B's spec / AttendingLayerCache protocols). Deterministic weights
   (fixed seed) so tests are reproducible. NO downloads.
2. `Tests/MLXLMTests/CBv2InvarianceTests.swift` — THE core suite
   (batch-composition invariance, report 12 item 5):
   - Greedy request R decoded alone == R decoded alongside 1/2/3 different
     neighbors == R decoded joining mid-stream and having neighbors finish
     around it. Token-exact over ≥64 steps on TinyTestModel.
   - Same with sliding-window layers crossing the window during the run.
   - Same with per-request seeds on stochastic sampling (when E merges;
     mark pending until then).
   - Join/leave churn storm: 50 random join/leave events; every request's
     output token-exact vs its solo run.
3. `Tests/MLXLMTests/CBv2ParityTests.swift` — v2 vs LEGACY engine (B=1):
   greedy outputs of TinyTestModel through the old `BatchedEngine` B=1 path
   vs v2 B=1 — token-exact (both are unpadded single-request; any
   divergence is a v2 bug). Plus offsets/RoPE position parity assertions.
4. `Tests/MLXLMTests/CBv2InvariantSuiteTests.swift` — executable encodings
   of report 10 §4 invariants 1–10 at the unit level (position invariance,
   mask/KV congruence N/A-by-construction assertions, rollback scrubbing,
   window-eviction boundaries, sink-path eligibility).
5. `benchmarks/CBv2Benchmark.swift` (follow the existing benchmarks/
   layout): decode TPS and TTFT at B ∈ {1,2,4}, prompt mixes {short 64,
   long 2048}, on TinyTestModel (CI-runnable) AND a `--model <path>` mode
   for real weights (Gemma-4 / GPT-OSS) for manual runs. Reports a
   markdown table: TPS/request, aggregate TPS, p50/p99 ITL, step CPU μs
   vs GPU μs (measure via CFAbsoluteTime around asyncEval boundaries).
6. A `docs/engine-v2/TESTPLAN.md` mapping every invariant → test name.

## Acceptance
All suites compile and run green with mocks/fixtures alone (`swift test
--filter CBv2`). Suites that need unmerged workstreams: implement against
protocols + mark with `XCTSkip("pending integration: WS-X")` so intent is
executable — integration flips them on.

## References
Report 10 §4 (the 11 invariants — your primary source), report 12
(critique items 4, 5, 18), plan §7 Phase 0.4. Corpus:
`/Users/gaj/Documents/Builds/d-inference/docs/research/batching-engine-2026-07/`.
