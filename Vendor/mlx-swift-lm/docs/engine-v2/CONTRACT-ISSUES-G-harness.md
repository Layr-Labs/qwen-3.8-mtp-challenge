# Contract issues — WS-G (correctness harness + benchmarks)

Per the integration protocol in `ARCHITECTURE.md`: where the frozen contract
(`CBv2Contracts.swift`) was insufficient, WS-G picked the closest conforming
shape and recorded the gap here. Integration resolves these.

## 1. No steppable-model protocol in the frozen contract

**Gap.** The contract defines the KV/attention surface
(`CBv2AttendingLayerCache`, `CBv2SequenceKV`, `CBv2KVBackend`) and the engine
surface (`CBv2Engine`), but not the protocol a MODEL conforms to so an engine
can drive it — "forward these `[B, L]` tokens through these per-layer
attending caches, give me `[B, L, vocab]` logits". That protocol
(`CBv2SteppableModel`) is WS-B's deliverable (`EngineLoopV2.swift`), unmerged
while WS-G builds.

**Shape chosen.** Test-local mirror in `Tests/MLXLMTests/CBv2Fixtures.swift`:

```swift
protocol CBv2HarnessSteppableModel: AnyObject {
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray
}
```

`TinyTestModel` conforms. At integration, re-point the conformance at WS-B's
real protocol (mechanical: same shape) and retire the mirror.

## 2. Legacy compiled-decode B=1 diverges from the legacy eager path (pre-existing; found by the parity suite)

**Finding, not a contract gap** — recorded here because it defines how the
parity gate must be run.

The spec for `CBv2ParityTests` says "v2 vs the old engine at B=1: any
divergence is a v2 bug". Building that suite exposed a divergence INSIDE the
legacy engine: with compiled decode enabled (the default —
`DARKBLOOM_COMPILED_DECODE` unset ⇒ ON), the Scheduler/EngineCore B=1 output
on `TinyTestModel` differs from the same engine's eager output whenever the
prompt is already past a sliding-window boundary when compiled decode is set
up (prompt length > windowSize at the prefill→decode transition):

- prompt 8 (< window 16): compiled == eager == v2 — token-exact.
- prompt 24 / 50 (> window 16): compiled diverges from eager (an extra
  repeated token early, different window contents after the next window
  crossing). Eager == v2 remains token-exact, and a direct drive of the
  engine's own cache substitution (`BatchKVCache` +
  `BatchRotatingKVCache(maxSize:)`, model-built masks) reproduces the EAGER
  output exactly — so the delta is introduced by the compiled path's
  cache promotion (`extractBatched(0)` → `Compilable*KVCache` ring seeding),
  not by v2 and not by the batched cache substitution.

Evidence: `swift test --filter CBv2ParityTests` — 3 failures with default
env, 7/7 pass with `DARKBLOOM_COMPILED_DECODE=0` (worktree
`engine-v2-g-harness`, TinyTestModel seed 0xC0FFEE, window 16, prefill chunk
16).

**Consequences for the harness (implemented):**

- v2 is asserted token-exact against the legacy EAGER reference
  unconditionally — that assertion is the WS-G gate and never skips.
- The engine-level assertion runs strictly whenever the engine output matches
  its own eager path; when the compiled path is active AND diverges, the
  engine-level assertion is skipped with a pointer to this item. Strict
  engine-level parity is available via `DARKBLOOM_COMPILED_DECODE=0 swift
  test --filter CBv2ParityTests`.
- The suite must NOT setenv/flip `DARKBLOOM_COMPILED_DECODE` in-process:
  `CompiledDecode.isEnabled` is a lazily-frozen process-global, and mutating
  it from a test would silently disable compiled-decode coverage for every
  legacy suite running later in the same test process.

**Follow-up owner.** The legacy compiled-decode path (d-inference provider /
`Libraries/MLXLMCommon/CompiledDecode.swift` +
`Compilable*KVCache`) — out of WS-G scope (`ContinuousBatching/` and the
compiled-decode files are frozen for feature worktrees). Suggested repro for
that team: `CBv2ParityTests.testGreedyParityPromptLongerThanWindow` with
compiled decode on vs off.
