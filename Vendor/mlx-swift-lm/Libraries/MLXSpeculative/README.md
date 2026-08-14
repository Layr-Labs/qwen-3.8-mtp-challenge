# MLXSpeculative

Speculative-decoding drafters and round-loops for `mlx-swift-lm`.

v1 ships Google's Gemma 4 Multi-Token Prediction (MTP) "assistant"
drafters, published at
`mlx-community/gemma-4-{E2B,E4B,26B-A4B,31B}-it-assistant-bf16`.

This branch also includes an experimental DFlash path for upstream-style
`z-lab/*-DFlash` drafters. The API is intentionally narrow while it is
being hardened: single-request greedy decoding, explicit drafter loading,
and target models that conform to `DFlashTargetModel`.

## What is MTP

MTP is a tightly-coupled speculative-decoding scheme:

- The drafter is a small model (4 kv-shared decoder layers) that borrows
  the target's input embeddings and last-layer hidden state. It has no
  KV cache of its own; every drafter layer reads K/V directly from the
  target's last non-shared full- and sliding-attention layers.
- On each round, the drafter generates `blockSize - 1` candidate tokens
  autoregressively from `(bonusToken, lastHidden)`, holding RoPE
  positions constant at the bonus token's absolute position.
- The target verifies `[bonus, c_0, ..., c_{K-1}]` in a single forward
  pass and emits its greedy prediction at each position. The drafter's
  candidates are accepted up to the first mismatch; the target's
  prediction at the mismatch position becomes the next bonus.
- At greedy (`temperature=0`) this produces output that is
  token-identical to running the target alone, with speedup proportional
  to the drafter's accept rate.

## Public API

### Single-request generation (B=1)

```swift
import MLXSpeculative

let drafter = try Gemma4AssistantDraftModel.load(
    from: assistantModelDirectory)
var parameters = GenerateParameters(maxTokens: 512, temperature: 0.8)
parameters.topP = 0.9
parameters.topK = 50
let stream = try generateGemma4MTP(
    input: lmInput,
    parameters: parameters,
    target: targetModelContext,     // ModelContext from MLXLLM
    drafter: drafter,
    rngSeed: 123
)
for try await generation in stream {
    switch generation {
    case .chunk(let text): print(text, terminator: "")
    case .info(let info): print("\n[tokens: \(info.tokensPerSecond) tok/s]")
    }
}
```

### DFlash generation (experimental)

DFlash drafters condition on selected target hidden states from multiple
layers and draft a masked block in one forward pass. The target verifies
the block, accepts the matching prefix, and rewinds rejected target-cache
state. The Swift port keeps the integration hidden behind an explicit API
until checkpoint compatibility is proven on real model pairs.

Supported target conformers in this branch:

- `Qwen3Model`
- `Qwen3MoEModel`
- `Qwen3NextModel`
- `Qwen35TextModel` / `Qwen35Model` / `Qwen35MoEModel`
- `Gemma4TextModel` / `Gemma4Model`
- `GPTOSSModel`

```swift
import MLXLLM
import MLXSpeculative

let target = modelContext                 // ModelContext from MLXLLM
let targetModel = target.model as! any DFlashTargetModel

let drafter = try await DFlashDraftModel.load(
    from: drafterDirectory,
    bindTo: targetModel
)

let stream = try generateDFlash(
    input: lmInput,
    parameters: GenerateParameters(maxTokens: 512, temperature: 0),
    target: target,
    drafter: drafter
)

for await generation in stream {
    switch generation {
    case .chunk(let text): print(text, terminator: "")
    case .info(let info): print("\n[tokens: \(info.tokensPerSecond) tok/s]")
    case .toolCall: break
    }
}
```

When `blockSize` is omitted, DFlash uses the configuration's recommended
block size. This is usually the checkpoint block size, clamped only by a
shorter sliding-window constraint. Passing `blockSize:` remains an explicit
override.

For benchmark parity with the upstream z-lab MLX backend, set
`MLX_GEMMA4_DFLASH_OFFICIAL_FAST=1`. This uses one vector target verify
per block and accepts the matching target-token prefix, matching the
throughput contract used by the Python MLX benchmark. It does not promise
an exact generated-token hash match against sequential baseline decoding.
Leave it unset when running strict parity diagnostics.

Single-request DFlash uses a CPU accept walk by default, matching the
upstream MLX implementation's small-list token comparison and avoiding extra
GPU work for the 15-token prefix check. Set `MLX_DFLASH_CPU_ACCEPT_WALK=0`
to force the older MLX `cumprod`/`sum` accept path for comparison.

The DFlash drafter fuses its MLP gate/up projection by default to reduce
per-round matmul launches. Set `MLX_DFLASH_DRAFTER_FUSE_MLP_GATE_UP=0` if
memory pressure or benchmark results favor the separate projections on a
specific checkpoint.

Gemma4 target expert gate/up fusion is on by default for the general routed
path. The DFlash weighted expert path defaults to separate expert gate/up
projections because that schedules faster on the 26B A4B DFlash benchmark.
Set `MLX_SWITCH_GLU_GEMMA4_WEIGHTED_FUSE_GATE_UP=1` to force the fused
weighted path, or set `MLX_SWITCH_GLU_GEMMA4_FUSE_GATE_UP=0` to force separate
projections everywhere for comparison.

The optimized Gemma4 verify fusion paths are capped at the K=16 shape by
default. For block-size experiments above 16, set
`MLX_GEMMA4_DFLASH_VERIFY_FUSION_MAX_ROWS=24` and
`MLX_SWITCH_GLU_GEMMA4_WEIGHTED_MAX_ROWS=24` to let the fused attention/MLP
and weighted expert paths run on larger verify blocks.

Router-only verify QMM can be enabled with
`--verify-qmm --verify-qmm-include router` for a speed-oriented benchmark
mode. It changes floating-point routing decisions enough to alter token
hashes and EOS behavior, so it remains opt-in rather than the default
quality path.

`DFlashDraftModel.load(from:downloader:id:bindTo:)` is available for Hub
downloads. It fetches only `config.json` and `*.safetensors`; tokenizers
and prompt formatting still come from the target `ModelContext`.

Target integration is split into two surfaces. `DFlashTargetModel` is the
minimal hidden-state capture and LM-head API that new target models need.
`DFlashTargetCacheRollbackProvider` is optional and exists for hybrid or
model-specific cache rollback without putting those cache details in the
generic DFlash generation loop.

Current DFlash limitations:

- Greedy only: `GenerateParameters.temperature` must be `0`.
- B=1 only.
- Checkpoint weights must use the upstream DFlash MLX naming shape
  (`fc`, `hidden_norm`, `layers`, `norm`) and must match the target
  hidden size, vocab size, and selected layer ids.
- Hybrid Qwen targets with `MambaCache` use snapshot plus accepted-prefix
  replay for rejected-token rollback. This is correct but not yet the
  optimized gated-delta rollback path used by upstream Python.

Real checkpoint smoke coverage is opt-in. Set:

```sh
MLX_SWIFT_LM_DFLASH_TARGET_DIR=/path/to/target
MLX_SWIFT_LM_DFLASH_DRAFTER_DIR=/path/to/z-lab-dflash-drafter
MLX_SWIFT_LM_DFLASH_PROMPT_TOKENS=1,2,3   # optional
swift test --filter DFlashRealCheckpointSmokeTests
```

The test loads the target with `LLMModelFactory`, loads and binds the
DFlash drafter, then runs a two-token greedy `generateDFlashTokens`
smoke pass.

### Batched (B>1) generation

Continuous-batching MTP round loop, one `BatchedGeneration` per
generated step with per-row slot state:

```swift
let stream = try runGemma4MTPRoundsBatched(
    target: targetModel,
    drafter: drafter,
    targetCache: prefilledBatchCache,   // BatchKVCache / BatchRotatingKVCache
    firstBonus: bonusPerRow,            // [Int], length B
    firstHidden: lastHiddenBxLxH,
    firstSharedKV: capturedSharedKV,
    maxTokens: 256,
    blockSize: 4,
    eosTokenIds: [1, 106]
)
for await step in stream {
    for slot in step.slots where slot.token != nil {
        handleToken(row: slot.row, token: slot.token!)
    }
}
```

### Benchmark primitives

Two measurement helpers for wall-clock comparisons. Callers supply a
loaded target + bound drafter; the helpers return timings + accept
histogram. No model loading or CLI in the library.

```swift
let baseline = measureBaselineThroughput(
    target: target, promptTokens: prompt, maxTokens: 256)
let mtp = try measureMTPThroughput(
    target: target, drafter: drafter,
    promptTokens: prompt, maxTokens: 256, blockSize: 4)
print("MTP speedup: \(mtp.tokensPerSecond / baseline.tokensPerSecond)x")
print("Accept histogram: \(mtp.acceptLengths ?? [])")
```

## Measured throughput (M3, 36 GB)

Single-batch greedy, 3 chat-templated prompts, 64 max_tokens, 16-token
warmup. Block size swept per model; speedups are shown as observed
ranges because laptop load can move short local benchmark runs by a few
percentage points.

| Model           | Best K | Baseline tok/s | MTP tok/s | Speedup range |
|-----------------|--------|----------------|-----------|---------------|
| E2B-bf16        | 5      | 22.0-22.2      | 24.3-25.0 | 1.10-1.14×    |
| E2B-4bit        | 3      | 49.7           | 62.3      | 1.25×         |
| E4B-bf16        | 5      | 11.8-12.0      | 12.6-12.9 | 1.06-1.09×    |
| 26B-A4B-4bit    | 3      | 29.1-29.4      | 36.9-37.2 | 1.26-1.28×    |

M5 Max 128 GB validation used the same prompt fixture and E2B 4-bit
target + 160 MB bf16 assistant pair:

| Model    | Max tokens | Best K | Baseline tok/s | MTP tok/s | Speedup |
|----------|------------|--------|----------------|-----------|---------|
| E2B-4bit | 12         | 3      | 94.7-95.0      | 136.0-137.8 | 1.43-1.46× |
| E2B-4bit | 64         | 3      | 88.8           | 107.0     | 1.20×   |

Block-size tuning is model- and quantization-dependent. E2B / E4B bf16
should sweep K in the 4-5 range, E2B 4-bit should start at K=3, and
26B-A4B should start at K=3: larger K increases accepted tokens per
round but was slower on M3 because drafter overhead and the larger verify
block outweighed the extra accepts. On an upstream-style short run
(`E2B-4bit`, `max_tokens=12`, K=4, B=1, 160 MB bf16 assistant), this
implementation measured `1.43-1.47x` on the same M3; K=5 dropped to
`1.22x` at `max_tokens=12` and `0.97x` at `max_tokens=64`. Sweeping K
showed K=3 was faster than K=4 for E2B-4bit on both M3 (`1.52x` vs
`1.44x` at 12 tokens) and M5 Max (`1.46x` vs `1.44x` at 12 tokens,
`1.20x` vs `1.15x` at 64 tokens).

Batched B=4 needs a separate sweep; the single-batch K does not always
transfer. In the current benchmark, E2B-bf16 does best at K=3
(`1.13x`), E4B-bf16 does best at K=2 (`1.06x`), while E2B-4bit and
26B-A4B-4bit regress in batched mode. On M5 Max, E2B-4bit B=4 was
`0.74x` at its best tested K=2 and `0.64x` at K=3. On M3, 26B-A4B-4bit
still regressed even at its best tested K=2 (`0.75x`). Treat batched
E2B-4bit and batched 26B-A4B MTP as disabled-by-default until the
drafter/verify balance improves.
`Gemma4MTPAutomaticPolicy.automatic(for:)` encodes these defaults: E2B
4-bit uses K=3 and stays on the B=1 path, E2B / E4B bf16 use K=5 for
B=1, E2B / E4B bf16 use the batched round loop for B>1, and 26B-A4B plus
unknown MoE models stay on the B=1 path. The single-request
`generateGemma4MTP` and `Gemma4MTPTokenIterator` APIs use the same policy
when `blockSize` is omitted; passing `blockSize:` remains an explicit
override.

Per-round accept rates improve with target size, as expected — the
drafter's predictions align more closely with a larger target's greedy
output:

| Model           | Accept/K at K=3 | Accept/K at K=4 |
|-----------------|-----------------|-----------------|
| E2B-bf16        | 0.64/2 (32%)    | 0.71/3 (24%)    |
| E4B-bf16        | 0.53/2 (27%)    | 0.61/3 (20%)    |
| 26B-A4B-4bit    | 1.25/2 (62%)    | 1.42/3 (47%)    |

Continuous-batching compaction was also measured with staggered
per-row budgets. The default M3 benchmark runs B=4; larger B is opt-in
because it needs more memory headroom. The path passed the current
unit/parity tests but was not faster per emitted token than padded
generation; the best observed case was near parity on the 26B-A4B 4-bit
target (`0.97x`). The compaction path is kept as infrastructure for
serving workloads and future optimization rather than claimed as a
throughput win today.

Reproduce with the `realModelThroughputBenchmark` test. It requires
`MTP_BENCH_DATA_DIR` and `MTP_BENCH_PROMPTS` env vars pointing to
local model directories and a pre-tokenized prompt fixture JSON.

## Correctness

At `temperature=0`, MTP output is bitwise equal to target-only greedy
output, with two caveats driven by bf16 numerics rather than the
algorithm:

1. **Near-uniform logit tails.** When the target starts repeating a
   single token (post-EOS padding, or degenerate prompts at small
   random-weight shapes), its top-1 and top-2 logits can be within bf16
   precision. In that regime the multi-position verify forward's argmax
   occasionally flips vs a serial single-position forward at the same
   position — both continuations are mathematically valid greedy
   choices, but not bitwise identical. The Python mlx-vlm reference
   exhibits the same behavior at 8/20 of our chat-templated prompts,
   so the parity gate is "Swift MTP == Swift baseline" rather than
   "Swift MTP == Python MTP".

2. **Batched attention.** Small-target batched (B>1) runs can diverge
   from B=1 baselines on the same prompt for the same reason — bf16
   scaled-dot-product attention's argmax is not bitwise stable across
   batch shapes at near-uniform logits.

Parity is enforced as a hard gate internally (Swift MTP tokens == Swift
baseline tokens) across the E2B, E4B, 26B-A4B MoE, iterator, stochastic,
and batched round-loop suites. The throughput numbers above were taken
on the M3 36 GB development machine used for those parity runs.

## Architecture notes

- **Shared-KV capture** (`Gemma4TextModel.forwardForMTP`): the target
  forward exposes a snapshot of the last non-shared full- and
  sliding-attention K/V tensors via `Gemma4SharedKVCapture`. The drafter
  consumes them directly; it never allocates KV of its own.
- **Pre-norm hidden**: the drafter's `pre_projection` is trained against
  the LAST decoder-layer output BEFORE `model.norm`. `forwardForMTP`
  returns this pre-norm hidden alongside post-norm logits so the
  drafter gets what it was trained against.
- **Per-row cache rewind**: B>1 rewind uses
  `BatchKVCache.zeroTailPerRow(keepLengths:)` plus
  `Gemma4SharedKV.zeroTailPerRow(from:keepLengths:)` to clear rejected
  tails when rows accept different numbers of drafter tokens. This path
  is covered by parity tests, but remains the main place to harden if
  B>1 serving becomes a priority: fully right-aligned per-row masking
  for all target cache types and shared-KV masks would remove the
  remaining dependence on zeroed tail slots.
- **MaskedEmbedder** (centroid-routed sparse LM head): used by E2B / E4B
  drafters (`use_ordered_embeddings=true`). Scores 2048 token clusters,
  materialises the top-K clusters' tokens (default 32 of 2048) and
  scatters logits back into the full vocab with sentinel values
  elsewhere. The sentinel is scalar-filled because the GPU-only broadcast
  variant regressed E2B B=1 throughput on M3.

## Model / drafter pairing

| Target | Drafter |
|---|---|
| `mlx-community/gemma-4-E2B-it-{bf16,4bit}` | `mlx-community/gemma-4-E2B-it-assistant-bf16` |
| `mlx-community/gemma-4-E4B-it-{bf16,4bit}` | `mlx-community/gemma-4-E4B-it-assistant-bf16` |
| `mlx-community/gemma-4-26B-A4B-it-{bf16,4bit}` | `mlx-community/gemma-4-26B-A4B-it-assistant-bf16` |
| `mlx-community/gemma-4-31B-it-{bf16,4bit}` | `mlx-community/gemma-4-31B-it-assistant-bf16` |

Drafter attention head dims (`head_dim`, `global_head_dim`) must match
the target for `scaled_dot_product_attention` to accept shared K/V —
real drafters are published this way; the pairing checks in
`Gemma4AssistantDraftModel.bind(target:)` enforce it at runtime.

## Known limitations

- B=1 supports greedy and stochastic sampling. Stochastic sampling uses
  rejection-based speculative sampling and honors `temperature`,
  `topP`, `topK`, and `minP`.
- B>1 currently supports greedy sampling only. Stochastic batched MTP is
  a follow-up.
- B>1 currently exposes the raw `runGemma4MTPRoundsBatched` round loop,
  not a tokenizer-decoding `generateGemma4MTPBatched` convenience API.
- Text-only. Multimodal prefill (images / audio) runs through the target
  unchanged; MTP kicks in on the text-decode tail.
- Drafter weights must match the target's input embedding + attention
  head shapes. Cross-family pairings (e.g. E2B drafter with 26B-A4B
  target) are rejected by `bind(target:)`.
