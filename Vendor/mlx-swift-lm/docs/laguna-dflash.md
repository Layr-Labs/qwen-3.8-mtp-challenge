# Laguna XS 2.1 DFlash draft model

This document is operator-facing: what the converted artifact is, how to
rebuild it, and how to run it through `mlx-bench dflash`.

## What the artifact is

Poolside publishes Laguna XS 2.1's DFlash block-draft speculator as a
`poolside/Laguna-XS-2.1-DFlash-NVFP4` checkpoint. It is a small speculative
decoder: 5 sliding-window decoder layers, block size 16, mask token id 12,
targeting 5 hidden states pulled from target-model layers `[1, 13, 25, 33,
39]`. It borrows its token embedding and LM head from the Laguna target model
via `bind(target:)` at load time rather than carrying its own — the converted
artifact's 68 tensors cover only the drafter's own layers (self-attention,
MLP, per-layer `input_layernorm`, `aux_hidden_norms`, `fc`, `hidden_norm`,
`norm`), not `embed_tokens`/`lm_head`.

`scripts/convert_laguna_dflash.py` converts the source Poolside NVFP4
checkpoint (safetensors) into the MLX Swift `DFlashDraftModel` layout:
`config.json` + `model.safetensors`, bf16 tensors, 68 keys. The converter
validates the source against an exact BF16 manifest — every tensor's name,
dtype, shape, and byte length are checked against the config-derived
expectation and against its own `data_offsets` span — and fails loudly on
any drift rather than silently reinterpreting or truncating a tensor.

The converted `config.json` sets `decoder_layer_type: "laguna_xs"` —
this is the switch, read by `DFlashConfiguration`/`DFlashDraftModel`
(`Libraries/MLXSpeculative/DFlashConfiguration.swift`,
`Libraries/MLXSpeculative/DFlashDraftModel.swift`), that activates the
Laguna-specific decoder-layer behaviors: per-head `g_proj` attention gating,
`aux_hidden_norms` (one RMSNorm per target hidden state, feeding the `fc`
projection that builds the drafter's context vector), and a per-layer
context `input_layernorm` (the context stream is renormalized per layer,
independent of the token stream's `input_layernorm`). Other decoder layer
types (e.g. `qwen3`) skip all three — see
`Tests/MLXLMTests/LagunaDFlashDraftModelTests.swift` for the behavioral
contrast and a reference-forward parity test.

The currently converted artifact lives at
`~/models/laguna-xs-2.1-dflash-mlx` (config.json + model.safetensors, 68
tensors). Its `config.json` carries an `_mlx_conversion` block recording the
exact source revision and source content hash this copy was converted from.

## Rebuilding the artifact

Reproduces the source download and conversion from scratch (Task 1, Step 3
of `docs/superpowers/plans/2026-07-25-laguna-dflash-mlx-conversion.md`):

```bash
mkdir -p ~/models/laguna-dflash-src
curl -sL https://huggingface.co/poolside/Laguna-XS-2.1-DFlash-NVFP4/raw/main/config.json -o ~/models/laguna-dflash-src/config.json
curl -L --progress-bar https://huggingface.co/poolside/Laguna-XS-2.1-DFlash-NVFP4/resolve/main/model.safetensors -o ~/models/laguna-dflash-src/model.safetensors
REV=$(curl -sI https://huggingface.co/poolside/Laguna-XS-2.1-DFlash-NVFP4/resolve/main/config.json | tr -d '\r' | awk -F': ' 'tolower($1)=="x-repo-commit"{print $2}')
python3 scripts/convert_laguna_dflash.py --source ~/models/laguna-dflash-src --output ~/models/laguna-xs-2.1-dflash-mlx --source-revision "$REV"
```

The converter pins `source_revision` (and a source content hash) into the
output `config.json`'s `_mlx_conversion` block so the converted artifact is
traceable back to the exact upstream commit it was built from.

## Running mlx-bench dflash for Laguna

```bash
swift build -c release
scripts/build-mlx-metallib.sh   # required: plain swift build does not produce mlx.metallib
.build/release/mlx-bench dflash \
  --target /path/to/laguna-xs-2.1-nvfp4-mlx \
  --drafter ~/models/laguna-xs-2.1-dflash-mlx
```

**`--target` must be an MLX-format Laguna checkpoint** — a directory with an
MLX-style `quantization` block in `config.json` plus per-tensor `.scales`
tensors in its safetensors (for example, the mlxfast-challenge reference
checkpoint). It must **not** be the raw `poolside/Laguna-XS-2.1-NVFP4`
Hugging Face repo directly: that repo's NVFP4 tensor layout/dtype is the
source format the *target*-model loader does not speak, and is unrelated to
the drafter conversion above (which only touches the DFlash drafter, not the
target). `--drafter` is the artifact produced above.

`--target`/`--drafter` can also be supplied via the environment
(`MLX_SWIFT_LM_DFLASH_TARGET_DIR` / `MLX_SWIFT_LM_DFLASH_DRAFTER_DIR`, or the
`DFLASH_BENCH_`-prefixed equivalents) instead of flags; see
`Sources/mlx-bench/DFlashBenchCommand.swift`.

## Env-gated tests

Two env-gated test suites exercise real, on-disk artifacts instead of the
synthetic tiny-config fixtures used by the rest of the DFlash test suite.
Both are skipped (not failed) when their env var(s) are unset, so `swift
test` / the default `xcodebuild test` run stays hermetic.

- `LagunaDFlashConvertedArtifactTests`
  (`Tests/MLXLMTests/LagunaDFlashDraftModelTests.swift`) — loads the
  converted Laguna drafter artifact with full parameter verification
  (`update(parameters:verify: [.all])`, i.e. every one of the 68 tensors
  must match a module parameter exactly) and checks its config and a few
  parameter shapes/dtypes. Gate: `MLX_SWIFT_LM_LAGUNA_DFLASH_DIR` (absolute
  path to the artifact directory).
- `DFlashRealCheckpointSmokeTests` (`Tests/MLXLMTests/DFlashGenerateTests.swift`)
  — loads a real target + drafter pair and runs a short generation. Gates:
  `MLX_SWIFT_LM_DFLASH_TARGET_DIR` and `MLX_SWIFT_LM_DFLASH_DRAFTER_DIR`
  (both must be set).

Example:

```bash
MLX_SWIFT_LM_LAGUNA_DFLASH_DIR=/absolute/path/to/laguna-xs-2.1-dflash-mlx \
  swift test --filter LagunaDFlashConvertedArtifactTests
```

Note for `xcodebuild test`: unlike `swift test`, `xcodebuild test` does not
forward arbitrary shell environment variables to the test host process.
Prefix the variable with `TEST_RUNNER_` (Xcode's standard mechanism for
passing environment through to a test bundle) when driving these gates
through `xcodebuild`:

```bash
TEST_RUNNER_MLX_SWIFT_LM_LAGUNA_DFLASH_DIR=/Users/you/models/laguna-xs-2.1-dflash-mlx \
  xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' \
  -only-testing:MLXLMTests/LagunaDFlashConvertedArtifactTests
```

Also note `xcodebuild` does not expand `~` in values passed on the command
line — use an absolute path for the directory env vars regardless of which
runner you use.

## Deferred: Hugging Face upload

Publishing the converted artifact to a Hugging Face repo (org/name TBD by
the operator) is a deferred manual step — it is not part of this
conversion pipeline or its test coverage. Until that happens, consumers of
the converted artifact use the local path produced by the conversion
command above.

## Measured results (M5-B, Apple M5 Max 128 GB, 2026-07-26)

Target: the mlxfast-challenge transformed Laguna XS 2.1 NVFP4 reference
checkpoint (`model_type: laguna`, quantization nvfp4 group 16, 5 shards /
20 GB), consumed via a symlink overlay that patches only
`tokenizer_config.json` (`tokenizer_class` → `GPT2Tokenizer`, because
swift-transformers does not map the checkpoint's `TokenizersBackend`; plus
the checkpoint's own `chat_template.jinja` injected as `chat_template`).
Drafter: the converted artifact from this repo's converter. Both loaded by
`.build/release/mlx-bench dflash` after `swift build -c release` and
`MLX_SWIFT_DIR=$PWD/.build/checkouts/mlx-swift ./scripts/build-mlx-metallib.sh`
(copy the resulting `mlx.metallib` next to the release binary).

Throughput/acceptance (256 generated tokens, chat-templated coding prompt,
`--warmup-tokens 32`):

| K | base tok/s | dflash tok/s | speedup | accepted/block |
|---|---|---|---|---|
| 4 | 83.4 | 130.2 | 1.56x | 2.19/3 |
| 5 | 83.4 | 133.2 | 1.60x | 2.59/4 |
| 6 | 83.4 | 146.8 | 1.76x | 3.11/5 |
| 8 | 83.4 | 155.1 | 1.86x | 3.55/7 |
| 16 | 83.4 | 150.7 | 1.81x | 4.10/15 |

Acceptance criteria: mean accepted 4.10 per 16-block (>= 4.0 target);
peak speedup 1.86x at K=8. IMPORTANT: prompts must go through the chat
template — the drafter was trained on chat-formatted data, and the same
prompt encoded raw (no template) roughly halves acceptance (1.90/15) and
drops speedup to ~1.1x.

Parity (`--parity-check`, 128 tokens): every DFlash-side comparison passed
in all rounds across block sizes 4/5/6/8/16. The only mismatches (5 runs,
2 distinct positions) were of kind `baseline_path_block_vs_sequential` —
the TARGET's own block-shaped forward vs its sequential forward flipping
near-tie argmaxes (identical top-5 sets, <= 0.4-logit deltas, recurring at
the same absolute positions regardless of block segmentation). That is
inherent NVFP4 kernel numerics at different matmul widths, not a DFlash
defect; it equally bounds any block-verify scheme on this target, so exact
token-identity with a *sequential* baseline is not an achievable criterion
on this model family.

Prefill (target-only, same M5-B session, 1,138-token chat-templated prompt):
0.313 s for the full prompt = **3,636 tok/s** (~0.28 ms/token). DFlash does
not alter prefill — the drafter participates only in decode. (`mlx-bench`
prints this as `prefill[i]` in the baseline pass.) Note acceptance is
prompt-dependent: a summarization-style prompt at 64 tokens measured only
~1.7 accepted/block vs ~3.6–4.1 for code-writing prompts at 256 tokens.
