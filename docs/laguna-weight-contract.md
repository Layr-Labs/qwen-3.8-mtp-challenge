# Poolside Laguna NVFP4 Weight Contract

This is the interface between the offline transform
(`Sources/MLXFastTransform/`) and the serial runtime (`LagunaConfig`,
`LagunaWeightLoader`, `LagunaRuntimeWeightCache`, and
`LagunaRuntimeModel`).

Source identity:

```text
poolside/Laguna-XS-2.1-NVFP4-mlx
revision 841778bda563a36104dd521e37d99218e46f4f25
R2 https://ds4.darkbloom.ai/laguna-xs-2.1-nvfp4-mlx
```

The source is text-only. Its five shards contain 912 tensors and
21,561,408,512 tensor-data bytes. The transform preserves the source
`model.*` / `lm_head.*` names and the source config (minus an empty
`vision_config` if present).

The public artifact contract is checked in without model data:

```text
fixtures/poolside_laguna_xs_2_1_nvfp4_config.json
fixtures/poolside_laguna_xs_2_1_nvfp4_tensor_inventory.json
```

The config fixture is the complete public `config.json`, normalized only for
JSON formatting. The inventory fixture was extracted with HTTP range requests
for the five safetensors headers plus the public index; no tensor payload was
downloaded. It records every name, dtype, shape, and index shard assignment,
the public config/index SHA256 values, each raw header SHA256, representative
records, per-shard hashes, and a full canonical inventory SHA256
`6da902713b7348d11f3b9664b2ad64703c2544eae6d028133f3d2c45be926848`.

## Directory and inventory

```text
weights/
  config.json
  model.safetensors.index.json
  model-00001-of-00005.safetensors
  ...
  model-00005-of-00005.safetensors
```

Every referenced shard must exist; every present safetensors shard must be
referenced; and each shard must contain exactly the tensors assigned to it by
the index. Tensor names must be unique. `rotary_emb.inv_freq` tables are
excluded.

When a source shard contains only selected tensors—as every Poolside shard
does—the transform may create an independent APFS copy-on-write clone. It must
fall back to a regular copy across filesystems and must never publish a
symlink.

## Quantization

Both `quantization` and `quantization_config` are required, must match, and
must contain exactly:

```json
{"group_size": 16, "bits": 4, "mode": "nvfp4"}
```

No per-tensor overrides are permitted.

Only routed and shared expert projections are quantized. For logical shape
`[rows, in]` or `[experts, rows, in]`:

```text
<stem>.weight  U32 [..., rows, in * 4 / 32]
<stem>.scales  U8  [..., rows, in / 16]
```

The U8 values are NVFP4 E4M3 group scales. There is no `.biases` companion.
The runtime promotes exactly every sparse layer's routed/shared expert
projection to `QuantizedLinear` / `QuantizedSwitchLinear` during model
construction and dispatches MLX with mode `.nvfp4`; metadata validation then
requires the matching `.scales` tensor for each such module.

## Tensor inventory

Geometry: hidden 2048, vocab 100352, 40 layers, head dimension 128, 8 KV
heads, 48 query heads on full-attention layers (0, 4, ..., 36), 64 query
heads on sliding layers, dense intermediate 8192 at layer 0, 256 experts,
expert/shared intermediate 512.

Top level:

- `model.embed_tokens.weight`: BF16 `[100352, 2048]`
- `model.norm.weight`: BF16 `[2048]`
- `lm_head.weight`: BF16 `[100352, 2048]`, untied

Every layer `model.layers.<N>`:

- `input_layernorm.weight`, `post_attention_layernorm.weight`: BF16 `[2048]`
- `self_attn.q_proj.weight`: BF16 `[H(N)*128, 2048]`
- `self_attn.k_proj.weight`, `v_proj.weight`: BF16 `[1024, 2048]`
- `self_attn.o_proj.weight`: BF16 `[2048, H(N)*128]`
- `self_attn.g_proj.weight`: BF16 `[H(N), 2048]`
- `self_attn.q_norm.weight`, `k_norm.weight`: BF16 `[128]`

Layer 0 dense MLP:

- `mlp.gate_proj.weight`, `up_proj.weight`: BF16 `[8192, 2048]`
- `mlp.down_proj.weight`: BF16 `[2048, 8192]`

Sparse layers 1–39:

- `mlp.gate.weight`: BF16 `[256, 2048]`
- `mlp.gate.e_score_correction_bias`: F32 `[256]`
- `mlp.switch_mlp.gate_proj` / `up_proj`: NVFP4 logical
  `[256, 512, 2048]`; stored weight `[256, 512, 256]`, scales
  `[256, 512, 128]`
- `mlp.switch_mlp.down_proj`: NVFP4 logical `[256, 2048, 512]`; stored
  weight `[256, 2048, 64]`, scales `[256, 2048, 32]`
- `mlp.shared_expert.gate_proj` / `up_proj`: NVFP4 logical `[512, 2048]`;
  stored weight `[512, 256]`, scales `[512, 128]`
- `mlp.shared_expert.down_proj`: NVFP4 logical `[2048, 512]`; stored
  weight `[2048, 64]`, scales `[2048, 32]`

Count check:

- 405 BF16 tensors
- 39 F32 tensors
- 234 U32 tensors
- 234 U8 tensors
- total 912

The `switch_mlp` tensors remain stacked by expert. They must not be split into
per-expert tensors. Affine `.biases`, compressed-tensors `weight_packed`,
`global_scale`, or KV-scale tensors, and any missing, extra, or renamed tensor
are different schemas and fail the exact inventory check.

## Config invariants

The runtime validates the frozen architecture before materializing weights:
model type `laguna`, vocab 100352, hidden 2048, dense intermediate 8192, 40
layers, 8 KV heads, head dimension 128, sliding window 512, 256 experts,
top-8 routing, expert/shared intermediate 512, untied head, per-head gating,
all 40 `gating_types` entries set to `per_head`, and the every-fourth
full-attention schedule. It also pins max position 262144, RMS epsilon 1e-6,
zero attention dropout, normalized top-k probabilities, routed factor 2.5,
`moe_apply_router_weight_on_input=false`, `router_aux_loss_coef=0`, and
`use_cache=true`. The public artifact explicitly stores the two zero-valued
fields but omits `qkv_bias` and `moe_router_logit_softcapping`; the exact
fixture preserves absent, JSON null, and numeric zero as distinct states.

RoPE remains:

- sliding: default, theta 10000, full head rotation
- full: YaRN, theta 500000, factor 32, original context 8192, beta 64/1,
  attention factor 1.0, partial rotary factor 0.5

`attention_factor` is parsed and pinned as public artifact metadata, but it is
not applied directly as the YaRN query/key multiplier. The vendored
mlx-swift-lm `YarnRoPE` implementation is the runtime authority: with its
default `mscale=1`, `mscale_all_dim=0`, and factor 32 it derives
`1 + 0.1 * log(32) = 1.3465736...`. Replacing that derived multiplier with
the literal public `attention_factor=1.0` changes runtime math and is forbidden.

Any affine checkpoint, group size other than 16, quantization override,
quantized router, affine bias companion, non-U8 NVFP4 scale, or non-BF16
dense projection fails closed before the first forward. The 48-layer,
top-10 Laguna S geometry is likewise a different artifact and fails closed.

## M5 upstream equivalence gate

Before regenerating correctness artifacts, operators compare one 512-token
prefill and eight serial teacher-forced decode steps against the vendored
`MLXLLM.LagunaModel`, with both module trees sharing the same loaded tensor
objects:

```bash
MLXFAST_RUN_LAGUNA_UPSTREAM_EQUIVALENCE=1 \
MLXFAST_LAGUNA_EQUIVALENCE_WEIGHTS_PATH=weights \
swift test --force-resolved-versions \
  --filter lagunaRuntimeMatchesVendoredUpstreamOnM5WhenEnabled
```

The test reports maximum/mean absolute logit error and both greedy tokens for
every step. Its default tolerance is exact (`0`); an operator may set
`MLXFAST_LAGUNA_EQUIVALENCE_MAX_ABS_ERROR` only to diagnose a mismatch, not to
weaken the exact-token correctness gates.
