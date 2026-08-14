# Qwen 3.6 27B 4-bit Weight Contract

> **HISTORY — this document describes the QWEN 3.6 source artifact.** It is
> kept under its 3.6 name and with its 3.6 numbers on purpose: the shard
> layout, tensor/byte counts and vision-tower split below were measured on
> `mlx-community/Qwen3.6-27B-4bit`, and rewriting them into a 3.8 document
> nobody measured would be worse than leaving them labelled. The ranked track's
> pinned source moved on 2026-08-14 to `EigenLabs/Qwen3.8-27B-4bit` @
> `eda45ab47f465d08d6558f0353a2346e2eb9d5b3` (our own mlx-0.32.0 conversion of the
> official bf16 base, replacing a third-party conversion adopted and then
> terminated the same day), whose 4-bit form is
> **text-only** — 1,847 tensors, all `language_model.`,
> no vision tower and no `mtp.` — so the *selection* half of the contract below
> is unchanged (the same 1,847 tensors, same names, prefix preserved) while the
> *drop* half no longer has anything to drop. The `manifest` line in the
> identity block names the fixture path, which was repointed at the 3.8 artifact
> and is currently a header-only stub. The transform interface itself, and the
> geometry it compiles against, are unchanged: the 3.8 tower was verified
> identical.

This is the interface between the offline transform
(`Sources/MLXFastTransform/`, family `.qwen35`) and the runtime
(`Qwen35Config`, `Qwen35WeightLoader`, `Qwen35RuntimeWeightCache`, and the
vendored `MLXLLM.Qwen35TextModel` the runtime worker drives).

Source identity:

```text
mlx-community/Qwen3.6-27B-4bit
revision c000ac2c2057d94be3fa931000c31723aac53282
manifest fixtures/reference_qwen3_8_27b_4bit.sha256
```

The artifact is named Qwen3.6; its immutable internal text architecture name
is `qwen3_5_text`, which is why every source file carries a `Qwen35` prefix.

Unlike Poolside Laguna, the source is **not** text-only: it is a multimodal
package. Its three shards contain 2,180 tensors and 16,054,262,240
tensor-data bytes, of which the transform selects the 1,847 `language_model.*`
tensors (15,132,802,048 bytes) and drops the 333 `vision_tower.*` tensors
(921,460,192 bytes). The transform preserves the source `language_model.*`
names verbatim; it does **not** strip the prefix.

The public artifact contract is checked in without model data:

```text
fixtures/qwen3_6_27b_config.json
fixtures/qwen3_6_27b_tensor_inventory.json
```

`ARTIFACT_SNAKE` is `qwen3_6_27b`. It deliberately omits the vendor segment
the Laguna fixtures carry (`poolside_laguna_xs_2_1_nvfp4`): mlx-community is
the *distributor* of an Alibaba model, not its author, so a `mlx_community_`
prefix would name the wrong party for the artifact being described. The two
filenames are consumed by `Tests/MLXFastTests/Model/Qwen35ArtifactFixtureSupport.swift`.

The config fixture is the complete public `config.json`, normalized only for
JSON formatting. The inventory fixture was extracted by reading only the three
safetensors headers (8-byte little-endian length prefix plus the JSON header)
and the public index; no tensor payload was read. It records every name,
dtype, shape, and index shard assignment, the public config/index SHA256
values (`ede24666...` / `13b84016...`), each raw header SHA256 and length, per-shard
canonical hashes and dtype counts, representative records, and a full canonical
inventory SHA256
`f17f15bb2d498ab22478bea86b8e1a3e7fd7d939c65104338b04367cd11e3f54`
over `[name,dtype,shape,shard_name]` records.

The test-side pair lives at `Tests/Fixtures/Qwen3627B4bit/`. Its
`header-inventory-contract.json` carries a **different** canonical digest,
`4c6f7bc7fc9a137d7020c893aa5d21f80c209e17156869864f6b95b3533721ea`, because its
records are `name<TAB>dtype<TAB>comma-separated-shape<LF>` with no shard name —
so the digest survives a re-sharded republish of the same tensors. The two are
computed under their own documented canonicalizations and must never be copied
into each other; `qwen36HeaderInventoryContractPinsTheShardIndependentDigest`
asserts they differ.

## Directory and inventory

```text
weights/
  config.json
  model.safetensors.index.json
  model-00001-of-00003.safetensors
  model-00002-of-00003.safetensors
  model-00003-of-00003.safetensors
```

Every referenced shard must exist; every present safetensors shard must be
referenced; and each shard must contain exactly the tensors assigned to it by
the index. Tensor names must be unique.

All 333 `vision_tower.*` tensors live in shard 1 (514 of its 847 tensors are
text-tower), so shard 1 is always rewritten through the subset copy while
shards 2 and 3 — 720 and 613 text-tower tensors, nothing else — may be taken
as independent APFS copy-on-write clones. The transform must fall back to a
regular copy across filesystems and must never publish a symlink.

The source ships its chat template as a separate `chat_template.jinja` rather
than inside `tokenizer_config.json`, so `.jinja` is part of the metadata
copy set.

## Quantization

`quantization` and `quantization_config` publish the SAME spec twice. At least
one must be present; when both are, they must agree exactly. The accepted
value is exactly:

```json
{"group_size": 64, "bits": 4, "mode": "affine"}
```

No per-tensor overrides are permitted. The emitted runtime `config.json`
carries a single `quantization` block and no `quantization_config`.

Almost everything is quantized here — every projection including the token
embedding and the untied `lm_head`; only norms and the gated-delta state
tensors stay raw. For logical shape `[out, in]`:

```text
<stem>.weight  U32  [out, in / 8]
<stem>.scales  BF16 [out, in / 64]
<stem>.biases  BF16 [out, in / 64]
```

Affine quantization needs BOTH companions. This is the load-bearing
difference from `docs/laguna-weight-contract.md`, whose NVFP4 contract
requires U8 scales and **forbids** `.biases`: the two validators cannot share
a packing rule. `in / 8` is `in * bits / 32` at 4 bits — eight nibbles per U32
word.

The runtime quantizes every module for which the loaded weights carry a
`.scales` entry, with `groupSize`, `bits`, and mode `.affine` taken from the
config, and then applies a strict `verify: [.all]` parameter update, so a
missing or extra tensor fails at load rather than at the first forward.

## Tensor inventory

Geometry: hidden 5120, intermediate 17408, vocab 248320, 64 layers, head
dimension 256, 24 query heads, 4 KV heads. The layer schedule is a 4-layer
repeat: index % 4 == 3 is **full attention** (16 of them: 3, 7, ..., 63), the
other three are **gated-delta linear attention** (48 of them) with 48 value
heads, 16 key heads, value/key head dimension 128, and convolution kernel
width 4. The `lm_head` is untied.

Top level (7 tensors):

- `language_model.model.embed_tokens.{weight,scales,biases}`: affine
  `[248320, 5120]` → U32 `[248320, 640]`, BF16 `[248320, 80]` ×2
- `language_model.model.norm.weight`: BF16 `[5120]`
- `language_model.lm_head.{weight,scales,biases}`: affine `[248320, 5120]`,
  same stored shapes, untied

Every layer `language_model.model.layers.<N>`:

- `input_layernorm.weight`, `post_attention_layernorm.weight`: BF16 `[5120]`
- `mlp.gate_proj`, `mlp.up_proj`: affine `[17408, 5120]` → U32
  `[17408, 640]`, BF16 `[17408, 80]` ×2
- `mlp.down_proj`: affine `[5120, 17408]` → U32 `[5120, 2176]`, BF16
  `[5120, 272]` ×2

Full-attention layers (N % 4 == 3), 25 tensors each:

- `self_attn.q_norm.weight`, `k_norm.weight`: BF16 `[256]`
- `self_attn.q_proj`: affine `[12288, 5120]` → U32 `[12288, 640]`, BF16
  `[12288, 80]` ×2. The output width is `24 * 256 * 2`: the per-head output
  gate shares this matrix with the query projection (`attn_output_gate: true`,
  `output_gate_type: "swish"`).
- `self_attn.k_proj`, `v_proj`: affine `[1024, 5120]` → U32 `[1024, 640]`,
  BF16 `[1024, 80]` ×2
- `self_attn.o_proj`: affine `[5120, 6144]` → U32 `[5120, 768]`, BF16
  `[5120, 96]` ×2

Linear-attention layers (all others), 30 tensors each:

- `linear_attn.conv1d.weight`: BF16 `[10240, 4, 1]`, where 10240 is
  `2 * 16 * 128 + 48 * 128` (both key streams plus the value stream)
- `linear_attn.A_log`, `linear_attn.dt_bias`: BF16 `[48]`
- `linear_attn.norm.weight`: BF16 `[128]`
- `linear_attn.in_proj_qkv`: affine `[10240, 5120]` → U32 `[10240, 640]`,
  BF16 `[10240, 80]` ×2
- `linear_attn.in_proj_z`: affine `[6144, 5120]` → U32 `[6144, 640]`, BF16
  `[6144, 80]` ×2
- `linear_attn.in_proj_b`, `in_proj_a`: affine `[48, 5120]` → U32 `[48, 640]`,
  BF16 `[48, 80]` ×2
- `linear_attn.out_proj`: affine `[5120, 6144]` → U32 `[5120, 768]`, BF16
  `[5120, 96]` ×2

Count check for the transformed (text-only) artifact:

- 7 + 48 × 30 + 16 × 25 = **1,847 tensors**
- 1,349 BF16, 498 U32

The published (pre-transform) checkpoint additionally carries 333
`vision_tower.*` tensors, for 2,180 total and a 1,682 BF16 / 498 U32 split.

`Sources/MLXFastTransform/Qwen35CheckpointValidation.swift` holds this
inventory as literals and checks it before the multi-GB copy;
`qwen35CheckpointValidationInventoryMatchesThePublicHeaders` proves those
literals are the real published headers. Any missing, extra, or renamed
tensor, any changed dtype or shape, any compressed-tensors `weight_packed` /
`global_scale` / KV-scale alias, and any `mtp.*` tensor fail the exact
inventory check.

## MTP

The source `text_config` declares `mtp_num_hidden_layers: 1` and
`mtp_use_dedicated_embeddings: false`, but the pinned backbone revision ships
**no** `mtp.*` tensors. The head is a separately pinned artifact
(`mlx-community/Qwen3.6-27B-MTP-4bit` @
`83795d546e9d328160e593fb0bf10b2bf2fe637e`), so:

- the transform never selects `mtp.*` and rejects a checkpoint that offers it;
- `Qwen35TextModel` is constructed with its optional MTP attachment disabled,
  which is what lets the strict parameter update cover the complete
  instantiated model;
- both config fields are still pinned by the runtime worker's gate, because a
  checkpoint that changed them would be a different artifact.

Attaching the head is phase 2 of `QWEN36-MTP-CHALLENGE-PLAN.md`, not part of
this contract.

## Config invariants

The transform writes the source `text_config` verbatim, plus the single
`quantization` block, and drops `vision_config`. The runtime worker's
pinned-config gate
(`validateRuntimeWorkerPinnedConfigurationData`, both harness trees) then
requires that exact key set — no more, no fewer, at the root and inside
`rope_parameters` and `quantization` — so a config carrying the pinned values
PLUS an extra behaviour-bearing field is rejected rather than silently
ignored by `Decodable`.

Pinned values: model type `qwen3_5_text`, vocab 248320, hidden 5120,
intermediate 17408, 64 layers, 24 attention heads, 4 KV heads, head dimension
256, linear 48/16 value/key heads at head dimension 128, convolution kernel 4,
`full_attention_interval` 4 with the matching 64-entry `layer_types`
schedule, RMS epsilon 1e-6, `hidden_act` silu, max position 262144,
`attention_bias` false, zero attention dropout, `attn_output_gate` true,
`output_gate_type` swish, BOS = EOS = 248044, initializer range 0.02,
`pad_token_id` **null** (not absent, not zero), untied embeddings,
`mamba_ssm_dtype` float32, dtype bfloat16, `use_cache` true, and the two
`mtp_*` fields above.

RoPE is **partial** and shared by every full-attention layer — there is no
per-layer-type RoPE table as Laguna has:

- type `default`, theta 10,000,000
- `partial_rotary_factor` 0.25, declared at both the top level and inside
  `rope_parameters`; both are pinned. At head dimension 256 that rotates the
  first **64** dimensions and passes the remaining 192 through unrotated.
- `mrope_interleaved` true, `mrope_section` `[11, 11, 10]`. The sections sum
  to 32 = 64 / 2 rotary pairs. Text-only inference uses a single position
  stream, so the multimodal sections do not change text math, but they are
  pinned because a checkpoint that changed them would not be this artifact.

`MLXFastConstants` mirrors the geometry (vocab, hidden, intermediate, layers,
attention heads) for the trusted targets that cannot import the editable model
target, and the gate reads it from there rather than duplicating literals — on
this branch those constants ARE the Qwen identity.

Any NVFP4 checkpoint, group size other than 64, quantization override,
missing affine `.biases` companion, non-BF16 scale, or changed layer schedule
fails closed before the first forward.

## Reference parity gate

Before regenerating correctness artifacts, operators compare the runtime's
streaming schedule against a single-shot reference forward on the real
transformed checkpoint:

```bash
MLXFAST_RUN_QWEN_REFERENCE_PARITY=1 \
MLXFAST_QWEN_REFERENCE_WEIGHTS_PATH=weights \
swift test --force-resolved-versions \
  --filter transformedCheckpointParity
```

The gate is opt-in and loads the real checkpoint, so it is skipped unless both
variables are set. It is not a substitute for the ranked correctness gates.
