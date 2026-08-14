#!/usr/bin/env python3
"""
Patch an existing mlx-community Qwen3.6 4-bit checkpoint with MTP weights
extracted from the original HuggingFace model.

The mlx-community conversions strip the `mtp.*` weight keys, but the original
HF checkpoint has them in specific shards. This script:

  1. Downloads only the MTP-containing shards from the original HF repo
  2. Extracts the MTP weights (stored as bfloat16 — no requantisation needed,
     since the Swift Linear layers load plain-float weights correctly)
  3. Saves a new `mtp-weights.safetensors` shard in the mlx checkpoint dir
     using the key prefix `language_model.mtp.*` expected by Qwen35Model
  4. Updates `model.safetensors.index.json` to reference the new shard

Usage:
    python3 scripts/patch-qwen36-mtp.py [--model 27b|35b]

Requirements:
    pip install huggingface_hub safetensors torch
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file as st_load, save_file as st_save


# ── model registry ────────────────────────────────────────────────────────────

MODELS = {
    "27b": {
        "mlx_repo": "mlx-community/Qwen3.6-27B-4bit",
        "hf_repo": "Qwen/Qwen3.6-27B",
    },
    "35b": {
        "mlx_repo": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "hf_repo": "Qwen/Qwen3.6-35B-A3B",
    },
}

# In Qwen35Model.sanitize, the text-model's MTP weights live under the
# `language_model` sub-module of the outer VLM model, so keys must be
# `language_model.mtp.*` (not `language_model.model.mtp.*`).
KEY_PREFIX = "language_model."

# Norm keys that need a +1 shift (HF stores them as deviation-from-1; MLXNN
# uses the weight directly, so we normalise to the MLX convention here).
# These must match the suffixes in Qwen35TextModel.sanitize's normKeys list.
NORM_SHIFT_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    ".pre_fc_norm_hidden.weight",
    ".pre_fc_norm_embedding.weight",
    "mtp.norm.weight",       # full suffix so model.norm.weight doesn't match
)


# ── helpers ───────────────────────────────────────────────────────────────────

def find_mtp_shards(hf_repo: str, cache_dir: str | None) -> tuple[list[str], list[str]]:
    """Return (shard_filenames, mtp_key_names) by inspecting the HF index."""
    index_path = hf_hub_download(
        repo_id=hf_repo, filename="model.safetensors.index.json", cache_dir=cache_dir
    )
    with open(index_path) as f:
        index = json.load(f)
    wmap: dict[str, str] = index["weight_map"]
    shards = sorted({v for k, v in wmap.items() if k.startswith("mtp.")})
    keys = sorted(k for k in wmap if k.startswith("mtp."))
    return shards, keys


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["27b", "35b"], default="27b")
    parser.add_argument("--mlx-dir", default=None,
                        help="Path to existing MLX checkpoint dir (default: auto-download)")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    cfg = MODELS[args.model]
    hf_repo = cfg["hf_repo"]
    mlx_repo = cfg["mlx_repo"]

    print(f"=== Patching {mlx_repo} with MTP weights from {hf_repo} ===\n")

    # 1. Locate/download the MLX checkpoint
    if args.mlx_dir:
        mlx_dir = Path(args.mlx_dir)
        print(f"Using MLX dir: {mlx_dir}")
    else:
        print(f"Downloading MLX checkpoint: {mlx_repo}")
        mlx_dir = Path(snapshot_download(mlx_repo, cache_dir=args.cache_dir))
        print(f"  -> {mlx_dir}")

    index_path = mlx_dir / "model.safetensors.index.json"
    if not index_path.exists():
        sys.exit(f"ERROR: {index_path} not found — is this an MLX checkpoint?")

    with open(index_path) as f:
        mlx_index = json.load(f)
    mlx_wmap: dict[str, str] = mlx_index["weight_map"]

    existing = [k for k in mlx_wmap if "mtp" in k]
    if existing:
        print(f"\nCheckpoint already has {len(existing)} MTP key(s) — nothing to do.")
        return

    # 2. Find which HF shards contain MTP weights
    print(f"\nInspecting HF index: {hf_repo}")
    mtp_shards, mtp_keys = find_mtp_shards(hf_repo, cache_dir=args.cache_dir)
    if not mtp_shards:
        sys.exit("ERROR: no mtp.* keys found in HF model index.")
    print(f"  {len(mtp_keys)} MTP keys in {len(mtp_shards)} shard(s): {mtp_shards}")

    # 3. Download only those shards
    print(f"\nDownloading {len(mtp_shards)} shard(s) ...")
    shard_paths = [
        hf_hub_download(repo_id=hf_repo, filename=s, cache_dir=args.cache_dir)
        for s in mtp_shards
    ]
    for p in shard_paths:
        print(f"  -> {p}")

    # 4. Extract mtp.* tensors and rekey to language_model.mtp.*
    print("\nExtracting MTP tensors ...")
    output: dict[str, torch.Tensor] = {}
    for shard_path in shard_paths:
        for k, v in st_load(shard_path, device="cpu").items():
            if not k.startswith("mtp."):
                continue

            # MoE fused expert conversion:
            # HF stores fused gate+up proj as experts.gate_up_proj [num_experts, 2*intermediate, hidden].
            # Swift expects switch_mlp.gate_proj and switch_mlp.up_proj each [num_experts, intermediate, hidden].
            # Also rename experts.down_proj -> switch_mlp.down_proj.
            if k.endswith(".mlp.experts.gate_up_proj"):
                prefix = k[: -len(".mlp.experts.gate_up_proj")]
                mid = v.shape[1] // 2
                gate = v[:, :mid, :]
                up = v[:, mid:, :]
                for suffix, tensor in [
                    (".mlp.switch_mlp.gate_proj.weight", gate),
                    (".mlp.switch_mlp.up_proj.weight", up),
                ]:
                    new_key = KEY_PREFIX + prefix + suffix
                    t = tensor.to(torch.float32)
                    print(f"  {k}  {tuple(v.shape)} split -> {new_key} {tuple(t.shape)}")
                    output[new_key] = t.to(torch.bfloat16)
                continue

            if k.endswith(".mlp.experts.down_proj"):
                prefix = k[: -len(".mlp.experts.down_proj")]
                new_key = KEY_PREFIX + prefix + ".mlp.switch_mlp.down_proj.weight"
                t = v.to(torch.float32)
                print(f"  {k}  {tuple(v.shape)} rename -> {new_key}")
                output[new_key] = t.to(torch.bfloat16)
                continue

            new_key = KEY_PREFIX + k
            t = v.to(torch.float32)
            # Apply +1 shift to 1-D norm weights.
            # HF stores RMSNorm gammas as deviation-from-1 (near 0 after training).
            # MLXNN uses the weight directly, so we add 1 to match convention.
            # The Swift sanitize won't shift these because hasUnsanitizedConv1d=False
            # for the already-converted mlx-community checkpoint.
            shifted = any(new_key.endswith(s) for s in NORM_SHIFT_SUFFIXES)
            if shifted and t.dim() == 1:
                t = t + 1.0
                print(f"  {k}  {tuple(v.shape)} +1 norm shift -> {new_key}")
            else:
                print(f"  {k}  {tuple(v.shape)} {v.dtype} -> {new_key}")
            output[new_key] = t.to(torch.bfloat16)

    if not output:
        sys.exit("ERROR: no mtp.* tensors found in downloaded shards.")

    # 5. Save new shard
    out_shard = "mtp-weights.safetensors"
    out_path = mlx_dir / out_shard
    print(f"\nSaving {out_path} ...")
    st_save(output, str(out_path))
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved {len(output)} tensors ({size_mb:.1f} MB)")

    # 6. Update index
    print("\nUpdating model.safetensors.index.json ...")
    for k in output:
        mlx_wmap[k] = out_shard
    with open(index_path, "w") as f:
        json.dump(mlx_index, f, indent=2, sort_keys=True)
    print(f"  Added {len(output)} keys -> {out_shard}")

    print(f"\n=== Done! Patched: {mlx_dir} ===")
    print(f'\nTest: .build/release/mlx-server --model "{mlx_dir}" --mtp --port 18765')


if __name__ == "__main__":
    main()
