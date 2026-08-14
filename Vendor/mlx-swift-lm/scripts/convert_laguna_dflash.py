#!/usr/bin/env python3
"""Convert a Laguna DFlash Hugging Face checkpoint into the flat MLX schema.

Stdlib only (no torch / numpy / safetensors / huggingface_hub). Every tensor
move is a raw byte copy -- this script never interprets BF16 values, it only
slices and re-packs the safetensors data buffer.

Safetensors format (implemented exactly):
    8 bytes little-endian u64 header length N
    N bytes of UTF-8 JSON header (may be trailing-whitespace padded)
    raw data buffer, addressed by the header's "data_offsets" (relative to
    the start of the data buffer)

Usage:
    python3 scripts/convert_laguna_dflash.py --selftest
    python3 scripts/convert_laguna_dflash.py --source <dir> --output <dir> \
        [--source-revision <sha>]
"""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import struct
import sys
import tempfile

CHUNK_SIZE = 64 * 1024 * 1024  # 64 MiB streaming chunk cap
ELEM_SIZE_BF16 = 2

SOURCE_REPO = "poolside/Laguna-XS-2.1-DFlash-NVFP4"
CONVERTER_ID = "convert_laguna_dflash.py v1"

# Non-qkv per-layer tensor suffixes that pass through unchanged.
_PASSTHROUGH_LAYER_SUFFIXES = (
    "self_attn.g_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)


def read_header(path):
    """Read a safetensors header.

    Returns (header, data_start): `header` is the parsed JSON header dict
    (tensor name -> {"dtype", "shape", "data_offsets"}, plus "__metadata__"
    if present -- callers that want tensor entries only should filter that
    key out themselves). `data_start` is the absolute file offset where the
    raw data buffer begins.
    """
    with open(path, "rb") as f:
        length_bytes = f.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"{path}: file too short to contain a safetensors header length")
        (header_len,) = struct.unpack("<Q", length_bytes)
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")
        # Safetensors headers may be space-padded to align the data buffer;
        # strip trailing whitespace before parsing.
        header_str = header_bytes.decode("utf-8").rstrip()
        header = json.loads(header_str)
        data_start = 8 + header_len
    return header, data_start


def expected_source_manifest(cfg):
    """Build the exact expected name -> (dtype, shape) map for the source checkpoint."""
    hidden_size = cfg["hidden_size"]
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    intermediate_size = cfg["intermediate_size"]
    num_layers = cfg["num_hidden_layers"]
    num_aux = len(cfg["dflash_config"]["target_layer_ids"])

    qkv_rows = (num_heads + 2 * num_kv_heads) * head_dim
    q_o_cols = num_heads * head_dim

    manifest = {}
    for i in range(num_layers):
        p = f"layers.{i}."
        manifest[p + "self_attn.qkv_proj.weight"] = ("BF16", [qkv_rows, hidden_size])
        manifest[p + "self_attn.g_proj.weight"] = ("BF16", [num_heads, hidden_size])
        manifest[p + "self_attn.q_norm.weight"] = ("BF16", [head_dim])
        manifest[p + "self_attn.k_norm.weight"] = ("BF16", [head_dim])
        manifest[p + "self_attn.o_proj.weight"] = ("BF16", [hidden_size, q_o_cols])
        manifest[p + "mlp.gate_proj.weight"] = ("BF16", [intermediate_size, hidden_size])
        manifest[p + "mlp.up_proj.weight"] = ("BF16", [intermediate_size, hidden_size])
        manifest[p + "mlp.down_proj.weight"] = ("BF16", [hidden_size, intermediate_size])
        manifest[p + "input_layernorm.weight"] = ("BF16", [hidden_size])
        manifest[p + "post_attention_layernorm.weight"] = ("BF16", [hidden_size])

    manifest["fc.weight"] = ("BF16", [hidden_size, num_aux * hidden_size])
    manifest["hidden_norm.weight"] = ("BF16", [hidden_size])
    manifest["norm.weight"] = ("BF16", [hidden_size])
    for j in range(num_aux):
        manifest[f"aux_hidden_norms.{j}.weight"] = ("BF16", [hidden_size])

    return manifest


def _sha256_file(path, chunk_size=CHUNK_SIZE):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stream_copy(src_f, dst_f, abs_offset, length, chunk_size=CHUNK_SIZE):
    src_f.seek(abs_offset)
    remaining = length
    while remaining > 0:
        take = min(chunk_size, remaining)
        chunk = src_f.read(take)
        if not chunk:
            raise IOError("unexpected EOF while streaming tensor bytes")
        dst_f.write(chunk)
        remaining -= len(chunk)


def _validate_source_config(cfg):
    architectures = cfg.get("architectures")
    assert architectures == ["DFlashLagunaForCausalLM"], (
        f"unexpected architectures: {architectures!r} (expected ['DFlashLagunaForCausalLM'])"
    )
    assert cfg.get("model_type") == "laguna", f"unexpected model_type: {cfg.get('model_type')!r}"
    assert cfg.get("draft_vocab_size") == cfg.get("vocab_size"), (
        f"draft_vocab_size ({cfg.get('draft_vocab_size')!r}) != vocab_size ({cfg.get('vocab_size')!r})"
    )

    layer_types = cfg.get("layer_types")
    assert layer_types, "config.layer_types is missing or empty"
    non_uniform = [lt for lt in layer_types if lt != "sliding_attention"]
    assert not non_uniform, (
        "drafter must be uniform sliding_attention; found non-uniform layer_types: "
        f"{layer_types!r}"
    )

    assert cfg.get("gating") == "per-head", f"unexpected gating: {cfg.get('gating')!r}"

    dflash_cfg = cfg["dflash_config"]
    vocab_size = cfg["vocab_size"]
    mask_token_id = dflash_cfg["mask_token_id"]
    num_target_layers = dflash_cfg["num_target_layers"]
    target_layer_ids = dflash_cfg["target_layer_ids"]

    assert mask_token_id < vocab_size, (
        f"mask_token_id ({mask_token_id}) must be < vocab_size ({vocab_size})"
    )
    bad_ids = [tid for tid in target_layer_ids if not (tid < num_target_layers)]
    assert not bad_ids, (
        f"target_layer_ids {bad_ids} must all be < num_target_layers ({num_target_layers})"
    )


def _diff_manifest(expected, actual_entries):
    """Return a list of human-readable discrepancy strings, empty if none."""
    mismatches = []
    expected_keys = set(expected.keys())
    actual_keys = set(actual_entries.keys())

    for key in sorted(expected_keys - actual_keys):
        mismatches.append(f"missing tensor: {key}")
    for key in sorted(actual_keys - expected_keys):
        mismatches.append(f"unexpected tensor: {key}")

    for key in sorted(expected_keys & actual_keys):
        exp_dtype, exp_shape = expected[key]
        actual = actual_entries[key]
        act_dtype = actual.get("dtype")
        act_shape = list(actual.get("shape", []))
        if act_dtype != exp_dtype:
            mismatches.append(f"{key}: dtype {act_dtype!r} != expected {exp_dtype!r}")
        if act_shape != list(exp_shape):
            mismatches.append(f"{key}: shape {act_shape} != expected {list(exp_shape)}")

    return mismatches


def convert(source_dir, output_dir, source_revision):
    source_dir = pathlib.Path(source_dir)
    output_dir = pathlib.Path(output_dir)

    config_path = source_dir / "config.json"
    weights_path = source_dir / "model.safetensors"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    _validate_source_config(cfg)

    header, data_start = read_header(weights_path)
    tensor_entries = {k: v for k, v in header.items() if k != "__metadata__"}

    expected = expected_source_manifest(cfg)
    mismatches = _diff_manifest(expected, tensor_entries)
    if mismatches:
        print("Source checkpoint manifest mismatch:", file=sys.stderr)
        for m in mismatches:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(1)

    hidden_size = cfg["hidden_size"]
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    num_layers = cfg["num_hidden_layers"]
    dflash_cfg = cfg["dflash_config"]
    target_layer_ids = dflash_cfg["target_layer_ids"]
    num_target_layers = dflash_cfg["num_target_layers"]
    block_size = dflash_cfg["block_size"]
    mask_token_id = dflash_cfg["mask_token_id"]

    row_bytes = hidden_size * ELEM_SIZE_BF16
    q_rows = num_heads * head_dim
    kv_rows = num_kv_heads * head_dim

    def _assert_exact_bf16_length(name, shape, byte_length):
        """Every tensor in this checkpoint is a validated BF16 tensor -- its
        source byte span must equal prod(shape) * 2 exactly. This is a
        distinct check from `_diff_manifest` above: that function compares
        each tensor's *declared* dtype/shape against the config-derived
        expectation, but never cross-checks a tensor's declared shape
        against its own `data_offsets` span. A header where those two
        disagree (truncated/padded/corrupted data_offsets, or a shape typo)
        would otherwise be copied silently and produce a mis-shaped or
        truncated output tensor. Fail loudly instead."""
        expected_length = ELEM_SIZE_BF16
        for dim in shape:
            expected_length *= dim
        assert byte_length == expected_length, (
            f"{name}: source byte length ({byte_length}) != prod(shape) * "
            f"{ELEM_SIZE_BF16} ({expected_length}) for shape {shape}"
        )

    # plan: output tensor name -> (dtype, shape, absolute_source_offset, length_bytes)
    plan = {}

    for i in range(num_layers):
        p = f"layers.{i}."
        qkv_entry = tensor_entries[p + "self_attn.qkv_proj.weight"]
        qkv_start, qkv_end = qkv_entry["data_offsets"]

        q_len = q_rows * row_bytes
        kv_len = kv_rows * row_bytes
        q_start = qkv_start
        k_start = q_start + q_len
        v_start = k_start + kv_len
        v_end = v_start + kv_len
        assert v_end == qkv_end, (
            f"{p}self_attn.qkv_proj.weight: row split ({v_end - qkv_start} bytes) "
            f"does not cover the full tensor ({qkv_end - qkv_start} bytes)"
        )

        q_name = p + "self_attn.q_proj.weight"
        k_name = p + "self_attn.k_proj.weight"
        v_name = p + "self_attn.v_proj.weight"
        _assert_exact_bf16_length(q_name, [q_rows, hidden_size], q_len)
        _assert_exact_bf16_length(k_name, [kv_rows, hidden_size], kv_len)
        _assert_exact_bf16_length(v_name, [kv_rows, hidden_size], kv_len)

        plan[q_name] = ("BF16", [q_rows, hidden_size], data_start + q_start, q_len)
        plan[k_name] = ("BF16", [kv_rows, hidden_size], data_start + k_start, kv_len)
        plan[v_name] = ("BF16", [kv_rows, hidden_size], data_start + v_start, kv_len)

        for suffix in _PASSTHROUGH_LAYER_SUFFIXES:
            key = p + suffix
            entry = tensor_entries[key]
            start, end = entry["data_offsets"]
            shape = list(entry["shape"])
            _assert_exact_bf16_length(key, shape, end - start)
            plan[key] = (entry["dtype"], shape, data_start + start, end - start)

    root_names = ["fc.weight", "hidden_norm.weight", "norm.weight"]
    root_names += [f"aux_hidden_norms.{j}.weight" for j in range(len(target_layer_ids))]
    for name in root_names:
        entry = tensor_entries[name]
        start, end = entry["data_offsets"]
        shape = list(entry["shape"])
        _assert_exact_bf16_length(name, shape, end - start)
        plan[name] = (entry["dtype"], shape, data_start + start, end - start)

    sorted_names = sorted(plan.keys())

    output_dir.mkdir(parents=True, exist_ok=True)
    out_weights_path = output_dir / "model.safetensors"
    out_config_path = output_dir / "config.json"

    source_sha256 = _sha256_file(weights_path)
    revision = source_revision if source_revision else "unknown"

    out_header = {}
    cursor = 0
    for name in sorted_names:
        dtype, shape, _, length = plan[name]
        out_header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + length]}
        cursor += length

    out_header["__metadata__"] = {
        "format": "mlx",
        "source_repo": SOURCE_REPO,
        "source_revision": revision,
        "source_sha256": source_sha256,
        "converter": CONVERTER_ID,
    }

    header_bytes = json.dumps(out_header, separators=(",", ":")).encode("utf-8")

    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(output_dir), prefix=".model.safetensors.", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as tmp_f, open(weights_path, "rb") as src_f:
            tmp_f.write(struct.pack("<Q", len(header_bytes)))
            tmp_f.write(header_bytes)
            for name in sorted_names:
                _, _, abs_offset, length = plan[name]
                _stream_copy(src_f, tmp_f, abs_offset, length)
        os.replace(tmp_name, str(out_weights_path))
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise

    converted_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    out_config = {
        "architectures": ["DFlashDraftModel"],
        "model_type": "laguna",
        "decoder_layer_type": "laguna_xs",
        "gating": cfg["gating"],
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "intermediate_size": cfg["intermediate_size"],
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "vocab_size": cfg["vocab_size"],
        "rms_norm_eps": cfg["rms_norm_eps"],
        "rope_theta": cfg["rope_theta"],
        "max_position_embeddings": cfg["max_position_embeddings"],
        "sliding_window": cfg["sliding_window"],
        "layer_types": cfg["layer_types"],
        "tie_word_embeddings": cfg.get("tie_word_embeddings", False),
        "block_size": block_size,
        "num_target_layers": num_target_layers,
        "dflash_config": {"target_layer_ids": target_layer_ids, "mask_token_id": mask_token_id},
        "_mlx_conversion": {
            "source_repo": SOURCE_REPO,
            "source_revision": revision,
            "source_sha256": source_sha256,
            "converter": CONVERTER_ID,
            "converted_at": converted_at,
        },
    }

    tmp_cfg_fd, tmp_cfg_name = tempfile.mkstemp(dir=str(output_dir), prefix=".config.json.", suffix=".tmp")
    try:
        with os.fdopen(tmp_cfg_fd, "w", encoding="utf-8") as tmp_f:
            json.dump(out_config, tmp_f, indent=2)
            tmp_f.write("\n")
        os.replace(tmp_cfg_name, str(out_config_path))
    except Exception:
        if os.path.exists(tmp_cfg_name):
            os.remove(tmp_cfg_name)
        raise

    print(f"Converted {len(tensor_entries)} source tensors -> {len(sorted_names)} output tensors")
    print(f"  source sha256: {source_sha256}")
    print(f"  source revision: {revision}")
    print(f"  wrote: {out_weights_path}")
    print(f"  wrote: {out_config_path}")


def selftest():
    elem_size = ELEM_SIZE_BF16

    with tempfile.TemporaryDirectory(prefix="convert_laguna_dflash_selftest_") as tmp:
        tmp_path = pathlib.Path(tmp)
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        source_dir.mkdir()

        cfg = {
            "architectures": ["DFlashLagunaForCausalLM"],
            "model_type": "laguna",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 32,
            "draft_vocab_size": 32,
            "rms_norm_eps": 1e-06,
            "rope_theta": 500000.0,
            "max_position_embeddings": 1024,
            "sliding_window": 8,
            "layer_types": ["sliding_attention", "sliding_attention"],
            "tie_word_embeddings": False,
            "gating": "per-head",
            "dflash_config": {
                "block_size": 4,
                "mask_token_id": 3,
                "num_target_layers": 4,
                "target_layer_ids": [0, 2],
                "causal": True,
            },
        }
        with open(source_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        manifest = expected_source_manifest(cfg)
        ordered_names = sorted(manifest.keys())

        # Fill each tensor with distinct deterministic bytes that vary BOTH
        # between tensors (via idx) and within a tensor (via the byte
        # position b). A single repeated fill byte per tensor would make
        # every byte inside a fused qkv_proj tensor identical, which cannot
        # distinguish a correct q/k/v row split from a bug that swaps the k
        # and v regions (both regions would just be the same repeated byte
        # either way). Varying by byte position means the q/k/v regions
        # each carry distinct, position-dependent content, so a swapped
        # k_start/v_start (or any other row-accounting bug) produces bytes
        # that don't match the expected slice of the source tensor.
        tensor_bytes = {}
        for idx, name in enumerate(ordered_names):
            _, shape = manifest[name]
            n_elems = 1
            for d in shape:
                n_elems *= d
            n_bytes = n_elems * elem_size
            tensor_bytes[name] = bytes(((idx * 31 + b) % 256) for b in range(n_bytes))

        src_header = {}
        cursor = 0
        for name in ordered_names:
            dtype, shape = manifest[name]
            length = len(tensor_bytes[name])
            src_header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [cursor, cursor + length]}
            cursor += length
        src_header_bytes = json.dumps(src_header).encode("utf-8")

        weights_path = source_dir / "model.safetensors"
        with open(weights_path, "wb") as f:
            f.write(struct.pack("<Q", len(src_header_bytes)))
            f.write(src_header_bytes)
            for name in ordered_names:
                f.write(tensor_bytes[name])

        convert(str(source_dir), str(output_dir), "selftest-rev")

        out_weights_path = output_dir / "model.safetensors"
        out_header, out_data_start = read_header(out_weights_path)
        out_meta = out_header.pop("__metadata__", None)
        assert out_meta is not None, "output header is missing __metadata__"
        assert out_meta["source_repo"] == SOURCE_REPO
        assert out_meta["converter"] == CONVERTER_ID
        assert out_meta["source_revision"] == "selftest-rev"
        expected_sha256 = _sha256_file(weights_path)
        assert out_meta["source_sha256"] == expected_sha256

        num_layers = cfg["num_hidden_layers"]
        num_aux = len(cfg["dflash_config"]["target_layer_ids"])
        expected_count = num_layers * 12 + (3 + num_aux)
        assert len(out_header) == expected_count, (
            f"expected {expected_count} output tensors, got {len(out_header)}"
        )

        with open(out_weights_path, "rb") as f:
            out_bytes = f.read()

        def out_tensor_bytes(name):
            start, end = out_header[name]["data_offsets"]
            return out_bytes[out_data_start + start : out_data_start + end]

        hidden_size = cfg["hidden_size"]
        num_heads = cfg["num_attention_heads"]
        num_kv_heads = cfg["num_key_value_heads"]
        head_dim = cfg["head_dim"]
        row_bytes = hidden_size * elem_size
        q_rows = num_heads * head_dim
        kv_rows = num_kv_heads * head_dim

        for i in range(num_layers):
            p = f"layers.{i}."
            qkv_bytes = tensor_bytes[p + "self_attn.qkv_proj.weight"]
            q_len = q_rows * row_bytes
            kv_len = kv_rows * row_bytes
            expected_q = qkv_bytes[:q_len]
            expected_k = qkv_bytes[q_len : q_len + kv_len]
            expected_v = qkv_bytes[q_len + kv_len : q_len + 2 * kv_len]

            assert out_tensor_bytes(p + "self_attn.q_proj.weight") == expected_q, f"{p}q_proj byte mismatch"
            assert out_tensor_bytes(p + "self_attn.k_proj.weight") == expected_k, f"{p}k_proj byte mismatch"
            assert out_tensor_bytes(p + "self_attn.v_proj.weight") == expected_v, f"{p}v_proj byte mismatch"

            for suffix in _PASSTHROUGH_LAYER_SUFFIXES:
                key = p + suffix
                assert out_tensor_bytes(key) == tensor_bytes[key], f"{key} byte mismatch (passthrough)"

        for name in ["fc.weight", "hidden_norm.weight", "norm.weight"]:
            assert out_tensor_bytes(name) == tensor_bytes[name], f"{name} byte mismatch (passthrough)"
        for j in range(num_aux):
            name = f"aux_hidden_norms.{j}.weight"
            assert out_tensor_bytes(name) == tensor_bytes[name], f"{name} byte mismatch (passthrough)"

        with open(output_dir / "config.json", "r", encoding="utf-8") as f:
            out_cfg = json.load(f)

        assert out_cfg["architectures"] == ["DFlashDraftModel"]
        assert out_cfg["model_type"] == "laguna"
        assert out_cfg["decoder_layer_type"] == "laguna_xs"
        assert out_cfg["gating"] == "per-head"
        assert out_cfg["block_size"] == cfg["dflash_config"]["block_size"]
        assert out_cfg["num_target_layers"] == cfg["dflash_config"]["num_target_layers"]
        assert out_cfg["dflash_config"] == {
            "target_layer_ids": cfg["dflash_config"]["target_layer_ids"],
            "mask_token_id": cfg["dflash_config"]["mask_token_id"],
        }
        assert out_cfg["hidden_size"] == hidden_size
        assert out_cfg["num_hidden_layers"] == num_layers
        assert out_cfg["tie_word_embeddings"] == cfg["tie_word_embeddings"]
        assert "_mlx_conversion" in out_cfg
        assert out_cfg["_mlx_conversion"]["source_sha256"] == expected_sha256

    print("SELFTEST OK")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Laguna DFlash HF checkpoint to the flat MLX schema."
    )
    parser.add_argument("--source", type=str, default=None, help="source checkpoint directory")
    parser.add_argument("--output", type=str, default=None, help="output directory")
    parser.add_argument(
        "--source-revision", type=str, default=None, help="source repo revision/commit sha"
    )
    parser.add_argument(
        "--selftest", action="store_true", help="run the built-in synthetic-checkpoint selftest and exit"
    )
    args = parser.parse_args()

    if args.selftest:
        selftest()
        sys.exit(0)

    if not args.source or not args.output:
        parser.error("--source and --output are required unless --selftest is given")

    convert(args.source, args.output, args.source_revision)
    sys.exit(0)


if __name__ == "__main__":
    main()
