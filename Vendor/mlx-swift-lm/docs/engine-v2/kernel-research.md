# Paged-attention kernel research (kernel-opt track)

Research + root-cause notes for the ContinuousBatchingV2 paged decode
kernels (`Libraries/MLXLMCommon/ContinuousBatchingV2/Paged/`). Three parts:

1. Apple-GPU hardware ground truth (per-family capability table, occupancy).
2. Reference-kernel survey (MLX sdpa_vector/steel, mistral.rs, vllm-metal,
   llama.cpp) with comparable threadgroup-memory/register budgets.
3. The measured root cause of the ~100x real-model slowdown
   (benchmarks/reports/gptoss-20b-mxfp4q8-main.md) and the design space for
   d512 (Gemma-4 global layers) within the 32 KB threadgroup-memory cap.

Profiling reproduction:
`DARKBLOOM_CBV2_PAGED_PROFILE=1 swift test --filter CBv2PagedProfileTests`
(driver: `Paged/PagedDecodeProfiler.swift`). Machine for all local numbers:
Apple M4 Max, 40-core GPU, 128 GB, macOS 26.5.

---

## 1. Apple GPU ground truth

### 1.1 Per-family capabilities

From the Metal Feature Set Tables (May 2026 edition), philipturner/
metal-benchmarks (measured, M1/M2 era), and Apple Tech Talk 111375:

| Family | Chips | max threadgroup mem (API) | max threads/tg | physical tg-mem / core | register file / core | simdgroup_matrix | simd reductions |
|---|---|---|---|---|---|---|---|
| Apple2–3 | A8–A10 | 16,352 B / 16 KB | 512 | n/a | n/a | no | no |
| Apple4–6 | A11–A13 | 32 KB | 1024 | n/a | n/a | no | no (quad/simd permute only) |
| Apple7 | **M1 family**, A14 | **32 KB** | 1024 | ~60 KB (measured) | ~208 KB (measured) | yes (f16/f32; bf16 emulated) | yes |
| Apple8 | **M2 family**, A15/A16 | 32 KB | 1024 | ~60 KB | ~208 KB | yes | yes (+shift/fill) |
| Apple9 | **M3/M4 families**, A17P/A18 | 32 KB | 1024 | unified/dynamic (unpublished) | dynamic (register file is a cache) | yes | yes |
| Apple10 | **M5**, A19 | 32 KB | 1024 | unpublished | dynamic | yes (+M4 tensor ops) | yes |

Key conclusions for this kernel:

- **32 KB is the API threadgroup-memory cap on every Apple Silicon Mac**
  (Apple7+ == M1 and later). Only pre-Mac A-series (Apple2/3) were 16 KB.
  The `PagedAttentionKernel.threadgroupMemoryLimit = 32 * 1024` constant is
  correct for the whole fleet; there is no per-family relaxation to chase.
- **simdgroup size 32; up to 128 32-bit registers per thread** (ISA cap,
  Dougall Johnson G13 docs). Register file ~208 KB/core on M1/M2 → one
  simdgroup at 128 regs/thread costs 16 KB → ~13 simdgroups/core at worst
  register pressure; ALU utilization saturates at 24 simds/core, and the
  compiler prefers spilling over dropping below 24.
- **Threadgroup-memory occupancy (Apple7/8)**: physical ~60 KB/core, so a
  32 KB/tg kernel runs 1 tg/core; 16 KB → 3/core; ≤ ~4 KB → tg-mem never
  the limiter. Our current part kernel at GPT-OSS shape (18.9 KB) caps at
  3 tg/core.
- **M3/M4 Dynamic Caching**: registers, threadgroup, tile, stack and buffer
  data share one on-chip pool, allocated by actual use; occupancy is
  adjusted dynamically to prevent spill. A large *declared* tg allocation
  no longer statically pins occupancy — but heavy actual use still
  throttles it. The 32 KB API cap is unchanged. Static occupancy math only
  binds on M1/M2.
- **simd_sum(f32) is expensive**: ~14.5 cycles throughput / ~28 latency vs
  ~2 for a shuffle — a per-(token, head) `simd_sum` is a real ALU cost in
  the pass-A inner loop (GQA=8 → 8 simd_sums per token per simdgroup).
  Quad reductions are ~2x cheaper; a shuffle tree can beat the builtin.
- **exp2 ~4.3 cycles**; the exp2-domain softmax (already used) is right.
- **M4 Max: 40 cores, 546 GB/s DRAM** (~80% achievable, STREAM-style).
- **Dispatch overhead**: MLX batches ~50 lazily-evaluated kernels per
  command buffer on Max-tier chips (`max_ops_per_buffer`); dispatches
  within one open compute encoder cost single-digit µs, command-buffer
  commit+wait round trips are tens of µs. Per-op host graph overhead in
  MLX is the constant to watch at 48+ dispatches/token.

### 1.2 Flash-decoding / split-K facts

- Flash-decoding = split KV into chunks, one (m, l)-annotated partial per
  chunk, second kernel merges. Standard price: O(splits x head_dim) extra
  partial traffic.
- Production split sizing: vLLM CUDA & mistral.rs & vllm-metal use fixed
  **512-token contiguous partitions**; llama.cpp uses **fixed 32
  workgroups** with strided 32-token blocks; MLX uses an adaptive block
  count (32–1024) keyed on device tier + context + `gqa*qL`. Only
  vllm-metal gates split-K on the real core count
  (`H_q * tokens < 8 * gpu_core_count`).
- DeepSeek MLA decode (DK=576/DV=512) exists on Metal in llama.cpp
  (f16/bf16 KV) and vllm-metal (`paged_mla_attention`) — d512-class decode
  is demonstrably feasible within 32 KB (see §4).

---

## 2. Reference kernels — comparable budgets

B=1 decode, H_q query heads, H_kv KV heads, GQA g, context L, head dim D,
half KV. (Sources: vendored MLX at libs/mlx-swift; mistral.rs
`mistralrs-paged-attn/src/metal/kernels/pagedattention.metal`; vllm-metal
`kernels_v2/*.metal`; llama.cpp `ggml-metal.metal` @ 4fc4ec55.)

| Kernel | tg threads | O accumulator | tg-mem bytes | acc regs/thread | grid @ B=1 | GQA KV sharing | split-K |
|---|---|---|---|---|---|---|---|
| MLX sdpa_vector (1-pass) | 1024 (32 sg) | registers | **4,352 (const, any D)** | 3D/32 | (H_q,1,1) | none | no |
| MLX sdpa_vector_2pass_1 | 32·g·qL | registers | **0** | 2D/32 | (H_kv,1,S) | **yes** (sg-per-Q-head lockstep) | S=32–1024 adaptive, strided |
| MLX sdpa_vector_2pass_2 | 1024 | registers | 4,096 | D/32 | (H_q,1,1) | — | merge pass |
| MLX steel (prefill) | 128 | registers (8x8 MMA frags) | 9.7–14.8 KB (D≤128) | D/4 | — | none | no |
| mistral.rs V2 | 256 (8 warps) | registers | 2D + 64 + max(2048, 16D) | ~D/16 | (H_q,1,⌈L/512⌉) | none | 512 contiguous |
| vllm-metal vector | 256 | registers (per-warp m,l,O) | D·szT + 64 + 32·max(BS, D+2) ⇒ ~17.5 KB @ D=512 | ⌈D/32⌉ | (H_q,1,P) | none | 512 contiguous, cores-aware gate |
| vllm-metal MLA (576/512) | 1024/512 | registers | **(2·G·32 + 1024)·4 ≈ 4.35 KB (const)** | G·16 | (H_q/G,1,1) | **yes** (G heads/tg, K regs reused) | no |
| llama.cpp vec (incl. 576/512) | 32·nsg | **threadgroup** (full padded-DV per sg) + transient regs | (PAD(DK,128)+128+2·PAD(DV,128))·nsg·2 ⇒ ≤14.3 KB @ 576/512 | DV·NE/32 transient | (1,H_q,32) | none | **fixed nwg=32**, strided |
| **ours (cbv2 part), pre-fix** | 32·NSG | registers acc[GQA][D/32] | **(GQA·D + NSG·GQA·(D+2))·4** | **GQA·D/32** | (H_kv,B,⌈L/256⌉) | **yes** (all GQA heads per KV load) | PTOK=256 contiguous |
| **ours, post head-split (§5)** | 32·NSG | registers acc[HPT][D/32] | (HPT·D + NSG·HPT·(D+2))·4 | HPT·D/32 ≤ 32 | (H_kv·GQA/HPT,B,⌈L/256⌉) | yes, HPT heads per KV load | PTOK=256 contiguous |

Why the reference caps are where they are:

- **MLX sdpa_vector caps at D=256** not because of smem (constant 4.35 KB)
  but register growth (3·D/32 floats/thread at 1024 threads) — they simply
  stopped instantiating. The guard lives in
  `scaled_dot_product_attention.cpp::use_fallback` (D ∈ {64,96,128,256},
  `qL ≤ 8`, `qL·gqa ≤ 32`).
- **STEEL caps at D=128** (guard: D ∈ {64,80,128}): O-fragment registers
  double at 256 and smem approaches 29 KB.
- **llama.cpp supports (576,512)** by keeping the persistent O in
  *threadgroup memory per simdgroup* with **no GQA multiplier** (one query
  head per threadgroup) and a small transient register accumulator.
- **vllm-metal MLA** fuses G ∈ {1,2} query heads per threadgroup and still
  keeps smem constant by using the **MLX 32x32 transpose-tile merge**.

The three cross-simdgroup merge schemes that avoid a full-D buffer per
(simdgroup, head) — the thing our RSTRIDE=D+2 red_smem does wrong:

1. **32x32 transpose tile** (MLX 1-pass/2-pass-2, vllm MLA): lane owns
   D/32 elements in registers; merge loops over element slots; round i
   writes `o[i]` to `tile[lane*32 + sg]`, barrier, reads transposed,
   `simd_sum` across lanes (which now enumerate simdgroups). Constant
   4 KB smem regardless of D and GQA; cost (D/32)·GQA rounds x 2 barriers.
2. **No intra-tg merge at all** (MLX 2pass_1): one simdgroup owns one query
   head end-to-end; splits only across threadgroups, merged by pass 2
   through device memory. smem = 0.
3. **Warp-count-bounded smem merge** (mistral.rs/llama.cpp/vllm vector):
   per-warp full-D regions, merged by tree — linear in D but multiplied by
   warp count (≤8), never by GQA, because one query head per threadgroup.

Rule extracted: **never put a GQA multiplier on a full-D threadgroup-memory
surface**. The two production kernels that fuse GQA (MLX 2pass_1, vllm MLA)
both have GQA-independent smem (0 / 4.35 KB).

---

## 3. Root cause of the ~100x GPT-OSS slowdown (measured)

### 3.1 Symptom recap

- Parity suites and the 1-layer microbenchmark
  (benchmarks/cbv2-paged-attention.md) are healthy: 1.07 ms/step at the
  exact GPT-OSS shape (B=1, ctx 512), beating per-row SDPA at B≥2.
- Real GPT-OSS-20B decode: ~2 s/token, 30 s step-watchdog kills every
  request (benchmarks/reports/gptoss-20b-mxfp4q8-main.md).
- Difference: the microbench ran with a ~2 MB pool; the real engine ran
  with `kvBytes = 16 GiB` → one group (all 24 layers share (kvh=8, d=64))
  → **one 8 GiB K slab + one 8 GiB V slab**.

### 3.2 Ablation numbers (PagedDecodeProfiler, M4 Max)

Slab-size sweep, 1 layer, B=1, ctx 512 — flat, no O(slab) term:

| slab | write+dispatch | dispatch-only | write-only |
|---|---|---|---|
| 0.06 GiB | 1.58 ms | 1.02 ms | 0.51 ms |
| 8.00 GiB | 2.18 ms | 1.60 ms | (noise ≤ 7.8) ms |

Layer sweep, fixed 2 GiB pool (1 GiB K + 1 GiB V slabs), B=1, ctx 512:

| layers | write+dispatch | step peak mem | write-only | step peak mem |
|---|---|---|---|---|
| 1 | 0.49 ms | 0.00 GiB | 0.33 ms | 0 |
| 2 | 1.86 ms | 2.00 GiB | 0.50 ms | 0 |
| 4 | 15.5 ms | 6.00 GiB | 0.89 ms | 0 |
| 8 | 62.3 ms | 8.00 GiB | 1.51 ms | 0 |
| 16 | 134.2 ms | 8.00 GiB | 3.06 ms | 0 |
| 24 | **236.4 ms** | 8.00 GiB | **4.64 ms** | 0 |

24-layer GPT-OSS emulation (alternating window-128/full, sinks), 16 GiB
pool — reproduces the field failure without weights:

| B | paged write+dispatch | per-row contiguous SDPA |
|---|---|---|
| 1 | **25,470 ms/step** | 4.7 ms/step |
| 4 | **1,999 ms/step** | 45.5 ms/step |

Attribution: write-only chains are linear and cheap with zero transient
memory — **slab slice-update writes donate fine, even chained 24 deep in
one graph**. Interleaving the attention-kernel dispatches makes every
subsequent write allocate a slab-sized buffer (step-peak ≈ 8 GiB) and
ms/step grows linearly in layer count with slope ≈ 2 slabs x bandwidth
(46 GiB / 0.236 s ≈ 195 GB/s — a plain GPU copy rate).

### 3.3 Mechanism (vendored MLX source)

`mlx/backend/metal/eval.cpp::gpu::eval`:

```cpp
std::unordered_set<std::shared_ptr<array::Data>> buffers;
for (auto& in : arr.inputs())    buffers.insert(in.data_shared_ptr());
for (auto& s  : arr.siblings())  buffers.insert(s.data_shared_ptr());
// Remove the output if it was donated to by an input
if (auto it = buffers.find(arr.data_shared_ptr()); it != buffers.end())
  buffers.erase(it);
...
command_buffer->addCompletedHandler([buffers = std::move(buffers)](...){});
```

Every op's **input** `Data` handles are captured in the command buffer's
completion handler and held until the GPU finishes the buffer. A donated
slice-update output is deliberately erased (why pure write chains keep
donating), but a **custom-kernel read pins the slab's data**. The next
layer's `slice_update` then fails `is_donatable()`
(`array.h`: `array_desc_.use_count() == 1 && data.use_count() == 1`)
and `SliceUpdate::eval_gpu` falls back to `copy_gpu(in, out, Vector)` —
a **full-slab copy**: 23 layers x {K,V} x 8 GiB ≈ 370 GiB of GPU copies
per decoded token ≈ 1.9 s at copy bandwidth. At 16 GiB pools the transient
copies also trip MLX's memory-limit throttle
(`transforms.cpp: get_active_memory() > get_memory_limit()` → serialize),
which is why B=1 measured even worse (25 s) than pure copy math predicts.

Corollaries:

- The failure needs ≥ 2 layers sharing a slab group in one step graph —
  which is why every parity test (small pools) and the 1-layer
  microbenchmark stayed green. This was a designed-in trap:
  `PagedKVPool.swift`'s header even documented "consumers MUST NOT retain
  … kernel inputs" but the *MLX runtime itself* is such a consumer.
- Any fix that keeps per-layer `slice_update` on multi-GiB shared slabs is
  dead on arrival: the retention window is the command buffer lifetime,
  which under 2-step pipelining always covers the next write.

### 3.4 Fix direction (what production systems do)

vLLM (`reshape_and_cache`), llama.cpp, and mistral.rs never version their
KV cache through functional updates: the cache is a stable buffer written
**in place by a kernel**. The MLX-shaped equivalent:

1. **Decode: fuse the KV write into pass A.** The `[B, kvh, D]` K/V tiles
   become kernel inputs; ONE threadgroup per (row, kv-head) — hgroup 0 of
   the row's last partition — stores the tile into the page slot, and NO
   threadgroup reads that slot from the slab in the same dispatch: the
   newest position is always loaded from the knew/vnew inputs
   (bit-identical bytes), so the write is race-free without any
   cross-threadgroup primitive. Other layers/rows touch disjoint bytes,
   and the slab MLXArray never changes version → nothing to donate, no
   copies, and 2 slice-update dispatches per layer disappear.
2. **Prefill/bulk: a dedicated write kernel** (`const_cast` the slab input,
   vLLM reshape_and_cache shape) emitting a tiny fence array. MLX encoders
   are `MTL::DispatchTypeConcurrent` with hazard tracking at declared
   buffer-set granularity (`device.cpp::set_input_array/register_output_
   array`), so in-place writes through an "input" are invisible to the
   tracker: every subsequent *reader of the same bytes* must consume the
   fence (directly or transitively) to get the memory barrier. Gathers
   consume it via a zero-cost index add; cross-step ordering is already
   guaranteed because step N+1's graph always consumes step N's outputs
   (chained decode) or finalize host-syncs them (mixed steps).

---

## 4. d512 (Gemma-4 global layers, GQA 8) design space

Current formula (Swift + shader in lockstep):
`(GQA·D + NSG·GQA·(D+2)) · 4` bytes → d512/GQA8 = 32,832 B at NSG=1 —
over the 32,768 B cap, statically ineligible. Both terms are wrong for
large D:

- `red_smem[NSG·GQA·(D+2)]` — a full-D merge record per (simdgroup, head):
  the anti-pattern no production kernel uses (§2).
- `acc[GQA][D/32]` per-thread registers = 128 floats at d512/GQA8 — over
  the 128-register ISA budget on its own → guaranteed spill.

Options with concrete budgets:

**A. Simdgroup-per-query-head + split-K across threadgroups**
(MLX 2pass_1 shape). 8 sgs/tg, sg s owns head s, all sgs walk the same KV
in lockstep (memory system serves one read). Regs ≈ 50 floats/thread;
smem 0; grid (H_kv, B, S). Best endgame, biggest rewrite; loses the
explicit K-register reuse across heads.

**B. Keep fused-GQA/split-KV shape; replace the merge with the 32x32
transpose tile.** smem = q_smem[GQA·D] + 4 KB tile + NSG·GQA·2 m/l floats:
d512/GQA8 → 16 KB + 4 KB + small ≈ **20.5 KB, fits at any NSG**; also
shrinks GPT-OSS d64/GQA8 from 18.9 KB → ~6.3 KB (3→9 tg/core on M1/M2).
Does NOT fix the 128-float register accumulator at d512/GQA8.

**C. GQA-split across threadgroups** (grid x = KVH·ceil(GQA/G)): each
threadgroup handles G ≤ GQA heads. smem = (G·D + NSG·G·(D+2))·4; regs =
G·D/32. d512 with G=2: smem = (1024 + NSG·1028)·4 → NSG=6 fits (28.7 KB);
regs 32 floats. KV is re-read GQA/G times (vLLM RFC 15351 measured ~30%
loss from GQA re-reads at g=8 — acceptable for eligibility, not optimal).

**D. llama.cpp-style smem-resident O per (sg, owned-head)**: only wins
when registers are needed for something else (e.g. quantized-KV dequant
scratch).

Chosen for C.2: **C**, the head split (LANDED — see §5): factor
G = HPT chosen so `G·D/32 ≤ 32` acc floats per thread (d512 → G=2,
d256 → G=4 at GQA 8; d64/d128 → G == GQA, kernel unchanged). The
eligibility formula models the split exactly (`headsPerThreadgroup` +
`partThreadgroupBytes`); `CBv2PagedEligibilityTests` pins the pairing.
**B**'s transpose-tile merge remains a follow-up: it would shrink the
smem formula for every shape and improve occupancy fleet-wide, but the
head split alone already brings every supported head dim under the cap.

### Split sizing note (B=1 parallelism)

At GPT-OSS shapes the grid is (8 kv-heads x parts): ctx 512 → 16
threadgroups on a 40-core GPU. PTOK=256 is fine at ctx ≥ 2K but thin
below; the references either split much harder at decode (llama.cpp:
always 32 splits; MLX: 64–1024 blocks) or gate on core count
(vllm-metal). Worth revisiting PTOK/adaptive splits only after the copy
fix lands — the copy elimination is 3 orders of magnitude, split tuning
is percent-level at short contexts.

## 5. Outcome (post-fix, 2026-07-02)

Both fixes landed on this branch (in-place writes bc358e2, head split
5087663). Measured results:

Profiler (same scenarios as §3.2):

| scenario | before | after |
|---|---|---|
| layer sweep, 24 layers, 2 GiB pool, B=1 | 236.4 ms/step (8 GiB step-peak) | **11.4 ms/step (0 transient)** |
| GPT-OSS emulation, 16 GiB pool, B=1 | 25,470 ms/step | **13.6 ms/step** |
| GPT-OSS emulation, 16 GiB pool, B=4 | 1,999 ms/step | **14.3 ms/step** |

Real weights (BenchCBv2, same-run v2 vs v2-paged, M4 Max — reports
`gptoss-20b-mxfp4q8-kernel-opt.md`, `gemma4-26b-a4b-8bit-kernel-opt.md`):

| model | metric | v2 contiguous | v2-paged |
|---|---|---|---|
| GPT-OSS-20B | B=1 decode TPS | 99.5 | 89.9 (0.90x) |
| GPT-OSS-20B | B=4 aggregate TPS | 84.4 | **86.5 (beats)** |
| Gemma-4-26B | B=1 decode TPS | 67.2 | 66.8 (0.99x) |
| Gemma-4-26B | B=4 aggregate TPS | 56.3 | **59.3 (beats)** |

- The B=1 targets (≤ 2x of contiguous) are met with wide margin; B≥4
  paged beats contiguous on both models — the point of paged decode.
- Gemma-4-26B runs the paged path for the FIRST time: its global layers
  (d512, GQA 8) were statically ineligible before the head split.
- The teardown SIGSEGV noted in the original GPT-OSS report did not
  reproduce: it followed watchdog-errored runs, which no longer occur.
- Cross-report caution: the original `-main.md` runs were recorded on a
  contended machine (sibling builds); only same-run engine comparisons
  are meaningful.

Follow-ups (not blocking):

- Transpose-tile merge (§2 scheme 1) would shrink pass-A smem to ~4 KB
  at every shape (d64/GQA8: 18.9 KB → ~6 KB) and lift occupancy on
  M1/M2; percent-level, superseded in urgency by the copy fix.
- PTOK/split tuning for very short contexts at B=1 (§4 note).
- Quantized KV pages (PagedKVPool design TODO) can reuse the bulk-write
  kernel's scatter shape with inline quantization.

## 6. References

- philipturner/metal-benchmarks, Dougall Johnson applegpu docs
- Apple Metal Feature Set Tables (2026-05); Tech Talks 111375/111374
- vLLM paged_attention_v2 + RFC #15351 (GQA-aware v3)
- mistral.rs mistralrs-paged-attn Metal kernels
- vllm-metal kernels_v2 (vector/tiled/MLA)
- llama.cpp ggml-metal flash-attn (incl. DK576/DV512 vec kernel)
- MLX sdpa_vector / steel_attention + scaled_dot_product_attention.cpp
- Flash-Decoding (Dao et al., PyTorch blog); FlashMLA (DeepSeek)
- d-inference/docs/research/batching-engine-2026-07/ (reports 04, 07, 08)
