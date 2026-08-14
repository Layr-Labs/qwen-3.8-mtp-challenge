# WS-C: Paged KV backend + Metal paged-attention kernel (optimization track)

Swappable replacement for WS-A's contiguous backend behind the SAME
contracts (`CBv2KVBackend`, `CBv2SequenceKV`, attention via a
`CBv2AttendingLayerCache` variant). Ship-optional: if kernel perf misses the
bar, v2 ships on the v1 backend and this merges later. Work independently;
do not block on other workstreams.

## Deliverables (`Libraries/MLXLMCommon/ContinuousBatchingV2/Paged/`)

1. `PagedKVPool.swift` — global slab pool per model: a FEW large MLXArray
   slab buffers per layer group (NEVER per-page buffers — the ~499k Metal
   resource-count ceiling is real, see report 04), page size 16 tokens
   (constant, revisit later), free-list allocator with O(1) alloc/free,
   refcounts per page (prefix sharing later), truthful bytesInUse.
   Capacity: configured bytes, wired via existing WiredMemory policies.
2. `PagedSequenceKV.swift` — `CBv2SequenceKV` holding a page table
   (`[Int32]` page ids); `update` writes new K/V into pages via a scatter
   (`reshape_and_cache` style — use bound dynamic slice update per page, or
   a small custom scatter kernel); snapshot/rollback/release = table ops.
3. `pagedattention.metal` + `PagedAttentionKernel.swift` — decode kernel via
   `MLXFast.metalKernel` (exposed in Swift — MLXFastKernel.swift:98). Port
   from mistral.rs (`mistralrs-paged-attn/src/metal/pagedattention.metal`)
   / vllm-metal designs — clone them to a temp dir to study. Requirements:
   GQA, per-request KV lengths, block tables, sliding window, softcap
   optional, per-head sinks (denominator-only), head_dim ∈ {64, 128, 256,
   512} (Gemma-4 global layers are 512 — verify shared-memory budget; fall
   back to composed path per head-dim if needed), fp16 KV first (quantized
   pages later).
4. `PagedLayerCache.swift` — `CBv2AttendingLayerCache` calling the kernel
   for B>1 decode; per-request SDPA fallback for prefill chunks (same as
   v1) until a varlen extend kernel exists (stretch goal, not required).
5. `PagedBackendBenchmark.swift` (+ a `benchmarks/` entry): kernel
   microbench vs v1 per-row SDPA and vs legacy dense batch: B ∈ {1,2,4,8},
   context ∈ {512, 4k, 16k}, report achieved GB/s and tokens/s. Print a
   markdown table. THIS IS THE GATE: paged decode must beat v1 per-row
   dispatch at B≥2 and be within 15% of single-request SDPA at B=1.
6. `CBv2KVQuantScheme` hook: leave quantized pages as a TODO with the
   design documented in the file header (fp16 pages are the v2.0 fallback
   for KV-quant configs).

## Correctness bar
Kernel parity vs a composed reference (matmul+softmax over gathered pages):
rel-err ≤ 1e-2 fp16 elementwise on random shapes covering GQA/window/sinks;
plus end-to-end greedy token-match ≥ 99.5% over 200 decode steps vs v1
backend on the harness tiny model.

## References
Report 08 §5 (kernel contract), report 07 (Metal ecosystem, mistral.rs/
vllm-metal), report 04 (metalKernel API, resource limits, no output
donation — KV writes must NOT go through the attention kernel). Corpus:
`/Users/gaj/Documents/Builds/d-inference/docs/research/batching-engine-2026-07/`.
