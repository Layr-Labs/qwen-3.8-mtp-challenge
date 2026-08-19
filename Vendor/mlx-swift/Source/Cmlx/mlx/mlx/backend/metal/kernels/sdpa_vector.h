// Copyright © 2024 Apple Inc.

#include <metal_simdgroup>

using namespace metal;

constant bool has_mask [[function_constant(20)]];
constant bool query_transposed [[function_constant(21)]];
constant bool do_causal [[function_constant(22)]];
constant bool bool_mask [[function_constant(23)]];
constant bool float_mask [[function_constant(24)]];
constant bool has_sinks [[function_constant(25)]];
constant int blocks [[function_constant(26)]];

// ---------------------------------------------------------------------------
// GQA K/V de-duplication for the decode-time vector kernels.
//
// TECHNIQUE. Ported from ml-explore/mlx#4077 ("Read each K/V byte once in gqa-8
// decode attention", commit fa0d4463), adapted to run under this repository's
// UNMODIFIED host dispatch.
//
// Both vector kernels give every QUERY head its own unit of work: sdpa_vector
// launches one threadgroup per (query head, query row), and sdpa_vector_2pass_1
// one SIMD group per (query head, query row) inside a per-KV-head threadgroup.
// Each unit streams the whole K/V range of its KV head, so at a GQA factor of G
// every K/V byte is fetched G times per query row -- 6x at q_seq_len 1 here,
// and 30x for a 5-row verify segment, where 120 threadgroups share 4 KV heads.
// Those units differ only in their scores; the K and V bytes they read are the
// same bytes. The fix is to let one unit carry NH consecutive query heads of
// the same KV group -- loading a K/V row once into registers and running NH
// independent online-softmax accumulators against it -- and to retire the NH-1
// units left with nothing to do.
//
// WHY THIS PORT DIFFERS FROM UPSTREAM. Upstream added a kernel name, a launch
// geometry and a host predicate that selects it. Here
// backend/metal/scaled_dot_product_attention.cpp is not editable, so the grid,
// the threadgroup shape, the kernel names and the buffer bindings all stay
// exactly as they are and only the assignment of heads to threadgroups changes.
// Upstream's kernel also is not bitwise equal to the stock one and does not
// claim to be: it walks a contiguous token sub-chunk ("t = s0; t < s1; t++")
// instead of the stock strided range, which reorders the softmax rescaling, and
// its test asserts allclose at 1e-4. The whole #4077 family can therefore only
// be adopted here as an ownership remap, never as a file copy.
//
// EXACTNESS ARGUMENT. For a fixed query head the stock result is determined by
// the ordered key positions the unit visits, the arithmetic applied at each,
// and the cross-unit reduction. The remap preserves all three: the token walk
// ("i = simd_gid; i < N; i += BN" one-pass, "i = block_idx; i < N; i += blocks"
// two-pass) depends on the SIMD group / block index and never on the head; the
// per-position arithmetic is the stock statement sequence with the
// head-independent K/V loads hoisted, staged in each kernel's own operand types
// (U for the one-pass K, which stock stages through "thread U k[]"; T for the
// two-pass K and for V in both, which stock multiplies straight from memory) so
// no promotion changes; do_causal reads the key index and query row but never
// the head; the one-pass cross-SIMD-group reduction is replayed per head over
// the same 32 partials in the same lanes; and the two-pass kernel writes its
// own (head, row, block) partial slots, leaving sdpa_vector_2pass_2 untouched.
//
// EARLY RETURN SAFETY. The one-pass kernel has threadgroup barriers, so its
// retirement is keyed on tid.x, uniform over the whole threadgroup: a retiring
// threadgroup reaches no barrier. The two-pass kernel has no barrier, and its
// retirement is keyed on tidtg.y, uniform over a SIMD group (the threadgroup is
// exactly one SIMD group wide in x), so every surviving simd_sum keeps a full
// SIMD group.
//
// MEASURED SCOPE. Verified bitwise identical to the pre-change kernels by an
// A/B harness: both metallibs built through tools/build-mlx-metallib.sh with
// only this file swapped, random bf16 inputs, output memcmp. 157 cases -- a
// dense q_seq_len == 1 sweep of kL 512..1024 across three seeds, q_seq_len
// 2..5 over seven kL values across three seeds, and the two-pass partials at
// kL 1024 and 2048. q_seq_len 1..5 is the whole reachable domain: use_fallback
// requires q_seq_len * gqa <= 32, so at gqa 6 nothing above 5 reaches these
// kernels at all.
//
// HOW TO MEASURE THIS, AND HOW NOT TO -- READ BEFORE CHANGING ANYTHING HERE.
// An earlier sheet reported mismatches at q_seq_len >= 2 and in the two-pass
// partials, and the dedup was briefly narrowed to single-row queries on that
// evidence. Those mismatches were an artifact of the harness, not of this code.
// The reference metallib was being compiled with a hand-rolled
// "xcrun metal -O3" while the candidate came from the real build. MLX compiles
// its kernels with -fno-fast-math and default optimization
// (mlx/backend/metal/kernels/CMakeLists.txt); -O3 turns fast-math ON, which
// changes fma contraction and the lowering of fast::exp, so IDENTICAL SOURCE
// produced different low bits. A null control -- the same source compiled both
// ways -- reproduced every one of those "mismatches", and building both sides
// through the real build reproduces none.
//
// The lesson worth keeping: an A/B between two differently-built binaries
// measures the build as well as the source. Validate any such oracle on
// identical source before trusting a single row of it.
//
// SCOPE. Dedup additionally requires no array mask and no sink logits, the only
// per-head inputs besides Q; excluding them keeps the merged loop free of
// per-head pointer state, and both are absent from the decode shapes this build
// serves. When any part of the guard fails, control falls through to the stock
// body, which is left untouched.
//
// WHY THIS PATH IS WORTH IT. q_seq_len == 1 carries all serial-control decode
// and every sequential draft step of the native MTP head, so it is the hot path
// for the configuration most likely to be submitted. At 24 query heads over 4
// KV heads, head_dim 256, bf16, across the 16 full-attention layers, a draft
// step re-reads roughly 300 MB of K/V at kL 768; NH = 2 halves that.
//
// CHOICE OF NH. head_dim is 256, so BD = 32 gives qk_per_thread =
// v_per_thread = 8 and each extra head costs 16 more 32-bit registers (8 for Q,
// 8 for the output accumulator). sdpa_vector is dispatched with a 1024-thread
// threadgroup and the host does NOT call check_kernel_threadgroup_size on that
// path, so a register count that pushes maxTotalThreadsPerThreadgroup under
// 1024 would fail at dispatch rather than at build time. NH = 2 keeps the hot
// working set near 64 registers, inside the budget that already sustains 1024
// threads, and it halves rather than decimates the resident threadgroup count:
// at qL = 1 this model launches 24 threadgroups against 20 GPU cores, so NH = 6
// would leave 4 and idle most of the machine while NH = 2 leaves 12. NH must
// divide the GQA factor; the guard checks that at runtime.
//
// Setting this to 1 folds both guards away at compile time and restores the
// stock kernels exactly, which is the A/B knob for measuring the change.
// ---------------------------------------------------------------------------
constexpr constant int sdpa_vector_heads_per_tg = 2;

// One-pass body for a threadgroup that carries NH consecutive query heads.
// Mirrors the loop and reduction of sdpa_vector statement for statement; see
// the exactness note above.
template <typename T, int D, int V, int NH>
METAL_FUNC void sdpa_vector_dedup_impl(
    const device T* queries,
    const device T* keys,
    const device T* values,
    device T* out,
    int head0,
    int q_seq_idx,
    int q_seq_len,
    int n_batch_heads,
    int gqa_factor,
    int N,
    size_t k_head_stride,
    size_t k_seq_stride,
    size_t v_head_stride,
    size_t v_seq_stride,
    float scale,
    threadgroup float* outputs,
    threadgroup float* max_scores,
    threadgroup float* sum_exp_scores,
    uint simd_gid,
    uint simd_lid) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[NH][qk_per_thread];
  thread U o[NH][v_per_thread];
  thread U max_score[NH];
  thread U sum_exp_score[NH];

  // All NH heads belong to one KV group, so they share a single K/V stream.
  const int kv_head_idx = head0 / gqa_factor;

  for (int h = 0; h < NH; h++) {
    const int o_offset = (head0 + h) * q_seq_len + q_seq_idx;
    const int q_offset =
        query_transposed ? n_batch_heads * q_seq_idx + head0 + h : o_offset;
    const device T* qh = queries + q_offset * D + simd_lid * qk_per_thread;
    for (int i = 0; i < qk_per_thread; i++) {
      q[h][i] = static_cast<U>(scale) * qh[i];
    }
    for (int i = 0; i < v_per_thread; i++) {
      o[h][i] = 0;
    }
    max_score[h] = Limits<U>::finite_min;
    sum_exp_score[h] = 0;
  }

  const device T* kp = keys + kv_head_idx * k_head_stride +
      simd_gid * k_seq_stride + simd_lid * qk_per_thread;
  const device T* vp = values + kv_head_idx * v_head_stride +
      simd_gid * v_seq_stride + simd_lid * v_per_thread;

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - q_seq_len + q_seq_idx);
    }
    if (use_key) {
      // Read the key and value once for all NH heads. k is staged in U and v
      // in T, matching the operand types of the stock kernel's multiplies.
      thread U k[qk_per_thread];
      thread T vr[v_per_thread];
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = kp[j];
      }
      for (int j = 0; j < v_per_thread; j++) {
        vr[j] = vp[j];
      }

      for (int h = 0; h < NH; h++) {
        // Compute the i-th score
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
          score += q[h][j] * k[j];
        }
        score = simd_sum(score);

        // Update the accumulators
        U new_max = max(max_score[h], score);
        U factor = fast::exp(max_score[h] - new_max);
        U exp_score = fast::exp(score - new_max);

        max_score[h] = new_max;
        sum_exp_score[h] = sum_exp_score[h] * factor + exp_score;

        // Update the output accumulator
        for (int j = 0; j < v_per_thread; j++) {
          o[h][j] = o[h][j] * factor + exp_score * vr[j];
        }
      }
    }

    // Move the pointers to the next kv
    kp += inner_k_stride;
    vp += inner_v_stride;
  }

  // Replay the stock cross-SIMD-group reduction once per head. The threadgroup
  // scratch is reused between heads: the previous head's last read of it is
  // separated from the next head's first write by the barriers inside the
  // v_per_thread loop, which every thread executes at least once.
  for (int h = 0; h < NH; h++) {
    U max_score_h = max_score[h];
    if (simd_lid == 0) {
      max_scores[simd_gid] = max_score_h;
      sum_exp_scores[simd_gid] = sum_exp_score[h];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score_h = max_scores[simd_lid];
    U new_max = simd_max(max_score_h);
    U factor = fast::exp(max_score_h - new_max);
    U sum_exp_score_h = simd_sum(sum_exp_scores[simd_lid] * factor);

    for (int i = 0; i < v_per_thread; i++) {
      outputs[simd_lid * BD + simd_gid] = o[h][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      o[h][i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
      o[h][i] = sum_exp_score_h == 0 ? o[h][i] : (o[h][i] / sum_exp_score_h);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // And write the output
    if (simd_lid == 0) {
      const int o_offset = (head0 + h) * q_seq_len + q_seq_idx;
      device T* op = out + o_offset * V + simd_gid * v_per_thread;
      for (int i = 0; i < v_per_thread; i++) {
        op[i] = static_cast<T>(o[h][i]);
      }
    }
  }
}

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const constant int& N [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride [[buffer(7)]],
    const constant size_t& v_head_stride [[buffer(8)]],
    const constant size_t& v_seq_stride [[buffer(9)]],
    const constant float& scale [[buffer(10)]],
    const device bool* bmask [[buffer(11), function_constant(bool_mask)]],
    const device T* fmask [[buffer(12), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(13), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(14), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(15), function_constant(has_mask)]],
    const device T* sinks [[buffer(16), function_constant(has_sinks)]],
    const constant int& num_q_heads
    [[buffer(17), function_constant(has_sinks)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Let one threadgroup carry NH consecutive query heads of a KV group and
  // retire the rest, so each K/V byte is fetched once per NH heads. tid.x is
  // uniform over the threadgroup, so a retiring threadgroup reaches no barrier.
  // Heads [head0, head0 + NH) share a KV head because NH divides gqa_factor and
  // head0 is NH-aligned; tpg.x being a multiple of NH keeps a group from
  // straddling the end of the grid or a batch boundary.
  //
  // Applies at every query-row count the vector path serves. use_fallback in
  // scaled_dot_product_attention.cpp requires q_seq_len * gqa <= 32, so at
  // gqa 6 that is q_seq_len 1..5 -- which is exactly why the promoted
  // AttentionUtils.swift splits verify into <= 5-row segments.
  constexpr int NH = sdpa_vector_heads_per_tg;
  if (NH > 1 && !has_mask && !has_sinks && (gqa_factor % NH) == 0 &&
      (int(tpg.x) % NH) == 0) {
    if ((int(tid.x) % NH) != 0) {
      return;
    }
    sdpa_vector_dedup_impl<T, D, V, NH>(
        queries,
        keys,
        values,
        out,
        int(tid.x),
        int(tid.y),
        int(tpg.y),
        int(tpg.x),
        gqa_factor,
        N,
        k_head_stride,
        k_seq_stride,
        v_head_stride,
        v_seq_stride,
        scale,
        outputs,
        max_scores,
        sum_exp_scores,
        simd_gid,
        simd_lid);
    return;
  }

  // Adjust positions
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;
  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx % num_q_heads]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Read the key
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
    if (bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// Two-pass first-pass body for a SIMD group that carries NH consecutive query
// heads. Mirrors the loop of sdpa_vector_2pass_1 statement for statement and
// writes the same (head, row, block) partial slots, so sdpa_vector_2pass_2 is
// unchanged. See the exactness note above.
template <typename T, int D, int V, int NH>
METAL_FUNC void sdpa_vector_2pass_1_dedup_impl(
    const device T* queries,
    const device T* keys,
    const device T* values,
    device T* out,
    device float* sums,
    device float* maxs,
    int head_in_group0,
    int kv_head_idx,
    int batch_idx,
    int block_idx,
    int num_kv_heads,
    int gqa_factor,
    int q_seq_len,
    int q_seq_idx,
    int N,
    size_t k_head_stride,
    size_t k_seq_stride,
    size_t v_head_stride,
    size_t v_seq_stride,
    float scale,
    uint simd_lid) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;

  typedef float U;

  thread U q[NH][qk_per_thread];
  thread U o[NH][v_per_thread];
  thread U max_score[NH];
  thread U sum_exp_score[NH];

  const int num_q_heads = num_kv_heads * gqa_factor;

  for (int h = 0; h < NH; h++) {
    const int q_head_idx = gqa_factor * kv_head_idx + head_in_group0 + h;
    const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;
    const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
    const int q_offset = query_transposed
        ? num_q_heads * q_seq_idx + q_batch_head_idx
        : o_offset;
    const device T* qh = queries + q_offset * D + simd_lid * qk_per_thread;
    for (int i = 0; i < qk_per_thread; i++) {
      q[h][i] = static_cast<U>(scale) * qh[i];
    }
    for (int i = 0; i < v_per_thread; i++) {
      o[h][i] = 0;
    }
    max_score[h] = Limits<U>::finite_min;
    sum_exp_score[h] = 0;
  }

  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
  const device T* kp = keys + kv_batch_head_idx * k_head_stride +
      block_idx * k_seq_stride + simd_lid * qk_per_thread;
  const device T* vp = values + kv_batch_head_idx * v_head_stride +
      block_idx * v_seq_stride + simd_lid * v_per_thread;

  // For each key
  for (int i = block_idx; i < N; i += blocks) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - q_seq_len + q_seq_idx);
    }
    if (use_key) {
      // Read the key and value once for all NH heads. Both are staged in T,
      // the operand type the stock kernel multiplies straight out of memory.
      thread T kr[qk_per_thread];
      thread T vr[v_per_thread];
      for (int j = 0; j < qk_per_thread; j++) {
        kr[j] = kp[j];
      }
      for (int j = 0; j < v_per_thread; j++) {
        vr[j] = vp[j];
      }

      for (int h = 0; h < NH; h++) {
        // Compute the i-th score
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
          score += q[h][j] * kr[j];
        }
        score = simd_sum(score);

        // Update the accumulators
        U new_max = max(max_score[h], score);
        U factor = fast::exp(max_score[h] - new_max);
        U exp_score = fast::exp(score - new_max);

        max_score[h] = new_max;
        sum_exp_score[h] = sum_exp_score[h] * factor + exp_score;

        // Update the output accumulator
        for (int j = 0; j < v_per_thread; j++) {
          o[h][j] = o[h][j] * factor + exp_score * vr[j];
        }
      }
    }

    // Move the pointers to the next kv
    kp += blocks * int(k_seq_stride);
    vp += blocks * int(v_seq_stride);
  }

  // Write the sum and max and outputs
  for (int h = 0; h < NH; h++) {
    const int q_head_idx = gqa_factor * kv_head_idx + head_in_group0 + h;
    const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;
    const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;

    if (simd_lid == 0) {
      sums[o_offset * blocks + block_idx] = sum_exp_score[h];
      maxs[o_offset * blocks + block_idx] = max_score[h];
    }

    device T* op = out + o_offset * blocks * V + block_idx * V +
        simd_lid * v_per_thread;
    for (int i = 0; i < v_per_thread; i++) {
      op[i] = static_cast<T>(o[h][i]);
    }
  }
}

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    const device bool* bmask [[buffer(13), function_constant(bool_mask)]],
    const device T* fmask [[buffer(14), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
    [[buffer(15), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
    [[buffer(16), function_constant(has_mask)]],
    const constant int& mask_head_stride
    [[buffer(17), function_constant(has_mask)]],
    const device T* sinks [[buffer(18), function_constant(has_sinks)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;

  typedef float U;

  // Let one SIMD group carry NH consecutive query heads of its KV group and
  // retire the rest. tidtg.y is uniform over a SIMD group (the threadgroup is
  // exactly one SIMD group wide in x) and this kernel has no barrier, so the
  // retirement leaves every surviving simd_sum with a full SIMD group.
  constexpr int NH = sdpa_vector_heads_per_tg;
  if (NH > 1 && !has_mask && !has_sinks && (int(tptg.y) % NH) == 0) {
    if ((int(tidtg.y) % NH) != 0) {
      return;
    }
    sdpa_vector_2pass_1_dedup_impl<T, D, V, NH>(
        queries,
        keys,
        values,
        out,
        sums,
        maxs,
        int(tidtg.y),
        int(tid.x),
        int(tid.y),
        int(tid.z),
        int(tpg.x),
        int(tptg.y),
        int(tptg.z),
        int(tidtg.z),
        N,
        k_head_stride,
        k_seq_stride,
        v_head_stride,
        v_seq_stride,
        scale,
        simd_lid);
    return;
  }

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  // Adjust positions
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int q_seq_len = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
  const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
  const int q_offset =
      query_transposed ? num_q_heads * q_seq_idx + q_batch_head_idx : o_offset;

  queries += q_offset * D + simd_lid * qk_per_thread;

  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
  keys += kv_batch_head_idx * k_head_stride + block_idx * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_batch_head_idx * v_head_stride + block_idx * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  // Read the query
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && block_idx == 0) {
    max_score = static_cast<U>(sinks[q_head_idx]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = block_idx; i < N; i += blocks) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - q_seq_len + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Compute the i-th score
      U score = 0;
      for (int i = 0; i < qk_per_thread; i++) {
        score += q[i] * keys[i];
      }
      score = simd_sum(score);

      if (float_mask) {
        score += fmask[0];
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int i = 0; i < v_per_thread; i++) {
        o[i] = o[i] * factor + exp_score * values[i];
      }
    }

    // Move the pointers to the next kv
    keys += blocks * int(k_seq_stride);
    values += blocks * int(v_seq_stride);
    if (bool_mask) {
      bmask += blocks * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += blocks * mask_kv_seq_stride;
    }
  }

  // Write the sum and max and outputs
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }

  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<T>(o[i]);
  }
}

template <typename T, int D>
[[kernel]] void sdpa_vector_2pass_2(
    const device T* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& blocks [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  typedef float U;

  thread U o[elem_per_thread] = {0};
  threadgroup U outputs[BN * BD];

  // Adjust positions
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int q_offset = head_idx * tpg.y + q_seq_idx;
  partials += q_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += q_offset * blocks;
  maxs += q_offset * blocks;
  out += q_offset * D + simd_gid * elem_per_thread;

  // Set defaults
  U sum_exp_score = 0.0;
  U max_score = Limits<U>::finite_min;

  // Reduce the max
  for (int b = 0; b < blocks / BN; ++b) {
    max_score = max(max_score, maxs[simd_lid + BN * b]);
  }
  max_score = simd_max(max_score);

  // Reduce the d
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
    sum_exp_score += factor * sums[simd_lid + BN * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);

  // Reduce the sum exp and partials
  for (int b = 0; b < blocks / BN; ++b) {
    U factor = fast::exp(maxs[simd_gid] - max_score);

    // Update the output accumulator
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] += factor * static_cast<U>(partials[i]);
    }
    maxs += BN;
    sums += BN;
    partials += BN * D;
  }

  // Use shared memory to transpose and reduce the final block
  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
