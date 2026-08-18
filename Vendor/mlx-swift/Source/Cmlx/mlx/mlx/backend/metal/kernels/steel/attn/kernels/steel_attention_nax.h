// Copyright © 2024-25 Apple Inc.

#include "mlx/backend/metal/kernels/steel/attn/nax.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/backend/metal/kernels/steel/attn/transforms.h"
#include "mlx/backend/metal/kernels/steel/utils.h"

using namespace mlx::steel;

// DARKBLOOM_ATTN_QHOIST default. DEFAULT OFF: unless the compiler is invoked
// with -DDARKBLOOM_ATTN_QHOIST=1, the kernel below is byte-for-byte the
// upstream algorithm.
//
// NOTE ON WHICH TWIN RUNS. steel/attn is a JIT family: the runtime-effective
// source is the string in Cmlx/mlx-generated/steel_attention_nax.cpp, and the
// DARKBLOOM_ATTN_QHOIST env var is read there, by
// get_steel_attention_nax_kernel in mlx/backend/metal/jit_kernels.cpp, which
// prepends the define to that string. This AOT header is kept in sync so the
// two twins do not diverge, but nothing wires the env var to
// tools/build-mlx-metallib.sh -- the AOT copy is reachable only by adding the
// define to the metallib build flags by hand.
#ifndef DARKBLOOM_ATTN_QHOIST
#define DARKBLOOM_ATTN_QHOIST 0
#endif

// DARKBLOOM_ATTN_QBLOCK_MAJOR default. DEFAULT ON for the standalone ranked
// candidate: remap the stock physical grid into query-block-major logical
// order. The mapping is a pure permutation of threadgroups; it does not alter
// any threadgroup's arithmetic or output. Compile with
// -DDARKBLOOM_ATTN_QBLOCK_MAJOR=0 for an emergency opt-out.
#ifndef DARKBLOOM_ATTN_QBLOCK_MAJOR
#define DARKBLOOM_ATTN_QBLOCK_MAJOR 1
#endif

// DARKBLOOM_ATTN_QBLOCK_ZIGZAG default. DEFAULT ON for the balanced
// qblock-major candidate: retain head-minor locality inside each query block,
// but present query blocks in high/low order rather than monotonically
// increasing causal work. Compile with
// -DDARKBLOOM_ATTN_QBLOCK_ZIGZAG=0 to recover ascending qblock-major.
#ifndef DARKBLOOM_ATTN_QBLOCK_ZIGZAG
#define DARKBLOOM_ATTN_QBLOCK_ZIGZAG 1
#endif

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];

constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];
constant bool has_sinks [[function_constant(302)]];

template <typename T>
struct TransformScale {
  T scale;
  METAL_FUNC TransformScale(T scale_) : scale(scale_) {}

  METAL_FUNC T apply(T x) const {
    return scale * x;
  }
};

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct SubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x - y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_nax(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device MaskType* mask [[buffer(6), function_constant(has_mask)]],
    const device T* sinks [[buffer(7), function_constant(has_sinks)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;
  (void)simd_lane_id;

  // Move to the logical query block and head. The host dispatches the stock
  // (NQ, H, B) grid. In the optional qblock-major arm, reinterpret its x-fast
  // physical linear index as (query block major, query head minor). Since
  // physical_linear ranges over exactly [0, NQ * H), quotient/remainder by H
  // is a bijection onto the same logical coordinate set. Each threadgroup
  // therefore retains its exact Q/K/V inputs, floating-point operation order,
  // and disjoint output rows; only GPU presentation order changes.
#if DARKBLOOM_ATTN_QBLOCK_MAJOR
  const ulong physical_linear =
      ulong(tid.y) * ulong(params->NQ) + ulong(tid.x);
  const ulong physical_qblock = physical_linear / ulong(params->H);
  const ulong logical_head =
      physical_linear - physical_qblock * ulong(params->H);
#if DARKBLOOM_ATTN_QBLOCK_ZIGZAG
  // Present causal work high, low, second-high, second-low, ... while keeping
  // every query block's heads contiguous. Even ranks map injectively onto the
  // upper half in descending order; odd ranks map onto the lower half in
  // ascending order, so their disjoint union is exactly [0, NQ).
  const ulong logical_qblock =
      (physical_qblock & 1ul) == 0
      ? ulong(params->NQ) - 1 - physical_qblock / 2
      : physical_qblock / 2;
#else
  const ulong logical_qblock = physical_qblock;
#endif
  ulong3 tidl{logical_qblock, logical_head, tid.z};
#else
  ulong3 tidl{tid.x, tid.y, tid.z};
#endif

  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      tidl.x * BQ * params->Q_strides[2]; // Sequence

  ulong kv_head_idx = int(tidl.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + // Batch
      kv_head_idx * params->K_strides[1]; // Head

  V += tidl.z * params->V_strides[0] + // Batch
      kv_head_idx * params->V_strides[1]; // Head

  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      tidl.x * BQ * params->O_strides[2]; // Sequence

  if (has_mask) {
    mask += tidl.z * mask_params->M_strides[0] + // Batch
        tidl.y * mask_params->M_strides[1]; // Head
  }

  const metal::uniform<float> scale2 =
      make_uniform(params->scale) * make_uniform(1.44269504089f);

  // Prepare MMA tiles
  constexpr short kU = 16;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kU);
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kU;
  // KV seq frags per warp
  constexpr short TK = BK / kU;

  static_assert(TQ == 1, "Check TQ");
  using otile_t = NAXTile<AccumType, TQ, TD>;
  otile_t Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short tm = kU * TQ * simd_group_id;
  Q += tm * int(params->Q_strides[2]);

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  // Init row reduction variables
  constexpr short kRowsPT = otile_t::kRowsPerThread;

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

  // Init to -Inf
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  if (has_sinks) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
      sum_score[i] = 1;
    }
  }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;

  if (do_causal) {
    int q_max = (int(tidl.x) + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);

    int q_min = int(tidl.x) * BQ + params->qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  // Per-simdgroup causal K-block elision (level 1, always on). kb_lim above
  // derives from the THREADGROUP's last row; this simdgroup owns only rows
  // [tidl.x * BQ + tm, tidl.x * BQ + tm + kU * TQ). K blocks at or beyond
  // sg_kb_lim lie entirely above its causal diagonal: the causal mask would
  // set every element of its Stile rows to neg_inf, making the P tile
  // exactly the all-+0.0 tile Stile.clear() already produces, new_max equal
  // to max_score (factor == exp2(+0.0) == 1.0), and the sum_score update a
  // +0.0 add into a value that is always >= +0.0. Skipping QK^T, the scale,
  // both masks and the softmax for those blocks is therefore bit-exact.
  // Level 2 also skips the P@V loads and MMAs: every Stile multiplicand is
  // +0.0, so those instructions can only add a signed zero to Otile. That
  // leaves every finite nonzero accumulator bit-identical; an exactly-zero
  // accumulator can differ only in its zero sign, which is numerically equal
  // through the final positive normalization and all downstream arithmetic.
  // The kb trip count and every barrier stay untouched (the P@V loop contains
  // a threadgroup_barrier at BD == 128, so a per-simdgroup trip count would be
  // undefined behaviour); sg_active is simdgroup-uniform (tidl.x and tm only).
  // Restricted to
  // do_causal && !has_mask so the all-masked proof rests on the causal mask
  // alone; the timed window passes no array mask.
  // Tighten the lower bound to this simdgroup's first query row. For every
  // block below this bound the causal predicate is the identity for every
  // element already loaded into Stile, so the select loop can be skipped
  // without changing a value or any arithmetic order.
  int sg_kb_min_causal = kb_min_causal;
  int sg_kb_lim = kb_lim;
  if (do_causal && !has_mask) {
    int sg_q_min =
        int(tidl.x) * BQ + params->qL_off + int(tm);
    sg_kb_min_causal = max(0, sg_q_min + 1) / BK;
    int sg_q_max =
        int(tidl.x) * BQ + params->qL_off + int(tm) + kU * TQ;
    sg_kb_lim = min(kb_lim, (sg_q_max + BK - 1) / BK);
  }

  const bool is_last_bq = int(tidl.x) == (params->NQ_aligned);
  // const bool is_last_tq = int(simd_group_id) >= (params->qL_rem / UQ);
  const bool is_last_q = is_last_bq;

  const short lim_rows_q = params->qL_rem - tm;
  const short lim_rows_k = params->kL_rem;

#if DARKBLOOM_ATTN_QHOIST
  // DARKBLOOM_ATTN_QHOIST -- hoist the loop-invariant Q fragments.
  //
  // The kb loop advances K and V (`K += BK * K_strides[2]` at the bottom) but
  // NEVER advances Q: the Q pointer is finalised above and is constant for the
  // whole loop. The QK^T phase nevertheless re-executed `Qtile.load(...)` on
  // every iteration, re-reading the identical TQ*TD fragments from device
  // memory ~9 times per q-block at the frozen 512-token prefill window.
  //
  // Staging them once here is a PURE HOIST: the loads use the same pointer,
  // the same offsets and the same bounds predicate as the in-loop version, and
  // the mma consumes the identical fragment values in the identical order. No
  // float arithmetic is touched, so no rounding boundary can move.
  //
  // Both loop nests are STEEL_PRAGMA_UNROLL, so every Qhoist index is a
  // compile-time constant and the array stays in registers. If either unroll
  // were ever dropped, Qhoist would spill to thread-local memory and this
  // would become a pessimisation, not an optimisation.
  //
  // COST (BQ=64 BK=32 BD=128 WM=4 WN=1 => TQ=1 TD=8, T=bfloat16): TQ*TD = 8
  // fragments x 8 elems x 2 B = 128 B/thread = 32 x 32-bit registers, live for
  // the whole loop instead of one fragment (4 registers) live for part of it,
  // so +28 registers/thread and +16 KB/threadgroup at 128 threads. See
  // notes/21-attn-analysis.md for why that is expected to fit: this kernel
  // allocates ZERO threadgroup memory, and on Apple family 9+ the on-chip pool
  // is shared between registers and threadgroup memory.
  typename NAXTile<T, 1, 1>::frag_type Qhoist[TQ * TD];

  STEEL_PRAGMA_UNROLL
  for (short iq = 0; iq < TQ; iq++) {
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      NAXTile<T, 1, 1> Qstage;
      const int Q_load_off = iq * kU * int(params->Q_strides[2]) + id * kU;

      if (!align_Q && is_last_q) {
        Qstage.load_rows(
            Q + Q_load_off, int(params->Q_strides[2]), lim_rows_q - iq * kU);
      } else {
        Qstage.load(Q + Q_load_off, int(params->Q_strides[2]));
      }

      Qhoist[iq * TD + id] = Qstage.frag_at(0, 0);
    }
  }
#endif

  // Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {
    const int is_last_k = (kb == (params->NK_aligned));

    // Do S = Q @ K.T
    using stile_t = NAXTile<AccumType, TQ, TK>;
    stile_t Stile;

    Stile.clear();

    // Causal elision: guard the score computation and the zero P@V work, but
    // never a barrier or the outer-loop pointer advance. See the sg_kb_lim
    // comment above for the exactness argument.
    const bool sg_active = kb < sg_kb_lim;
    if (sg_active) {

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik += 2) {
        // Upstream ml-explore/mlx 3541c66b (PR #3843): unrolling the
        // head-dim loop by 4, rather than fully, lets the compiler
        // interleave the next K-tile loads with the running mma chain
        // instead of hoisting all TD loads up front; +12% kernel throughput
        // at head_dim 128 on M5 Max, outputs bitwise identical.
#pragma clang loop unroll_count(4)
        for (short id = 0; id < TD; id++) {
          NAXTile<T, 1, 1> Qtile;
          NAXTile<T, 2, 1> Ktile;

#if !DARKBLOOM_ATTN_QHOIST
          const int Q_load_off = iq * kU * int(params->Q_strides[2]) + id * kU;
#endif
          const int K_load_off = ik * kU * int(params->K_strides[2]) + id * kU;

#if DARKBLOOM_ATTN_QHOIST
          // Q is loop-invariant: the kb loop advances K and V but never Q, so
          // the load in the #else branch re-read the same addresses on every
          // one of the ~9 K-block iterations. Consume the fragment staged
          // before the loop instead.
          //
          // EXACTNESS: the staged load used the same base pointer, the same
          // Q_load_off, the same stride and the same bounds predicate, so the
          // bits in Qhoist are the bits this load would have returned. The mma
          // below consumes them in the identical order. NO FLOAT ARITHMETIC IS
          // TOUCHED AT ALL -- nothing is reassociated, no accumulation order
          // changes, no rounding boundary moves. The only difference is WHEN
          // the device read happened.
          Qtile.frag_at(0, 0) = Qhoist[iq * TD + id];
#else
          if (!align_Q && is_last_q) {
            Qtile.load_rows(
                Q + Q_load_off,
                int(params->Q_strides[2]),
                lim_rows_q - iq * kU);
          } else {
            Qtile.load(Q + Q_load_off, int(params->Q_strides[2]));
          }
#endif

          if (!align_K && is_last_k) {
            Ktile.load_rows(
                K + K_load_off,
                int(params->K_strides[2]),
                lim_rows_k - ik * kU);
          } else {
            Ktile.load(K + K_load_off, int(params->K_strides[2]));
          }

          stile_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qtile.frag_at(0, 0),
              metal::false_type{},
              Ktile.frag_at(0, 0),
              Ktile.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    // Scale S
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= float(scale2);
    }

    // Mask out length sequence
    if (!align_K && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = ik * kU + sn;

          thread auto& fg = Stile.frag_at(iq, ik);

          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < params->kL_rem) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= sg_kb_min_causal) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = int(tidl.x) * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);

          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto r =
                  base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
              const auto c = base_col + ik * kU + jj + sn;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = int(tidl.x) * BQ + tm;
      const int base_col = kb * BK;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, AccumType>;
      using mtile_t = NAXTile<melem_t, TQ, TK>;
      using mfrag_t = typename mtile_t::frag_type;

      if (base_row + BQ <= params->qL && base_col + BK <= params->kL) {
        for (short iq = 0; iq < TQ; iq++) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            const int row_pos = base_row + iq * kU;
            const int col_pos = base_col + ik * kU;

            mfrag_t mfrag;
            mtile_t::NAXFrag_t::load(
                mfrag,
                mask,
                int64_t(mask_params->M_strides[2]),
                Int<1>{},
                row_pos,
                col_pos);

            thread auto& fg = Stile.frag_at(iq, ik);

            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
              if constexpr (is_bool) {
                fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
              } else {
                fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
              }
            }
          }
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            const int row_pos = base_row + iq * kU;
            const int col_pos = base_col + ik * kU;

            mfrag_t mfrag;
            mtile_t::NAXFrag_t::load_safe(
                mfrag,
                mask,
                int64_t(mask_params->M_strides[2]),
                Int<1>{},
                params->qL,
                params->kL,
                row_pos,
                col_pos);

            thread auto& fg = Stile.frag_at(iq, ik);

            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
              if constexpr (is_bool) {
                fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
              } else {
                fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
              }
            }
          }
        }
      }
    }

    // Do softmax

    // Temp variables
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    // Row Sum
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }

    Stile.template row_reduce<SumOp>(sum_score);

    // Update O
    Otile.template row_bin_op<MulOp>(factor);
    }

    simdgroup_barrier(mem_flags::mem_none);

    // Do O = P @ V
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id += 2) {
        if constexpr (BD == 128) {
          if (id == 4) {
            threadgroup_barrier(mem_flags::mem_none);
          }
        }

        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if (sg_active) {
          NAXTile<T, 1, 2> Vtile;

          const int V_load_off = ik * kU * int(params->V_strides[2]) + id * kU;

          if (!align_K && is_last_k) {
            Vtile.load_rows(
                V + V_load_off,
                int(params->V_strides[2]),
                lim_rows_k - ik * kU);
          } else {
            Vtile.load(V + V_load_off, int(params->V_strides[2]));
          }

          otile_t::NAXFrag_t::mma(
              Otile.frag_at(iq, id),
              Otile.frag_at(iq, id + 1),
              Stile.frag_at(iq, ik),
              metal::false_type{},
              Vtile.frag_at(0, 0),
              Vtile.frag_at(0, 1),
              metal::false_type{});
          }
        }
      }
    }

    // Prepare for next iteration
    K += BK * int(params->K_strides[2]);
    V += BK * int(params->V_strides[2]);
  }

  // Normalize output

  threadgroup_barrier(mem_flags::mem_none);

  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }

  Otile.template row_bin_op<MulOp>(rcp);

  // Store results
  O += tm * int(params->O_strides[2]);

  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;

    Otile.store_rows(O, int(params->O_strides[2]), lim_rows_q);
  } else {
    Otile.store(O, int(params->O_strides[2]));
  }
}
