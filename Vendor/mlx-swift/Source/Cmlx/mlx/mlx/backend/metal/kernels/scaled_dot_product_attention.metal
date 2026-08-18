#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sdpa_vector.h"

using namespace metal;

// SDPA vector instantiations
#define instantiate_sdpa_vector_aggregation(type, value_dim) \
  instantiate_kernel(                                        \
      "sdpa_vector_2pass_2_" #type "_" #value_dim,           \
      sdpa_vector_2pass_2,                                   \
      type,                                                  \
      value_dim)

#define instantiate_sdpa_vector(type, qk_dim, value_dim)       \
  instantiate_kernel(                                          \
      "sdpa_vector_" #type "_" #qk_dim "_" #value_dim,         \
      sdpa_vector,                                             \
      type,                                                    \
      qk_dim,                                                  \
      value_dim)                                               \
  instantiate_kernel(                                          \
      "sdpa_vector_2pass_1_" #type "_" #qk_dim "_" #value_dim, \
      sdpa_vector_2pass_1,                                     \
      type,                                                    \
      qk_dim,                                                  \
      value_dim)                                               \
  instantiate_sdpa_vector_measure(type, qk_dim, value_dim)

// DARKBLOOM_AOT_SDPA_MEASURE: the named `_planes*` entry points are
// measurement scaffolding. They are reachable ONLY through the local host
// patch to `backend/metal/scaled_dot_product_attention.cpp`, which is outside
// editablePaths and must be reverted before submit -- so in a packaged build
// nothing can dispatch them. Compiling them anyway would put three dead
// specializations per (type, qk_dim, value_dim) into the ranked metallib and
// hand the static review gates a pile of apparatus to reason about. Undefined
// by default; build the measurement metallib with -DDARKBLOOM_AOT_SDPA_MEASURE.
#ifdef DARKBLOOM_AOT_SDPA_MEASURE
#define instantiate_sdpa_vector_measure(type, qk_dim, value_dim) \
  instantiate_sdpa_vector_planes(type, qk_dim, value_dim)
#else
#define instantiate_sdpa_vector_measure(type, qk_dim, value_dim)
#endif

// DARKBLOOM: one named entry point per exchange-plane width, so all three arms
// of the DARKBLOOM_AOT_SDPA_PLANES experiment live in a SINGLE metallib and an
// A/B/C series can interleave without a rebuild between runs. The stock
// dispatcher only ever asks for the unsuffixed name above (whose width is the
// DARKBLOOM_AOT_SDPA_PLANES macro in sdpa_vector.h -- that is what ships);
// these suffixed names are reachable only through the measurement-only host
// patch, which is not packaged. The trailing `false` is the DEFAULT_ENTRY
// template argument: it keeps each named variant a distinct specialization
// from the unsuffixed default even when the widths coincide.
#define instantiate_sdpa_vector_planes(type, qk_dim, value_dim)          \
  instantiate_kernel(                                                    \
      "sdpa_vector_" #type "_" #qk_dim "_" #value_dim "_planes1",        \
      sdpa_vector, type, qk_dim, value_dim, 1, false)                    \
  instantiate_kernel(                                                    \
      "sdpa_vector_" #type "_" #qk_dim "_" #value_dim "_planes2",        \
      sdpa_vector, type, qk_dim, value_dim, 2, false)                    \
  instantiate_kernel(                                                    \
      "sdpa_vector_" #type "_" #qk_dim "_" #value_dim "_planes4",        \
      sdpa_vector, type, qk_dim, value_dim, 4, false)

#define instantiate_sdpa_vector_heads(type)      \
  instantiate_sdpa_vector(type, 64, 64)          \
  instantiate_sdpa_vector(type, 96, 96)          \
  instantiate_sdpa_vector(type, 128, 128)        \
  instantiate_sdpa_vector(type, 256, 256)        \
  instantiate_sdpa_vector_aggregation(type, 64)  \
  instantiate_sdpa_vector_aggregation(type, 96)  \
  instantiate_sdpa_vector_aggregation(type, 128) \
  instantiate_sdpa_vector_aggregation(type, 256)

instantiate_sdpa_vector_heads(float)
instantiate_sdpa_vector_heads(bfloat16_t)
instantiate_sdpa_vector_heads(float16_t)
    // clang-format on
