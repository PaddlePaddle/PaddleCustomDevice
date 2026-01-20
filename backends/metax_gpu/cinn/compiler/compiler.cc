// PaddleCustomDevice/backends/metax_gpu/cinn/compiler/compiler.cc

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>
#include <ctime>
#include <atomic>

// Host 端头文件，仅供 compiler.cc 使用
#include "paddle/phi/backends/device_ext.h"

namespace paddle {
namespace custom_device {
namespace metax {

// ============================================================
// 1. Runtime Source (JIT 源码头文件 - Device 端代码)
// ============================================================
static const char* kMacaRuntimeSource = R"MACA_SOURCE(
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>

extern "C" {

#define WARP_SIZE 64

#if defined(__MACACC_RTC__) || defined(__HIPCC_RTC__) || defined(__CUDACC_RTC__)
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
#endif

// 兼容 CINN 生成代码中对 __half 的引用
typedef __half float16;

#define CINN_INT32_MAX 2147483647
#define CINN_INT32_MIN -2147483648

#define cinn_max(a, b) ((a) > (b) ? (a) : (b))
#define cinn_min(a, b) ((a) < (b) ? (a) : (b))

// ===============================================================
// 1. Bool / Int8 / UInt8 / Int16 Operations
// ===============================================================
#define FN_BOOL(func) cinn_custom_device_##func##_bool
__device__ inline bool FN_BOOL(bitwise_and)(bool a, bool b) { return a & b; }
__device__ inline bool FN_BOOL(bitwise_or)(bool a, bool b) { return a | b; }
__device__ inline bool FN_BOOL(bitwise_xor)(bool a, bool b) { return a ^ b; }
__device__ inline bool FN_BOOL(bitwise_not)(bool a) { return !a; }

#define FN_UINT8(func) cinn_custom_device_##func##_uint8
__device__ inline uint8_t FN_UINT8(bitwise_and)(uint8_t a, uint8_t b) { return a & b; }
__device__ inline uint8_t FN_UINT8(bitwise_or)(uint8_t a, uint8_t b) { return a | b; }
__device__ inline uint8_t FN_UINT8(bitwise_xor)(uint8_t a, uint8_t b) { return a ^ b; }
__device__ inline uint8_t FN_UINT8(bitwise_not)(uint8_t a) { return ~a; }
__device__ inline uint8_t FN_UINT8(logical_right_shift)(uint8_t a, uint8_t b) { return ((uint8_t)a >> b); }

#define FN_INT8(func) cinn_custom_device_##func##_int8
__device__ inline int8_t FN_INT8(bitwise_and)(int8_t a, int8_t b) { return a & b; }
__device__ inline int8_t FN_INT8(bitwise_or)(int8_t a, int8_t b) { return a | b; }
__device__ inline int8_t FN_INT8(bitwise_xor)(int8_t a, int8_t b) { return a ^ b; }
__device__ inline int8_t FN_INT8(bitwise_not)(int8_t a) { return ~a; }
__device__ inline int8_t FN_INT8(logical_right_shift)(int8_t a, int8_t b) { return ((uint8_t)a >> b); }

#define FN_INT16(func) cinn_custom_device_##func##_int16
__device__ inline int16_t FN_INT16(bitwise_and)(int16_t a, int16_t b) { return a & b; }
__device__ inline int16_t FN_INT16(bitwise_or)(int16_t a, int16_t b) { return a | b; }
__device__ inline int16_t FN_INT16(bitwise_xor)(int16_t a, int16_t b) { return a ^ b; }
__device__ inline int16_t FN_INT16(bitwise_not)(int16_t a) { return ~a; }
__device__ inline int16_t FN_INT16(logical_right_shift)(int16_t a, int16_t b) { return ((uint16_t)a >> b); }

// ===============================================================
// 2. Reduce Binary Operations (CINN CodeGen Requirement)
// ===============================================================

// --- FP64 (Double) ---
__device__ inline double cinn_sum_fp64(const double left, const double right) { return left + right; }
__device__ inline double cinn_prod_fp64(const double left, const double right) { return left * right; }
__device__ inline double cinn_max_fp64(const double left, const double right) { return max(left, right); }
__device__ inline double cinn_min_fp64(const double left, const double right) { return min(left, right); }

// --- FP32 (Float) ---
__device__ inline float cinn_sum_fp32(const float left, const float right) { return left + right; }
__device__ inline float cinn_prod_fp32(const float left, const float right) { return left * right; }
__device__ inline float cinn_max_fp32(const float left, const float right) { return max(left, right); }
__device__ inline float cinn_min_fp32(const float left, const float right) { return min(left, right); }

// --- Int32 ---
__device__ inline int cinn_sum_int32(const int left, const int right) { return left + right; }
__device__ inline int cinn_prod_int32(const int left, const int right) { return left * right; }
__device__ inline int cinn_max_int32(const int left, const int right) { return max(left, right); }
__device__ inline int cinn_min_int32(const int left, const int right) { return min(left, right); }

// --- Int64 ---
__device__ inline int64_t cinn_sum_int64(const int64_t left, const int64_t right) { return left + right; }
__device__ inline int64_t cinn_prod_int64(const int64_t left, const int64_t right) { return left * right; }
__device__ inline int64_t cinn_max_int64(const int64_t left, const int64_t right) { return max(left, right); }
__device__ inline int64_t cinn_min_int64(const int64_t left, const int64_t right) { return min(left, right); }

// --- Bool ---
__device__ inline bool cinn_all_bool(const bool left, const bool right) { return left && right; }
__device__ inline bool cinn_any_bool(const bool left, const bool right) { return left || right; }
__device__ inline bool cinn_all(const bool left, const bool right) { return left && right; }
__device__ inline bool cinn_any(const bool left, const bool right) { return left || right; }

// --- FP16 (Half) ---
// 注意：必须使用 __hadd 等 intrinsics，不能直接用 +
__device__ inline float16 cinn_sum_fp16(const float16 left, const float16 right) { return __hadd(left, right); }
__device__ inline float16 cinn_prod_fp16(const float16 left, const float16 right) { return __hmul(left, right); }
__device__ inline float16 cinn_max_fp16(const float16 left, const float16 right) { return __hgt(left, right) ? left : right; }
__device__ inline float16 cinn_min_fp16(const float16 left, const float16 right) { return __hlt(left, right) ? left : right; }

// --- BF16 (BFloat16) ---
// 【注意】如果 mxcc 不支持 __nv_bfloat16，这部分需要注释掉或报错
#if defined(__MACACC__) || defined(__CUDACC__) // 假设支持
// 暂时留空，如果报错请注释掉 BF16 部分
// __device__ inline __nv_bfloat16 cinn_sum_bf16(...) ...
#endif

// ===============================================================
// 3. Reduce Initialization Macros
// ===============================================================

#define EXPAND_REDUCE_FP64_MACRO(MACRO, ...)            \
  MACRO(sum_fp64, 0.0, double, ##__VA_ARGS__)           \
  MACRO(prod_fp64, 1.0, double, ##__VA_ARGS__)          \
  MACRO(max_fp64, -1.79769e+308, double, ##__VA_ARGS__) \
  MACRO(min_fp64, 1.79769e+308, double, ##__VA_ARGS__)

#define EXPAND_REDUCE_FP32_MACRO(MACRO, ...)      \
  MACRO(sum_fp32, 0.0f, float, ##__VA_ARGS__)     \
  MACRO(prod_fp32, 1.0f, float, ##__VA_ARGS__)    \
  MACRO(max_fp32, -3.40282e+38f, float, ##__VA_ARGS__) \
  MACRO(min_fp32, 3.40282e+38f, float, ##__VA_ARGS__)

#define EXPAND_REDUCE_INT32_MACRO(MACRO, ...)       \
  MACRO(sum_int32, 0, int, ##__VA_ARGS__)           \
  MACRO(prod_int32, 1, int, ##__VA_ARGS__)          \
  MACRO(max_int32, -2147483648, int, ##__VA_ARGS__) \
  MACRO(min_int32, 2147483647, int, ##__VA_ARGS__)

#define EXPAND_REDUCE_INT64_MACRO(MACRO, ...)                 \
  MACRO(sum_int64, 0, int64_t, ##__VA_ARGS__)                 \
  MACRO(prod_int64, 1, int64_t, ##__VA_ARGS__)                \
  MACRO(max_int64, -9223372036854775807LL - 1, int64_t, ##__VA_ARGS__) \
  MACRO(min_int64, 9223372036854775807LL, int64_t, ##__VA_ARGS__)

#define EXPAND_REDUCE_BOOL_MACRO(MACRO, ...)    \
  MACRO(all, true, bool, ##__VA_ARGS__)         \
  MACRO(any, false, bool, ##__VA_ARGS__)

// FP16 初始值 (使用 hex 转换)
#define EXPAND_REDUCE_FP16_MACRO(MACRO, ...)              \
  MACRO(sum_fp16, 0.0, float16, ##__VA_ARGS__)            \
  MACRO(prod_fp16, 1.0, float16, ##__VA_ARGS__)           \
  MACRO(max_fp16, -65504.0, float16, ##__VA_ARGS__)       \
  MACRO(min_fp16, 65504.0, float16, ##__VA_ARGS__)


// ===============================================================
// 4. Warp Shuffle Wrappers (Using Legacy API & Full Down Strategy)
// ===============================================================

// 【核心修复】Warp Reduce 逻辑重写
// 1. 弃用 XOR 模式：因为在 64-thread warp 下，跨 32 边界的 XOR 可能存在未定义行为或硬件 bug。
// 2. 统一使用 DOWN 模式：__shfl_down 是单向规约，Lane 0 总是能收集到数据的，更加稳健。
// 3. 严格的边界检查：确保 fetch 的来源线程在 Block 范围内，否则使用 INIT_VAL 填充。

#define CINN_WARP_SHUFFLE_INTERNAL_IMPL(REDUCE_TYPE, INIT_VAL, DTYPE)         \
  __device__ inline DTYPE cinn_warp_shuffle_##REDUCE_TYPE##_internal(         \
      const DTYPE value) {                                                    \
    DTYPE tmp_val = value;                                                    \
    unsigned int thread_id = threadIdx.x;                                     \
    unsigned int block_dim = blockDim.x;                                      \
    /* 始终使用 Down Shuffle 进行规约 (Log2 复杂度) */                          \
    for (unsigned int offset = WARP_SIZE / 2; offset >= 1; offset /= 2) {     \
        DTYPE shfl_res = cinn_warp_shuffle_down_##DTYPE##_wrapper(tmp_val, offset); \
        /* 检查数据来源是否有效：当前线程+offset 必须还在 Block 范围内 */             \
        /* 如果 Block 大小不是 WARP_SIZE 的倍数，这一步至关重要 */                  \
        DTYPE neighbor = (thread_id + offset < block_dim) ? shfl_res : (DTYPE)(INIT_VAL); \
        tmp_val = cinn_##REDUCE_TYPE(tmp_val, neighbor);                      \
    }                                                                         \
    /* 广播：虽然 Down Shuffle 只有 Lane 0 结果正确，但这里为了兼容 XOR 语义 */    \
    /* 我们用 shfl 0 把 Lane 0 的结果广播给所有人 (CINN Block Reduce 需要) */     \
    return __shfl(tmp_val, 0);                                                \
  }

// --- Warp Shuffle Primitives (Legacy API without mask) ---

__device__ inline float cinn_warp_shuffle_down_float_wrapper(float v, int factor) { return __shfl_down(v, factor); }
__device__ inline int cinn_warp_shuffle_down_int_wrapper(int v, int factor) { return __shfl_down(v, factor); }
__device__ inline bool cinn_warp_shuffle_down_bool_wrapper(bool v, int factor) { return __shfl_down(v, factor); }

__device__ inline double cinn_warp_shuffle_down_double_wrapper(double v, int factor) {
  unsigned long long int val_u64 = *(unsigned long long int*)&v;
  int lo = (int)val_u64; int hi = (int)(val_u64 >> 32);
  lo = __shfl_down(lo, factor);
  hi = __shfl_down(hi, factor);
  unsigned long long int res_u64 = ((unsigned long long int)hi << 32) | (unsigned int)lo;
  return *(double*)&res_u64;
}

__device__ inline int64_t cinn_warp_shuffle_down_int64_t_wrapper(int64_t v, int factor) {
  int lo = (int)v; int hi = (int)(v >> 32);
  lo = __shfl_down(lo, factor);
  hi = __shfl_down(hi, factor);
  return ((int64_t)hi << 32) | (unsigned int)lo;
}

__device__ inline float16 cinn_warp_shuffle_down_float16_wrapper(float16 v, int factor) {
  unsigned short val = __half_as_ushort(v);
  unsigned short res = (unsigned short)__shfl_down((int)val, factor);
  return __ushort_as_half(res);
}

// Expand Warp Shuffle
EXPAND_REDUCE_INT32_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
EXPAND_REDUCE_INT64_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
EXPAND_REDUCE_FP32_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
EXPAND_REDUCE_FP64_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
EXPAND_REDUCE_BOOL_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
EXPAND_REDUCE_FP16_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)

// ===============================================================
// 5. Block Reduce & Discrete Reduce & Grid Reduce
// ===============================================================

// Block Reduce Implementation
// 1. Warp Reduce -> SHM
// 2. Warp 0 reads SHM and Pads with Identity
// 3. Warp 0 Reduce
// 4. Broadcast
#define CINN_BLOCK_REDUCE_IMPL(DTYPE, INIT_VAL, cinn_warp_shuffle_internal)       \
  /* 1. Warp Reduce */                                                            \
  DTYPE tmp_val = cinn_warp_shuffle_internal(value);                              \
  if (return_warp || blockDim.x <= WARP_SIZE) {                                   \
    return tmp_val;                                                               \
  }                                                                               \
  __syncthreads();                                                                \
  /* 2. Write Warp results to SHM (Lane 0 only) */                                \
  if (threadIdx.x % WARP_SIZE == 0) {                                             \
    shm[threadIdx.x / WARP_SIZE] = tmp_val;                                       \
  }                                                                               \
  __syncthreads();                                                                \
  /* 3. Inter-Warp Reduce (Warp 0 only) */                                        \
  if (threadIdx.x < WARP_SIZE) {                                                  \
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;                     \
    /* Pad with Identity value for idle threads in Warp 0 */                        \
    DTYPE reduce_val = (DTYPE)(INIT_VAL);                                         \
    if (threadIdx.x < num_warps) {                                                \
      reduce_val = shm[threadIdx.x];                                              \
    }                                                                             \
    /* Reduce across all threads in Warp 0 */                                     \
    reduce_val = cinn_warp_shuffle_internal(reduce_val);                          \
    if (threadIdx.x == 0) {                                                       \
      shm[0] = reduce_val;                                                        \
    }                                                                             \
  }                                                                               \
  __syncthreads();                                                                \
  return shm[0];

#define CINN_BLOCK_REDUCE_MACRO(REDUCE_TYPE, INIT_VAL, DTYPE)                  \
  __device__ inline DTYPE cinn_block_reduce_##REDUCE_TYPE(                     \
      const DTYPE value, DTYPE *shm, bool return_warp = false) {               \
    CINN_BLOCK_REDUCE_IMPL(DTYPE, INIT_VAL, cinn_warp_shuffle_##REDUCE_TYPE##_internal); \
  }

EXPAND_REDUCE_INT32_MACRO(CINN_BLOCK_REDUCE_MACRO)
EXPAND_REDUCE_INT64_MACRO(CINN_BLOCK_REDUCE_MACRO)
EXPAND_REDUCE_FP32_MACRO(CINN_BLOCK_REDUCE_MACRO)
EXPAND_REDUCE_FP64_MACRO(CINN_BLOCK_REDUCE_MACRO)
EXPAND_REDUCE_BOOL_MACRO(CINN_BLOCK_REDUCE_MACRO)
EXPAND_REDUCE_FP16_MACRO(CINN_BLOCK_REDUCE_MACRO)

#define CINN_DISCRETE_REDUCE_IMPL(REDUCE_TYPE, value)                          \
  int tid = threadIdx.y * blockDim.x + threadIdx.x;                            \
  __syncthreads();                                                             \
  shm[tid] = value;                                                            \
  __syncthreads();                                                             \
  for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {                \
    if (threadIdx.y < offset) {                                                \
      shm[tid] = cinn_##REDUCE_TYPE(shm[tid], shm[tid + offset * blockDim.x]); \
    }                                                                          \
    __syncthreads();                                                           \
  }                                                                            \
  return shm[threadIdx.x];

#define CINN_DISCRETE_REDUCE_MACRO(REDUCE_TYPE, INIT_VAL, DTYPE)      \
  __device__ inline DTYPE cinn_discrete_reduce_##REDUCE_TYPE(         \
      const DTYPE value, DTYPE *shm) {                                \
    CINN_DISCRETE_REDUCE_IMPL(REDUCE_TYPE, value);                    \
  }

EXPAND_REDUCE_INT32_MACRO(CINN_DISCRETE_REDUCE_MACRO)
EXPAND_REDUCE_INT64_MACRO(CINN_DISCRETE_REDUCE_MACRO)
EXPAND_REDUCE_FP32_MACRO(CINN_DISCRETE_REDUCE_MACRO)
EXPAND_REDUCE_FP64_MACRO(CINN_DISCRETE_REDUCE_MACRO)
EXPAND_REDUCE_BOOL_MACRO(CINN_DISCRETE_REDUCE_MACRO)
EXPAND_REDUCE_FP16_MACRO(CINN_DISCRETE_REDUCE_MACRO)

#define CINN_GRID_REDUCE_IMPL(REDUCE_TYPE, init_value, DTYPE)               \
  DTYPE tmp_val = init_value;                                               \
  for (int y = 0; y < gridDim.y; y++) {                                     \
    tmp_val =                                                               \
        cinn_##REDUCE_TYPE(tmp_val, mem[y * spatial_size + spatial_index]); \
  }                                                                         \
  return tmp_val;

#define CINN_GRID_REDUCE_MACRO(REDUCE_TYPE, INIT_VAL, DTYPE)           \
  __device__ inline DTYPE cinn_grid_reduce_##REDUCE_TYPE(              \
      const DTYPE *mem, int spatial_size, int spatial_index) {         \
    CINN_GRID_REDUCE_IMPL(REDUCE_TYPE, (DTYPE)(INIT_VAL), DTYPE);      \
  }

EXPAND_REDUCE_INT32_MACRO(CINN_GRID_REDUCE_MACRO)
EXPAND_REDUCE_INT64_MACRO(CINN_GRID_REDUCE_MACRO)
EXPAND_REDUCE_FP32_MACRO(CINN_GRID_REDUCE_MACRO)
EXPAND_REDUCE_FP64_MACRO(CINN_GRID_REDUCE_MACRO)
EXPAND_REDUCE_BOOL_MACRO(CINN_GRID_REDUCE_MACRO)
EXPAND_REDUCE_FP16_MACRO(CINN_GRID_REDUCE_MACRO)

__device__ inline bool cinn_grid_reduce_update_semaphore(int *semaphores) {
  __shared__ bool done;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    int old = atomicAdd(&semaphores[blockIdx.x], 1);
    done = (old == (gridDim.y - 1));
  }
  __syncthreads();
  return done;
}
// ===============================================================
// 6. Standard Math Functions 
// ===============================================================
// ===============================================================
// Float64 (Double) Math Functions
// ===============================================================
#define FN_FP64(func) cinn_custom_device_##func##_fp64

__device__ inline double FN_FP64(sin)(double x) { return sin(x); }
__device__ inline double FN_FP64(cos)(double x) { return cos(x); }
__device__ inline double FN_FP64(tan)(double x) { return tan(x); }
__device__ inline double FN_FP64(exp)(double x) { return exp(x); }
__device__ inline double FN_FP64(log)(double x) { return log(x); }
__device__ inline double FN_FP64(log2)(double x) { return log2(x); }
__device__ inline double FN_FP64(log10)(double x) { return log10(x); }
__device__ inline double FN_FP64(sqrt)(double x) { return sqrt(x); }
__device__ inline double FN_FP64(rsqrt)(double x) { return rsqrt(x); }
__device__ inline double FN_FP64(abs)(double x) { return fabs(x); }
__device__ inline double FN_FP64(floor)(double x) { return floor(x); }
__device__ inline double FN_FP64(ceil)(double x) { return ceil(x); }
__device__ inline double FN_FP64(round)(double x) { return round(x); }
__device__ inline double FN_FP64(trunc)(double x) { return trunc(x); }
__device__ inline double FN_FP64(pow)(double a, double b) { return pow(a, b); }
__device__ inline double FN_FP64(mod)(double a, double b) { return fmod(a, b); }
__device__ inline double FN_FP64(fma)(double a, double b, double c) { return fma(a, b, c); }
__device__ inline bool FN_FP64(isnan)(double x) { return isnan(x); }
__device__ inline bool FN_FP64(isinf)(double x) { return isinf(x); }
__device__ inline bool FN_FP64(isfinite)(double x) { return isfinite(x); }
__device__ inline double FN_FP64(acos)(double x) { return acos(x); }
__device__ inline double FN_FP64(acosh)(double x) { return acosh(x); }
__device__ inline double FN_FP64(asin)(double x) { return asin(x); }
__device__ inline double FN_FP64(asinh)(double x) { return asinh(x); }
__device__ inline double FN_FP64(atan)(double x) { return atan(x); }
__device__ inline double FN_FP64(atanh)(double x) { return atanh(x); }
__device__ inline double FN_FP64(cbrt)(double x) { return cbrt(x); }
__device__ inline double FN_FP64(cosh)(double x) { return cosh(x); }
__device__ inline double FN_FP64(erf)(double x) { return erf(x); }
__device__ inline double FN_FP64(log1p)(double x) { return log1p(x); }
__device__ inline double FN_FP64(sigmoid)(double x) { return 1.0 / (1.0 + exp(-x)); }
__device__ inline double FN_FP64(sinh)(double x) { return sinh(x); }
__device__ inline double FN_FP64(tanh)(double x) { return tanh(x); }

// ===============================================================
// Float32 Math Functions
// ===============================================================
#define FN_FP32(func) cinn_custom_device_##func##_fp32

__device__ inline float FN_FP32(sin)(float x) { return sinf(x); }
__device__ inline float FN_FP32(cos)(float x) { return cosf(x); }
__device__ inline float FN_FP32(tan)(float x) { return tanf(x); }
__device__ inline float FN_FP32(exp)(float x) { return expf(x); }
__device__ inline float FN_FP32(log)(float x) { return logf(x); }
__device__ inline float FN_FP32(sqrt)(float x) { return sqrtf(x); }
__device__ inline float FN_FP32(rsqrt)(float x) { return rsqrtf(x); }
__device__ inline float FN_FP32(pow)(float a, float b) { return powf(a, b); }
__device__ inline float FN_FP32(floor)(float x) { return floorf(x); }
__device__ inline float FN_FP32(ceil)(float x) { return ceilf(x); }
__device__ inline float FN_FP32(round)(float x) { return roundf(x); }
__device__ inline float FN_FP32(trunc)(float x) { return truncf(x); }
__device__ inline float FN_FP32(abs)(float x) { return fabsf(x); }
__device__ inline float FN_FP32(mod)(float a, float b) { return fmodf(a, b); }
__device__ inline float FN_FP32(fma)(float a, float b, float c) { return fmaf(a, b, c); }
__device__ inline bool FN_FP32(isnan)(float x) { return isnan(x); }
__device__ inline bool FN_FP32(isinf)(float x) { return isinf(x); }
__device__ inline bool FN_FP32(isfinite)(float x) { return isfinite(x); }
__device__ inline float FN_FP32(acos)(float x) { return acosf(x); }
__device__ inline float FN_FP32(acosh)(float x) { return acoshf(x); }
__device__ inline float FN_FP32(asin)(float x) { return asinf(x); }
__device__ inline float FN_FP32(asinh)(float x) { return asinhf(x); }
__device__ inline float FN_FP32(atan)(float x) { return atanf(x); }
__device__ inline float FN_FP32(atanh)(float x) { return atanhf(x); }
__device__ inline float FN_FP32(cbrt)(float x) { return cbrtf(x); }
__device__ inline float FN_FP32(cosh)(float x) { return coshf(x); }
__device__ inline float FN_FP32(erf)(float x) { return erff(x); }
__device__ inline float FN_FP32(log2)(float x) { return log2f(x); }
__device__ inline float FN_FP32(log10)(float x) { return log10f(x); }
__device__ inline float FN_FP32(log1p)(float x) { return log1pf(x); }
__device__ inline float FN_FP32(sigmoid)(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ inline float FN_FP32(sinh)(float x) { return sinhf(x); }
__device__ inline float FN_FP32(tanh)(float x) { return tanhf(x); }
__device__ inline float FN_FP32(left_shift)(float a, float b) {
  return (float)((int)a << (int)b);
}
__device__ inline float FN_FP32(right_shift)(float a, float b) {
  return (float)((int)a >> (int)b);
}

// ===============================================================
// Int32 Functions
// ===============================================================
#define FN_INT32(func) cinn_custom_device_##func##_int32
__device__ inline int FN_INT32(bitwise_not)(int a) { return ~a; }
__device__ inline int FN_INT32(clz)(int a) { return __clz(a); }
__device__ inline int FN_INT32(popc)(int a) { return __popc(a); }
__device__ inline int FN_INT32(mod)(int a, int b) { 
  int res = a % b;
  if ((res != 0) && ((b ^ res) < 0)) res += b;
  return res;
}
__device__ inline int FN_INT32(max)(int a, int b) { return cinn_max(a, b); }
__device__ inline int FN_INT32(min)(int a, int b) { return cinn_min(a, b); }
__device__ inline int FN_INT32(left_shift)(int a, int b) { return a << b; }
__device__ inline int FN_INT32(right_shift)(int a, int b) { return a >> b; }
__device__ inline int FN_INT32(bitwise_and)(int a, int b) { return a & b; }
__device__ inline int FN_INT32(bitwise_or)(int a, int b) { return a | b; }
__device__ inline int FN_INT32(bitwise_xor)(int a, int b) { return a ^ b; }
__device__ inline int FN_INT32(logical_right_shift)(int a, int b) { return (unsigned int)a >> b; }
__device__ inline int FN_INT32(trunc)(int a) { return a; }
__device__ inline int FN_INT32(pow)(int a, int b) {
  if (a == 0 && b < 0) return -1;
  float res = powf(__int2float_rd(a), __int2float_rd(b));
  return __float2int_rn(res);
}
__device__ inline int FN_INT32(arithmetic_right_shift)(int a, int b) { return a >> b; }

// ===============================================================
// Int64 Functions
// ===============================================================
#define FN_INT64(func) cinn_custom_device_##func##_int64
__device__ inline int64_t FN_INT64(bitwise_and)(int64_t a, int64_t b) { return a & b; }
__device__ inline int64_t FN_INT64(bitwise_or)(int64_t a, int64_t b) { return a | b; }
__device__ inline int64_t FN_INT64(bitwise_xor)(int64_t a, int64_t b) { return a ^ b; }
__device__ inline int64_t FN_INT64(bitwise_not)(int64_t a) { return ~a; }
__device__ inline int64_t FN_INT64(clz)(int64_t a) { return __clzll(a); }
__device__ inline int64_t FN_INT64(popc)(int64_t a) { return __popcll(a); }
__device__ inline int64_t FN_INT64(logical_right_shift)(int64_t a, int64_t b) { return ((uint64_t)a >> b); }
__device__ inline int64_t FN_INT64(trunc)(int64_t a) { return a; }
__device__ inline int64_t FN_INT64(mod)(int64_t a, int64_t b) { int64_t res = a % b; if ((res != 0) && ((b ^ res) < 0)) res += b; return res; }
__device__ inline int64_t FN_INT64(pow)(int64_t a, int64_t b) { double res = pow(__ll2double_rd(a), __ll2double_rd(b)); return __double2ll_rn(res); }

// ===============================================================
// Float16 (Half) Functions
// ===============================================================
#define FN_FP16(func) cinn_custom_device_##func##_fp16

#define FN_FP16(func) cinn_custom_device_##func##_fp16
__device__ inline float16 FN_FP16(ceil)(float16 x) { return hceil(x); }
__device__ inline float16 FN_FP16(floor)(float16 x) { return hfloor(x); }
__device__ inline float16 FN_FP16(round)(float16 x) { return __float2half(roundf(__half2float(x))); }
__device__ inline float16 FN_FP16(trunc)(float16 x) { return htrunc(x); }
__device__ inline float16 FN_FP16(sin)(float16 x) { return hsin(x); }
__device__ inline float16 FN_FP16(cos)(float16 x) { return hcos(x); }
__device__ inline float16 FN_FP16(exp)(float16 x) { return hexp(x); }
__device__ inline float16 FN_FP16(log)(float16 x) { return hlog(x); }
__device__ inline float16 FN_FP16(log2)(float16 x) { return hlog2(x); }
__device__ inline float16 FN_FP16(log10)(float16 x) { return hlog10(x); }
__device__ inline float16 FN_FP16(sqrt)(float16 x) { return hsqrt(x); }
__device__ inline float16 FN_FP16(rsqrt)(float16 x) { return hrsqrt(x); }
__device__ inline float16 FN_FP16(cbrt)(float16 x) { return __float2half(cbrtf(__half2float(x))); }
__device__ inline float16 FN_FP16(abs)(float16 x) { return __float2half(fabsf(__half2float(x))); }
__device__ inline bool FN_FP16(isnan)(float16 x) { return __hisnan(x); }
__device__ inline bool FN_FP16(isinf)(float16 x) { return __hisinf(x); }
__device__ inline bool FN_FP16(isfinite)(float16 x) { return !__hisinf(x) && !__hisnan(x); }
__device__ inline float16 FN_FP16(erf)(float16 x) { return __float2half(erff(__half2float(x))); }
__device__ inline float16 FN_FP16(tan)(float16 x) { return __float2half(tanf(__half2float(x))); }
__device__ inline float16 FN_FP16(sinh)(float16 x) { return __float2half(sinhf(__half2float(x))); }
__device__ inline float16 FN_FP16(cosh)(float16 x) { return __float2half(coshf(__half2float(x))); }
__device__ inline float16 FN_FP16(tanh)(float16 x) { return __float2half(tanhf(__half2float(x))); }
__device__ inline float16 FN_FP16(asin)(float16 x) { return __float2half(asinf(__half2float(x))); }
__device__ inline float16 FN_FP16(acos)(float16 x) { return __float2half(acosf(__half2float(x))); }
__device__ inline float16 FN_FP16(atan)(float16 x) { return __float2half(atanf(__half2float(x))); }
__device__ inline float16 FN_FP16(asinh)(float16 x) { return __float2half(asinhf(__half2float(x))); }
__device__ inline float16 FN_FP16(acosh)(float16 x) { return __float2half(acoshf(__half2float(x))); }
__device__ inline float16 FN_FP16(atanh)(float16 x) { return __float2half(atanhf(__half2float(x))); }
__device__ inline float16 FN_FP16(sigmoid)(float16 x) { return __float2half(1.0f / (1.0f + expf(-__half2float(x)))); }
__device__ inline float16 FN_FP16(mod)(float16 a, float16 b) { return __float2half(fmodf(__half2float(a), __half2float(b))); }
__device__ inline float16 FN_FP16(pow)(float16 a, float16 b) { return __float2half(powf(__half2float(a), __half2float(b))); }
__device__ inline float16 FN_FP16(add)(float16 a, float16 b) { return __hadd(a, b); }
__device__ inline float16 FN_FP16(sub)(float16 a, float16 b) { return __hsub(a, b); }
__device__ inline float16 FN_FP16(mul)(float16 a, float16 b) { return __hmul(a, b); }
__device__ inline float16 FN_FP16(div)(float16 a, float16 b) { return __hdiv(a, b); }
__device__ inline float16 FN_FP16(neg)(float16 a) { return __hneg(a); }
__device__ inline float16 FN_FP16(fma)(float16 a, float16 b, float16 c) { return __hfma(a, b, c); }
__device__ inline float16 FN_FP16(max)(float16 a, float16 b) { return __hgt(a, b) ? a : b; }
__device__ inline float16 FN_FP16(min)(float16 a, float16 b) { return __hlt(a, b) ? a : b; }

// ===============================================================
// Warp Shuffle Functions (用于 Reduce 算子)
// ===============================================================
#define FN_SHUFFLE(func) cinn_custom_device_##func

__device__ inline float FN_SHUFFLE(warp_shuffle_xor_fp32)(float v, int factor) {
  return __shfl_xor_sync(0xffffffff, v, factor);
}
__device__ inline float FN_SHUFFLE(warp_shuffle_up_fp32)(float v, int factor) {
  return __shfl_up_sync(0xffffffff, v, factor);
}
__device__ inline float FN_SHUFFLE(warp_shuffle_down_fp32)(float v, int factor) {
  return __shfl_down_sync(0xffffffff, v, factor);
}

__device__ inline int FN_SHUFFLE(warp_shuffle_xor_int32)(int v, int factor) {
  return __shfl_xor_sync(0xffffffff, v, factor);
}
__device__ inline int FN_SHUFFLE(warp_shuffle_up_int32)(int v, int factor) {
  return __shfl_up_sync(0xffffffff, v, factor);
}
__device__ inline int FN_SHUFFLE(warp_shuffle_down_int32)(int v, int factor) {
  return __shfl_down_sync(0xffffffff, v, factor);
}

// MACA/CUDA 的 shfl 指令通常只支持 32位，__half 需要强转或使用 intrinsics
__device__ inline __half FN_SHUFFLE(warp_shuffle_xor_fp16)(__half v, int factor) {
  unsigned short val = __half_as_ushort(v);
  unsigned short res = (unsigned short)__shfl_xor_sync(0xffffffff, (int)val, factor);
  return __ushort_as_half(res);
}
__device__ inline __half FN_SHUFFLE(warp_shuffle_up_fp16)(__half v, int factor) {
  unsigned short val = __half_as_ushort(v);
  unsigned short res = (unsigned short)__shfl_up_sync(0xffffffff, (int)val, factor);
  return __ushort_as_half(res);
}
__device__ inline __half FN_SHUFFLE(warp_shuffle_down_fp16)(__half v, int factor) {
  unsigned short val = __half_as_ushort(v);
  unsigned short res = (unsigned short)__shfl_down_sync(0xffffffff, (int)val, factor);
  return __ushort_as_half(res);
}

// ===============================================================
// 7. Index Operations: Find, Sort & Resize Helpers
// ===============================================================
#define __cinn_custom_device_find_kernel(buf, size, num, begin, stride)  \
  do {                                                                   \
    for (int i = (size - 1) * stride + begin; i >= begin; i -= stride) { \
      if (buf[i] == num) return (i - begin) / stride;                    \
    }                                                                    \
    return -1;                                                           \
  } while (0)

__device__ inline int cinn_custom_device_find_int(const int *buf, int size, int num) {
  __cinn_custom_device_find_kernel(buf, size, num, 0, 1);
}
__device__ inline int cinn_custom_device_find_float(const float *buf, int size, float num) {
  __cinn_custom_device_find_kernel(buf, size, num, 0, 1);
}
__device__ inline int cinn_custom_device_find_int_nd(const int *buf, int size, int num, int begin, int stride) {
  __cinn_custom_device_find_kernel(buf, size, num, begin, stride);
}
__device__ inline int cinn_custom_device_find_float_nd(const float *buf, int size, float num, int begin, int stride) {
  __cinn_custom_device_find_kernel(buf, size, num, begin, stride);
}
#undef __cinn_custom_device_find_kernel

__device__ inline int cinn_custom_device_next_smallest_int32(int *buf, int size, int num, int begin, int stride) {
  int id = -1;
  for (int i = begin; i < begin + size * stride; i += stride) {
    if (id == -1 || buf[i] < buf[id]) {
      id = i;
    }
  }
  if (id != -1) {
    buf[id] = CINN_INT32_MAX;
    return (id - begin) / stride;
  }
  return -1;
}

#define __cinn_custom_device_find_from_kernel(buf, size, num, begin) \
  do {                                                     \
    for (int i = begin; i < size; ++i) {                   \
      if (buf[i] == num) return i;                         \
    }                                                      \
    return -1;                                             \
  } while (0)

__device__ inline int cinn_custom_device_find_int_from(const int *buf, int size, int num, int begin) {
  __cinn_custom_device_find_from_kernel(buf, size, num, begin);
}
__device__ inline int cinn_custom_device_find_float_from(const float *buf, int size, float num, int begin) {
  __cinn_custom_device_find_from_kernel(buf, size, num, begin);
}
#undef __cinn_custom_device_find_from_kernel

#define CINN_CUSTOM_DEVICE_LT_NUM(TYPE_SUFFIX, TYPE)                       \
  __device__ inline int cinn_custom_device_lt_num_##TYPE_SUFFIX(const TYPE *buf,     \
                                                      const int size,      \
                                                      const TYPE num,      \
                                                      const int offset,    \
                                                      const int stride) {  \
    int out = 0;                                                           \
    for (int i = (size - 1) * stride + offset; i >= offset; i -= stride) { \
      if (buf[i] < num) out++;                                             \
    }                                                                      \
    return out;                                                            \
  }

CINN_CUSTOM_DEVICE_LT_NUM(fp32, float)
CINN_CUSTOM_DEVICE_LT_NUM(fp64, double)
CINN_CUSTOM_DEVICE_LT_NUM(uint8, uint8_t)
CINN_CUSTOM_DEVICE_LT_NUM(int16, int16_t)
CINN_CUSTOM_DEVICE_LT_NUM(int32, int)
CINN_CUSTOM_DEVICE_LT_NUM(int64, int64_t)
CINN_CUSTOM_DEVICE_LT_NUM(fp16, float16)
#undef CINN_CUSTOM_DEVICE_LT_NUM

#define CINN_CUSTOM_DEVICE_GT_NUM(TYPE_SUFFIX, TYPE)                       \
  __device__ inline int cinn_custom_device_gt_num_##TYPE_SUFFIX(const TYPE *buf,     \
                                                      const int size,      \
                                                      const TYPE num,      \
                                                      const int offset,    \
                                                      const int stride) {  \
    int out = 0;                                                           \
    for (int i = (size - 1) * stride + offset; i >= offset; i -= stride) { \
      if (buf[i] > num) out++;                                             \
    }                                                                      \
    return out;                                                            \
  }

CINN_CUSTOM_DEVICE_GT_NUM(fp32, float)
CINN_CUSTOM_DEVICE_GT_NUM(fp64, double)
CINN_CUSTOM_DEVICE_GT_NUM(uint8, uint8_t)
CINN_CUSTOM_DEVICE_GT_NUM(int16, int16_t)
CINN_CUSTOM_DEVICE_GT_NUM(int32, int)
CINN_CUSTOM_DEVICE_GT_NUM(int64, int64_t)
CINN_CUSTOM_DEVICE_GT_NUM(fp16, float16)
#undef CINN_CUSTOM_DEVICE_GT_NUM

#define CINN_CUSTOM_DEVICE_INDEX_ADD(TYPE_SUFFIX, TYPE)                      \
  __device__ inline TYPE cinn_custom_device_index_add_##TYPE_SUFFIX(         \
      const TYPE x,                                                          \
      const int axis_indice,                                                 \
      const TYPE *__restrict__ y,                                            \
      const int offset,                                                      \
      const int stride,                                                      \
      const int *__restrict__ index,                                         \
      const int index_size) {                                                \
    TYPE res = x;                                                            \
    int idx = -1;                                                            \
    do {                                                                     \
      idx = cinn_custom_device_find_int_from(index, index_size, axis_indice, idx + 1); \
      if (idx >= 0) {                                                        \
        res = res + y[offset + idx * stride];                                \
      }                                                                      \
    } while (idx != -1);                                                     \
    return res;                                                              \
  }

CINN_CUSTOM_DEVICE_INDEX_ADD(bool, bool)
CINN_CUSTOM_DEVICE_INDEX_ADD(int8, int8_t)
CINN_CUSTOM_DEVICE_INDEX_ADD(int32, int32_t)
CINN_CUSTOM_DEVICE_INDEX_ADD(int64, int64_t)
CINN_CUSTOM_DEVICE_INDEX_ADD(fp32, float)
CINN_CUSTOM_DEVICE_INDEX_ADD(fp64, double)
CINN_CUSTOM_DEVICE_INDEX_ADD(fp16, float16)
#undef CINN_CUSTOM_DEVICE_INDEX_ADD

__device__ int cinn_custom_device_resize_bilinear(const int *buf,
                                        const int c_size,
                                        const int in_h,
                                        const int in_w,
                                        const int out_h,
                                        const int out_w,
                                        const int n,
                                        const int c,
                                        const int y,
                                        const int x) {
  float scale_y = static_cast<float>(in_h) / out_h;
  float scale_x = static_cast<float>(in_w) / out_w;
  float in_y = (y + 0.5F) * scale_y - 0.5F;
  float in_x = (x + 0.5F) * scale_x - 0.5F;
  int in_y_int = static_cast<int>(floorf(in_y));
  int in_x_int = static_cast<int>(floorf(in_x));
  float y_lerp = in_y - in_y_int;
  float x_lerp = in_x - in_x_int;
  float p[2][2];

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      int near_y = in_y_int + i;
      int near_x = in_x_int + j;
      near_y = max(min(near_y, in_h - 1), 0);
      near_x = max(min(near_x, in_w - 1), 0);
      p[i][j] = buf[n * c_size * in_h * in_w + c * in_h * in_w + near_y * in_w +
                    near_x];
    }
  }

  float top = p[0][0] * (1.0F - x_lerp) + p[0][1] * x_lerp;
  float bottom = p[1][0] * (1.0F - x_lerp) + p[1][1] * x_lerp;
  float value = top * (1.0F - y_lerp) + bottom * y_lerp;
  return value;
}

__device__ int cinn_custom_device_resize_bicubic(const int *buf,
                                       const int c_size,
                                       const int in_h,
                                       const int in_w,
                                       const int out_h,
                                       const int out_w,
                                       const int n,
                                       const int c,
                                       const int y,
                                       const int x) {
  float scale_y = static_cast<float>(in_h) / out_h;
  float scale_x = static_cast<float>(in_w) / out_w;
  float in_y = (y + 0.5F) * scale_y - 0.5F;
  float in_x = (x + 0.5F) * scale_x - 0.5F;
  int in_y_int = static_cast<int>(floorf(in_y));
  int in_x_int = static_cast<int>(floorf(in_x));
  float y_fract = in_y - floorf(in_y);
  float x_fract = in_x - floorf(in_x);
  float p[4][4];

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      int near_y = in_y_int + i - 1;
      int near_x = in_x_int + j - 1;
      near_y = max(min(near_y, in_h - 1), 0);
      near_x = max(min(near_x, in_w - 1), 0);
      p[i][j] = buf[n * c_size * in_h * in_w + c * in_h * in_w + near_y * in_w +
                    near_x];
    }
  }

  float alpha = -0.5F;
  float w[2][4];

  for (int i = 0; i < 2; ++i) {
    float t = (i == 0 ? x_fract : y_fract);
    float t2 = t * t;
    float t3 = t * t * t;
    w[i][0] = alpha * (t3 - 2 * t2 + t);
    w[i][1] = (alpha + 2) * t3 - (3 + alpha) * t2 + 1;
    w[i][2] = -(alpha + 2) * t3 + (3 + 2 * alpha) * t2 - alpha * t;
    w[i][3] = -alpha * t3 + alpha * t2;
  }

  float col[4];

  for (int i = 0; i < 4; ++i) {
    col[i] = 0.0F;
    for (int j = 0; j < 4; ++j) {
      col[i] += p[i][j] * w[0][j];
    }
  }

  float value = 0.0F;

  for (int i = 0; i < 4; ++i) {
    value += col[i] * w[1][i];
  }

  return value;
}

} // extern "C"
)MACA_SOURCE";


// ============================================================
// 2. 接口实现
// ============================================================

// 全局原子计数器，确保文件名唯一
static std::atomic<uint64_t> g_compile_counter{0};

const char* MetaxGetRuntimeSource(void* dev_ptr) {
    return kMacaRuntimeSource;
}

C_Status MetaxCompile(void* dev_ptr, const char* code, char* out_path, size_t len) {
    // 0. 生成随机文件名
    // 【关键修复】使用 进程ID + 原子计数器 生成唯一文件名
    // 彻底解决多线程编译时的文件名冲突问题
    uint64_t file_id = g_compile_counter.fetch_add(1);
    std::string file_prefix = "cinn_metax_" + std::to_string(getpid()) + "_" + std::to_string(file_id);
    
    // 生成临时文件路径
    std::string src_path = "/tmp/" + file_prefix + ".cu";
    std::string obj_path = "/tmp/" + file_prefix + ".co";

    // 注意：即使 CINN 传了 out_path 进来，通常也是空的或者期望我们填写的
    // 所以我们尽量使用自己生成的 obj_path，最后再拷贝回去

    // 1. 写入源码
    {
        // 使用 truncate 模式打开，虽然文件名唯一，但以防万一
        std::ofstream src_file(src_path, std::ios::trunc);
        if (!src_file.is_open()) {
            std::cerr << "[MetaX] Failed to open temp file: " << src_path << std::endl;
            return C_Status::C_FAILED;
        }
        src_file << kMacaRuntimeSource << "\n";
        src_file << code;
        src_file.close();
    }

    // 2. 准备编译器路径
    const char* maca_path_env = std::getenv("MACA_PATH");
    std::string maca_path = maca_path_env ? std::string(maca_path_env) : "/opt/maca";
    
    std::string mxcc_cmd = maca_path + "/mxgpu_llvm/bin/mxcc";
    if (access(mxcc_cmd.c_str(), X_OK) != 0) {
         mxcc_cmd = maca_path + "/bin/mxcc";
         if (access(mxcc_cmd.c_str(), X_OK) != 0) mxcc_cmd = "mxcc";
    }

    // 3. 构建编译命令
    // 注意：加了空格防止粘连
    std::string cmd = mxcc_cmd + " -O3 -std=c++17 -w --fatbin --offload-arch=native -fvisibility=default";
    cmd += " -I" + maca_path + "/include";
    cmd += " -I" + maca_path + "/tools/cu-bridge/include";
    cmd += " -o " + obj_path;
    cmd += " " + src_path;

    // 4. 执行
    std::cout << "Command: " << cmd << std::endl;
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "[MetaX] JIT Compilation Failed! Code: " << ret << std::endl;
        std::cerr << "Command: " << cmd << std::endl;
        return C_Status::C_FAILED;
    }

    // 5. 确保文件存在
    if (access(obj_path.c_str(), F_OK) != 0) {
        std::cerr << "[MetaX] Output file missing: " << obj_path << std::endl;
        return C_Status::C_FAILED;
    }

    // =================================================================
    // 6. 【关键修复】将生成的二进制路径回填给 CINN 框架
    // =================================================================
    if (out_path && len > 0) {
        // 使用 strncpy 安全拷贝
        std::strncpy(out_path, obj_path.c_str(), len - 1);
        out_path[len - 1] = '\0'; // 确保 null 结尾
        // 打印调试信息，确认回填成功
        std::cout << "[MetaX Success] Compiled: " << out_path << std::endl;
    } else {
        std::cerr << "[MetaX Error] Invalid out_path buffer!" << std::endl;
        return C_Status::C_FAILED;
    }

    // 7. 清理源码 (调试成功后可开启)
    std::remove(src_path.c_str());

    return C_Status::C_SUCCESS;
}

} // namespace metax
} // namespace custom_device
} // namespace paddle