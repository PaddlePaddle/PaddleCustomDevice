// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#ifdef PADDLE_WITH_ASCENDC

#ifdef __ASCEND_C__
// compiled by the ascendc compiler
#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT
#endif
#ifndef __aicpu__
#define __aicpu__ [aicpu]  // NOLINT
#endif
#ifndef __gm__
#define __gm__ __attribute__((cce_global))
#endif
#ifndef __ca__
#define __ca__ __attribute__((cce_cube_a))
#endif
#ifndef __cb__
#define __cb__ __attribute__((cce_cube_b))
#endif
#ifndef __cc__
#define __cc__ __attribute__((cce_cube_c))
#endif
#ifndef __ubuf__
#define __ubuf__ __attribute__((cce_unif_buff))
#endif
#ifndef __cbuf__
#define __cbuf__ __attribute__((cce_cube_buff))
#endif
#ifndef __fbuf__
#define __fbuf__ __attribute__((cce_fixpipe_buff))
#endif
#ifndef __global__
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t* __restrict__
#endif

#else
// compiled by the hsot compiler
#ifndef __aicore__
#define __aicore__
#endif
#ifndef __aicpu__
#define __aicpu__
#endif
#ifndef __gm__
#define __gm__
#endif
#ifndef __ca__
#define __ca__
#endif
#ifndef __cb__
#define __cb__
#endif
#ifndef __cc__
#define __cc__
#endif
#ifndef __ubuf__
#define __ubuf__
#endif
#ifndef __cbuf__
#define __cbuf__
#endif
#ifndef __fbuf__
#define __fbuf__
#endif
#ifndef __global__
#define __global__
#endif

#include "runtime/runtime.h"

#endif  // __ASCEND_C__

namespace custom_device {
namespace ascendc {

inline __aicore__ size_t __div_ceil(size_t m, size_t n) {
  return (m + n - 1) / n;
}

inline __aicore__ size_t __div_floor(size_t m, size_t n) { return m / n; }

inline __aicore__ size_t __align_up(size_t m, size_t n) {
  return __div_ceil(m, n) * n;
}

inline __aicore__ size_t __align_down(size_t m, size_t n) {
  return __div_floor(m, n) * n;
}

#ifdef __ASCEND_C__
template <typename TilingDataT>
inline __aicore__ __ubuf__ TilingDataT* __get_tiling_ubuf(GM_ADDR tiling_dat) {
  __ubuf__ TilingDataT* tiling_ubuf =
      reinterpret_cast<__ubuf__ TilingDataT*>(get_imm(0));
  copy_gm_to_ubuf(reinterpret_cast<__ubuf__ uint8_t*>(tiling_ubuf),
                  tiling_dat,
                  0,
                  1,
                  __div_ceil(sizeof(TilingDataT), 32),
                  0,
                  0);
  pipe_barrier(PIPE_ALL);
  return tiling_ubuf;
}
#else
template <typename TilingDataT>
inline void* __get_tiling_device(const phi::CustomContext& dev_ctx,
                                 TilingDataT* tilingHost) {
  uint32_t tilingSize = sizeof(TilingDataT);
  phi::DenseTensor tiling;
  tiling.Resize({tilingSize});
  void* tilingDevice = dev_ctx.template Alloc<uint8_t>(&tiling);

  C_Device_st device{dev_ctx.GetPlace().GetDeviceId()};
  C_Stream stream{reinterpret_cast<C_Stream>(dev_ctx.stream())};
  AsyncMemCpyH2D(&device,
                 stream,
                 tilingDevice,
                 reinterpret_cast<void*>(tilingHost),
                 tilingSize);
  return tilingDevice;
}

#endif

inline size_t GetBlockNum() { return 48; }

}  // namespace ascendc
}  // namespace custom_device

#endif
