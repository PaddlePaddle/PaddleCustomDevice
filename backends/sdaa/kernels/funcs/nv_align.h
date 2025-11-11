// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include <algorithm>
#include <string>
#include <vector>

#include "kernels/funcs/sdaa_funcs.h"
#include "paddle/phi/extension.h"

#ifndef BACKENDS_SDAA_KERNELS_FUNCS_NV_ALIGN_H_
#define BACKENDS_SDAA_KERNELS_FUNCS_NV_ALIGN_H_

static const char* ALIGN_NV = "RANDOM_ALIGN_NV_DEVICE";

namespace custom_kernel {

inline void GetSeed(const phi::CustomContext& dev_ctx,
                    const paddle::optional<phi::DenseTensor>& seed_tensor,
                    int seed,
                    bool fix_seed,
                    const int offset,
                    uint64_t* seed_data,
                    uint64_t* increment) {
  auto gen_cuda = dev_ctx.GetGenerator();
  if (seed_tensor) {
    std::vector<int> seeds;
    TensorToVector(dev_ctx, seed_tensor.get(), dev_ctx, &seeds);
    *seed_data = static_cast<uint64_t>(seeds[0]);
    *increment = offset;
  } else if (!fix_seed) {
    auto seed_offset = gen_cuda->IncrementOffset(offset);
    *seed_data = seed_offset.first;
    *increment = seed_offset.second;
  } else {
    *seed_data = seed;
    *increment = offset;
  }
}

template <typename T = int64_t>
inline T DivUp(T a, T b) {
  return (a + b - 1) / b;
}

// https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
//   for round integer value into next highest power of 2.
inline int64_t RoundToNextHighPowOfTwo(int64_t n, int64_t min_val = 1) {
  n--;
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max(min_val, (n + 1));
}

inline int64_t RoundToPowerOfTwo(int64_t n) {
  constexpr int64_t min_val = 32;
  int64_t num = RoundToNextHighPowOfTwo(n, min_val);
  constexpr int64_t max_val = 1024;
  return std::min(max_val, num);
}

inline void GetGPUConfig(const char* mode_char,
                         int* max_threads,
                         int* sm_count) {
  std::string mode(mode_char);
  std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
  PADDLE_ENFORCE_EQ(
      mode == "a100" || mode == "v100",
      true,
      phi::errors::InvalidArgument(
          "Only a100 and v100 alignement is supported, but got %s", mode));

  if (mode == "a100") {
    *max_threads = 2048;
    *sm_count = 108;
  } else {
    *max_threads = 2048;
    *sm_count = 80;
  }
}

inline void GetBlockGrid(size_t size,
                         int max_threads,
                         int sm_count,
                         int vec_size,
                         int* block_size,
                         int* grid_size) {
  int threads = max_threads;
  int limit_threads = std::min(512, max_threads);
  int64_t active_threads_num = size / vec_size;
  if (active_threads_num / (sm_count << 1) < limit_threads) {
    threads = RoundToPowerOfTwo(active_threads_num / (sm_count << 1));
  } else if (active_threads_num / (sm_count << 2) < limit_threads) {
    threads = RoundToPowerOfTwo(active_threads_num / (sm_count << 2));
  }
  // Number of threads per block shall be larger than 64.
  threads = std::max(64, threads);
  int blocks = DivUp<int64_t>(DivUp<int64_t>(size, vec_size), threads);
  if (blocks > 65536) {
    blocks = 65536;
  }
  *block_size = threads;
  *grid_size = blocks;
}
}  // namespace custom_kernel
#endif  // BACKENDS_SDAA_KERNELS_FUNCS_NV_ALIGN_H_
