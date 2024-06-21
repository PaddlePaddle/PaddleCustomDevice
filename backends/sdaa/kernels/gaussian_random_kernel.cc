// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>

#include "kernels/funcs/nv_align.h"
#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"  // 自定义Kernel依赖头文件

namespace custom_kernel {

template <typename Context>
void GaussianRandomAlign(const Context& dev_ctx,
                         size_t numel,
                         float mean,
                         float stddev,
                         const char* mode,
                         phi::DenseTensor* out) {
  // Align sdaa with NV device
  uint64_t seed_data;
  int max_threads, sm_count;

  // only float and half, so vec_size is 4
  constexpr int vec_size = 4;
  size_t block_size = 256;
  size_t expect_grid_size = (numel + block_size - 1) / block_size;

  custom_kernel::GetGPUConfig(mode, &max_threads, &sm_count);
  size_t max_grid_size = (max_threads / block_size) * sm_count;
  size_t grid_size =
      expect_grid_size > max_grid_size ? max_grid_size : expect_grid_size;

  size_t total_thread = block_size * grid_size;
  size_t curand4_loop_times =
      (numel + vec_size * total_thread - 1) / (vec_size * total_thread);
  // 'increment' should be multiple of 4
  uint64_t increment = curand4_loop_times * vec_size;

  auto gen_cuda = dev_ctx.GetGenerator();
  auto seed_offset = gen_cuda->IncrementOffset(increment);
  uint64_t seed = seed_offset.first;
  uint64_t offset = seed_offset.second;

  VLOG(4) << "gaussian: size=" << out->numel() << ", vec_size=" << vec_size
          << ", block_size=" << block_size << ", grid_size=" << grid_size
          << ", seed=" << seed << ", offset=" << offset;

  phi::DenseTensor float_temp;
  if (out->dtype() == phi::DataType::FLOAT16) {
    float_temp.Resize(out->dims());
    dev_ctx.template Alloc<float>(&float_temp);
  } else {
    float_temp = *out;
  }
  sdaaStream_t custom_stream = GetStreamFromCTX(dev_ctx);
  sdcops::pd_normal_kernel(float_temp.data(),
                           numel,
                           mean,
                           stddev,
                           seed,
                           offset,
                           total_thread,
                           custom_stream);
  phi::Copy(dev_ctx, float_temp, out->place(), false, out);
}

template <typename T, typename Context>
void GaussianRandomKernel(const Context& dev_ctx,
                          const phi::IntArray& shape,
                          float mean,
                          float std,
                          int seed,
                          phi::DataType dtype,
                          phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA GaussianRandomKernel";

  auto shape_vec = shape.GetData();
  out->Resize(phi::make_ddim(shape_vec));
  dev_ctx.template Alloc<T>(out);

  if (out->numel() == 0) {
    return;
  }

  const char* value = std::getenv(ALIGN_NV);
  if (value && seed == 0) {
    GaussianRandomAlign(dev_ctx, out->numel(), mean, std, value, out);
    return;
  }
  std::normal_distribution<float> dist(mean, std);
  phi::DenseTensor host_temp;
  host_temp.Resize(out->dims());
  float* data = dev_ctx.template HostAlloc<float>(&host_temp);
  if (seed == 0) {
    auto engine = dev_ctx.GetGenerator()->GetCPUEngine();
    for (size_t i = 0; i < out->numel(); i++) {
      data[i] = dist(*engine);
    }
  } else {
    std::minstd_rand rand;
    rand.seed(seed);
    for (size_t i = 0; i < out->numel(); i++) {
      data[i] = dist(rand);
    }
  }
  if (out->dtype() == phi::DataType::FLOAT16) {
    phi::DenseTensor float_temp;
    float_temp.Resize(out->dims());
    dev_ctx.template Alloc<float>(&float_temp);
    phi::Copy(dev_ctx, host_temp, float_temp.place(), false, &float_temp);
    sdaa_ops::doCastTensor(dev_ctx, float_temp, out);
  } else {
    phi::Copy(dev_ctx, host_temp, out->place(), false, out);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(gaussian,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::GaussianRandomKernel,
                          float,
                          phi::dtype::float16) {}
