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

#include "dnn_support.hpp"
#include <random>
#include "paddle/phi/capi/all.h"


namespace custom_kernel {

template <typename T>
inline void UniformRealDistribution(T *data,
                                    const int64_t &size,
                                    const float &min,
                                    const float &max,
                                    std::shared_ptr<std::mt19937_64> engine) {
  std::uniform_real_distribution<T> dist(static_cast<T>(min),
                                         static_cast<T>(max));
  for (int64_t i = 0; i < size; ++i) {
    data[i] = dist(*engine);
  }
}

template <typename T>
void UniformRandomRawKernel(const phi::Context &dev_ctx,
                            const phi::IntArray &shape,
                            phi::DataType dtype,
                            const phi::Scalar &min,
                            const phi::Scalar &max,
                            int seed,
                            int diag_num,
                            int diag_step,
                            float diag_val,
                            phi::DenseTensor *out) {
   show_kernel("UniformRandom-SYCL type=" << dnn_support::type2String<T>::name());

  auto shape_data = shape.GetData();
  out->Resize(std::vector<int64_t>(shape_data.begin(), shape_data.end()));
  auto out_data = dev_ctx.template Alloc<T>(out);
  auto numel = out->numel();

  // // 1. CPU implement
  phi::DenseTensor cpu_out;
  cpu_out.Resize(std::vector<int64_t>(shape_data.begin(), shape_data.end()));
  cpu_out.set_dtype(out->dtype());
  auto cpu_data = dev_ctx.template HostAlloc<T>(&cpu_out);

  std::shared_ptr<std::mt19937_64> engine;
  engine = std::make_shared<std::mt19937_64>();
  engine->seed(seed);

  UniformRealDistribution<T>(
      cpu_data, numel, min.to<float>(), max.to<float>(), engine);
  if (diag_num > 0) {
    PD_CHECK(
        numel,
        (diag_num - 1) * (diag_step + 1),
            "ShapeInvalid: the diagonal's elements is equal (num-1) "
            "* (step-1) with num %d, step %d,"
            "It should be smaller than %d, but received %d",
            diag_num,
            diag_step,
            (diag_num - 1) * (diag_step + 1),
            numel);
    for (int64_t i = 0; i < diag_num; ++i) {
      int64_t pos = i * diag_step + i;
      cpu_data[pos] = diag_val;
    }
  }

  // 2. CPU Copy to IntelGPU
  auto* q = static_cast<sycl::queue*>(dev_ctx.stream());
  q->memcpy(out_data, cpu_data, numel*sizeof(T));
}

template <typename T>
void UniformRandomKernel(const phi::Context &dev_ctx,
                         const phi::IntArray &shape,
                         phi::DataType dtype,
                        //  float min,
                        //  float max,
                         const phi::Scalar &min,
                         const phi::Scalar &max,
                         int seed,
                         phi::DenseTensor *out) {
  show_kernel("UniformRandom-SYCL type=" << dnn_support::type2String<T>::name());
   custom_kernel::UniformRandomRawKernel<T>(
      dev_ctx, shape, dtype, min, max, seed, 0, 0, 0.0f, out);
}
}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(uniform_random_raw,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::UniformRandomRawKernel,
                    float) {}

PD_BUILD_PHI_KERNEL(uniform_random,
                    intel_gpu,
                    ALL_LAYOUT,
                    custom_kernel::UniformRandomKernel,
                    float) {}
