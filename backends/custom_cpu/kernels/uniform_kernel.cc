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
void UniformRawKernel(const phi::Context &dev_ctx,
                      const phi::IntArray &shape,
                      phi::DataType dtype,
                      const phi::Scalar &min,
                      const phi::Scalar &max,
                      int seed,
                      int diag_num,
                      int diag_step,
                      float diag_val,
                      phi::DenseTensor *out) {
  auto shape_data = shape.GetData();

  out->Resize(std::vector<int64_t>(shape_data.begin(), shape_data.end()));
  T *data = dev_ctx.template Alloc<T>(out);
  auto size = out->numel();
  std::shared_ptr<std::mt19937_64> engine;

  engine = std::make_shared<std::mt19937_64>();
  engine->seed(seed);

  UniformRealDistribution<T>(
      data, size, min.to<float>(), max.to<float>(), engine);
  if (diag_num > 0) {
    PD_CHECK(size > (diag_num - 1) * (diag_step + 1),
             "ShapeInvalid: the diagonal's elements is equal (num-1) "
             "* (step-1) with num %d, step %d,"
             "It should be smaller than %d, but received %d",
             diag_num,
             diag_step,
             (diag_num - 1) * (diag_step + 1),
             size);
    for (int64_t i = 0; i < diag_num; ++i) {
      int64_t pos = i * diag_step + i;
      data[pos] = diag_val;
    }
  }
}

template <typename T>
void UniformKernel(const phi::Context &dev_ctx,
                   const phi::IntArray &shape,
                   phi::DataType dtype,
                   const phi::Scalar &min,
                   const phi::Scalar &max,
                   int seed,
                   phi::DenseTensor *out) {
  UniformRawKernel<T>(dev_ctx, shape, dtype, min, max, seed, 0, 0, 0.0f, out);
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(uniform_raw,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::UniformRawKernel,
                    float,
                    double) {}

PD_BUILD_PHI_KERNEL(uniform,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::UniformKernel,
                    float,
                    double) {}
