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

#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void GaussianKernel(const Context& ctx,
                    const phi::IntArray& shape,
                    float mean,
                    float std,
                    int seed,
                    phi::DataType dtype,
                    phi::DenseTensor* out) {
  ctx.template Alloc<T>(out);

  phi::DenseTensor cpu_tensor;
  phi::DenseTensorMeta cpu_meta = {out->dtype(), out->dims()};
  cpu_tensor.set_meta(cpu_meta);
  T* cpu_data = ctx.template HostAlloc<T>(&cpu_tensor);
  std::normal_distribution<T> dist(mean, std);

  int64_t size = out->numel();

  std::shared_ptr<std::mt19937_64> engine;
  if (seed) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);
  } else {
    engine = ctx.GetGenerator()->GetCPUEngine();
  }

  for (int64_t i = 0; i < size; ++i) {
    cpu_data[i] = dist(*engine);
  }
  TensorCopy(ctx, cpu_tensor, true, out);
}

}  // namespace custom_kernel

// PD_REGISTER_PLUGIN_KERNEL(
//     gaussian, gcu, ALL_LAYOUT, custom_kernel::GaussianKernel, float) {}
