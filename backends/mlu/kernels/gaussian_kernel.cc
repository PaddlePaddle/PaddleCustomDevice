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

#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void GaussianKernel(const Context& dev_ctx,
                    const phi::IntArray& shape,
                    float mean,
                    float std,
                    int seed,
                    phi::DataType dtype,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  int real_seed =
      seed != 0 ? seed : static_cast<int>(dev_ctx.GetGenerator()->Random64());
  auto dev_id = static_cast<int64_t>(dev_ctx.GetPlace().GetDeviceId());
  auto generator_desc = GetMLURandomGenerator(dev_ctx, dev_id, real_seed);

  MLUCnnl::RandGenerateNormal(dev_ctx,
                              generator_desc->get(),
                              ToCnnlDataType<T>(),
                              out->numel(),
                              mean,
                              std,
                              GetBasePtr(&generator_desc->get_state()),
                              GetBasePtr(out));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(gaussian,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::GaussianKernel,
                          float,
                          phi::dtype::float16) {}
