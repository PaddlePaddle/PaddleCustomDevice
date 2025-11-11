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
void MultinomialKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::Scalar& num,
                       bool replacement,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<int64_t>(out);
  MLUCnnlTensorDesc desc_x(x);
  MLUCnnlTensorDesc desc_out(*out);

  int real_seed = static_cast<int>(dev_ctx.GetGenerator()->Random64());
  auto dev_id = static_cast<int64_t>(dev_ctx.GetPlace().GetDeviceId());
  auto generator_desc = GetMLURandomGenerator(dev_ctx, dev_id, real_seed);

  MLUCnnl::RandGenerateMultinomial(dev_ctx,
                                   generator_desc->get(),
                                   desc_x.get(),
                                   GetBasePtr(&x),
                                   replacement,
                                   false,
                                   GetBasePtr(&generator_desc->get_state()),
                                   desc_out.get(),
                                   GetBasePtr(out));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(multinomial,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::MultinomialKernel,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
