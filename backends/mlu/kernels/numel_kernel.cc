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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void NumelKernel(const Context& dev_ctx,
                 const phi::DenseTensor& input,
                 phi::DenseTensor* out) {
  dev_ctx.template Alloc<int64_t>(out);
  int64_t size = input.numel();
  FillMLUTensorWithHostValue<int64_t>(dev_ctx, size, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(numel,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::NumelKernel,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          float,
                          double,
                          bool) {}
