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

#include "kernels/funcs/reduce_op.h"

namespace custom_kernel {

template <typename T, typename Context>
void WhereKernel(const Context& dev_ctx,
                 const phi::DenseTensor& condition,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc y_desc(y);
  MLUCnnlTensorDesc condition_desc(condition);
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnl::Select(dev_ctx,
                  condition_desc.get(),
                  GetBasePtr(&condition),
                  x_desc.get(),
                  GetBasePtr(&x),
                  y_desc.get(),
                  GetBasePtr(&y),
                  out_desc.get(),
                  GetBasePtr(out));
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    where, mlu, ALL_LAYOUT, custom_kernel::WhereKernel, int32_t, float) {}
