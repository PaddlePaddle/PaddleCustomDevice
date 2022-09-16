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

namespace custom_kernel {

template <typename T, typename Context>
void CumsumKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& axis_scalar,
                  bool flatten,
                  bool exclusive,
                  bool reverse,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto axis = axis_scalar.to<int>();

  Tensor input_tensor = x;
  if (flatten) {
    PADDLE_ENFORCE_EQ(
        axis,
        -1,
        phi::errors::InvalidArgument(
            "when flatten is true, attr axis must be default %d, but got %d",
            -1,
            axis));

    input_tensor.Resize(phi::make_ddim({x.numel()}));
  }

  const int true_axis = (axis < 0) ? input_tensor.dims().size() + axis : axis;
  MLUCnnlTensorDesc input_desc(input_tensor);
  MLUCnnlTensorDesc out_desc(*out);

  MLUCnnl::Cumsum(dev_ctx,
                  true_axis,
                  exclusive,
                  reverse,
                  input_desc.get(),
                  GetBasePtr(&input_tensor),
                  out_desc.get(),
                  GetBasePtr(out));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cumsum,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::CumsumKernel,
                          int,
                          float,
                          phi::dtype::float16) {}
