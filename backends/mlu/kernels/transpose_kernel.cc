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
void TransposeKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const std::vector<int>& axis,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  TransposeFromMLUTensor<T>(
      dev_ctx, axis, &x, out, false /*need_reshape_or_alloc*/);
}

template <typename T, typename Context>
void TransposeGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& dout,
                         const std::vector<int>& axis,
                         phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  std::vector<int> reversed_axis(axis);
  for (size_t i = 0; i < axis.size(); i++) {
    reversed_axis[axis[i]] = i;
  }
  TransposeFromMLUTensor<T>(
      dev_ctx, reversed_axis, &dout, dx, false /*need_reshape_or_alloc*/);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(transpose,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::TransposeKernel,
                          int,
                          uint8_t,
                          int8_t,
                          int16_t,
                          float,
                          phi::dtype::float16,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(transpose_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::TransposeGradKernel,
                          int,
                          uint8_t,
                          int8_t,
                          int16_t,
                          float,
                          phi::dtype::float16,
                          bool) {}
