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

#include "kernels/funcs/elementwise_utils.h"

namespace custom_kernel {

template <typename T, typename Context>
void MaximumRawKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      int axis,
                      phi::DenseTensor* out) {
  MLUBinaryOp<MAXIMUM, T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void MaximumKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::MaximumRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void MaximumGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       const phi::DenseTensor& dout,
                       int axis,
                       phi::DenseTensor* dx,
                       phi::DenseTensor* dy) {
  MLUMinMaxGradHelper<MAXIMUM_GRAD, T>(dev_ctx, x, y, dout, axis, dx, dy);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(maximum_raw,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::MaximumRawKernel,
                          int,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(maximum,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::MaximumKernel,
                          int,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(maximum_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::MaximumGradKernel,
                          int,
                          float,
                          phi::dtype::float16) {}
