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

#include "kernels/funcs/sdaa_baseop.h"
#include "kernels/funcs/sdaa_funcs.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void SubtractRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA SubtractKernel";
  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doElementSub(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void SubtractKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::SubtractRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void SubtractGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        const phi::DenseTensor& dout,
                        int axis,
                        phi::DenseTensor* dx,
                        phi::DenseTensor* dy) {
  VLOG(4) << "Call SDAA SubtractGradKernel";
  auto out_dims_vec = phi::vectorize<int64_t>(dout.dims());
  std::vector<int64_t> x_dims_vec, y_dims_vec;
  broadcastDims<int64_t>(x.dims(), y.dims(), axis, &x_dims_vec, &y_dims_vec);

  // this kernel is inplace op:(dout->dx), if calculate dx first， dout's value
  // may change ,lead to dy get the wrong value
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
    auto reduce_dims = findReduceDims(y_dims_vec, out_dims_vec);
    sdaa_ops::doSumTensor(dev_ctx, dout, reduce_dims, dy);
    sdaa_ops::doNegTensor(dev_ctx, *dy, dy);
  }
  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    auto reduce_dims = findReduceDims(x_dims_vec, out_dims_vec);
    sdaa_ops::doSumTensor(dev_ctx, dout, reduce_dims, dx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(subtract_raw,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SubtractRawKernel,
                          int64_t,
                          int,
                          int16_t,
                          int8_t,
                          uint8_t,
                          phi::dtype::float16,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(subtract,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SubtractKernel,
                          int64_t,
                          int,
                          int16_t,
                          int8_t,
                          uint8_t,
                          phi::dtype::float16,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(subtract_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SubtractGradKernel,
                          float,
                          phi::dtype::float16) {}
