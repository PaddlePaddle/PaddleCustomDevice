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
void DivideRawKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     int axis,
                     phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA DivideRawKernel";
  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doElementDiv(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void DivideKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::DivideRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void DivideGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      const phi::DenseTensor& out,
                      const phi::DenseTensor& dout,
                      int axis,
                      phi::DenseTensor* dx,
                      phi::DenseTensor* dy) {
  VLOG(4) << "Call SDAA DivideGradKernel";

  auto out_dims_vec = phi::vectorize<int64_t>(dout.dims());
  std::vector<int64_t> x_dims_vec, y_dims_vec;
  broadcastDims<int64_t>(x.dims(), y.dims(), axis, &x_dims_vec, &y_dims_vec);
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
    phi::DenseTensor temp_out;
    phi::DenseTensorMeta temp_out_meta = {dout.dtype(), dout.dims()};
    temp_out.set_meta(temp_out_meta);
    dev_ctx.template Alloc<T>(&temp_out);
    sdaa_ops::doElementMul(dev_ctx, dout, out, -1, &temp_out);
    if (dy->dims() == dout.dims()) {
      sdaa_ops::doElementDiv(dev_ctx, temp_out, y, -1, dy);
    } else {
      auto reduce_dims = findReduceDims(y_dims_vec, out_dims_vec);
      sdaa_ops::doSumTensor(dev_ctx, temp_out, reduce_dims, dy);
      sdaa_ops::doElementDiv(dev_ctx, *dy, y, -1, dy);
    }
    sdaa_ops::doNegTensor(dev_ctx, *dy, dy);
  }
  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    phi::DenseTensor y_temp(y);
    y_temp.Resize(phi::make_ddim(y_dims_vec));

    if (dx->dims() == dout.dims()) {
      sdaa_ops::doElementDiv(dev_ctx, dout, y_temp, -1, dx);
    } else {
      phi::DenseTensor x_temp;
      x_temp.Resize(phi::make_ddim(out_dims_vec));
      dev_ctx.template Alloc<T>(&x_temp);
      sdaa_ops::doElementDiv(dev_ctx, dout, y_temp, -1, &x_temp);
      auto reduce_dims = findReduceDims(x_dims_vec, out_dims_vec);
      sdaa_ops::doSumTensor(dev_ctx, x_temp, reduce_dims, dx);
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(divide_raw,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::DivideRawKernel,
                          int64_t,
                          int,
                          int16_t,
                          int8_t,
                          uint8_t,
                          phi::dtype::float16,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(divide,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::DivideKernel,
                          int64_t,
                          int,
                          int16_t,
                          int8_t,
                          uint8_t,
                          phi::dtype::float16,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(divide_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::DivideGradKernel,
                          float,
                          phi::dtype::float16) {}
