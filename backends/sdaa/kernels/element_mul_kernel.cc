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
void MultiplyRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA MultiplyKernel";
  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doElementMul(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void MultiplyKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  if (isEnvEnable("HIGH_PERFORMANCE_CONV") &&
      (&x != out && x.storage_properties_initialized())) {
    VLOG(1) << "Multiply spread conv's filter at " << &x;
    // only open to adamw optimzer with ClipGradByGlobalNorm
    // called by: new_grad = paddle.multiply(g, clip_input)
    PADDLE_ENFORCE_EQ(y.numel(),
                      1,
                      phi::errors::InvalidArgument(
                          "grad tensor's storage propertiey only supported in "
                          "optimizer.Adamw with ClipGradByGlobalNorm"));

    SDAAStorageProperties properties =
        x.storage_properties<SDAAStorageProperties>();

    sdaa_ops::doAddStorageProperties(dev_ctx, out, properties);
  }

  int axis = -1;
  custom_kernel::MultiplyRawKernel<T>(dev_ctx, x, y, axis, out);
}  // namespace custom_kernel

template <typename T, typename Context>
void MultiplyGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        const phi::DenseTensor& dout,
                        int axis,
                        phi::DenseTensor* dx,
                        phi::DenseTensor* dy) {
  VLOG(4) << "Call SDAA MultiplyGradKernel";

  auto out_dims_vec = phi::vectorize<int64_t>(dout.dims());
  std::vector<int64_t> x_dims_vec, y_dims_vec;
  broadcastDims<int64_t>(x.dims(), y.dims(), axis, &x_dims_vec, &y_dims_vec);
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
    if (dy->dims() == dout.dims()) {
      sdaa_ops::doElementMul(dev_ctx, dout, x, axis, dy);
    } else {
      phi::DenseTensor y_temp;
      y_temp.Resize(dout.dims());
      dev_ctx.template Alloc<T>(&y_temp);
      sdaa_ops::doElementMul(dev_ctx,
                             dout,
                             x,
                             x.dims().size() == dout.dims().size() ? -1 : axis,
                             &y_temp);
      auto reduce_dims = findReduceDims(y_dims_vec, out_dims_vec);
      sdaa_ops::doSumTensor(dev_ctx, y_temp, reduce_dims, dy);
    }
  }
  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    if (dx->dims() == dout.dims()) {
      sdaa_ops::doElementMul(dev_ctx, dout, y, axis, dx);
    } else {
      phi::DenseTensor x_temp;
      x_temp.Resize(dout.dims());
      dev_ctx.template Alloc<T>(&x_temp);
      sdaa_ops::doElementMul(dev_ctx,
                             dout,
                             y,
                             y.dims().size() == dout.dims().size() ? -1 : axis,
                             &x_temp);
      auto reduce_dims = findReduceDims(x_dims_vec, out_dims_vec);
      sdaa_ops::doSumTensor(dev_ctx, x_temp, reduce_dims, dx);
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(multiply_raw,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::MultiplyRawKernel,
                          int64_t,
                          int,
                          int16_t,
                          int8_t,
                          uint8_t,
                          bool,
                          phi::dtype::float16,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(multiply,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::MultiplyKernel,
                          int64_t,
                          int,
                          int16_t,
                          int8_t,
                          uint8_t,
                          bool,
                          phi::dtype::float16,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(multiply_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::MultiplyGradKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          int,
                          int64_t) {}
