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
void AddRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA AddKernel";
  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doElementAdd(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out) {
  if (x.dtype() == phi::DataType::FLOAT32 &&
      y.dtype() == phi::DataType::FLOAT16) {
    VLOG(4) << "Call SDAA Multi-Precision AddKernel";

    PADDLE_ENFORCE_EQ(x.dims(),
                      y.dims(),
                      phi::errors::InvalidArgument(
                          "The input tensor x and y dims must be same "
                          "on %s",
                          dev_ctx.GetPlace()));

    dev_ctx.template Alloc<float>(out);
    sdaaStream_t custom_stream = GetStreamFromCTX(dev_ctx);
    auto dims = phi::vectorize<int>(x.dims());
    TCUS_CHECK(sdcops::multi_precision_add(x.data(),
                                           y.data(),
                                           out->data(),
                                           dims.data(),
                                           dims.size(),
                                           custom_stream));
  } else {
    int axis = -1;
    custom_kernel::AddRawKernel<T>(dev_ctx, x, y, axis, out);
  }
}
template <typename T, typename Context>
void AddGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   const phi::DenseTensor& dout,
                   int axis,
                   phi::DenseTensor* dx,
                   phi::DenseTensor* dy) {
  VLOG(4) << "Call SDAA AddGradKernel";

  auto out_dims_vec = phi::vectorize<int64_t>(dout.dims());
  std::vector<int64_t> x_dims_vec, y_dims_vec;
  broadcastDims(x.dims(), y.dims(), axis, &x_dims_vec, &y_dims_vec);

  // under inplace situation, dx and dout would share address, causing wrong
  // calculation of dy, so dy first
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
    if (dy->dims() != dout.dims()) {
      auto reduce_dims = findReduceDims(y_dims_vec, out_dims_vec);
      sdaa_ops::doSumTensor(dev_ctx, dout, reduce_dims, dy);
    } else {
      phi::Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dy);
    }
  }

  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    if (dx->dims() != dout.dims()) {
      auto reduce_dims = findReduceDims(x_dims_vec, out_dims_vec);
      sdaa_ops::doSumTensor(dev_ctx, dout, reduce_dims, dx);
    } else {
      phi::Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
    }
  }
}

template <typename T, typename Context>
void GradAddKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA GradAddKernel";

  custom_kernel::AddRawKernel<T>(dev_ctx, x, y, -1, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(add_raw,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::AddRawKernel,
                          int64_t,
                          int,
                          int16_t,
                          int8_t,
                          uint8_t,
                          phi::dtype::float16,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(add,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::AddKernel,
                          int64_t,
                          int,
                          int16_t,
                          int8_t,
                          uint8_t,
                          phi::dtype::float16,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(add_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::AddGradKernel,
                          float,
                          phi::dtype::float16,
                          double,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(grad_add,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::GradAddKernel,
                          int64_t,
                          int,
                          int16_t,
                          int8_t,
                          uint8_t,
                          phi::dtype::float16,
                          float,
                          double) {}
