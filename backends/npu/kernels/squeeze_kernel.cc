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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void SqueezeInferKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::IntArray& axes_int_array,
                        phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();

  auto out_dims = out->dims();
  dev_ctx.template Alloc<T>(out);

  TensorCopy(dev_ctx, x, false, out);

  out->Resize(out_dims);
}

template <typename T, typename Context>
void SqueezeKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::IntArray& axes_int_array,
                   phi::DenseTensor* out,
                   phi::DenseTensor* xshape) {
  custom_kernel::SqueezeInferKernel<T, Context>(
      dev_ctx, x, axes_int_array, out);
}

template <typename T, typename Context>
void SqueezeGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& xshape,
                       const phi::DenseTensor& dout,
                       const phi::IntArray& axes_int_array,
                       phi::DenseTensor* dx) {
  auto stream = dev_ctx.stream();

  auto xshape_dims = xshape.dims();
  auto x_dims = phi::slice_ddim(xshape_dims, 1, xshape_dims.size());

  TensorCopy(dev_ctx, dout, false, dx);
  dx->Resize(x_dims);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(squeeze_infer,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SqueezeInferKernel,
                          bool,
                          int,
                          uint8_t,
                          int8_t,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(squeeze,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SqueezeKernel,
                          bool,
                          int,
                          uint8_t,
                          int8_t,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(squeeze_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SqueezeGradKernel,
                          bool,
                          int,
                          uint8_t,
                          int8_t,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}
