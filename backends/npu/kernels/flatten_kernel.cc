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

#include "npu_funcs.h"
#include "npu_op_runner.h"

#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace custom_kernel {

template <typename T, typename Context>
void FlattenKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int start_axis,
                   int stop_axis,
                   phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  const auto& runner =
      NpuOpRunner("FlattenV2",
                  {x},
                  {*out},
                  {{"axis", static_cast<int32_t>(start_axis)},
                   {"end_axis", static_cast<int32_t>(stop_axis)}});
  auto stream = dev_ctx.stream();
  runner.Run(stream);
}

template <typename T, typename Context>
void FlattenGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out_grad,
                       const phi::DenseTensor& xshape,
                       phi::DenseTensor* x_grad) {
  auto xshape_dims = xshape.dims();
  auto x_dims = phi::slice_ddim(xshape_dims, 1, xshape_dims.size());

  TensorCopy(dev_ctx, out_grad, false, x_grad);
  x_grad->Resize(x_dims);
}

template <typename T, typename Context>
void FlattenWithXShape(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       int start_axis,
                       int stop_axis,
                       phi::DenseTensor* out,
                       phi::DenseTensor* xshape) {
  custom_kernel::FlattenKernel<T, Context>(
      dev_ctx, x, start_axis, stop_axis, out);
  phi::funcs::SetXShape(x, xshape);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(flatten,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::FlattenKernel,
                          float,
                          double,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(flatten_with_xshape,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::FlattenWithXShape,
                          float,
                          double,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(flatten_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::FlattenGradKernel,
                          float,
                          double,
                          int16_t,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::float16) {}
