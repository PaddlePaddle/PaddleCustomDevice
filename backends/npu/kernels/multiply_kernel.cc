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
void MultiplyRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  bool direct_compute = false;
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  if (x_dims.size() >= y_dims.size()) {
    direct_compute = x_dims.size() == (y_dims.size() + axis);
  } else {
    direct_compute = y_dims.size() == (x_dims.size() + axis);
  }
  aclrtStream stream = static_cast<aclrtStream>(dev_ctx.stream());

  if (direct_compute) {
    const auto& runner = NpuOpRunner("Mul", {x, y}, {*out}, {});
    runner.Run(stream);
  } else {
    phi::DenseTensor trans_x, trans_y;
    NpuElementWiseOpBroadcast<T>(dev_ctx, &x, &y, axis, &trans_x, &trans_y);
    const auto& runner = NpuOpRunner("Mul", {trans_x, trans_y}, {*out}, {});
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void MultipyKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::MultiplyRawKernel<T>(dev_ctx, x, y, axis, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(multiply,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::MultipyKernel,
                          int8_t,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(multiply_raw,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::MultiplyRawKernel,
                          int8_t,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float,
                          double) {}
