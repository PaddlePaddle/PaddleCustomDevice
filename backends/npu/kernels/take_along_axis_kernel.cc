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
void TakeAlongAxisKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& index,
                         int axis,
                         phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  const auto& runner =
      NpuOpRunner("GatherElements", {x, index}, {*out}, {{"dim", axis}});
  runner.Run(stream);
}

template <typename T, typename Context>
void TakeAlongAxisGradKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::DenseTensor& index,
                             const phi::DenseTensor& out_grad,
                             int axis,
                             phi::DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("ScatterAddWithAxis",
                                   {*x_grad, index, out_grad},
                                   {*x_grad},
                                   {{"axis", axis}});
  runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(take_along_axis,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::TakeAlongAxisKernel,
                          int,
                          int64_t,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(take_along_axis_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::TakeAlongAxisGradKernel,
                          int,
                          int64_t,
                          float,
                          double) {}
