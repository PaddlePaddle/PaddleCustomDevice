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
void LogicalNotNPUKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("LogicalNot", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void LogicalOrNPUKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("LogicalOr", {x, y}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void LogicalAndNPUKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& y,
                         phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("LogicalAnd", {x, y}, {*out}, {});
  runner.Run(stream);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    logical_not, ascend, ALL_LAYOUT, custom_kernel::LogicalNotNPUKernel, bool) {
}

PD_REGISTER_PLUGIN_KERNEL(
    logical_or, ascend, ALL_LAYOUT, custom_kernel::LogicalOrNPUKernel, bool) {}

PD_REGISTER_PLUGIN_KERNEL(
    logical_and, ascend, ALL_LAYOUT, custom_kernel::LogicalAndNPUKernel, bool) {
}
