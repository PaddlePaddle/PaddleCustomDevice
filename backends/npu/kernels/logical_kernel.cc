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
  dev_ctx.template Alloc<bool>(out);

  auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                    const std::vector<phi::DenseTensor>& outputs,
                    const NPUAttributeMap& attrs,
                    const phi::CustomContext& dev_ctx) {
    const auto& runner =
        NpuOpRunner("LogicalNot", {inputs[0]}, {outputs[0]}, attrs);
    runner.Run(dev_ctx.stream());
  };

  NpuOpRunner::TypeAdapter({x},
                           {*out},
                           {},
                           dev_ctx,
                           op_func,
                           {phi::DataType::BOOL},
                           {phi::DataType::BOOL});
}

template <typename T, typename Context>
void LogicalOrNPUKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);

  auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                    const std::vector<phi::DenseTensor>& outputs,
                    const NPUAttributeMap& attrs,
                    const phi::CustomContext& dev_ctx) {
    const auto& runner =
        NpuOpRunner("LogicalOr", {inputs[0], inputs[1]}, {outputs[0]}, attrs);
    runner.Run(dev_ctx.stream());
  };
  NpuOpRunner::TypeAdapter({x, y},
                           {*out},
                           {},
                           dev_ctx,
                           op_func,
                           {phi::DataType::BOOL, phi::DataType::BOOL},
                           {phi::DataType::BOOL});
}

template <typename T, typename Context>
void LogicalAndNPUKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& y,
                         phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);

  auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                    const std::vector<phi::DenseTensor>& outputs,
                    const NPUAttributeMap& attrs,
                    const phi::CustomContext& dev_ctx) {
    const auto& runner =
        NpuOpRunner("LogicalAnd", {inputs[0], inputs[1]}, {outputs[0]}, {});
    runner.Run(dev_ctx.stream());
  };
  NpuOpRunner::TypeAdapter({x, y},
                           {*out},
                           {},
                           dev_ctx,
                           op_func,
                           {phi::DataType::BOOL, phi::DataType::BOOL},
                           {phi::DataType::BOOL});
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(logical_not,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LogicalNotNPUKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(logical_or,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LogicalOrNPUKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(logical_and,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LogicalAndNPUKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
