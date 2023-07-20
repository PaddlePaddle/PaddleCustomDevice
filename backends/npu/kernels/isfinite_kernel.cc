// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
void IsinfKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("IsInf", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void IsnanKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("IsNan", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void IsfiniteKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("IsFinite", {x}, {*out}, {});
  runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(isinf,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::IsinfKernel,
                          phi::dtype::float16,
                          float,
                          double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(isnan,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::IsnanKernel,
                          phi::dtype::float16,
                          float,
                          double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(isfinite,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::IsfiniteKernel,
                          phi::dtype::float16,
                          float,
                          double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
