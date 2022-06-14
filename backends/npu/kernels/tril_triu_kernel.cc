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
void TrilTriuKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    int diagonal,
                    bool lower,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  std::string op_type = lower ? "Tril" : "Triu";

  NPUAttributeMap attr_input = {{"diagonal", diagonal}};

  auto op_func_tril = [](const std::vector<phi::DenseTensor>& inputs,
                         const std::vector<phi::DenseTensor>& outputs,
                         const NPUAttributeMap& attrs,
                         const phi::CustomContext& dev_ctx) {
    const auto& runner = NpuOpRunner("Tril", inputs, outputs, attrs);
    runner.Run(dev_ctx.stream());
  };

  auto op_func_triu = [](const std::vector<phi::DenseTensor>& inputs,
                         const std::vector<phi::DenseTensor>& outputs,
                         const NPUAttributeMap& attrs,
                         const phi::CustomContext& dev_ctx) {
    const auto& runner = NpuOpRunner("Triu", inputs, outputs, attrs);
    runner.Run(dev_ctx.stream());
  };

  if (x.dtype() == paddle::experimental::DataType::BOOL) {
    if (lower) {
      NpuOpRunner::TypeAdapter({x},
                               {*out},
                               attr_input,
                               dev_ctx,
                               op_func_tril,
                               {paddle::experimental::DataType::UINT8},
                               {paddle::experimental::DataType::UINT8});
    } else {
      NpuOpRunner::TypeAdapter({x},
                               {*out},
                               attr_input,
                               dev_ctx,
                               op_func_triu,
                               {paddle::experimental::DataType::UINT8},
                               {paddle::experimental::DataType::UINT8});
    }
  } else {
    const auto& runner = NpuOpRunner(op_type, {x}, {*out}, attr_input);
    runner.Run(dev_ctx.stream());
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(tril_triu,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::TrilTriuKernel,
                          int,
                          bool,
                          float,
                          phi::dtype::float16,
                          double) {}
