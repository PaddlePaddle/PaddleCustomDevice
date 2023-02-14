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
  const auto* x_data = x.data<T>();
  auto* out_data = dev_ctx.template Alloc<T>(out);

  const auto& dims = x.dims();
  const auto H = dims[dims.size() - 2];
  const auto W = dims[dims.size() - 1];

  std::string op_type = lower ? "Tril" : "Triu";
  NPUAttributeMap attr_input = {{"diagonal", diagonal}};

  auto op_func_tril = [](const std::vector<phi::DenseTensor>& inputs,
                         const std::vector<phi::DenseTensor>& outputs,
                         const NPUAttributeMap& attrs,
                         const Context& dev_ctx) {
    const auto& runner = NpuOpRunner("Tril", inputs, outputs, attrs);
    runner.Run(dev_ctx.stream());
  };

  auto op_func_triu = [](const std::vector<phi::DenseTensor>& inputs,
                         const std::vector<phi::DenseTensor>& outputs,
                         const NPUAttributeMap& attrs,
                         const Context& dev_ctx) {
    const auto& runner = NpuOpRunner("Triu", inputs, outputs, attrs);
    runner.Run(dev_ctx.stream());
  };
  if (x.dtype() == phi::DenseTensorMeta::DataType::BOOL) {
    if (lower) {
      NpuOpRunner::TypeAdapter({x},
                               {*out},
                               attr_input,
                               dev_ctx,
                               op_func_tril,
                               {phi::DenseTensorMeta::DataType::UINT8},
                               {phi::DenseTensorMeta::DataType::UINT8});
    } else {
      NpuOpRunner::TypeAdapter({x},
                               {*out},
                               attr_input,
                               dev_ctx,
                               op_func_triu,
                               {phi::DenseTensorMeta::DataType::UINT8},
                               {phi::DenseTensorMeta::DataType::UINT8});
    }
  } else {
    const auto& runner = NpuOpRunner(op_type, {x}, {*out}, attr_input);
    runner.Run(dev_ctx.stream());
  }
}

template <typename T, typename Context>
void TrilKernel(const Context& ctx,
                const phi::DenseTensor& x,
                int diagonal,
                phi::DenseTensor* out) {
  custom_kernel::TrilTriuKernel<T, Context>(ctx, x, diagonal, true, out);
}

template <typename T, typename Context>
void TriuKernel(const Context& ctx,
                const phi::DenseTensor& x,
                int diagonal,
                phi::DenseTensor* out) {
  custom_kernel::TrilTriuKernel<T, Context>(ctx, x, diagonal, false, out);
}

template <typename T, typename Context>
void TrilTriuGradKernel(const Context& ctx,
                        const phi::DenseTensor& out_grad,
                        int diagonal,
                        bool lower,
                        phi::DenseTensor* x_grad) {
  custom_kernel::TrilTriuKernel<T, Context>(
      ctx, out_grad, diagonal, lower, x_grad);
}

template <typename T, typename Context>
void TrilGradKernel(const Context& ctx,
                    const phi::DenseTensor& out_grad,
                    int diagonal,
                    phi::DenseTensor* x_grad) {
  custom_kernel::TrilTriuGradKernel<T, Context>(
      ctx, out_grad, diagonal, true, x_grad);
}

template <typename T, typename Context>
void TriuGradKernel(const Context& ctx,
                    const phi::DenseTensor& out_grad,
                    int diagonal,
                    phi::DenseTensor* x_grad) {
  custom_kernel::TrilTriuGradKernel<T, Context>(
      ctx, out_grad, diagonal, false, x_grad);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(tril_triu,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::TrilTriuKernel,
                          bool,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tril,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::TrilKernel,
                          bool,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(triu,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::TriuKernel,
                          bool,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tril_triu_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::TrilTriuGradKernel,
                          bool,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tril_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::TrilGradKernel,
                          bool,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(triu_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::TriuGradKernel,
                          bool,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16) {}
