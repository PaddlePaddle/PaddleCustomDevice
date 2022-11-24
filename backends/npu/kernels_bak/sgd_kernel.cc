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
void SGDKernel(const Context& dev_ctx,
               const phi::DenseTensor& param_var,
               const phi::DenseTensor& learning_rate,
               const phi::DenseTensor& grad_var,
               const paddle::optional<phi::DenseTensor>& master_param,
               bool multi_precision,
               phi::DenseTensor* param_out,
               phi::DenseTensor* master_param_out) {
  aclrtStream stream = static_cast<aclrtStream>(dev_ctx.stream());
  dev_ctx.template Alloc<T>(param_out);

  const auto& runner = NpuOpRunner("ApplyGradientDescent",
                                   {param_var, learning_rate, grad_var},
                                   {*param_out},
                                   {});
  runner.Run(stream);

  // NOTE(zhiqiu): ApplyGradientDescent updates params inplace, so
  // if param and param_out is not same, we need to do copy.
  if (param_out->data<T>() != param_var.data<T>()) {
    TensorCopy(dev_ctx, param_var, false, param_out);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(sgd,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SGDKernel,
                          phi::dtype::float16,
                          float,
                          double) {}
