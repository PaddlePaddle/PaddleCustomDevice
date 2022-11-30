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
#include "kernels/funcs/op_command.h"

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
  if (param_out != &param_var) {
    PADDLE_THROW(phi::errors::Unavailable(
        "SGDKernel is an inplace op, but the param_out "
        "and param_var is different."));
  }

  experimental::OpCommandHelper::MarkAsParameter(param_out);
  experimental::OpCommand("ApplyGradientDescent")
      .Input(param_var)
      .Input(learning_rate)
      .Input(grad_var)
      .Run(dev_ctx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(sgd,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SGDKernel,
                          phi::dtype::float16,
                          float,
                          double) {}
