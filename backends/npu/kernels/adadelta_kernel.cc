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
void AdadeltaKernel(const Context& dev_ctx,
                    const phi::DenseTensor& param,
                    const phi::DenseTensor& grad,
                    const phi::DenseTensor& avg_squared_grad,
                    const phi::DenseTensor& avg_squared_update,
                    const phi::DenseTensor& learning_rate,
                    const paddle::optional<phi::DenseTensor>& master_param,
                    float rho,
                    float epsilon,
                    bool multi_precision,
                    phi::DenseTensor* param_out,
                    phi::DenseTensor* avg_squared_grad_out,
                    phi::DenseTensor* avg_squared_update_out,
                    phi::DenseTensor* master_param_outs) {
  PADDLE_ENFORCE_EQ(
      multi_precision,
      false,
      phi::errors::InvalidArgument(
          "Paddle Custom NPU does not support muilt precision for adadelta."));

  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(avg_squared_grad_out);
  dev_ctx.template Alloc<T>(avg_squared_update_out);

  auto stream = dev_ctx.stream();
  NpuOpRunner runner;
  runner.SetType("ApplyAdadelta")
      .AddInput(param)
      .AddInput(avg_squared_grad)
      .AddInput(avg_squared_update)
      .AddInput(learning_rate)
      .AddInput(dev_ctx, std::vector<float>({rho}))
      .AddInput(dev_ctx, std::vector<float>({epsilon}))
      .AddInput(grad)
      .AddOutput(*param_out)
      .AddAttr("use_locking", true)
      .Run(stream);
  dev_ctx.Wait();

  // NOTE(songkai05):CANN op ApplyAdadelta update param, avg_squared_grad,
  // avg_squared_update inplace, so we need to copy these inputs to
  // corresponding outputs after op execution.
  TensorCopy(dev_ctx, param, true, param_out);
  TensorCopy(dev_ctx, avg_squared_grad, true, avg_squared_grad_out);
  TensorCopy(dev_ctx, avg_squared_update, true, avg_squared_update_out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    adadelta, npu, ALL_LAYOUT, custom_kernel::AdadeltaKernel, float, double) {}
