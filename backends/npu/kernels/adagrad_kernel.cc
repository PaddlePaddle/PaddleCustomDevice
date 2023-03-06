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
void AdagradDenseKernel(const Context& dev_ctx,
                        const phi::DenseTensor& param,
                        const phi::DenseTensor& grad,
                        const phi::DenseTensor& moment,
                        const phi::DenseTensor& learning_rate,
                        const paddle::optional<phi::DenseTensor>& master_param,
                        float epsilon,
                        bool multi_precision,
                        phi::DenseTensor* param_out,
                        phi::DenseTensor* moment_out,
                        phi::DenseTensor* master_param_outs) {
  // TODO(qili93): add fp16 support based on Paddle PR#50078
  PADDLE_ENFORCE_EQ(
      multi_precision,
      false,
      phi::errors::InvalidArgument("Paddle Custom NPU only support "
                                   "multi_precision = false, but got = <%d>",
                                   multi_precision));
  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(moment_out);
  auto stream = dev_ctx.stream();

  std::vector<T> epsilon_vec(1, static_cast<T>(epsilon));
  NpuOpRunner runner;
  runner.SetType("ApplyAdagradV2")
      .AddInput(param)
      .AddInput(moment)
      .AddInput(learning_rate)
      .AddInput(dev_ctx, std::move(epsilon_vec))
      .AddInput(grad)
      .AddOutput(*param_out)
      .AddAttr("update_slots", true)
      .AddAttr("use_locking", true)
      .Run(stream);
  dev_ctx.Wait();
  // NOTE(songkai05): param_out can't get right value by call ApplyAdagradV2,
  // but the first input, param can be updated correctly, so need to copy it to
  // param_out.
  TensorCopy(dev_ctx, param, true, param_out);
  // NOTE(songkai05): CANN op ApplyAdagradV2 does not return moment_out as an
  // output, but updates moment directly, so we copy moment to moment_out.
  TensorCopy(dev_ctx, moment, true, moment_out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    adagrad, npu, ALL_LAYOUT, custom_kernel::AdagradDenseKernel, float) {}
