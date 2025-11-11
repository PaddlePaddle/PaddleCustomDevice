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
void RmspropDenseKernel(const Context& dev_ctx,
                        const phi::DenseTensor& param,
                        const phi::DenseTensor& mean_square,
                        const phi::DenseTensor& grad,
                        const phi::DenseTensor& moment,
                        const phi::DenseTensor& learning_rate,
                        const paddle::optional<phi::DenseTensor>& mean_grad,
                        const paddle::optional<phi::DenseTensor>& master_param,
                        float epsilon,
                        float decay,
                        float momentum,
                        bool centered,
                        bool multi_precision,
                        phi::DenseTensor* param_out,
                        phi::DenseTensor* moment_out,
                        phi::DenseTensor* mean_square_out,
                        phi::DenseTensor* mean_grad_out,
                        phi::DenseTensor* master_param_outs) {
  // TODO(qili93): add fp16 support based on Paddle PR#50132
  PADDLE_ENFORCE_EQ(
      multi_precision,
      false,
      phi::errors::InvalidArgument("Paddle Custom NPU only support "
                                   "multi_precision = false, but got = <%d>",
                                   multi_precision));
  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(moment_out);
  dev_ctx.template Alloc<T>(mean_square_out);

  auto stream = dev_ctx.stream();

  if (centered) {
    NPUAttributeMap attr_input = {{"use_locking", false}};

    phi::DenseTensorMeta tmp_meta = {phi::DataType::FLOAT32, {1}};

    phi::DenseTensor rho_tmp;
    rho_tmp.set_meta(tmp_meta);
    dev_ctx.template Alloc<T>(&rho_tmp);
    FillNpuTensorWithConstant<T>(&rho_tmp, dev_ctx, decay);

    phi::DenseTensor momentum_tmp;
    momentum_tmp.set_meta(tmp_meta);
    dev_ctx.template Alloc<T>(&momentum_tmp);
    FillNpuTensorWithConstant<T>(&momentum_tmp, dev_ctx, momentum);

    phi::DenseTensor epsilon_tmp;
    epsilon_tmp.set_meta(tmp_meta);
    dev_ctx.template Alloc<T>(&epsilon_tmp);
    FillNpuTensorWithConstant<T>(&epsilon_tmp, dev_ctx, epsilon);

    dev_ctx.template Alloc<T>(mean_grad_out);
    const auto& runner_applycenterrmsprop =
        NpuOpRunner(std::string("ApplyCenteredRMSPropD"),
                    {param,
                     *mean_grad,
                     mean_square,
                     moment,
                     learning_rate,
                     rho_tmp,
                     momentum_tmp,
                     epsilon_tmp,
                     grad},
                    {*param_out, *mean_grad_out, *mean_square_out, *moment_out},
                    {attr_input});
    runner_applycenterrmsprop.Run(stream);
  } else {
    NPUAttributeMap attr_input = {
        {"rho", decay}, {"momentum", momentum}, {"epsilon", epsilon}};
    const auto& runner_applyrmsprop =
        NpuOpRunner(std::string("ApplyRMSPropD"),
                    {param, mean_square, moment, learning_rate, grad},
                    {*param_out, *mean_square_out, *moment_out},
                    {attr_input});
    runner_applyrmsprop.Run(stream);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    rmsprop, npu, ALL_LAYOUT, custom_kernel::RmspropDenseKernel, float) {}
