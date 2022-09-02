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

enum class RegularizationType {
  kNONE = 0,
  kL1DECAY = 1,  // do not need support right now
  kL2DECAY = 2,
};

void CheckInputs(
    const std::vector<const phi::DenseTensor*>& param,
    const std::vector<const phi::DenseTensor*>& grad,
    const std::vector<const phi::DenseTensor*>& velocity,
    const std::vector<const phi::DenseTensor*>& learning_rate,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& master_param,
    float mu,
    bool use_nesterov,
    const std::vector<std::string>& regularization_method,
    const std::vector<float>& regularization_coeff,
    bool multi_precision,
    float rescale_grad,
    std::vector<phi::DenseTensor*> param_out,
    std::vector<phi::DenseTensor*> velocity_out,
    std::vector<phi::DenseTensor*> master_param_out) {
  size_t n = param.size();
  PADDLE_ENFORCE_EQ(n,
                    param_out.size(),
                    phi::errors::InvalidArgument(
                        "The size of Output(ParamOut) must be equal to "
                        "Input(Param), but got the size of Output(ParamOut) "
                        "is %d, the size of Input(Param) is %d.",
                        param_out.size(),
                        n));
  for (size_t i = 0; i < n; ++i) {
    PADDLE_ENFORCE_EQ(param[i],
                      param_out[i],
                      phi::errors::InvalidArgument(
                          "The size of Input(Param) and Output(ParamOut) "
                          "must be the same Tensors."));
  }
  PADDLE_ENFORCE_EQ(
      n,
      grad.size(),
      phi::errors::InvalidArgument(
          "The size of Input(Grad) must be equal to Input(Param), but got "
          "the size of Input(Grad) is %d, the size of Input(Param) is %d.",
          grad.size(),
          n));
  PADDLE_ENFORCE_EQ(n,
                    velocity.size(),
                    phi::errors::InvalidArgument(
                        "The size of Input(Velocity) must be equal to "
                        "Input(Param), but got the size of Input(Velocity) "
                        "is %d, the size of Input(Param) is %d.",
                        velocity.size(),
                        n));
  PADDLE_ENFORCE_EQ(
      n,
      velocity_out.size(),
      phi::errors::InvalidArgument(
          "The size of Output(VelocityOut) must be "
          "equal to Input(Param), but got the size of Output(VelocityOut) is "
          "%d, the size of Input(Param) is %d.",
          velocity_out.size(),
          n));
  for (size_t i = 0; i < n; ++i) {
    PADDLE_ENFORCE_EQ(velocity[i],
                      velocity_out[i],
                      phi::errors::InvalidArgument(
                          "Input(Velocity) and Output(VelocityOut) must be "
                          "the same Tensors."));
  }
  if (learning_rate.size() != 1) {
    PADDLE_ENFORCE_EQ(
        n,
        learning_rate.size(),
        phi::errors::InvalidArgument(
            "If the size of Input(LearningRate) is not 1, the size of "
            "Input(LearningRate) must be "
            "equal to Input(Param), but got the size of Input(LearningRate) "
            "is %d, the size of Input(Param) is %d.",
            learning_rate.size(),
            n));
  }
  if (regularization_method.size() != 0) {
    PADDLE_ENFORCE_EQ(
        n,
        regularization_method.size(),
        phi::errors::InvalidArgument(
            "The size of Attr(regularization_method) must be equal "
            "to Input(Param), but got the size of "
            "Attr(regularization_method) is %d, the size of Input(Param) is "
            "%d.",
            regularization_method.size(),
            n));
    PADDLE_ENFORCE_EQ(
        n,
        regularization_coeff.size(),
        phi::errors::InvalidArgument(
            "The size of Attr(regularization_coeff) must be equal "
            "to Input(Param), but got the size of Attr(regularization_coeff) "
            "is %d, the size of Input(Param) is %d.",
            regularization_coeff.size(),
            n));
  }
}
template <typename T, typename Context>
void MergedMomentumKernel(
    const Context& dev_ctx,
    const std::vector<const phi::DenseTensor*>& param,
    const std::vector<const phi::DenseTensor*>& grad,
    const std::vector<const phi::DenseTensor*>& velocity,
    const std::vector<const phi::DenseTensor*>& learning_rate,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& master_param,
    float mu,
    bool use_nesterov,
    const std::vector<std::string>& regularization_method,
    const std::vector<float>& regularization_coeff,
    bool multi_precision,
    float rescale_grad,
    std::vector<phi::DenseTensor*> param_out,
    std::vector<phi::DenseTensor*> velocity_out,
    std::vector<phi::DenseTensor*> master_param_out) {
  CheckInputs(param,
              grad,
              velocity,
              learning_rate,
              master_param,
              mu,
              use_nesterov,
              regularization_method,
              regularization_coeff,
              multi_precision,
              rescale_grad,
              param_out,
              velocity_out,
              master_param_out);
  phi::DenseTensor mu_tensor;
  mu_tensor.Resize(phi::make_ddim({1}));
  dev_ctx.template Alloc<T>(&mu_tensor);
  FillNpuTensorWithConstant<T>(&mu_tensor, dev_ctx, static_cast<T>(mu));

  size_t n = param.size();
  for (size_t idx = 0; idx < n; ++idx) {
    RegularizationType regularization_flag =
        regularization_method.size() > 0 &&
                regularization_method[idx] == "l2_decay"
            ? RegularizationType::kL2DECAY
            : RegularizationType::kNONE;
    float regularization_coeff_data = 0.0;
    if (regularization_coeff.size() != 0) {
      regularization_coeff_data = regularization_coeff[idx];
    }

    auto lr_data =
        learning_rate.size() > 1 ? learning_rate[idx] : learning_rate[0];
    auto param_data = param[idx];
    auto param_out_data = param_out[idx];
    auto velocity_data = velocity[idx];
    auto velocity_out_data = velocity_out[idx];

    auto grad_data = grad[idx];
    phi::DenseTensor regularized_grad;
    if (regularization_flag == RegularizationType::kL2DECAY) {
      regularized_grad.Resize(grad_data->dims());
      dev_ctx.template Alloc<T>(&regularized_grad);
      const auto& runner1 = NpuOpRunner("Muls",
                                        {*param_data},
                                        {regularized_grad},
                                        {{"value", regularization_coeff_data}});
      runner1.Run(dev_ctx.stream());
      const auto& runner2 = NpuOpRunner(
          "Add", {regularized_grad, *grad_data}, {regularized_grad}, {});
      runner2.Run(dev_ctx.stream());
    } else {
      regularized_grad = *grad_data;
      // regularized_grad.ShareDataWith(*grad_data);
    }
    TensorCopy(dev_ctx, *param_data, false, param_out_data);
    TensorCopy(dev_ctx, *velocity_data, false, velocity_out_data);
    // NOTE: ApplyMomentum will change the input
    const auto& runner = NpuOpRunner("ApplyMomentum",
                                     {*param_out_data,
                                      *velocity_out_data,
                                      *lr_data,
                                      regularized_grad,
                                      mu_tensor},
                                     {*param_out_data},
                                     {{"use_nesterov", use_nesterov}});
    runner.Run(dev_ctx.stream());
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(merged_momentum,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::MergedMomentumKernel,
                          float,
                          phi::dtype::float16) {}
