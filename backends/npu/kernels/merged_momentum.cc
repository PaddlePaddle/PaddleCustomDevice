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
void FullLikeKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::Scalar& val,
                    phi::DataType dtype,
                    phi::DenseTensor* out);

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
  phi::DenseTensor host_mu;
  experimental::OpCommandHelper::ScalarToHostTensor(dev_ctx, mu, &host_mu);

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
      experimental::OpCommand("Axpy")
          .Input(*grad_data)
          .Input(*param_data)
          .Output(regularized_grad)
          .Attr("alpha", regularization_coeff_data)
          .Run(dev_ctx);
    } else {
      experimental::OpCommandHelper::Assign(
          dev_ctx, *grad_data, &regularized_grad);
    }
    experimental::OpCommandHelper::Assign(dev_ctx, *param_data, param_out_data);
    experimental::OpCommandHelper::Assign(
        dev_ctx, *velocity_data, velocity_out_data);

    phi::DenseTensor tmp_out;
    tmp_out.Resize(param_out_data->dims());
    dev_ctx.template Alloc<T>(&tmp_out);
    experimental::OpCommandHelper::MarkAsParameter(param_out_data);
    experimental::OpCommandHelper::MarkAsParameter(velocity_out_data);
    experimental::OpCommand("ApplyMomentum")
        .Input(*param_out_data,
               experimental::TensorDescMaker("var", *param_out_data)
                   .SetDataLayout(phi::DataLayout::ANY))
        .Input(*velocity_out_data,
               experimental::TensorDescMaker("accum", *velocity_out_data)
                   .SetDataLayout(phi::DataLayout::ANY))
        .Input(*lr_data)
        .Input(regularized_grad)
        .ScalarInput(host_mu)
        .Attr("use_nesterov", use_nesterov)
        .Output(tmp_out)
        .Run(dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(merged_momentum,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MergedMomentumKernel,
                          float,
                          phi::dtype::float16) {}
