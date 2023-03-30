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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {
enum class RegularizationType {
  kNONE = 0,
  kL1DECAY = 1,  // do not need support right now
  kL2DECAY = 2,
};

template <typename T, typename Context>
void MomentumKernel(const Context &dev_ctx,
                    const phi::DenseTensor &param,
                    const phi::DenseTensor &grad,
                    const phi::DenseTensor &velocity,
                    const phi::DenseTensor &learning_rate,
                    const paddle::optional<phi::DenseTensor> &master_param,
                    float mu_f,
                    bool use_nesterov,
                    const std::string &regularization_method,
                    float regularization_coeff,
                    bool multi_precision,
                    float rescale_grad,
                    phi::DenseTensor *param_out,
                    phi::DenseTensor *velocity_out,
                    phi::DenseTensor *master_param_out) {
  auto mu = static_cast<T>(mu_f);

  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<T>(velocity_out);

  phi::DenseTensor mu_tensor;
  mu_tensor.Resize({1});
  dev_ctx.template Alloc<T>(&mu_tensor);
  FillMLUTensorWithHostValue(dev_ctx, mu, &mu_tensor);

  phi::DenseTensor regularized_grad;
  MLUCnnlTensorDesc param_desc(param);
  if (regularization_method == "l2_decay") {
    regularized_grad.Resize(grad.dims());
    dev_ctx.template Alloc<T>(&regularized_grad);
    MLUCnnlOpTensorDesc op_tensor_desc(
        CNNL_OP_TENSOR_ADD, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);
    MLUCnnl::OpTensor(dev_ctx,
                      op_tensor_desc.get(),
                      param_desc.get(),
                      GetBasePtr(&param),
                      param_desc.get(),
                      GetBasePtr(&grad),
                      param_desc.get(),
                      GetBasePtr(&regularized_grad),
                      ToCnnlDataType<T>(),
                      regularization_coeff);
  } else {
    regularized_grad = grad;
  }
  TensorCopy(dev_ctx, param, false, param_out);
  TensorCopy(dev_ctx, velocity, false, velocity_out);
  MLUCnnl::ApplyMomentum(dev_ctx,
                         param_desc.get(),
                         GetBasePtr(&regularized_grad),
                         use_nesterov,
                         GetBasePtr(&learning_rate),
                         GetBasePtr(&mu_tensor),
                         GetBasePtr(param_out),
                         GetBasePtr(velocity_out));
}

template <typename T, typename Context>
void MergedMomentumKernel(
    const Context &dev_ctx,
    const std::vector<const phi::DenseTensor *> &param,
    const std::vector<const phi::DenseTensor *> &grad,
    const std::vector<const phi::DenseTensor *> &velocity,
    const std::vector<const phi::DenseTensor *> &learning_rate,
    const paddle::optional<std::vector<const phi::DenseTensor *>> &master_param,
    float mu,
    bool use_nesterov,
    const std::vector<std::string> &regularization_method,
    const std::vector<float> &regularization_coeff,
    bool multi_precision,
    float rescale_grad,
    std::vector<phi::DenseTensor *> param_out,
    std::vector<phi::DenseTensor *> velocity_out,
    std::vector<phi::DenseTensor *> master_param_out) {
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

  VLOG(5) << "use_nesterov: " << use_nesterov
          << ",  regularization_method.size(): " << regularization_method.size()
          << ",  regularization_coeff.size(): " << regularization_coeff.size();

  Tensor mu_tensor;
  mu_tensor.Resize({1});
  dev_ctx.template Alloc<T>(&mu_tensor);
  MLUCnnlTensorDesc mu_tensor_desc(mu_tensor);
  MLUCnnl::Fill(dev_ctx,
                CNNL_POINTER_MODE_HOST,
                &mu,
                mu_tensor_desc.get(),
                GetBasePtr(&mu_tensor));

  for (size_t idx = 0; idx < n; ++idx) {
    RegularizationType regularization_flag =
        regularization_method.size() > 0 &&
                regularization_method[idx] == "l2_decay"
            ? RegularizationType::kL2DECAY
            : RegularizationType::kNONE;
    T reg_coeff_ = static_cast<T>(0.0);
    if (regularization_coeff.size() != 0) {
      reg_coeff_ = static_cast<T>(regularization_coeff[idx]);
    }

    auto lr_ = learning_rate.size() > 1 ? learning_rate[idx] : learning_rate[0];
    auto param_out_ = param_out[idx];
    auto velocity_out_ = velocity_out[idx];

    auto grad_ = grad[idx];
    Tensor regularized_grad;
    MLUCnnlTensorDesc param_desc(*param_out_);
    const char *reg_method_ = regularization_method[idx].c_str();
    if (regularization_flag == RegularizationType::kL2DECAY) {
      regularized_grad.Resize(param_out_->dims());
      dev_ctx.template Alloc<T>(&regularized_grad);
      MLUCnnlOpTensorDesc op_tensor_desc(
          CNNL_OP_TENSOR_ADD, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);
      MLUCnnl::OpTensor(dev_ctx,
                        op_tensor_desc.get(),
                        param_desc.get(),
                        GetBasePtr(param_out_),
                        param_desc.get(),
                        GetBasePtr(grad_),
                        param_desc.get(),
                        GetBasePtr(&regularized_grad),
                        ToCnnlDataType<T>(),
                        reg_coeff_);
    } else {
      regularized_grad = *grad_;
    }
    MLUCnnl::ApplyMomentum(dev_ctx,
                           param_desc.get(),
                           GetBasePtr(&regularized_grad),
                           use_nesterov,
                           GetBasePtr(lr_),
                           GetBasePtr(&mu_tensor),
                           GetBasePtr(param_out_),
                           GetBasePtr(velocity_out_));
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(momentum,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::MomentumKernel,
                          phi::dtype::float16,
                          float) {}
PD_REGISTER_PLUGIN_KERNEL(merged_momentum,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::MergedMomentumKernel,
                          float,
                          phi::dtype::float16) {}
