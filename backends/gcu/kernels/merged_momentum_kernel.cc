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

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {
static void CheckInputs(
    const std::vector<const phi::DenseTensor*>& param,
    const std::vector<const phi::DenseTensor*>& grad,
    const std::vector<const phi::DenseTensor*>& velocity,
    const std::vector<const phi::DenseTensor*>& learning_rate,
    const std::vector<std::string>& regularization_method,
    const std::vector<float>& regularization_coeff,
    std::vector<phi::DenseTensor*> param_out,
    std::vector<phi::DenseTensor*> velocity_out) {
  size_t param_num = param.size();
  PADDLE_ENFORCE_GT(param_num, 0);
  PADDLE_ENFORCE_EQ(
      param_num,
      grad.size(),
      phi::errors::InvalidArgument(
          "The size of Input(Grad) must be equal to Input(Param), but got "
          "the size of Input(Grad) is %zu, the size of Input(Param) is %zu.",
          grad.size(),
          param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    velocity.size(),
                    phi::errors::InvalidArgument(
                        "The size of Input(Velocity) must be equal to "
                        "Input(Param), but got the size of Input(Velocity) "
                        "is %zu, the size of Input(Param) is %zu.",
                        velocity.size(),
                        param_num));
  if (learning_rate.size() != 1) {
    PADDLE_ENFORCE_EQ(
        param_num,
        learning_rate.size(),
        phi::errors::InvalidArgument(
            "If the size of Input(LearningRate) is not 1, the size of "
            "Input(LearningRate) must be "
            "equal to Input(Param), but got the size of Input(LearningRate) "
            "is %zu, the size of Input(Param) is %zu.",
            learning_rate.size(),
            param_num));
  }
  if (regularization_method.size() != 0) {
    PADDLE_ENFORCE_EQ(
        param_num,
        regularization_method.size(),
        phi::errors::InvalidArgument(
            "The size of Attr(regularization_method) must be equal "
            "to Input(Param), but got the size of "
            "Attr(regularization_method) is %zu, the size of Input(Param) is "
            "%zu.",
            regularization_method.size(),
            param_num));
    PADDLE_ENFORCE_EQ(
        param_num,
        regularization_coeff.size(),
        phi::errors::InvalidArgument(
            "The size of Attr(regularization_coeff) must be equal "
            "to Input(Param), but got the size of Attr(regularization_coeff) "
            "is %zu, the size of Input(Param) is %zu.",
            regularization_coeff.size(),
            param_num));
  }
  PADDLE_ENFORCE_EQ(param_num,
                    param_out.size(),
                    phi::errors::InvalidArgument(
                        "The size of Output(ParamOut) must be equal to "
                        "Input(Param), but got the size of Output(ParamOut) "
                        "is %zu, the size of Input(Param) is %zu.",
                        param_out.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(
      param_num,
      velocity_out.size(),
      phi::errors::InvalidArgument(
          "The size of Output(VelocityOut) must be "
          "equal to Input(Param), but got the size of Output(VelocityOut) is "
          "%zu, the size of Input(Param) is %zu.",
          velocity_out.size(),
          param_num));
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
  PADDLE_GCU_KERNEL_TRACE("merged_momentum");
  CheckInputs(param,
              grad,
              velocity,
              learning_rate,
              regularization_method,
              regularization_coeff,
              param_out,
              velocity_out);

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    size_t param_num = param.size();
    TensorNameMap input_names;
    TensorValueMap inputs;
    TensorNameMap output_names;
    TensorValueMap outputs;
    input_names["Param"].reserve(param_num);
    input_names["Grad"].reserve(param_num);
    input_names["Velocity"].reserve(param_num);
    input_names["LearningRate"].reserve(param_num);
    inputs["Param"].reserve(param_num);
    inputs["Grad"].reserve(param_num);
    inputs["Velocity"].reserve(param_num);
    inputs["LearningRate"].reserve(param_num);
    output_names["VelocityOut"].reserve(param_num);
    output_names["ParamOut"].reserve(param_num);
    outputs["VelocityOut"].reserve(param_num);
    outputs["ParamOut"].reserve(param_num);
    std::vector<std::shared_ptr<phi::DenseTensor>> param_outs_tmp;
    std::vector<std::shared_ptr<phi::DenseTensor>> velocity_outs_tmp;
    param_outs_tmp.reserve(param_num);
    velocity_outs_tmp.reserve(param_num);

    for (size_t i = 0; i < param_num; ++i) {
      input_names["Param"].emplace_back(std::string("param") +
                                        std::to_string(i));
      input_names["Grad"].emplace_back(std::string("grad") + std::to_string(i));
      input_names["Velocity"].emplace_back(std::string("velocity") +
                                           std::to_string(i));
      input_names["LearningRate"].emplace_back(std::string("learning_rate") +
                                               std::to_string(i));

      inputs["Param"].emplace_back(const_cast<DenseTensor*>(param[i]));
      inputs["Grad"].emplace_back(const_cast<DenseTensor*>(grad[i]));
      inputs["Velocity"].emplace_back(const_cast<DenseTensor*>(velocity[i]));
      size_t lr_idx = (learning_rate.size() == 1) ? 0 : i;
      inputs["LearningRate"].emplace_back(
          const_cast<DenseTensor*>(learning_rate[lr_idx]));

      output_names["ParamOut"].emplace_back(std::string("param_out") +
                                            std::to_string(i));
      output_names["VelocityOut"].emplace_back(std::string("velocity_out") +
                                               std::to_string(i));

      auto param_out_tmp = std::make_shared<phi::DenseTensor>();
      param_out_tmp->set_meta(param_out[i]->meta());
      dev_ctx.template Alloc<T>(param_out_tmp.get());
      param_outs_tmp.emplace_back(param_out_tmp);

      auto velocity_out_tmp = std::make_shared<phi::DenseTensor>();
      velocity_out_tmp->set_meta(velocity_out[i]->meta());
      dev_ctx.template Alloc<T>(velocity_out_tmp.get());
      velocity_outs_tmp.emplace_back(velocity_out_tmp);

      outputs["ParamOut"].emplace_back(param_out_tmp.get());
      outputs["VelocityOut"].emplace_back(velocity_out_tmp.get());
    }

    auto regularization_methods =
        (regularization_method.size() == 1)
            ? std::vector<std::string>(param_num, regularization_method[0])
            : regularization_method;
    auto regularization_coeffs =
        (regularization_coeff.size() == 1)
            ? std::vector<float>(param_num, regularization_coeff[0])
            : regularization_coeff;

    GcuAttributeMap attrs;
    attrs["mu"] = mu;
    attrs["use_nesterov"] = use_nesterov;
    attrs["regularization_method"] = regularization_methods;
    attrs["regularization_coeff"] = regularization_coeffs;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "merged_momentum",
              dev_ctx);

    for (size_t i = 0; i < param_num; ++i) {
      dev_ctx.template Alloc<T>(param_out[i]);
      dev_ctx.template Alloc<T>(velocity_out[i]);

      TensorCopy(dev_ctx, *(param_outs_tmp[i]), false, param_out[i]);
      TensorCopy(dev_ctx, *(velocity_outs_tmp[i]), false, velocity_out[i]);

      // if param and param_out is not same, we need to do copy.
      if (param_out[i]->data<T>() != param[i]->data<T>()) {
        TensorCopy(dev_ctx,
                   *(param_out[i]),
                   false,
                   const_cast<DenseTensor*>(param[i]));
      }
      if (velocity_out[i]->data<T>() != velocity[i]->data<T>()) {
        TensorCopy(dev_ctx,
                   *(velocity_out[i]),
                   false,
                   const_cast<DenseTensor*>(velocity[i]));
      }
    }
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(merged_momentum,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MergedMomentumKernel,
                          float,
                          phi::dtype::float16) {}
