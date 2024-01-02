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

#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {
static void CheckInputs(
    const std::vector<const phi::DenseTensor*>& param,
    const std::vector<const phi::DenseTensor*>& grad,
    const std::vector<const phi::DenseTensor*>& learning_rate,
    const std::vector<const phi::DenseTensor*>& moment1,
    const std::vector<const phi::DenseTensor*>& moment2,
    const std::vector<const phi::DenseTensor*>& beta1_pow,
    const std::vector<const phi::DenseTensor*>& beta2_pow,
    std::vector<phi::DenseTensor*> param_out,
    std::vector<phi::DenseTensor*> moment1_out,
    std::vector<phi::DenseTensor*> moment2_out,
    std::vector<phi::DenseTensor*> beta1_pow_out,
    std::vector<phi::DenseTensor*> beta2_pow_out) {
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
  PADDLE_ENFORCE_EQ(param_num,
                    moment1.size(),
                    phi::errors::InvalidArgument(
                        "The size of Input(Moment1) must be equal to "
                        "Input(Param), but got the size of Input(Moment1) "
                        "is %zu, the size of Input(Param) is %zu.",
                        moment1.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    moment2.size(),
                    phi::errors::InvalidArgument(
                        "The size of Input(Moment2) must be equal to "
                        "Input(Param), but got the size of Input(Moment2) "
                        "is %zu, the size of Input(Param) is %zu.",
                        moment2.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    beta1_pow.size(),
                    phi::errors::InvalidArgument(
                        "The size of Input(Beta1Pow) must be equal to "
                        "Input(Param), but got the size of Input(Beta1Pow) "
                        "is %zu, the size of Input(Param) is %zu.",
                        beta1_pow.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    beta2_pow.size(),
                    phi::errors::InvalidArgument(
                        "The size of Input(Beta2Pow) must be equal to "
                        "Input(Param), but got the size of Input(Beta2Pow) "
                        "is %zu, the size of Input(Param) is %zu.",
                        beta2_pow.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    param_out.size(),
                    phi::errors::InvalidArgument(
                        "The size of Output(ParamOut) must be equal to "
                        "Input(Param), but got the size of Output(ParamOut) "
                        "is %zu, the size of Input(Param) is %zu.",
                        param_out.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    moment1_out.size(),
                    phi::errors::InvalidArgument(
                        "The size of Output(Moment1Out) must be equal to "
                        "Input(Param), but got the size of Output(Moment1Out) "
                        "is %zu, the size of Input(Param) is %zu.",
                        moment1_out.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    moment2_out.size(),
                    phi::errors::InvalidArgument(
                        "The size of Output(Moment2Out) must be equal to "
                        "Input(Param), but got the size of Output(Moment2Out) "
                        "is %zu, the size of Input(Param) is %zu.",
                        moment2_out.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    beta1_pow.size(),
                    phi::errors::InvalidArgument(
                        "The size of Output(Beta1PowOut) must be equal to "
                        "Input(Param), but got the size of Output(Beta1PowOut) "
                        "is %zu, the size of Input(Param) is %zu.",
                        beta1_pow_out.size(),
                        param_num));
  PADDLE_ENFORCE_EQ(param_num,
                    beta2_pow.size(),
                    phi::errors::InvalidArgument(
                        "The size of Output(Beta2PowOut) must be equal to "
                        "Input(Param), but got the size of Output(Beta2PowOut) "
                        "is %zu, the size of Input(Param) is %zu.",
                        beta2_pow_out.size(),
                        param_num));
}

template <typename T, typename Context>
void MergedAdamKernel(
    const Context& dev_ctx,
    const std::vector<const phi::DenseTensor*>& param,
    const std::vector<const phi::DenseTensor*>& grad,
    const std::vector<const phi::DenseTensor*>& learning_rate,
    const std::vector<const phi::DenseTensor*>& moment1,
    const std::vector<const phi::DenseTensor*>& moment2,
    const std::vector<const phi::DenseTensor*>& beta1_pow,
    const std::vector<const phi::DenseTensor*>& beta2_pow,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& master_param,
    const phi::Scalar& beta1,
    const phi::Scalar& beta2,
    const phi::Scalar& epsilon,
    bool multi_precision,
    bool use_global_beta_pow,
    std::vector<phi::DenseTensor*> param_out,
    std::vector<phi::DenseTensor*> moment1_out,
    std::vector<phi::DenseTensor*> moment2_out,
    std::vector<phi::DenseTensor*> beta1_pow_out,
    std::vector<phi::DenseTensor*> beta2_pow_out,
    std::vector<phi::DenseTensor*> master_param_out) {
  CheckInputs(param,
              grad,
              learning_rate,
              moment1,
              moment2,
              beta1_pow,
              beta2_pow,
              param_out,
              moment1_out,
              moment2_out,
              beta1_pow_out,
              beta2_pow_out);

  size_t param_num = param.size();

  // beta1_pow and beta2_pow may on CPU and not transform place.
  std::vector<std::shared_ptr<phi::DenseTensor>> beta1_pow_gcu;
  if (beta1_pow[0]->place().GetType() == phi::AllocationType::CPU) {
    for (size_t i = 0; i < param_num; ++i) {
      auto beta1_pow_tmp = std::make_shared<phi::DenseTensor>();
      T beta1 = *(beta1_pow[i]->data<T>());
      beta1_pow_tmp->Resize({1});
      dev_ctx.template Alloc<T>(beta1_pow_tmp.get());
      FillGcuTensorWithConstant<T>(beta1_pow_tmp.get(), dev_ctx, beta1);
      beta1_pow_gcu.emplace_back(beta1_pow_tmp);
    }
  }

  std::vector<std::shared_ptr<phi::DenseTensor>> beta2_pow_gcu;
  if (beta2_pow[0]->place().GetType() == phi::AllocationType::CPU) {
    for (size_t i = 0; i < param_num; ++i) {
      auto beta2_pow_tmp = std::make_shared<phi::DenseTensor>();
      T beta2 = *(beta2_pow[i]->data<T>());
      beta2_pow_tmp->Resize({1});
      dev_ctx.template Alloc<T>(beta2_pow_tmp.get());
      FillGcuTensorWithConstant<T>(beta2_pow_tmp.get(), dev_ctx, beta2);
      beta2_pow_gcu.emplace_back(beta2_pow_tmp);
    }
  }

  TensorNameMap input_names;
  TensorValueMap inputs;
  TensorNameMap output_names;
  TensorValueMap outputs;
  input_names["Param"].reserve(param_num);
  input_names["Grad"].reserve(param_num);
  input_names["LearningRate"].reserve(param_num);
  input_names["Moment1"].reserve(param_num);
  input_names["Moment2"].reserve(param_num);
  input_names["Beta1Pow"].reserve(param_num);
  input_names["Beta2Pow"].reserve(param_num);
  inputs["Param"].reserve(param_num);
  inputs["Grad"].reserve(param_num);
  inputs["LearningRate"].reserve(param_num);
  inputs["Moment1"].reserve(param_num);
  inputs["Moment2"].reserve(param_num);
  inputs["Beta1Pow"].reserve(param_num);
  inputs["Beta2Pow"].reserve(param_num);

  output_names["ParamOut"].reserve(param_num);
  output_names["Moment1Out"].reserve(param_num);
  output_names["Moment2Out"].reserve(param_num);
  output_names["Beta1PowOut"].reserve(param_num);
  output_names["Beta2PowOut"].reserve(param_num);
  outputs["ParamOut"].reserve(param_num);
  outputs["Moment1Out"].reserve(param_num);
  outputs["Moment2Out"].reserve(param_num);
  outputs["Beta1PowOut"].reserve(param_num);
  outputs["Beta2PowOut"].reserve(param_num);

  std::vector<std::shared_ptr<phi::DenseTensor>> param_outs_tmp;
  std::vector<std::shared_ptr<phi::DenseTensor>> moment1_outs_tmp;
  std::vector<std::shared_ptr<phi::DenseTensor>> moment2_outs_tmp;
  std::vector<std::shared_ptr<phi::DenseTensor>> beta1_pow_outs_tmp;
  std::vector<std::shared_ptr<phi::DenseTensor>> beta2_pow_outs_tmp;
  param_outs_tmp.reserve(param_num);
  moment1_outs_tmp.reserve(param_num);
  moment2_outs_tmp.reserve(param_num);
  beta1_pow_outs_tmp.reserve(param_num);
  beta2_pow_outs_tmp.reserve(param_num);

  for (size_t i = 0; i < param_num; ++i) {
    input_names["Param"].emplace_back(std::string("param") + std::to_string(i));
    input_names["Grad"].emplace_back(std::string("grad") + std::to_string(i));
    input_names["LearningRate"].emplace_back(std::string("learning_rate") +
                                             std::to_string(i));
    input_names["Moment1"].emplace_back(std::string("moment1_in") +
                                        std::to_string(i));
    input_names["Moment2"].emplace_back(std::string("moment2_in") +
                                        std::to_string(i));
    input_names["Beta1Pow"].emplace_back(std::string("beta1_pow") +
                                         std::to_string(i));
    input_names["Beta2Pow"].emplace_back(std::string("beta2_pow") +
                                         std::to_string(i));

    inputs["Param"].emplace_back(const_cast<DenseTensor*>(param[i]));
    inputs["Grad"].emplace_back(const_cast<DenseTensor*>(grad[i]));
    size_t lr_idx = (learning_rate.size() == 1) ? 0 : i;
    inputs["LearningRate"].emplace_back(
        const_cast<DenseTensor*>(learning_rate[lr_idx]));
    inputs["Moment1"].emplace_back(const_cast<DenseTensor*>(moment1[i]));
    inputs["Moment2"].emplace_back(const_cast<DenseTensor*>(moment2[i]));
    if (beta1_pow_gcu.empty()) {
      inputs["Beta1Pow"].emplace_back(const_cast<DenseTensor*>(beta1_pow[i]));
    } else {
      inputs["Beta1Pow"].emplace_back(beta1_pow_gcu[i].get());
    }
    if (beta2_pow_gcu.empty()) {
      inputs["Beta2Pow"].emplace_back(const_cast<DenseTensor*>(beta2_pow[i]));
    } else {
      inputs["Beta2Pow"].emplace_back(beta2_pow_gcu[i].get());
    }

    output_names["ParamOut"].emplace_back(std::string("param_out") +
                                          std::to_string(i));
    output_names["Moment1Out"].emplace_back(std::string("moment1_out") +
                                            std::to_string(i));
    output_names["Moment2Out"].emplace_back(std::string("moment2_out") +
                                            std::to_string(i));
    output_names["Beta1PowOut"].emplace_back(std::string("beta1_pow_out") +
                                             std::to_string(i));
    output_names["Beta2PowOut"].emplace_back(std::string("beta2_pow_out") +
                                             std::to_string(i));

    auto param_out_tmp = std::make_shared<phi::DenseTensor>();
    param_out_tmp->set_meta(param_out[i]->meta());
    dev_ctx.template Alloc<T>(param_out_tmp.get());
    param_outs_tmp.emplace_back(param_out_tmp);

    auto moment1_out_tmp = std::make_shared<phi::DenseTensor>();
    moment1_out_tmp->set_meta(moment1_out[i]->meta());
    dev_ctx.template Alloc<T>(moment1_out_tmp.get());
    moment1_outs_tmp.emplace_back(moment1_out_tmp);

    auto moment2_out_tmp = std::make_shared<phi::DenseTensor>();
    moment2_out_tmp->set_meta(moment2_out[i]->meta());
    dev_ctx.template Alloc<T>(moment2_out_tmp.get());
    moment2_outs_tmp.emplace_back(moment2_out_tmp);

    auto beta1_pow_out_tmp = std::make_shared<phi::DenseTensor>();
    beta1_pow_out_tmp->set_meta(beta1_pow_out[i]->meta());
    dev_ctx.template Alloc<T>(beta1_pow_out_tmp.get());
    beta1_pow_outs_tmp.emplace_back(beta1_pow_out_tmp);

    auto beta2_pow_out_tmp = std::make_shared<phi::DenseTensor>();
    beta2_pow_out_tmp->set_meta(beta2_pow_out[i]->meta());
    dev_ctx.template Alloc<T>(beta2_pow_out_tmp.get());
    beta2_pow_outs_tmp.emplace_back(beta2_pow_out_tmp);

    outputs["ParamOut"].emplace_back(param_out_tmp.get());
    outputs["Moment1Out"].emplace_back(moment1_out_tmp.get());
    outputs["Moment2Out"].emplace_back(moment2_out_tmp.get());
    outputs["Beta1PowOut"].emplace_back(beta1_pow_out_tmp.get());
    outputs["Beta2PowOut"].emplace_back(beta2_pow_out_tmp.get());
  }

  GcuAttributeMap attrs;
  attrs["beta1"] = beta1.to<float>();
  attrs["beta2"] = beta2.to<float>();
  attrs["epsilon"] = epsilon.to<float>();
  attrs["multi_precision"] = multi_precision;
  attrs["use_global_beta_pow"] = use_global_beta_pow;

  GcuRunner(input_names,
            inputs,
            output_names,
            outputs,
            attrs,
            "merged_adam",
            dev_ctx,
            false);

  for (size_t i = 0; i < param_num; ++i) {
    dev_ctx.template Alloc<T>(param_out[i]);
    dev_ctx.template Alloc<T>(moment1_out[i]);
    dev_ctx.template Alloc<T>(moment2_out[i]);

    TensorCopy(dev_ctx, *(param_outs_tmp[i]), false, param_out[i]);
    TensorCopy(dev_ctx, *(moment1_outs_tmp[i]), false, moment1_out[i]);
    TensorCopy(dev_ctx, *(moment2_outs_tmp[i]), false, moment2_out[i]);

    if (!use_global_beta_pow) {
      dev_ctx.template Alloc<T>(beta1_pow_out[i]);
      dev_ctx.template Alloc<T>(beta2_pow_out[i]);
      TensorCopy(dev_ctx, *(beta1_pow_outs_tmp[i]), false, beta1_pow_out[i]);
      TensorCopy(dev_ctx, *(beta2_pow_outs_tmp[i]), false, beta2_pow_out[i]);
    }

    // if param and param_out is not same, we need to do copy.
    if (param_out[i]->data<T>() != param[i]->data<T>()) {
      TensorCopy(
          dev_ctx, *(param_out[i]), false, const_cast<DenseTensor*>(param[i]));
    }
    if (moment1_out[i]->data<T>() != moment1[i]->data<T>()) {
      TensorCopy(dev_ctx,
                 *(moment1_out[i]),
                 false,
                 const_cast<DenseTensor*>(moment1[i]));
    }
    if (moment2_out[i]->data<T>() != moment2[i]->data<T>()) {
      TensorCopy(dev_ctx,
                 *(moment2_out[i]),
                 false,
                 const_cast<DenseTensor*>(moment2[i]));
    }
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(merged_adam,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MergedAdamKernel,
                          float,
                          double) {
  // Skip beta1_pow, beta2_pow data transform
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);

  if (kernel_key.dtype() == phi::DataType::FLOAT16 ||
      kernel_key.dtype() == phi::DataType::BFLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
  }
  kernel->OutputAt(3).SetBackend(phi::Backend::UNDEFINED);
  kernel->OutputAt(4).SetBackend(phi::Backend::UNDEFINED);
}
