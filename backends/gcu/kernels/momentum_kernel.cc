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

#include "kernels/common_ops/common_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {
template <typename T, typename Context>
void MomentumKernel(const Context& dev_ctx,
                    const phi::DenseTensor& param,
                    const phi::DenseTensor& grad,
                    const phi::DenseTensor& velocity,
                    const phi::DenseTensor& learning_rate,
                    const paddle::optional<phi::DenseTensor>& master_param,
                    float mu_f,
                    bool use_nesterov,
                    const std::string& regularization_method,
                    float regularization_coeff,
                    bool multi_precision,
                    float rescale_grad,
                    phi::DenseTensor* param_out,
                    phi::DenseTensor* velocity_out,
                    phi::DenseTensor* master_param_out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "momentum", momentum);
    phi::DenseTensor tmp_grad = grad;
    if (regularization_method == "l2_decay") {
      auto regular_coeff_op = full_like(dev_ctx, param, regularization_coeff);
      auto regular_param = mul_compute(dev_ctx, param, regular_coeff_op);
      tmp_grad = add_compute(dev_ctx, tmp_grad, regular_param);
    } else if (regularization_method.size() != 0) {
      PADDLE_THROW(
          phi::errors::InvalidArgument("GCU Not support regularization"));
    }
    auto mu_op = full_like(dev_ctx, velocity, mu_f);

    *velocity_out =
        add_compute(dev_ctx, mul_compute(dev_ctx, mu_op, velocity), tmp_grad);

    if (use_nesterov) {
      // param_out = param - (tmp_grad + mu_op * velocity_out) * learning_rate;
      auto tmp = mul_compute(dev_ctx, mu_op, *velocity_out);
      tmp = add_compute(dev_ctx, tmp_grad, tmp);
      tmp = mul_compute(dev_ctx, tmp, learning_rate);
      *param_out = sub_compute(dev_ctx, param, tmp);
    } else {
      // param_out = param - learning_rate * velocity_out;
      auto tmp = mul_compute(dev_ctx, learning_rate, *velocity_out);
      *param_out = sub_compute(dev_ctx, param, tmp);
    }
    PADDLE_GCU_KERNEL_END("momentum", momentum);
  } else {
    dev_ctx.template Alloc<T>(param_out);
    dev_ctx.template Alloc<T>(velocity_out);

    TensorNameMap input_names;
    input_names["Param"] = {"param"};
    input_names["Grad"] = {"grad"};
    input_names["Velocity"] = {"velocity"};
    input_names["LearningRate"] = {"learning_rate"};

    TensorValueMap inputs;
    inputs["Param"] = {const_cast<DenseTensor*>(&param)};
    inputs["Grad"] = {const_cast<DenseTensor*>(&grad)};
    inputs["Velocity"] = {const_cast<DenseTensor*>(&velocity)};
    inputs["LearningRate"] = {const_cast<DenseTensor*>(&learning_rate)};

    phi::DenseTensor param_out_tmp;
    param_out_tmp.set_meta(param_out->meta());
    dev_ctx.template Alloc<T>(&param_out_tmp);

    phi::DenseTensor velocity_out_tmp;
    velocity_out_tmp.set_meta(velocity_out->meta());
    dev_ctx.template Alloc<T>(&velocity_out_tmp);

    TensorNameMap output_names;
    output_names["VelocityOut"] = {"velocity_out"};
    output_names["ParamOut"] = {"param_out"};

    TensorValueMap outputs;
    outputs["VelocityOut"] = {&velocity_out_tmp};
    outputs["ParamOut"] = {&param_out_tmp};

    GcuAttributeMap attrs;
    attrs["mu"] = mu_f;
    attrs["use_nesterov"] = use_nesterov;
    attrs["regularization_method"] = regularization_method;
    attrs["regularization_coeff"] = regularization_coeff;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "momentum", dev_ctx);

    TensorCopy(dev_ctx, param_out_tmp, false, param_out);
    TensorCopy(dev_ctx, velocity_out_tmp, false, velocity_out);

    // if param and param_out is not same, we need to do copy.
    if (param_out->data<T>() != param.data<T>()) {
      TensorCopy(dev_ctx, *param_out, false, const_cast<DenseTensor*>(&param));
    }
    if (velocity_out->data<T>() != velocity.data<T>()) {
      TensorCopy(
          dev_ctx, *velocity_out, false, const_cast<DenseTensor*>(&velocity));
    }
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(momentum,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MomentumKernel,
                          phi::dtype::float16,
                          float,
                          double) {}
