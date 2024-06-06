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
template <typename T, typename Context>
void InstanceNormKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const paddle::optional<phi::DenseTensor>& scale,
                        const paddle::optional<phi::DenseTensor>& bias,
                        float epsilon_f,
                        phi::DenseTensor* y,
                        phi::DenseTensor* saved_mean,
                        phi::DenseTensor* saved_variance) {
  PADDLE_GCU_KERNEL_TRACE("instance_norm");
  dev_ctx.template Alloc<T>(y);
  dev_ctx.template Alloc<T>(saved_mean);
  dev_ctx.template Alloc<T>(saved_variance);

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    if (scale) {
      input_names["Scale"] = {"scale"};
      inputs["Scale"] = {const_cast<DenseTensor*>(scale.get_ptr())};
    }
    if (bias) {
      input_names["Bias"] = {"bias"};
      inputs["Bias"] = {const_cast<DenseTensor*>(bias.get_ptr())};
    }

    TensorNameMap output_names;
    output_names["Y"] = {"y"};
    output_names["SavedMean"] = {"saved_mean"};
    output_names["SavedVariance"] = {"saved_variance"};

    TensorValueMap outputs;
    outputs["Y"] = {y};
    outputs["SavedMean"] = {saved_mean};
    outputs["SavedVariance"] = {saved_variance};

    GcuAttributeMap attrs;
    attrs["epsilon"] = epsilon_f;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "instance_norm",
              dev_ctx);
  }
}

template <typename T, typename Context>
void InstanceNormGradKernel(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            const paddle::optional<phi::DenseTensor>& scale,
                            const phi::DenseTensor& saved_mean,
                            const phi::DenseTensor& saved_variance,
                            const phi::DenseTensor& d_y,
                            float epsilon,
                            phi::DenseTensor* d_x,
                            phi::DenseTensor* d_scale,
                            phi::DenseTensor* d_bias) {
  PADDLE_GCU_KERNEL_TRACE("instance_norm_grad");
  dev_ctx.template Alloc<T>(d_x);
  dev_ctx.template Alloc<T>(d_scale);
  dev_ctx.template Alloc<T>(d_bias);

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["SavedMean"] = {"saved_mean"};
    input_names["SavedVariance"] = {"saved_variance"};
    input_names[GradVarName("Y")] = {"d_y"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["SavedMean"] = {const_cast<DenseTensor*>(&saved_mean)};
    inputs["SavedVariance"] = {const_cast<DenseTensor*>(&saved_variance)};
    inputs[GradVarName("Y")] = {const_cast<DenseTensor*>(&d_y)};

    if (scale) {
      input_names["Scale"] = {"scale"};
      inputs["Scale"] = {const_cast<DenseTensor*>(scale.get_ptr())};
    }

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"d_x"};
    output_names[GradVarName("Scale")] = {"d_scale"};
    output_names[GradVarName("Bias")] = {"d_bias"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {d_x};
    outputs[GradVarName("Scale")] = {d_scale};
    outputs[GradVarName("Bias")] = {d_bias};

    GcuAttributeMap attrs;
    attrs["epsilon"] = epsilon;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "instance_norm_grad",
              dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(instance_norm,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::InstanceNormKernel,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(instance_norm_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::InstanceNormGradKernel,
                          float,
                          double) {}
