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

#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {

using Tensor = phi::DenseTensor;

template <typename T, typename Context>
void LogLossKernel(const Context& dev_ctx,
                   const phi::DenseTensor& input,
                   const phi::DenseTensor& label,
                   float epsilon,
                   phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  TensorNameMap input_names;
  input_names["Predicted"] = {"input"};
  input_names["Labels"] = {"label"};

  TensorValueMap inputs;
  inputs["Predicted"] = {const_cast<DenseTensor*>(&input)};
  inputs["Labels"] = {const_cast<DenseTensor*>(&label)};

  TensorNameMap output_names;
  output_names["Loss"] = {"out"};

  TensorValueMap outputs;
  outputs["Loss"] = {out};

  GcuAttributeMap attrs;
  attrs["epsilon"] = epsilon;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, "log_loss", dev_ctx);
}

template <typename T, typename Context>
void LogLossGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& input,
                       const phi::DenseTensor& label,
                       const phi::DenseTensor& out_grad,
                       float epsilon,
                       phi::DenseTensor* in_grad) {
  dev_ctx.template Alloc<T>(in_grad);

  TensorNameMap input_names;
  input_names["Predicted"] = {"input"};
  input_names["Labels"] = {"label"};
  input_names[GradVarName("Loss")] = {"out_grad"};

  TensorValueMap inputs;
  inputs["Predicted"] = {const_cast<DenseTensor*>(&input)};
  inputs["Labels"] = {const_cast<DenseTensor*>(&label)};
  inputs[GradVarName("Loss")] = {const_cast<DenseTensor*>(&out_grad)};

  TensorNameMap output_names;
  output_names[GradVarName("Predicted")] = {"in_grad"};

  TensorValueMap outputs;
  outputs[GradVarName("Predicted")] = {in_grad};

  GcuAttributeMap attrs;
  attrs["epsilon"] = epsilon;

  GcuRunner(input_names,
            inputs,
            output_names,
            outputs,
            attrs,
            "log_loss_grad",
            dev_ctx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    log_loss, gcu, ALL_LAYOUT, custom_kernel::LogLossKernel, float) {}
PD_REGISTER_PLUGIN_KERNEL(
    log_loss_grad, gcu, ALL_LAYOUT, custom_kernel::LogLossGradKernel, float) {}
