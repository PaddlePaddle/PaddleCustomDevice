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
void SigmoidCrossEntropyWithLogitsKernel(const Context& dev_ctx,
                                         const phi::DenseTensor& x,
                                         const phi::DenseTensor& label,
                                         bool normalize,
                                         int ignore_index,
                                         phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("sigmoid_cross_entropy_with_logits");
  dev_ctx.template Alloc<T>(out);
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Label"] = {"label"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Label"] = {const_cast<DenseTensor*>(&label)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["normalize"] = normalize;
    attrs["ignore_index"] = ignore_index;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "sigmoid_cross_entropy_with_logits",
              dev_ctx);
  }
}

template <typename T, typename Context>
void SigmoidCrossEntropyWithLogitsGradKernel(const Context& dev_ctx,
                                             const phi::DenseTensor& x,
                                             const phi::DenseTensor& label,
                                             const phi::DenseTensor& dout,
                                             bool normalize,
                                             int ignore_index,
                                             phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("sigmoid_cross_entropy_with_logits_grad");
  dev_ctx.template Alloc<T>(dx);

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Label"] = {"label"};
    input_names[GradVarName("Out")] = {"dout"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Label"] = {const_cast<DenseTensor*>(&label)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&dout)};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"dx"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {dx};

    GcuAttributeMap attrs;
    attrs["normalize"] = normalize;
    attrs["ignore_index"] = ignore_index;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "sigmoid_cross_entropy_with_logits_grad",
              dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(sigmoid_cross_entropy_with_logits,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SigmoidCrossEntropyWithLogitsKernel,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(
    sigmoid_cross_entropy_with_logits_grad,
    gcu,
    ALL_LAYOUT,
    custom_kernel::SigmoidCrossEntropyWithLogitsGradKernel,
    float,
    double) {}
