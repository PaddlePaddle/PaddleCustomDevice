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
void StackKernel(const Context& dev_ctx,
                 const std::vector<const phi::DenseTensor*>& x,
                 int axis,
                 phi::DenseTensor* y) {
  dev_ctx.template Alloc<T>(y);
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "stack", stack);
    std::vector<phi::DenseTensor> inputs;
    for (auto& input : x) {
      inputs.push_back(*(const_cast<phi::DenseTensor*>(input)));
    }
    stack(dev_ctx, inputs, axis, *y);
    PADDLE_GCU_KERNEL_END("stack", stack);
  } else {
    TensorNameMap input_names;
    TensorValueMap inputs;
    std::vector<std::string> names;
    names.reserve(x.size());
    std::vector<phi::DenseTensor*> values;
    values.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      names.emplace_back(std::string("x_") + std::to_string(i));
      values.emplace_back(const_cast<DenseTensor*>(x[i]));
    }
    input_names["X"] = names;
    inputs["X"] = values;

    TensorNameMap output_names;
    output_names["Y"] = {"y"};

    TensorValueMap outputs;
    outputs["Y"] = {y};

    GcuAttributeMap attrs;
    attrs["axis"] = axis;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "stack", dev_ctx);
  }
}

template <typename T, typename Context>
void StackGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& dy,
                     int axis,
                     std::vector<phi::DenseTensor*> dx) {
  TensorNameMap input_names;
  TensorValueMap inputs;

  input_names[GradVarName("Y")] = {"dy"};
  inputs[GradVarName("Y")] = {const_cast<DenseTensor*>(&dy)};

  TensorNameMap output_names;
  TensorValueMap outputs;

  std::vector<std::string> names;
  names.reserve(dx.size());
  std::vector<phi::DenseTensor*> values;
  values.reserve(dx.size());
  for (size_t i = 0; i < dx.size(); ++i) {
    dev_ctx.template Alloc<T>(dx[i]);
    names.emplace_back(std::string("x_grad_") + std::to_string(i));
    values.emplace_back(dx[i]);
  }
  output_names[GradVarName("X")] = names;
  outputs[GradVarName("X")] = values;

  GcuAttributeMap attrs;
  attrs["axis"] = axis;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, "stack_grad", dev_ctx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(stack,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::StackKernel,
                          int,
                          int64_t,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(stack_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::StackGradKernel,
                          int,
                          int64_t,
                          float,
                          double) {}
