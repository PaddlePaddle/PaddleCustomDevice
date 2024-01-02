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

#include "kernels/common_ops/elementwise_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {
template <typename T, typename Context>
void LogicalAndKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "logical_and", logical_and);
    and_compute(static_cast<const phi::CustomContext&>(dev_ctx), x, y, out);
    PADDLE_GCU_KERNEL_END("logical_and", logical_and);
  } else {
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Y"] = {"y"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Y"] = {const_cast<DenseTensor*>(&y)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "logical_and",
              dev_ctx);
  }
}

template <typename T, typename Context>
void LogicalNotKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  TensorNameMap input_names;
  input_names["X"] = {"x"};

  TensorValueMap inputs;
  inputs["X"] = {const_cast<DenseTensor*>(&x)};

  TensorNameMap output_names;
  output_names["Out"] = {"out"};

  TensorValueMap outputs;
  outputs["Out"] = {out};

  GcuAttributeMap attrs;

  GcuRunner(input_names,
            inputs,
            output_names,
            outputs,
            attrs,
            "logical_not",
            dev_ctx);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    logical_and, gcu, ALL_LAYOUT, custom_kernel::LogicalAndKernel, bool) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(
    logical_not, gcu, ALL_LAYOUT, custom_kernel::LogicalNotKernel, bool) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
