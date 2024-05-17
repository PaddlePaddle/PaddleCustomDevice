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
void WhereKernel(const Context& dev_ctx,
                 const phi::DenseTensor& condition,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("where");
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    phi::DenseTensor input_x = MaybeCreateOrTrans64To32bits(dev_ctx, x);
    phi::DenseTensor input_y = MaybeCreateOrTrans64To32bits(dev_ctx, y);
    phi::DenseTensor output_z =
        MaybeCreateOrTrans64To32bits(dev_ctx, *out, false);
    LAUNCH_TOPSATENOP(
        topsatenWhere, dev_ctx, output_z, condition, input_x, input_y);
    MaybeTransResult(dev_ctx, output_z, out);

  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Y"] = {"y"};
    input_names["Condition"] = {"condition"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Y"] = {const_cast<DenseTensor*>(&y)};
    inputs["Condition"] = {const_cast<DenseTensor*>(&condition)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuRunner(input_names, inputs, output_names, outputs, {}, "where", dev_ctx);
  }
}

template <typename T, typename Context>
void WhereGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& condition,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     const phi::DenseTensor& out_grad,
                     phi::DenseTensor* x_grad,
                     phi::DenseTensor* y_grad) {
  PADDLE_GCU_KERNEL_TRACE("where_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Y"] = {"y"};
    input_names["Condition"] = {"condition"};
    input_names[GradVarName("Out")] = {"out_grad"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Y"] = {const_cast<DenseTensor*>(&y)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};
    inputs["Condition"] = {const_cast<DenseTensor*>(&condition)};

    TensorNameMap output_names;
    TensorValueMap outputs;
    if (x_grad) {
      dev_ctx.template Alloc<T>(x_grad);
      output_names[GradVarName("X")] = {"x_grad"};
      outputs[GradVarName("X")] = {x_grad};
    }
    if (y_grad) {
      dev_ctx.template Alloc<T>(y_grad);
      output_names[GradVarName("Y")] = {"y_grad"};
      outputs[GradVarName("Y")] = {y_grad};
    }

    GcuRunner(
        input_names, inputs, output_names, outputs, {}, "where_grad", dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(where,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::WhereKernel,
                          int32_t,
                          int64_t,
                          double,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(where_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::WhereGradKernel,
                          int32_t,
                          int64_t,
                          double,
                          float) {}
