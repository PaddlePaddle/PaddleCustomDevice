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
void TransposeKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const std::vector<int>& axis,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "transpose", transpose);
    std::vector<int64_t> perm(axis.begin(), axis.end());
    transpose(dev_ctx, x, *out, perm);
    PADDLE_GCU_KERNEL_END("transpose", transpose);
  } else {
    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["axis"] = axis;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "transpose2",
              dev_ctx);
  }
}

template <typename T, typename Context>
void TransposeGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& dout,
                         const std::vector<int>& axis,
                         phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "transpose_grad", transpose_grad);
    std::vector<int64_t> reversed_permutation(axis.begin(), axis.end());
    for (size_t i = 0; i < axis.size(); ++i) {
      reversed_permutation[axis[i]] = i;
    }
    transpose(dev_ctx, dout, *dx, reversed_permutation);
    PADDLE_GCU_KERNEL_END("transpose_grad", transpose_grad);
  } else {
    TensorNameMap input_names;
    input_names[GradVarName("Out")] = {"dout"};

    TensorValueMap inputs;
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&dout)};

    TensorNameMap output_names;
    TensorValueMap outputs;

    output_names[GradVarName("X")] = {"dx"};
    outputs[GradVarName("X")] = {dx};

    GcuAttributeMap attrs;
    attrs["axis"] = axis;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "transpose2_grad",
              dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(transpose,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::TransposeKernel,
                          int,
                          int64_t,
                          uint8_t,
                          int8_t,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(transpose_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::TransposeGradKernel,
                          int,
                          int64_t,
                          uint8_t,
                          int8_t,
                          float,
                          double,
                          phi::dtype::float16) {}
