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
void BitwiseAndKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("bitwise_and");
  dev_ctx.template Alloc<T>(out);
  if (LaunchAOTKernel()) {
    LAUNCH_TOPSATENOP(topsatenBitwiseAnd, dev_ctx, *out, x, y);

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}

template <typename T, typename Context>
void BitwiseNotKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("bitwise_not");
  dev_ctx.template Alloc<T>(out);
  if (LaunchAOTKernel()) {
    LAUNCH_TOPSATENOP(topsatenBitwiseNot, dev_ctx, *out, x);

  } else {  // kernel impl base on JIT
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
              "bitwise_not",
              dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    bitwise_and, gcu, ALL_LAYOUT, custom_kernel::BitwiseAndKernel, bool, int) {}

PD_REGISTER_PLUGIN_KERNEL(
    bitwise_not, gcu, ALL_LAYOUT, custom_kernel::BitwiseNotKernel, bool, int) {}
