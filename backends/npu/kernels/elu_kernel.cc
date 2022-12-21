// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void EluKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               float alpha,
               phi::DenseTensor* out) {
  float scaleValue = 1.0f;
  float inputScaleValue = 1.0f;
  NPUAttributeMap attr_input = {{"alpha", alpha},
                                {"scale", scaleValue},
                                {"input_scale", inputScaleValue}};
  dev_ctx.template Alloc<T>(out);
  const auto& runner = NpuOpRunner("Elu", {x}, {*out}, attr_input);
  auto stream = dev_ctx.stream();
  runner.Run(stream);
}

template <typename T, typename Context>
void EluGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,  // output
                   const phi::DenseTensor& out,
                   const phi::DenseTensor& dout,
                   float alpha,
                   phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();
  const auto& runner =
      NpuOpRunner("EluGradV2", {dout, out}, {*dx}, {{"alpha", alpha}});
  runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(elu,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::EluKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(elu_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::EluGradKernel,
                          float,
                          phi::dtype::float16) {}
