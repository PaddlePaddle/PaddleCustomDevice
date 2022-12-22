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
void CeluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                float alpha,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  const auto& runner = NpuOpRunner("CeluV2", {x}, {*out}, {{"alpha", alpha}});
  auto stream = dev_ctx.stream();
  runner.Run(stream);
}

template <typename T, typename Context>
void CeluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& dout,
                    float alpha,
                    phi::DenseTensor* dx) {
  auto stream = dev_ctx.stream();
  phi::DenseTensor tmp_out;
  phi::DenseTensorMeta meta = {x.dtype(), x.dims()};
  tmp_out.set_meta(meta);

  float times = alpha != 0.0 ? (1 / alpha) : 0.0;
  dev_ctx.template Alloc<T>(&tmp_out);

  const auto& runner =
      NpuOpRunner("CeluV2", {x}, {tmp_out}, {{"alpha", alpha}});
  runner.Run(stream);

  const auto& runner_mul =
      NpuOpRunner("Muls", {tmp_out}, {tmp_out}, {{"value", times}});
  runner_mul.Run(stream);

  dev_ctx.template Alloc<T>(dx);
  const auto& runner_1 = NpuOpRunner(
      "EluGradV2", {dout, tmp_out}, {*dx}, {{"alpha", times * alpha}});
  runner_1.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(celu,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CeluKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(celu_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CeluGradKernel,
                          float,
                          phi::dtype::float16) {}
