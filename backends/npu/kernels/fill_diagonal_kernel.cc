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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void FillDiagonalKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        float value,
                        int offset,
                        bool wrap,
                        phi::DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      offset,
      0,
      phi::errors::InvalidArgument("Paddle Custom NPU only support offset = 0 "
                                   "for fill_diagonal, but got offset = %d",
                                   offset));

  T temp_var = static_cast<T>(value);
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  NpuOpRunner runner;
  runner.SetType("FillDiagonal")
      .AddInput(x)
      .AddOutput(*out)
      .AddAttr("fill_value", temp_var)
      .AddAttr("wrap", wrap);
  runner.Run(stream);
}

template <typename T, typename Context>
void FillDiagonalGradKernel(const Context& dev_ctx,
                            const phi::DenseTensor& out_grad,
                            float value,
                            int offset,
                            bool wrap,
                            phi::DenseTensor* x_grad) {
  PADDLE_ENFORCE_EQ(
      offset,
      0,
      phi::errors::InvalidArgument("Paddle Custom NPU only support offset = 0 "
                                   "for fill_diagonal, but got offset = %d",
                                   offset));

  dev_ctx.template Alloc<T>(x_grad);
  auto stream = dev_ctx.stream();

  NpuOpRunner runner;
  runner.SetType("FillDiagonal")
      .AddInput(out_grad)
      .AddOutput(*x_grad)
      .AddAttr("fill_value", static_cast<T>(0))
      .AddAttr("wrap", wrap);
  runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(fill_diagonal,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FillDiagonalKernel,
                          float,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(fill_diagonal_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FillDiagonalGradKernel,
                          float,
                          int,
                          int64_t) {}
