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
void LogSoftmaxKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      int axis,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const int rank = x.dims().size();
  axis = CanonicalAxis(axis, rank);

  if (rank == 0) {
    dev_ctx.template Alloc<T>(out);
    FillNpuTensorWithConstant<T>(out, dev_ctx, static_cast<T>(0));
    return;
  }

  if (x.numel() != 0) {
    const auto& runner = NpuOpRunner(
        "LogSoftmaxV2", {x}, {*out}, {{"axes", std::vector<int>{axis}}});
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void LogSoftmaxGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& out,
                          const phi::DenseTensor& dout,
                          int axis,
                          phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  const int rank = dout.dims().size();
  axis = CanonicalAxis(axis, rank);

  if (rank == 0) {
    FillNpuTensorWithConstant<T>(dx, dev_ctx, static_cast<T>(0));
    return;
  }

  if (dout.numel() != 0) {
    const auto& runner = NpuOpRunner("LogSoftmaxGrad",
                                     {dout, out},
                                     {*dx},
                                     {{"axis", std::vector<int>{axis}}});
    runner.Run(stream);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(log_softmax,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LogSoftmaxKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(log_softmax_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LogSoftmaxGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}
