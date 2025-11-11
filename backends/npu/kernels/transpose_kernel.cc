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

#include "paddle/phi/kernels/transpose_kernel.h"

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void NPUIdentityKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const int format,
                       phi::DenseTensor* out);

template <typename T, typename Context>
void TransposeKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const std::vector<int>& axis,
                     phi::DenseTensor* out) {
  phi::DenseTensor x_tmp;
  // TODO(songkai05): CANN does not support trans from NC1HWC0 to ND between
  // Transpose_in_0 and Transpose, so we trans NC1HWC0 to its original format
  // first temporarily.
  if (x.storage_properties_initialized()) {
    phi::DenseTensorMeta meta = {x.dtype(), x.dims()};
    x_tmp.set_meta(meta);
    custom_kernel::NPUIdentityKernel<T, Context>(
        dev_ctx, x, ConvertToNpuFormat(x.layout()), &x_tmp);
  } else {
    x_tmp = x;
  }

  dev_ctx.template Alloc<T>(out);

  if (axis.size() == 0) {
    phi::Copy<Context>(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    return;
  }

  auto stream = dev_ctx.stream();
  NpuOpRunner runner;
  runner.SetType("Transpose")
      .AddInput(x_tmp)
      .AddInput(dev_ctx, std::move(axis))
      .AddOutput(*out);
  runner.Run(stream);
}

template <typename T, typename Context>
void TransposeGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& dout,
                         const std::vector<int>& axis,
                         phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  if (axis.size() == 0) {
    phi::Copy<Context>(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
    return;
  }

  std::vector<int> reversed_axis(axis);
  for (size_t i = 0; i < axis.size(); i++) {
    reversed_axis[axis[i]] = i;
  }
  NpuOpRunner runner;
  runner.SetType("Transpose")
      .AddInput(dout)
      .AddInput(dev_ctx, std::move(reversed_axis))
      .AddOutput(*dx);
  runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(transpose,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::TransposeKernel,
                          int,
                          int64_t,
                          uint8_t,
                          int8_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(transpose_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::TransposeGradKernel,
                          int,
                          int64_t,
                          uint8_t,
                          int8_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
