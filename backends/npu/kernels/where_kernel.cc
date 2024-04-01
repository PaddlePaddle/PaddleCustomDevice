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
void AclopWhereKernel(const Context& dev_ctx,
                      const phi::DenseTensor& condition,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  const auto& runner = NpuOpRunner("Select", {condition, x, y}, {*out}, {});

  auto stream = dev_ctx.stream();
  runner.Run(stream);
}

template <typename T, typename Context>
void WhereKernel(const Context& dev_ctx,
                 const phi::DenseTensor& condition,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out) {
  DO_COMPATIBILITY(aclnnSWhere,
                   (custom_kernel::AclopWhereKernel<T, Context>(
                       dev_ctx, condition, x, y, out)));
  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnSWhere, dev_ctx, condition, x, y, *out);
}

template <typename T, typename Context>
void WhereGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& condition,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     const phi::DenseTensor& out_grad,
                     phi::DenseTensor* x_grad,
                     phi::DenseTensor* y_grad) {
  if (x_grad != nullptr) {
    dev_ctx.template Alloc<T>(x_grad);
  }
  if (y_grad != nullptr) {
    dev_ctx.template Alloc<T>(y_grad);
  }

  auto stream = dev_ctx.stream();

  phi::DenseTensor tensor_zeros;
  phi::DenseTensorMeta zeros_meta = {out_grad.dtype(), out_grad.dims()};
  tensor_zeros.set_meta(zeros_meta);
  dev_ctx.template Alloc<T>(&tensor_zeros);

  const auto& runner = NpuOpRunner("ZerosLike", {out_grad}, {tensor_zeros}, {});
  runner.Run(stream);

  if (x_grad != nullptr) {
    const auto& runner = NpuOpRunner(
        "Select", {condition, out_grad, tensor_zeros}, {*x_grad}, {});
    runner.Run(stream);
  }
  if (y_grad != nullptr) {
    const auto& runner = NpuOpRunner(
        "Select", {condition, tensor_zeros, out_grad}, {*y_grad}, {});
    runner.Run(stream);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(where,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::WhereKernel,
                          int32_t,
                          int64_t,
                          double,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(where_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::WhereGradKernel,
                          int32_t,
                          int64_t,
                          double,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
