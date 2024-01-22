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
#include "paddle/phi/common/type_traits.h"

namespace custom_kernel {

template <typename T, typename Context>
void AclopAbsKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("Abs", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void AbsKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnAbs, (custom_kernel::AclopAbsKernel<T, Context>(dev_ctx, x, out)));

  dev_ctx.template Alloc<T>(out);
  EXEC_NPU_CMD(aclnnAbs, dev_ctx, x, *out);
}

template <typename T, typename Context>
void AclopAbsGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& dout,
                        phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("AbsGrad", {x, dout}, {*dx}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void AbsGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  DO_COMPATIBILITY(
      aclnnSign,
      (custom_kernel::AclopAbsGradKernel<T, Context>(dev_ctx, x, dout, dx)));

  dev_ctx.template Alloc<T>(dx);
  EXEC_NPU_CMD(aclnnSign, dev_ctx, x, *dx);
  EXEC_NPU_CMD(aclnnInplaceMul, dev_ctx, *dx, dout);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(abs,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AbsKernel,
                          float,
                          phi::dtype::bfloat16,
                          phi::dtype::float16,
                          double,
                          int64_t) {
  kernel->InputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

PD_REGISTER_PLUGIN_KERNEL(abs_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AbsGradKernel,
                          float,
                          double,
                          phi::dtype::bfloat16,
                          phi::dtype::float16,
                          int64_t) {
  kernel->InputAt(1).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}
