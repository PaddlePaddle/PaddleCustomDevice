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
void AclopBCELossKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& labels,
                        phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto stream = dev_ctx.stream();
  const auto& runner =
      NpuOpRunner("BinaryCrossEntropy",
                  {x, labels},
                  {*out},
                  {{"reduction", static_cast<std::string>("none")}});
  runner.Run(stream);
}

template <typename T, typename Context>
void BCELossKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& labels,
                   phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnBinaryCrossEntropy,
      (custom_kernel::AclopBCELossKernel<T, Context>(dev_ctx, x, labels, out)));
  dev_ctx.template Alloc<T>(out);
  int64_t reduction = 0;
  phi::DenseTensor* weight = nullptr;
  EXEC_NPU_CMD(
      aclnnBinaryCrossEntropy, dev_ctx, x, labels, weight, reduction, *out);
}

template <typename T, typename Context>
void AclopBCELossGradKernel(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            const phi::DenseTensor& labels,
                            const phi::DenseTensor& dout,
                            phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  auto stream = dev_ctx.stream();
  const auto& runner =
      NpuOpRunner("BinaryCrossEntropyGrad",
                  {x, labels, dout},
                  {*dx},
                  {{"reduction", static_cast<std::string>("none")}});
  runner.Run(stream);
}

template <typename T, typename Context>
void BCELossGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& labels,
                       const phi::DenseTensor& dout,
                       phi::DenseTensor* dx) {
  DO_COMPATIBILITY(aclnnBinaryCrossEntropyBackward,
                   (custom_kernel::AclopBCELossGradKernel<T, Context>(
                       dev_ctx, x, labels, dout, dx)));
  dev_ctx.template Alloc<T>(dx);
  int64_t reduction = 0;
  phi::DenseTensor* weight = nullptr;
  EXEC_NPU_CMD(aclnnBinaryCrossEntropyBackward,
               dev_ctx,
               dout,
               x,
               labels,
               weight,
               reduction,
               *dx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    bce_loss, npu, ALL_LAYOUT, custom_kernel::BCELossKernel, float, double) {}

PD_REGISTER_PLUGIN_KERNEL(bce_loss_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::BCELossGradKernel,
                          float,
                          double) {}
