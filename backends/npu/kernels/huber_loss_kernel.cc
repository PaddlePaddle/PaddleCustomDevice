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
void SubtractKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out);

template <typename T, typename Context>
void MulsKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const float scaling,
                phi::DenseTensor* out);

template <typename T, typename Context>
void FullLikeKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::Scalar& val,
                    phi::DataType dtype,
                    phi::DenseTensor* out);

template <typename T, typename Context>
void HuberLossSub(const Context& dev_ctx,
                  const phi::DenseTensor* x,
                  const phi::DenseTensor* y,
                  phi::DenseTensor* z) {
  //  Calculate z = x - y
  z->Resize(x->dims());
  custom_kernel::SubtractKernel<T, Context>(dev_ctx, *x, *y, z);
}

template <typename T, typename Context>
void HuberLossMuls(const Context& dev_ctx,
                   const phi::DenseTensor* x,
                   float scalar,
                   phi::DenseTensor* y) {
  //  Calculate y = x + scale
  y->Resize(x->dims());
  custom_kernel::MulsKernel<T, Context>(dev_ctx, *x, scalar, y);
}

template <typename T, typename Context>
void HuberLossZerosLike(const Context& dev_ctx,
                        const phi::DenseTensor* x,
                        phi::DenseTensor* y) {
  y->Resize(x->dims());
  phi::Scalar zeros = static_cast<T>(0);
  custom_kernel::FullLikeKernel<T, Context>(dev_ctx, *x, zeros, x->dtype(), y);
}

template <typename T, typename Context>
void AclopSmoothL1LossKernel(const Context& dev_ctx,
                             const phi::DenseTensor* x,
                             const phi::DenseTensor* y,
                             float delta,
                             phi::DenseTensor* z) {
  dev_ctx.template Alloc<T>(z);
  const auto& runner =
      NpuOpRunner("SmoothL1Loss", {*x, *y}, {*z}, {{"sigma", delta}});
  runner.Run(dev_ctx.stream());
}

template <typename T, typename Context>
void SmoothL1LossKernel(const Context& dev_ctx,
                        const phi::DenseTensor* x,
                        const phi::DenseTensor* y,
                        float delta,
                        phi::DenseTensor* z) {
  DO_COMPATIBILITY(aclnnSmoothL1Loss,
                   (custom_kernel::AclopSmoothL1LossKernel<T, Context>(
                       dev_ctx, x, y, delta, z)));

  dev_ctx.template Alloc<T>(z);
  int64_t reduction = 0;  // none
  EXEC_NPU_CMD(aclnnSmoothL1Loss, dev_ctx, *x, *y, reduction, delta, *z);
}

template <typename T, typename Context>
void HuberLossSmoothL1Loss(const Context& dev_ctx,
                           const phi::DenseTensor* x,
                           const phi::DenseTensor* y,
                           float delta,
                           phi::DenseTensor* z) {
  z->Resize(x->dims());
  custom_kernel::SmoothL1LossKernel<T, Context>(dev_ctx, x, y, delta, z);
}

template <typename T, typename Context>
void AclopSmoothL1LossGrad(const Context& dev_ctx,
                           const phi::DenseTensor* pred,
                           const phi::DenseTensor* lab,
                           const phi::DenseTensor* dout,
                           float sigma,
                           phi::DenseTensor* grad) {
  dev_ctx.template Alloc<T>(grad);
  const auto& runner = NpuOpRunner(
      "SmoothL1LossGrad", {*pred, *lab, *dout}, {*grad}, {{"sigma", sigma}});
  runner.Run(dev_ctx.stream());
}

template <typename T, typename Context>
void SmoothL1LossGrad(const Context& dev_ctx,
                      const phi::DenseTensor* pred,
                      const phi::DenseTensor* lab,
                      const phi::DenseTensor* dout,
                      float sigma,
                      phi::DenseTensor* grad) {
  DO_COMPATIBILITY(aclnnSmoothL1LossBackward,
                   (custom_kernel::AclopSmoothL1LossGrad<T, Context>(
                       dev_ctx, pred, lab, dout, sigma, grad)));

  dev_ctx.template Alloc<T>(grad);
  int64_t reduction = 0;  // none
  EXEC_NPU_CMD(aclnnSmoothL1LossBackward,
               dev_ctx,
               *dout,
               *pred,
               *lab,
               reduction,
               sigma,
               *grad);
}

template <typename T, typename Context>
void HuberLossSmoothL1LossGrad(const Context& dev_ctx,
                               const phi::DenseTensor* pred,
                               const phi::DenseTensor* lab,
                               const phi::DenseTensor* dout,
                               float sigma,
                               phi::DenseTensor* grad) {
  grad->Resize(pred->dims());
  custom_kernel::SmoothL1LossGrad<T, Context>(
      dev_ctx, pred, lab, dout, sigma, grad);
}

template <typename T, typename Context>
void HuberLossKernel(const Context& dev_ctx,
                     const phi::DenseTensor& input,
                     const phi::DenseTensor& label,
                     float delta,
                     phi::DenseTensor* out,
                     phi::DenseTensor* residual) {
  auto stream = dev_ctx.stream();
  HuberLossSub<T, Context>(dev_ctx, &label, &input, residual);

  HuberLossSmoothL1Loss<T, Context>(dev_ctx, &input, &label, delta, out);
  HuberLossMuls<T, Context>(dev_ctx, out, delta, out);
}

template <typename T, typename Context>
void HuberLossGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& residual,
                         const phi::DenseTensor& dout,
                         float delta,
                         phi::DenseTensor* dx,
                         phi::DenseTensor* dy) {
  phi::DenseTensor t_grad_rd;
  if (dx || dy) {
    phi::DenseTensor t_zero;
    HuberLossZerosLike<T, Context>(dev_ctx, &residual, &t_zero);
    HuberLossSmoothL1LossGrad<T, Context>(
        dev_ctx, &residual, &t_zero, &dout, delta, &t_grad_rd);
  }
  if (dx) {
    HuberLossMuls<T, Context>(dev_ctx, &t_grad_rd, -delta, dx);
  }
  if (dy) {
    HuberLossMuls<T, Context>(dev_ctx, &t_grad_rd, delta, dy);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(huber_loss,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::HuberLossKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(huber_loss_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::HuberLossGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
