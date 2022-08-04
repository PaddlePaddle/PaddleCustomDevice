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
void HuberLossSub(const Context& dev_ctx,
                  const phi::DenseTensor* x,
                  const phi::DenseTensor* y,
                  phi::DenseTensor* z) {
  //  Calculate z = x - y
  z->Resize(x->dims());
  dev_ctx.template Alloc<T>(z);
  const auto& runner = NpuOpRunner("Sub", {*x, *y}, {*z}, {});
  runner.Run(dev_ctx.stream());
}

template <typename T, typename Context>
void HuberLossMuls(const Context& dev_ctx,
                   const phi::DenseTensor* x,
                   float scalar,
                   phi::DenseTensor* y) {
  //  Calculate y = x + scale
  y->Resize(x->dims());
  dev_ctx.template Alloc<T>(y);
  const auto& runner = NpuOpRunner("Muls", {*x}, {*y}, {{"value", scalar}});
  runner.Run(dev_ctx.stream());
}

template <typename T, typename Context>
void HuberLossZerosLike(const Context& dev_ctx,
                        const phi::DenseTensor* x,
                        phi::DenseTensor* y) {
  y->Resize(x->dims());
  dev_ctx.template Alloc<T>(y);
  const auto& runner = NpuOpRunner("ZerosLike", {*x}, {*y}, {});
  runner.Run(dev_ctx.stream());
}

template <typename T, typename Context>
void HuberLossSmoothL1Loss(const Context& dev_ctx,
                           const phi::DenseTensor* x,
                           const phi::DenseTensor* y,
                           float delta,
                           phi::DenseTensor* z) {
  z->Resize(x->dims());
  dev_ctx.template Alloc<T>(z);
  const auto& runner =
      NpuOpRunner("SmoothL1Loss", {*x, *y}, {*z}, {{"sigma", delta}});
  runner.Run(dev_ctx.stream());
}

template <typename T, typename Context>
void HuberLossSmoothL1LossGrad(const Context& dev_ctx,
                               const phi::DenseTensor* pred,
                               const phi::DenseTensor* lab,
                               const phi::DenseTensor* dout,
                               float sigma,
                               phi::DenseTensor* grad) {
  grad->Resize(pred->dims());
  dev_ctx.template Alloc<T>(grad);
  const auto& runner = NpuOpRunner(
      "SmoothL1LossGrad", {*pred, *lab, *dout}, {*grad}, {{"sigma", sigma}});
  runner.Run(dev_ctx.stream());
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
  auto stream = dev_ctx.stream();

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
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::HuberLossKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(huber_loss_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::HuberLossGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
