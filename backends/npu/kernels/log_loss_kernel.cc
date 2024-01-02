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

using Tensor = phi::DenseTensor;

template <typename T, typename Context>
void LogLossAdds(const Context& dev_ctx,
                 const aclrtStream& stream,
                 const Tensor* x,
                 float scale,
                 Tensor* y) {
  //  Calculate y = x + scale
  y->Resize(x->dims());
  dev_ctx.template Alloc<T>(y);
  const auto& runner = NpuOpRunner("Adds", {*x}, {*y}, {{"value", scale}});
  runner.Run(stream);
}

template <typename T, typename Context>
void LogLossMuls(const Context& dev_ctx,
                 const aclrtStream& stream,
                 const Tensor* x,
                 float scale,
                 Tensor* y) {
  //  Calculate y = x + scale
  y->Resize(x->dims());
  dev_ctx.template Alloc<T>(y);
  const auto& runner = NpuOpRunner("Muls", {*x}, {*y}, {{"value", scale}});
  runner.Run(stream);
}

template <typename T, typename Context>
void LogLossBCE(const Context& dev_ctx,
                const aclrtStream& stream,
                const Tensor* x,
                const Tensor* y,
                Tensor* z) {
  z->Resize(x->dims());
  dev_ctx.template Alloc<T>(z);
  const auto& runner =
      NpuOpRunner("BinaryCrossEntropy",
                  {*x, *y},
                  {*z},
                  {{"reduction", static_cast<std::string>("none")}});
  runner.Run(stream);
}

template <typename T, typename Context>
void LogLossBCEGrad(const Context& dev_ctx,
                    const aclrtStream& stream,
                    const Tensor* x,
                    const Tensor* y,
                    const Tensor* dout,
                    Tensor* dx) {
  dx->Resize(x->dims());
  dev_ctx.template Alloc<T>(dx);
  const auto& runner =
      NpuOpRunner("BinaryCrossEntropyGrad",
                  {*x, *y, *dout},
                  {*dx},
                  {{"reduction", static_cast<std::string>("none")}});
  runner.Run(stream);
}

template <typename T, typename Context>
void LogLossKernel(const Context& dev_ctx,
                   const phi::DenseTensor& input,
                   const phi::DenseTensor& label,
                   float epsilon,
                   phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();

  float factor = 1 / (1 + 2 * epsilon);
  float coef = std::log(factor);

  LogLossAdds<T>(dev_ctx, stream, &input, epsilon, out);
  LogLossMuls<T>(dev_ctx, stream, out, factor, out);
  LogLossBCE<T>(dev_ctx, stream, out, &label, out);
  LogLossAdds<T>(dev_ctx, stream, out, coef, out);
}

template <typename T, typename Context>
void LogLossGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& input,
                       const phi::DenseTensor& label,
                       const phi::DenseTensor& out_grad,
                       float epsilon,
                       phi::DenseTensor* in_grad) {
  auto stream = dev_ctx.stream();

  if (in_grad) {
    LogLossBCEGrad<T>(dev_ctx, stream, &input, &label, &out_grad, in_grad);
    LogLossMuls<T>(dev_ctx, stream, in_grad, 1 / (1 + 2 * epsilon), in_grad);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    log_loss, npu, ALL_LAYOUT, custom_kernel::LogLossKernel, float) {}
PD_REGISTER_PLUGIN_KERNEL(
    log_loss_grad, npu, ALL_LAYOUT, custom_kernel::LogLossGradKernel, float) {}
