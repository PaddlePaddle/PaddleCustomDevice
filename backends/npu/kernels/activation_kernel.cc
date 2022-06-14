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
void AtanKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("Atan", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void AtanGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("AtanGrad", {x, dout}, {*dx}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void ExpKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("Exp", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void ExpGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& out,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("Mul", {dout, out}, {*dx}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void ReluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  const auto& runner = NpuOpRunner("Relu", {x}, {*out}, {});

  auto stream = dev_ctx.stream();
  runner.Run(stream);
}

template <typename T, typename Context>
void ReluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  auto stream = dev_ctx.stream();
  dev_ctx.template Alloc<T>(dx);
  const auto& runner = NpuOpRunner("ReluGrad", {dout, out}, {*dx}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void LeakyReluKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     float alpha,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto stream = dev_ctx.stream();
  const auto& runner =
      NpuOpRunner("LeakyRelu", {x}, {*out}, {{"negative_slope", alpha}});
  runner.Run(stream);
}

template <typename T, typename Context>
void LeakyReluGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& dout,
                         float alpha,
                         phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner(
      "LeakyReluGrad", {dout, x}, {*dx}, {{"negative_slope", alpha}});
  runner.Run(stream);
}

template <typename T, typename Context>
void SigmoidKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("Sigmoid", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void SigmoidGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& dout,
                       phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("SigmoidGrad", {out, dout}, {*dx}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void SqrtKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("Sqrt", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void SqrtGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("SqrtGrad", {out, dout}, {*dx}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void SquareKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("Square", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void SquareGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& dout,
                      phi::DenseTensor* dx) {
  auto stream = dev_ctx.stream();

  auto factor = static_cast<float>(2.0);

  // Step 1: Compute x_muls_factor = factor * x
  phi::DenseTensor x_muls_factor;
  phi::DenseTensorMeta x_muls_factor_meta = {x.dtype(), x.dims()};
  x_muls_factor.set_meta(x_muls_factor_meta);
  dev_ctx.template Alloc<T>(&x_muls_factor);

  const auto& runner_muls_1 =
      NpuOpRunner("Muls", {x}, {x_muls_factor}, {{"value", factor}});
  runner_muls_1.Run(stream);

  // Step 2: Compute dx = dout * factor * x
  dev_ctx.template Alloc<T>(dx);
  const auto& runner_mul_2 =
      NpuOpRunner("Mul", {dout, x_muls_factor}, {*dx}, {});
  runner_mul_2.Run(stream);

void SwishKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 float beta,
                 phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const auto& muls_runner = NpuOpRunner("Muls", {x}, {*out}, {{"value", beta}});
  muls_runner.Run(stream);

  const auto& sigmoid_runner = NpuOpRunner("Sigmoid", {*out}, {*out}, {});
  sigmoid_runner.Run(stream);

  const auto& mul_runner = NpuOpRunner("Mul", {x, *out}, {*out});
  mul_runner.Run(stream);
}

template <typename T, typename Context>
void SwishGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& dout,
                     float beta,
                     phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  phi::DenseTensor beta_x, sigmoid_out, swish_out;
  phi::DenseTensorMeta beta_x_meta = {x.dtype(), x.dims()};
  beta_x.set_meta(beta_x_meta);
  dev_ctx.template Alloc<T>(&beta_x);
  phi::DenseTensorMeta sigmoid_out_meta = {x.dtype(), x.dims()};
  sigmoid_out.set_meta(sigmoid_out_meta);
  dev_ctx.template Alloc<T>(&sigmoid_out);
  phi::DenseTensorMeta swish_out_meta = {x.dtype(), x.dims()};
  swish_out.set_meta(swish_out_meta);
  dev_ctx.template Alloc<T>(&swish_out);

  const auto& muls_runner =
      NpuOpRunner("Muls", {x}, {beta_x}, {{"value", beta}});
  muls_runner.Run(stream);

  const auto& sigmoid_runner =
      NpuOpRunner("Sigmoid", {beta_x}, {sigmoid_out}, {});
  sigmoid_runner.Run(stream);

  const auto& mul_runner =
      NpuOpRunner("Mul", {sigmoid_out, x}, {swish_out}, {});
  mul_runner.Run(stream);
  const auto& muls_runner2 =
      NpuOpRunner("Muls", {swish_out}, {swish_out}, {{"value", beta}});
  muls_runner2.Run(stream);

  const auto& mul_runner1 =
      NpuOpRunner("Mul", {sigmoid_out, swish_out}, {*dx}, {});
  mul_runner1.Run(stream);

  const auto& sub_runner = NpuOpRunner("Sub", {swish_out, *dx}, {*dx}, {});
  sub_runner.Run(stream);

  const auto& add_runner = NpuOpRunner("Add", {sigmoid_out, *dx}, {*dx}, {});
  add_runner.Run(stream);

  const auto& mul_runner2 = NpuOpRunner("Mul", {dout, *dx}, {*dx}, {});
  mul_runner2.Run(stream);
}

template <typename T, typename Context>
void TanhKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("Tanh", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void TanhGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("TanhGrad", {out, dout}, {*dx}, {});
  runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(atan,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::AtanKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(atan_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::AtanGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(
    exp, ascend, ALL_LAYOUT, custom_kernel::ExpKernel, float, double) {}

PD_REGISTER_PLUGIN_KERNEL(
    exp_grad, ascend, ALL_LAYOUT, custom_kernel::ExpGradKernel, float, double) {
}

PD_REGISTER_PLUGIN_KERNEL(
    relu, ascend, ALL_LAYOUT, custom_kernel::ReluKernel, float, double) {}

PD_REGISTER_PLUGIN_KERNEL(relu_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::ReluGradKernel,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(leaky_relu,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::LeakyReluKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(leaky_relu_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::LeakyReluGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(sigmoid,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::SigmoidKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(sigmoid_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::SigmoidGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(sqrt,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::SqrtKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(sqrt_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::SqrtGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(square,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::SquareKernel,

PD_REGISTER_PLUGIN_KERNEL(square_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::SquareGradKernel,

PD_REGISTER_PLUGIN_KERNEL(swish,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::SwishKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(swish_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::SwishGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(tanh,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::TanhKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(tanh_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::TanhGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}
