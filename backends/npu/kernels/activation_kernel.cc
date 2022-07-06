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
void CosKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("Cos", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void CosGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  phi::DenseTensor sin_out;
  phi::DenseTensorMeta meta = {x.dtype(), x.dims()};
  sin_out.set_meta(meta);
  dev_ctx.template Alloc<T>(&sin_out);

  const auto& runner = NpuOpRunner("Sin", {x}, {sin_out}, {});
  runner.Run(stream);

  const auto& runner_dx = NpuOpRunner("Mul", {dout, sin_out}, {*dx}, {});
  runner_dx.Run(stream);

  phi::DenseTensor tmp;
  phi::DenseTensorMeta tmp_meta = {x.dtype(), {1, 1}};
  tmp.set_meta(tmp_meta);
  dev_ctx.template Alloc<T>(&tmp);
  float factor = -1.;
  FillNpuTensorWithConstant<T>(&tmp, dev_ctx, static_cast<T>(factor));

  const auto& runner_dx_ = NpuOpRunner("Xdivy", {*dx, tmp}, {*dx}, {});
  runner_dx_.Run(stream);
}

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
void GeluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                bool approximate,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("Gelu", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void GeluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& out_grad,
                    bool approximate,
                    phi::DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  auto stream = dev_ctx.stream();

  // NOTE(pangyoki): In the original implementation of GeluGrad op, the input
  // is {*dout, *x, out}, where out = Gelu(x). However, we find that variable
  // `out` was not actually used. In order to improve performance, the
  // useless GELU operation was deleted.
  // We directly use `*dout` as a placeholder to replace `out`, it will not
  // be used in calculations.
  const auto& runner_dx =
      NpuOpRunner("GeluGrad", {out_grad, x, out_grad}, {*x_grad}, {});
  runner_dx.Run(stream);
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
                    const phi::DenseTensor& out_grad,
                    phi::DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  auto stream = dev_ctx.stream();

  const auto& runner_dx =
      NpuOpRunner("TanhGrad", {out, out_grad}, {*x_grad}, {});
  runner_dx.Run(stream);
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
void LogKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  phi::DenseTensor one;
  phi::DenseTensorMeta one_meta = {x.dtype(), x.dims()};
  one.set_meta(one_meta);
  dev_ctx.template Alloc<T>(&one);
  const auto& runner_one = NpuOpRunner("OnesLike", {x}, {one}, {});
  runner_one.Run(stream);

  phi::DenseTensor sub;
  phi::DenseTensorMeta sub_meta = {x.dtype(), x.dims()};
  sub.set_meta(sub_meta);
  dev_ctx.template Alloc<T>(&sub);
  const auto& runner_sub = NpuOpRunner("Sub", {x, one}, {sub}, {});
  runner_sub.Run(stream);

  const auto& runner_out = NpuOpRunner("Log1p", {sub}, {*out}, {});
  runner_out.Run(stream);
}

template <typename T, typename Context>
void LogGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("DivNoNan", {dout, x}, {*dx}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void PowKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::Scalar& factor_scalar,
               phi::DenseTensor* out) {
  auto factor = factor_scalar.to<float>();
  dev_ctx.template Alloc<T>(out);

  const auto& runner = NpuOpRunner("Power",
                                   {x},
                                   {*out},
                                   {{"power", factor},
                                    {"scale", static_cast<float>(1.0)},
                                    {"shift", static_cast<float>(0.0)}});
  auto stream = dev_ctx.stream();

  runner.Run(stream);
}

template <typename T, typename Context>
void PowGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   const phi::Scalar& factor_scalar,
                   phi::DenseTensor* dx) {
  auto factor = factor_scalar.to<float>();

  auto x_dims = x.dims();

  auto stream = dev_ctx.stream();

  // dx = dout * factor * x.pow(factor-1)

  // Step1: Compute x_pow = x.pow(factor-1)
  phi::DenseTensor x_pow;
  phi::DenseTensorMeta x_pow_meta = {x.dtype(), x_dims};
  x_pow.set_meta(x_pow_meta);
  dev_ctx.template Alloc<T>(&x_pow);

  const auto& runner_pow = NpuOpRunner(
      "Power", {x}, {x_pow}, {{"power", factor - static_cast<float>(1)}});
  runner_pow.Run(stream);

  // Step 2: Construct a broadcast factor, which has the same shape with x.

  // 2.1 Get a factor tensor with shape [1].
  phi::DenseTensor factor_tensor;
  phi::DenseTensorMeta factor_tensor_meta = {x.dtype(), {1}};
  factor_tensor.set_meta(factor_tensor_meta);
  dev_ctx.template Alloc<T>(&factor_tensor);

  FillNpuTensorWithConstant<T>(&factor_tensor, dev_ctx, static_cast<T>(factor));

  // 2.2 Get the factor which has the shape with x and the same value with
  // factor.
  phi::DenseTensor factor_bc_tensor;
  phi::DenseTensorMeta factor_bc_tensor_meta = {x.dtype(), x_dims};
  factor_bc_tensor.set_meta(factor_bc_tensor_meta);
  dev_ctx.template Alloc<T>(&factor_bc_tensor);
  const auto& runner_bc = NpuOpRunner("FillD",
                                      {factor_tensor},
                                      {factor_bc_tensor},
                                      {{"dims", phi::vectorize(x_dims)}});
  runner_bc.Run(stream);

  // Step 3: Compute x_power_mul_factor = factor * x.pow(factor-1)
  phi::DenseTensor x_power_mul_factor;
  phi::DenseTensorMeta x_power_mul_factor_meta = {x.dtype(), x_dims};
  x_power_mul_factor.set_meta(x_power_mul_factor_meta);
  dev_ctx.template Alloc<T>(&x_power_mul_factor);
  const auto& runner_mul_1 =
      NpuOpRunner("Mul", {factor_bc_tensor, x_pow}, {x_power_mul_factor}, {});
  runner_mul_1.Run(stream);

  // Step 4: Compute dx = dout * factor * x.pow(factor-1)
  dev_ctx.template Alloc<T>(dx);
  const auto& runner_mul_2 =
      NpuOpRunner("Mul", {dout, x_power_mul_factor}, {*dx}, {});
  runner_mul_2.Run(stream);
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
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cos,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::CosKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(cos_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::CosGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(atan,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::AtanKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(atan_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::AtanGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(
    exp, ascend, ALL_LAYOUT, custom_kernel::ExpKernel, float, double) {}

PD_REGISTER_PLUGIN_KERNEL(
    exp_grad, ascend, ALL_LAYOUT, custom_kernel::ExpGradKernel, float, double) {
}

PD_REGISTER_PLUGIN_KERNEL(relu,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::ReluKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::ReluGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

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

PD_REGISTER_PLUGIN_KERNEL(pow,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::PowKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(pow_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::PowGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(log,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::LogKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(log_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::LogGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gelu,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::GeluKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gelu_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::GeluGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tanh,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::TanhKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tanh_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::TanhGradKernel,
                          float,
                          phi::dtype::float16) {}

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
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(square_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::SquareGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}
