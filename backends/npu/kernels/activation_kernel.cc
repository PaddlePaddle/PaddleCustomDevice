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
#include "kernels/funcs/op_command.h"

namespace custom_kernel {

template <typename T, typename Context>
void CosKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  experimental::OpCommand("Cos").Input(x).Output(*out).Run(dev_ctx);
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
  experimental::OpCommand("Atan").Input(x).Output(*out).Run(dev_ctx);
}

template <typename T, typename Context>
void AtanGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  experimental::OpCommand("AtanGrad")
      .Input(x)
      .Input(dout)
      .Output(*dx)
      .Run(dev_ctx);
}

template <typename T, typename Context>
void ExpKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  experimental::OpCommand("Exp").Input(x).Output(*out).Run(dev_ctx);
}

template <typename T, typename Context>
void ExpGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& out,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  experimental::OpCommand("Mul").Input(dout).Input(out).Output(*dx).Run(
      dev_ctx);
}

template <typename T, typename Context>
void FloorKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("Floor", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void FloorGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();
  const auto& runner =
      NpuOpRunner("Fills", {*dx}, {*dx}, {{"value", static_cast<float>(0)}});
  runner.Run(stream);
}

template <typename T, typename Context>
void SinKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  experimental::OpCommand("Sin").Input(x).Output(*out).Run(dev_ctx);
}

// Swish = x * sigmoid(beta * x)
template <typename T, typename Context>
void SwishRawKernel(const Context& dev_ctx,
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
void SwishKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  custom_kernel::SwishRawKernel<T, Context>(dev_ctx, x, 1.0, out);
}

template <typename T, typename Context>
void SwishGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& dout,
                     float beta,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  phi::DenseTensor beta_x, sigmoid_out, swish_out;
  beta_x.Resize(x.dims());
  sigmoid_out.Resize(x.dims());
  swish_out.Resize(x.dims());
  dev_ctx.template Alloc<T>(&beta_x);
  dev_ctx.template Alloc<T>(&sigmoid_out);
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
      NpuOpRunner("Mul", {sigmoid_out, swish_out}, {*out}, {});
  mul_runner1.Run(stream);

  const auto& sub_runner = NpuOpRunner("Sub", {swish_out, *out}, {*out}, {});
  sub_runner.Run(stream);

  const auto& add_runner = NpuOpRunner("Add", {sigmoid_out, *out}, {*out}, {});
  add_runner.Run(stream);

  const auto& mul_runner2 = NpuOpRunner("Mul", {dout, *out}, {*out}, {});
  mul_runner2.Run(stream);
}

template <typename T, typename Context>
void ReluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  experimental::OpCommand("Relu").Input(x).Output(*out).Run(dev_ctx);
}

template <typename T, typename Context>
void ReluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  experimental::OpCommand("ReluGrad")
      .Input(dout)
      .Input(out)
      .Output(*dx)
      .Run(dev_ctx);
}

template <typename T, typename Context>
void Relu6RawKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    float attr,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  experimental::OpCommand("Relu6").Input(x).Output(*out).Run(dev_ctx);
}

template <typename T, typename Context>
void Relu6Kernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  custom_kernel::Relu6RawKernel<T, Context>(dev_ctx, x, 6.0, out);
}

template <typename T, typename Context>
void Relu6GradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& out,
                     const phi::DenseTensor& dout,
                     float attr,
                     phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  experimental::OpCommand("Relu6Grad")
      .Input(dout)
      .Input(out)
      .Output(*dx)
      .Run(dev_ctx);
}

template <typename T, typename Context>
void LeakyReluKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     float alpha,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  experimental::OpCommand("LeakyRelu")
      .Input(x)
      .Output(*out)
      .Attr("negative_slope", alpha)
      .Run(dev_ctx);
}

template <typename T, typename Context>
void LeakyReluGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& dout,
                         float alpha,
                         phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  experimental::OpCommand("LeakyReluGrad")
      .Input(dout)
      .Input(x)
      .Output(*dx)
      .Attr("negative_slope", alpha)
      .Run(dev_ctx);
}

template <typename T, typename Context>
void GeluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                bool approximate,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  experimental::OpCommand("Gelu").Input(x).Output(*out).Run(dev_ctx);
}

template <typename T, typename Context>
void GeluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& out_grad,
                    bool approximate,
                    phi::DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);

  // NOTE(pangyoki): In the original implementation of GeluGrad op, the input
  // is {*dout, *x, out}, where out = Gelu(x). However, we find that variable
  // `out` was not actually used. In order to improve performance, the
  // useless GELU operation was deleted.
  // We directly use `*dout` as a placeholder to replace `out`, it will not
  // be used in calculations.
  experimental::OpCommand("GeluGrad")
      .Input(out_grad)
      .Input(x)
      .Input(out_grad)
      .Output(*x_grad)
      .Run(dev_ctx);
}

template <typename T, typename Context>
void TanhKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  experimental::OpCommand("Tanh").Input(x).Output(*out).Run(dev_ctx);
}

template <typename T, typename Context>
void TanhGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& out_grad,
                    phi::DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  experimental::OpCommand("TanhGrad")
      .Input(out)
      .Input(out_grad)
      .Output(*x_grad)
      .Run(dev_ctx);
}

template <typename T, typename Context>
void SigmoidKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  experimental::OpCommand("Sigmoid").Input(x).Output(*out).Run(dev_ctx);
}

template <typename T, typename Context>
void SigmoidGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& dout,
                       phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  experimental::OpCommand("SigmoidGrad")
      .Input(out)
      .Input(dout)
      .Output(*dx)
      .Run(dev_ctx);
}

template <typename T, typename Context>
void SqrtKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  experimental::OpCommand("Sqrt").Input(x).Output(*out).Run(dev_ctx);
}

template <typename T, typename Context>
void SqrtGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  experimental::OpCommand("SqrtGrad")
      .Input(out)
      .Input(dout)
      .Output(*dx)
      .Run(dev_ctx);
}

template <typename T, typename Context>
void LogKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  experimental::OpCommand("Log").Input(x).Output(*out).Run(dev_ctx);
}

template <typename T, typename Context>
void LogGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  experimental::OpCommand("DivNoNan")
      .Input(dout)
      .Input(x)
      .Output(*dx)
      .Run(dev_ctx);
}

template <typename T, typename Context>
void PowKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::Scalar& factor_scalar,
               phi::DenseTensor* out) {
  auto factor = factor_scalar.to<float>();
  experimental::OpCommand("Power")
      .Input(x)
      .Output(*out)
      .Attr("power", factor)
      .Attr("scale", 1.0f)
      .Attr("scale", 0.0f)
      .Run(dev_ctx);
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
  experimental::OpCommand("Square").Input(x).Output(*out).Run(dev_ctx);
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

template <typename T, typename Context>
void HardSigmoidKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       float slope,
                       float offset,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  NPUAttributeMap attr_input = {{"alpha", slope}, {"beta", offset}};
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("HardSigmoid", {x}, {*out}, attr_input);
  runner.Run(stream);
}

template <typename T, typename Context>
void HardSigmoidGradKernel(const Context& dev_ctx,
                           const phi::DenseTensor& out,
                           const phi::DenseTensor& dout,
                           float slope,
                           float offset,
                           phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  experimental::OpCommand("HardSigmoidGrad")
      .Input(dout)
      .Input(out)
      .Output(*dx)
      .Attr("alpha", slope)
      .Attr("beta", offset)
      .Run(dev_ctx);
}

template <typename T, typename Context>
void HardSwishRawKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        float threshold,
                        float scale,
                        float offset,
                        phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  phi::DenseTensorMeta meta_1 = {x.dtype(), {1}};
  phi::DenseTensorMeta meta_x = {x.dtype(), x.dims()};
  phi::DenseTensor tensor_offset;
  tensor_offset.set_meta(meta_1);
  dev_ctx.template Alloc<T>(&tensor_offset);
  FillNpuTensorWithConstant<T>(&tensor_offset, dev_ctx, static_cast<T>(offset));

  phi::DenseTensor add_offset_val;
  add_offset_val.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&add_offset_val);
  const auto& runner_add =
      NpuOpRunner("AddV2", {x, tensor_offset}, {add_offset_val});
  runner_add.Run(stream);

  phi::DenseTensor tensor_threshold;
  tensor_threshold.set_meta(meta_1);
  dev_ctx.template Alloc<T>(&tensor_threshold);
  FillNpuTensorWithConstant<T>(
      &tensor_threshold, dev_ctx, static_cast<T>(threshold));

  phi::DenseTensor tensor_zero;
  tensor_zero.set_meta(meta_1);
  dev_ctx.template Alloc<T>(&tensor_zero);
  FillNpuTensorWithConstant<T>(&tensor_zero, dev_ctx, static_cast<T>(0.0));

  phi::DenseTensor clip_val;
  clip_val.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&clip_val);
  const auto& runner_clip =
      NpuOpRunner("ClipByValue",
                  {add_offset_val, tensor_zero, tensor_threshold},
                  {clip_val});
  runner_clip.Run(stream);

  phi::DenseTensor tensor_scale_tmp;
  tensor_scale_tmp.set_meta(meta_1);
  dev_ctx.template Alloc<T>(&tensor_scale_tmp);
  FillNpuTensorWithConstant<T>(
      &tensor_scale_tmp, dev_ctx, static_cast<T>(scale));
  phi::DenseTensor tensor_scale;
  tensor_scale.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&tensor_scale);
  const auto& runner_fill = NpuOpRunner("FillD",
                                        {tensor_scale_tmp},
                                        {tensor_scale},
                                        {{"dims", phi::vectorize(x.dims())}});
  runner_fill.Run(stream);

  phi::DenseTensor div_val;
  div_val.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&div_val);
  const auto& runner_div =
      NpuOpRunner("Div", {clip_val, tensor_scale}, {div_val});
  runner_div.Run(stream);

  const auto& runner_mul = NpuOpRunner("Mul", {x, div_val}, {*out});
  runner_mul.Run(stream);
}

template <typename T, typename Context>
void HardSwishKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     phi::DenseTensor* out) {
  custom_kernel::HardSwishRawKernel<T, Context>(dev_ctx, x, 6, 6, 3, out);
}

template <typename T, typename Context>
void HardSwishGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& dout,
                         float threshold,
                         float scale,
                         float offset,
                         phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  phi::DenseTensorMeta meta_1 = {x.dtype(), {1}};
  phi::DenseTensorMeta meta_x = {x.dtype(), x.dims()};
  phi::DenseTensor tensor_offset;
  tensor_offset.set_meta(meta_1);
  dev_ctx.template Alloc<T>(&tensor_offset);
  FillNpuTensorWithConstant<T>(&tensor_offset, dev_ctx, static_cast<T>(offset));

  phi::DenseTensor add_offset_val;
  add_offset_val.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&add_offset_val);
  const auto& runner_add =
      NpuOpRunner("AddV2", {x, tensor_offset}, {add_offset_val});
  runner_add.Run(stream);

  phi::DenseTensor tmp1;
  tmp1.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&tmp1);
  const auto& runner_pow1 =
      NpuOpRunner("Power", {x}, {tmp1}, {{"scale", 2.0f}, {"shift", offset}});
  runner_pow1.Run(stream);

  phi::DenseTensor tmp2;
  tmp2.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&tmp2);
  const auto& runner_ht_grad =
      NpuOpRunner("HardtanhGrad",
                  {add_offset_val, tmp1},
                  {tmp2},
                  {{"min_val", 0.0f}, {"max_val", threshold}});
  runner_ht_grad.Run(stream);

  phi::DenseTensor tmp3;
  tmp3.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&tmp3);
  const auto& runner_pow2 = NpuOpRunner(
      "Power", {tmp2}, {tmp3}, {{"scale", 1.0f / scale}, {"shift", 1.0f}});
  runner_pow2.Run(stream);

  phi::DenseTensor tensor_threshold_tmp;
  tensor_threshold_tmp.set_meta(meta_1);
  dev_ctx.template Alloc<T>(&tensor_threshold_tmp);
  FillNpuTensorWithConstant<T>(
      &tensor_threshold_tmp, dev_ctx, static_cast<T>(threshold));
  phi::DenseTensor tensor_threshold;
  tensor_threshold.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&tensor_threshold);
  const auto& runner_fill = NpuOpRunner("FillD",
                                        {tensor_threshold_tmp},
                                        {tensor_threshold},
                                        {{"dims", phi::vectorize(x.dims())}});
  runner_fill.Run(stream);

  phi::DenseTensor tmp_bool;
  phi::DenseTensorMeta meta_tmp = {phi::DataType::BOOL, x.dims()};
  tmp_bool.set_meta(meta_tmp);
  dev_ctx.template Alloc<bool>(&tmp_bool);
  const auto& runner_less =
      NpuOpRunner("Less", {add_offset_val, tensor_threshold}, {tmp_bool});
  runner_less.Run(stream);
  phi::DenseTensor tmp4;
  tmp4.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&tmp4);
  auto dst_dtype = ConvertToNpuDtype(x.dtype());
  const auto& runner_cast = NpuOpRunner(
      "Cast", {tmp_bool}, {tmp4}, {{"dst_type", static_cast<int>(dst_dtype)}});
  runner_cast.Run(stream);

  phi::DenseTensor tmp5;
  tmp5.set_meta(meta_x);
  dev_ctx.template Alloc<T>(&tmp5);
  const auto& runner_sub = NpuOpRunner("Sub", {tmp3, tmp4}, {tmp5});
  runner_sub.Run(stream);

  const auto& runner_final = NpuOpRunner("Mul", {tmp5, dout}, {*dx});
  runner_final.Run(stream);
}

template <typename T, typename Context>
void ReciprocalKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("Reciprocal", {x}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void ReciprocalGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& out,
                          const phi::DenseTensor& dout,
                          phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  experimental::OpCommand("ReciprocalGrad")
      .Input(out)
      .Input(dout)
      .Output(*dx)
      .Run(dev_ctx);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cos,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CosKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(cos_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CosGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(atan,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AtanKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(atan_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AtanGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(
    exp, npu, ALL_LAYOUT, custom_kernel::ExpKernel, float, double) {}

PD_REGISTER_PLUGIN_KERNEL(
    exp_grad, npu, ALL_LAYOUT, custom_kernel::ExpGradKernel, float, double) {}

PD_REGISTER_PLUGIN_KERNEL(sin,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SinKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(swish,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SwishKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(swish_raw,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SwishRawKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(swish_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SwishGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ReluKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ReluGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu6,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::Relu6Kernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu6_raw,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::Relu6RawKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu6_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::Relu6GradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(leaky_relu,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LeakyReluKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(leaky_relu_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LeakyReluGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(pow,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::PowKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(pow_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::PowGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(log,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LogKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(log_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LogGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(floor,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FloorKernel,
                          double,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(floor_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FloorGradKernel,
                          double,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gelu,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::GeluKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gelu_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::GeluGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tanh,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::TanhKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tanh_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::TanhGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(sigmoid,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SigmoidKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(sigmoid_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SigmoidGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(sqrt,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SqrtKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(sqrt_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SqrtGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(square,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SquareKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(square_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SquareGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(hard_sigmoid,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::HardSigmoidKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hard_sigmoid_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::HardSigmoidGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hardswish,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::HardSwishKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hardswish_raw,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::HardSwishRawKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hardswish_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::HardSwishGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(reciprocal,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ReciprocalKernel,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(reciprocal_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ReciprocalGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {}
