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

#include <iostream>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void ReluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA ReluKernel";
  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doActivationForward(dev_ctx,
                                x,
                                0.,
                                ActivationMode::relu,
                                NanPropagation::not_propagate_nan,
                                out);
}

template <typename T, typename Context>
void ReluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA ReluGradKernel";
  dev_ctx.template Alloc<T>(dx);
  sdaa_ops::doActivationBackward(dev_ctx,
                                 out,
                                 dout,
                                 0.,
                                 ActivationMode::relu,
                                 NanPropagation::not_propagate_nan,
                                 dx);
}

template <typename T, typename Context>
void SigmoidKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA SigmoidKernel";
  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doActivationForward(dev_ctx,
                                x,
                                0.,
                                ActivationMode::sigmoid,
                                NanPropagation::not_propagate_nan,
                                out);
}

template <typename T, typename Context>
void SigmoidGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& dout,
                       phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA SigmoidGradKernel";
  dev_ctx.template Alloc<T>(dx);
  sdaa_ops::doActivationBackward(dev_ctx,
                                 out,
                                 dout,
                                 0.,
                                 ActivationMode::sigmoid,
                                 NanPropagation::not_propagate_nan,
                                 dx);
}

template <typename T, typename Context>
void TanhKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA TanhKernel";
  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doActivationForward(dev_ctx,
                                x,
                                0.,
                                ActivationMode::tanh,
                                NanPropagation::not_propagate_nan,
                                out);
}

template <typename T, typename Context>
void TanhGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA TanhGradKernel";
  dev_ctx.template Alloc<T>(dx);
  sdaa_ops::doActivationBackward(dev_ctx,
                                 out,
                                 dout,
                                 0.,
                                 ActivationMode::tanh,
                                 NanPropagation::not_propagate_nan,
                                 dx);
}

template <typename T, typename Context>
void ExpKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA ExpKernel";
  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doUnaryOpTensor(dev_ctx, x, 1.0, UnaryOpMode::EXP, out);
}

template <typename T, typename Context>
void ExpGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& out,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA ExpGradKernel";
  dev_ctx.template Alloc<T>(dx);

  sdaa_ops::doElementMul(dev_ctx, dout, out, -1, dx);
}

template <typename T, typename Context>
void GeluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                bool approximate,
                phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA GeluKernel";
  dev_ctx.template Alloc<T>(out);

  if (approximate) {
    sdaa_ops::doActivationForward(dev_ctx,
                                  x,
                                  0.,
                                  ActivationMode::gelu_approximate,
                                  NanPropagation::not_propagate_nan,
                                  out);
  } else {
    sdaa_ops::doActivationForward(dev_ctx,
                                  x,
                                  0.,
                                  ActivationMode::gelu,
                                  NanPropagation::not_propagate_nan,
                                  out);
  }
}

template <typename T, typename Context>
void GeluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& out_grad,
                    bool approximate,
                    phi::DenseTensor* x_grad) {
  VLOG(4) << "Call SDAA GeluGradKernel";
  dev_ctx.template Alloc<T>(x_grad);
  if (approximate) {
    sdaa_ops::doActivationBackward(dev_ctx,
                                   out,
                                   out_grad,
                                   0.,
                                   ActivationMode::gelu_approximate,
                                   NanPropagation::not_propagate_nan,
                                   x_grad);
  } else {
    sdaa_ops::doActivationBackward(dev_ctx,
                                   out,
                                   out_grad,
                                   0.,
                                   ActivationMode::gelu,
                                   NanPropagation::not_propagate_nan,
                                   x_grad);
  }
}

template <typename T, typename Context>
void LeakyReluKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     float alpha,
                     phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA LeakyReluKernel";
  dev_ctx.template Alloc<T>(out);
  double alp = alpha;
  sdaa_ops::doActivationForward(dev_ctx,
                                x,
                                alp,
                                ActivationMode::leaky_relu,
                                NanPropagation::not_propagate_nan,
                                out);
}

template <typename T, typename Context>
void LeakyReluGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& dout,
                         float alpha,
                         phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA LeakyReluGradKernel";
  dev_ctx.template Alloc<T>(dx);
  double alp = alpha;
  sdaa_ops::doActivationBackward(dev_ctx,
                                 x,
                                 dout,
                                 alp,
                                 ActivationMode::leaky_relu,
                                 NanPropagation::not_propagate_nan,
                                 dx);
}

template <typename T, typename Context>
void SqrtGrad(const Context& dev_ctx,
              const phi::DenseTensor& out,
              const phi::DenseTensor& dout,
              phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  // dx = 0.5 * dout / out
  sdaa_ops::doElementDiv(dev_ctx, dout, out, -1, dx);

  float alpha = 0.5;
  sdaa_ops::doScaleTensor(dev_ctx, *dx, alpha, 0.0, true, false, dx);
}

template <typename T, typename Context>
void SqrtKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA SqrtKernel";
  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doUnaryOpTensor(dev_ctx, x, 1.0, UnaryOpMode::SQRT, out);
}

template <typename T, typename Context>
void SqrtGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA SqrtGradKernel";
  SqrtGrad<T>(dev_ctx, out, dout, dx);
}

template <typename T, typename Context>
void RsqrtGrad(const Context& dev_ctx,
               const phi::DenseTensor& out,
               const phi::DenseTensor& dout,
               phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  // dx = -0.5 * dout * out * out * out
  float alpha = -0.5f;
  sdaa_ops::doScaleTensor(dev_ctx, dout, alpha, 0.0, false, false, dx);
  sdaa_ops::doElementMul(dev_ctx, *dx, out, -1, dx);
  sdaa_ops::doElementMul(dev_ctx, *dx, out, -1, dx);
  sdaa_ops::doElementMul(dev_ctx, *dx, out, -1, dx);
}

template <typename T, typename Context>
void RsqrtKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA RsqrtKernel";
  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doUnaryOpTensor(dev_ctx, x, 1.0, UnaryOpMode::RSQRT, out);
}

template <typename T, typename Context>
void RsqrtGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& out,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA RsqrtGradKernel";
  RsqrtGrad<T>(dev_ctx, out, dout, dx);
}

template <typename T, typename Context>
void PowKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::Scalar& factor_scalar,
               phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA PowKernel";
  auto factor = factor_scalar.to<float>();
  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doUnaryOpTensor(dev_ctx, x, factor, UnaryOpMode::POW, out);
}

template <typename T, typename Context>
void PowGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   const phi::Scalar& factor_scalar,
                   phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA PowGradKernel";
  auto factor = factor_scalar.to<float>();
  auto x_dims = x.dims();

  // dx = dout * factor * x.pow(factor - 1)
  // step 1: compute x_pow = x.pow(factor - 1)
  phi::DenseTensor x_pow;
  phi::DenseTensorMeta x_pow_meta = {x.dtype(), x_dims};
  x_pow.set_meta(x_pow_meta);
  dev_ctx.template Alloc<T>(&x_pow);
  float factor_x_pow = factor - static_cast<float>(1);
  sdaa_ops::doUnaryOpTensor(dev_ctx, x, factor_x_pow, UnaryOpMode::POW, &x_pow);

  // step 2: compute x_power_mul_factor = factor * x.pow(factor - 1)
  sdaa_ops::doScaleTensor(dev_ctx, x_pow, factor, 0.0, true, false, &x_pow);

  // step 3: compute dx = dout * factor * x.pow(factor - 1)
  dev_ctx.template Alloc<T>(dx);
  sdaa_ops::doElementMul(dev_ctx, dout, x_pow, -1, dx);
}

template <typename T, typename Context>
void LogKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA LogKernel";
  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doUnaryOpTensor(dev_ctx, x, 1.0, UnaryOpMode::LOG, out);
}

template <typename T, typename Context>
void LogGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA LogGradKernel";
  dev_ctx.template Alloc<T>(dx);
  phi::DenseTensor dx_temp;
  dx_temp.Resize(dx->dims());
  dev_ctx.template Alloc<T>(&dx_temp);

  sdaa_ops::doUnaryOpTensor(dev_ctx, x, 1.0, UnaryOpMode::RDIV, &dx_temp);

  sdaa_ops::doElementMul(dev_ctx, dx_temp, dout, -1, dx);
}

template <typename T, typename Context>
void ReciprocalKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA ReciprocalKernel";
  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doReciprocalTensor(dev_ctx, x, out);
}

template <typename T, typename Context>
void ReciprocalGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& out,
                          const phi::DenseTensor& dout,
                          phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA ReciprocalGradKernel";

  dev_ctx.template Alloc<T>(dx);
  phi::DenseTensor out_temp;
  out_temp.Resize(out.dims());
  dev_ctx.template Alloc<T>(&out_temp);

  sdaa_ops::doElementMul(dev_ctx, out, out, -1, &out_temp);
  sdaa_ops::doElementMul(dev_ctx, out_temp, dout, -1, dx);
  sdaa_ops::doNegTensor(dev_ctx, *dx, dx);
}

template <typename T, typename Context>
void SiluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA SiluKernel";
  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doActivationForward(dev_ctx,
                                x,
                                0.,
                                ActivationMode::silu,
                                NanPropagation::not_propagate_nan,
                                out);
}

template <typename T, typename Context>
void SiluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  VLOG(4) << "CALL SDAA SiluGradKernel";
  dev_ctx.template Alloc<T>(dx);
  sdaa_ops::doActivationBackward(dev_ctx,
                                 x,
                                 dout,
                                 0.,
                                 ActivationMode::silu,
                                 NanPropagation::not_propagate_nan,
                                 dx);
}

template <typename T, typename Context>
void doHardSwish(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  float threshold = 6;
  float scale = 6;
  float offset = 3;

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());

  phi::DenseTensor* x_ = const_cast<phi::DenseTensor*>(&x);

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  TECODNN_CHECK(tecodnnHardSwishForward(tecodnnHandle,
                                        threshold,
                                        scale,
                                        offset,
                                        Desc,
                                        x_->data(),
                                        Desc,
                                        out->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(Desc));
}

template <typename T, typename Context>
void doHardSwishGrad(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  float threshold = 6;
  float scale = 6;
  float offset = 3;

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  phi::DenseTensor* x_ = const_cast<phi::DenseTensor*>(&x);
  phi::DenseTensor* dout_ = const_cast<phi::DenseTensor*>(&dout);

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  TECODNN_CHECK(tecodnnHardSwishBackward(tecodnnHandle,
                                         threshold,
                                         scale,
                                         offset,
                                         Desc,
                                         dout_->data(),
                                         Desc,
                                         x_->data(),
                                         Desc,
                                         dx->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(Desc));
}

template <typename T, typename Context>
void HardSwishKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA HardSwishKernel";

  dev_ctx.template Alloc<T>(out);

  custom_kernel::doHardSwish<T, Context>(dev_ctx, x, out);
}

template <typename T, typename Context>
void HardSwishGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& dout,
                         phi::DenseTensor* dx) {
  VLOG(4) << "CALL SDAA HardSwishGradKernel";
  dev_ctx.template Alloc<T>(dx);

  custom_kernel::doHardSwishGrad<T, Context>(dev_ctx, x, dout, dx);
}

template <typename T, typename Context>
void HardSigmoidKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       float slope,
                       float offset,
                       phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA HardSigmoidKernel";

  dev_ctx.template Alloc<T>(out);

  std::vector<int> dims = phi::vectorize<int>(x.dims());

  tecodnnHandle_t handle = GetHandleFromCTX(dev_ctx);

  tecodnnTensorDescriptor_t desc =
      sdaa_ops::GetTecodnnTensorDesc(dims, x.dtype(), TensorFormat::Undefined);

  TECODNN_CHECK(tecodnnHardSigmoidForward(
      handle, slope, offset, desc, x.data(), desc, out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(desc));
}

template <typename T, typename Context>
void HardSigmoidGradKernel(const Context& dev_ctx,
                           const phi::DenseTensor& out,
                           const phi::DenseTensor& dout,
                           float slope,
                           float offset,
                           phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA HardSigmoidGradKernel";

  dev_ctx.template Alloc<T>(dx);

  std::vector<int> dims = phi::vectorize<int>(out.dims());

  tecodnnHandle_t handle = GetHandleFromCTX(dev_ctx);

  tecodnnTensorDescriptor_t desc = sdaa_ops::GetTecodnnTensorDesc(
      dims, out.dtype(), TensorFormat::Undefined);

  TECODNN_CHECK(tecodnnHardSigmoidBackward(handle,
                                           slope,
                                           offset,
                                           desc,
                                           const_cast<void*>(dout.data()),
                                           desc,
                                           const_cast<void*>(out.data()),
                                           desc,
                                           dx->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(desc));
}

template <typename T, typename Context>
void SoftsignKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA SoftsignKernel";

  int N = 1, C = x.numel(), H = 1, W = 1;
  std::vector<int> dims = {N, H, W, C};

  dev_ctx.template Alloc<T>(out);

  tecodnnHandle_t handle = GetHandleFromCTX(dev_ctx);

  tecodnnTensorDescriptor_t desc =
      sdaa_ops::GetTecodnnTensorDesc(dims, x.dtype(), TensorFormat::NHWC);

  const float alpha = 1.0f, beta = 0.0f;
  TECODNN_CHECK(tecodnnSoftsignForward(
      handle, &alpha, desc, x.data(), &beta, desc, out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(desc));
}

template <typename T, typename Context>
void SoftsignGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& dout,
                        phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA SoftsignGradKernel";

  int N = 1, C = x.numel(), H = 1, W = 1;
  std::vector<int> dims = {N, H, W, C};

  dev_ctx.template Alloc<T>(dx);

  tecodnnHandle_t handle = GetHandleFromCTX(dev_ctx);

  tecodnnTensorDescriptor_t desc =
      sdaa_ops::GetTecodnnTensorDesc(dims, x.dtype(), TensorFormat::NHWC);

  const float alpha = 1.0f, beta = 0.0f;
  TECODNN_CHECK(tecodnnSoftsignBackward(handle,
                                        &alpha,
                                        desc,
                                        dout.data(),
                                        desc,
                                        x.data(),
                                        &beta,
                                        desc,
                                        dx->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(desc));
}

template <typename T, typename Context>
void SoftplusKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    float beta,
                    float threshold,
                    phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA SoftplusKernel";

  int N = 1, C = x.numel(), H = 1, W = 1;
  std::vector<int> dims = {N, H, W, C};

  dev_ctx.template Alloc<T>(out);

  tecodnnHandle_t handle = GetHandleFromCTX(dev_ctx);

  tecodnnTensorDescriptor_t desc =
      sdaa_ops::GetTecodnnTensorDesc(dims, x.dtype(), TensorFormat::NHWC);

  const float coef = beta;
  const float alpha = 1.0f, beta_ = 0.0f;
  TECODNN_CHECK(tecodnnSoftplusForward(handle,
                                       &threshold,
                                       &coef,
                                       &alpha,
                                       desc,
                                       x.data(),
                                       &beta_,
                                       desc,
                                       out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(desc));
}

template <typename T, typename Context>
void SoftplusGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& dout,
                        float beta,
                        float threshold,
                        phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA SoftplusGradKernel";

  int N = 1, C = x.numel(), H = 1, W = 1;
  std::vector<int> dims = {N, H, W, C};

  dev_ctx.template Alloc<T>(dx);

  tecodnnHandle_t handle = GetHandleFromCTX(dev_ctx);

  tecodnnTensorDescriptor_t desc =
      sdaa_ops::GetTecodnnTensorDesc(dims, x.dtype(), TensorFormat::NHWC);

  const float coef = beta;
  const float alpha = 1.0f, beta_ = 0.0f;
  TECODNN_CHECK(tecodnnSoftplusBackward(handle,
                                        &threshold,
                                        &coef,
                                        &alpha,
                                        desc,
                                        dout.data(),
                                        desc,
                                        x.data(),
                                        &beta_,
                                        desc,
                                        dx->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(desc));
}

template <typename T, typename Context>
void SinKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA SinKernel";
  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doUnaryOpTensor(dev_ctx, x, 0.0, UnaryOpMode::SIN, out);
}

template <typename T, typename Context>
void SinGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA SinGradKernel";
  phi::DenseTensor x_cos;
  phi::DenseTensorMeta x_cos_meta = {x.dtype(), x.dims()};
  x_cos.set_meta(x_cos_meta);

  dev_ctx.template Alloc<T>(&x_cos);
  sdaa_ops::doUnaryOpTensor(dev_ctx, x, 0.0, UnaryOpMode::COS, &x_cos);

  dev_ctx.template Alloc<T>(dx);
  sdaa_ops::doElementMul(dev_ctx, dout, x_cos, -1, dx);
}

template <typename T, typename Context>
void CosKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA CosKernel";
  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doUnaryOpTensor(dev_ctx, x, 0.0, UnaryOpMode::COS, out);
}

template <typename T, typename Context>
void CosGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA CosGradKernel";
  phi::DenseTensor x_sin;
  phi::DenseTensorMeta x_sin_meta = {x.dtype(), x.dims()};
  x_sin.set_meta(x_sin_meta);

  dev_ctx.template Alloc<T>(&x_sin);
  sdaa_ops::doUnaryOpTensor(dev_ctx, x, 0.0, UnaryOpMode::SIN, &x_sin);

  sdaa_ops::doNegTensor(dev_ctx, x_sin, &x_sin);

  dev_ctx.template Alloc<T>(dx);
  sdaa_ops::doElementMul(dev_ctx, dout, x_sin, -1, dx);
}

template <typename T, typename Context>
void SquareKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA SquareKernel";
  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doUnaryOpTensor(dev_ctx, x, 0.0, UnaryOpMode::SQUARE, out);
}

template <typename T, typename Context>
void SquareGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& dout,
                      phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA SquareGradKernel";
  phi::DenseTensor double_x;
  double_x.set_meta(x.meta());
  dev_ctx.template Alloc<T>(&double_x);
  sdaa_ops::doUnaryOpTensor(dev_ctx, x, 2.0, UnaryOpMode::MUL_A, &double_x);

  dev_ctx.template Alloc<T>(dx);
  sdaa_ops::doElementMul(dev_ctx, dout, double_x, -1, dx);
}

template <typename T, typename Context>
void AtanKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA AtanKernel";
  dev_ctx.template Alloc<T>(out);

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  TECODNN_CHECK(
      tecodnnAtanTensor(tecodnnHandle, x_Desc, x.data(), x_Desc, out->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
}

// dx = dout * 1 / (1 + x.pow(2))
template <typename T, typename Context>
void AtanGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA AtanGradKernel";
  // Step1: Compute x_pow = x.pow(2)
  phi::DenseTensor x_pow;
  x_pow.Resize(x.dims());
  dev_ctx.template Alloc<T>(&x_pow);
  sdaa_ops::doUnaryOpTensor(dev_ctx, x, 0.0, UnaryOpMode::SQUARE, &x_pow);

  // Step2: x_pow_1 = x_pow + 1
  phi::DenseTensor x_pow_1;
  x_pow_1.Resize(x.dims());
  dev_ctx.template Alloc<T>(&x_pow_1);
  sdaa_ops::doUnaryOpTensor(dev_ctx, x_pow, 1.0, UnaryOpMode::ADD_A, &x_pow_1);

  // Step3: dx = dout / x_pow_1
  dev_ctx.template Alloc<T>(dx);
  sdaa_ops::doElementDiv(dev_ctx, dout, x_pow_1, -1, dx);
}

template <typename T, typename Context>
void CeilKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA CeilKernel";
  dev_ctx.template Alloc<T>(out);

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  TECODNN_CHECK(
      tecodnnCeil(tecodnnHandle, x_Desc, x.data(), x_Desc, out->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
}

template <typename T, typename Context>
void SwishRawKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    float beta,
                    phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA SwishRawKernel";
  dev_ctx.template Alloc<T>(out);

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  TECODNN_CHECK(tecodnnSwishForward(
      tecodnnHandle, beta, x_Desc, x.data(), x_Desc, out->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
}

template <typename T, typename Context>
void SwishKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA SwishKernel";
  custom_kernel::SwishRawKernel<T, Context>(dev_ctx, x, 1.0, out);
}

template <typename T, typename Context>
void SwishGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA SwishGradKernel";
  dev_ctx.template Alloc<T>(dx);

  float beta = 1.0;
  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  TECODNN_CHECK(tecodnnSwishBackward(tecodnnHandle,
                                     beta,
                                     x_Desc,
                                     x.data(),
                                     x_Desc,
                                     dout.data(),
                                     x_Desc,
                                     dx->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
}

template <typename T, typename Context>
void FloorKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA FloorKernel";
  dev_ctx.template Alloc<T>(out);

  sdaa_ops::doUnaryOpTensor(dev_ctx, x, 0.0, UnaryOpMode::FLOOR, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(relu,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::ReluKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::ReluGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(sigmoid,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SigmoidKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(sigmoid_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SigmoidGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tanh,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::TanhKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tanh_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::TanhGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(exp,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::ExpKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(exp_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::ExpGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gelu,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::GeluKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gelu_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::GeluGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(leaky_relu,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::LeakyReluKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(leaky_relu_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::LeakyReluGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(sqrt,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SqrtKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(sqrt_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SqrtGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(pow,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::PowKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(pow_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::PowGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(log,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::LogKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(log_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::LogGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(reciprocal,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::ReciprocalKernel,
                          double,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(reciprocal_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::ReciprocalGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(silu,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SiluKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(silu_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SiluGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hardswish,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::HardSwishKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hardswish_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::HardSwishGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hardsigmoid,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::HardSigmoidKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hardsigmoid_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::HardSigmoidGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(softsign,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SoftsignKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(softsign_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SoftsignGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(softplus,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SoftplusKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(softplus_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SoftplusGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(sin,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SinKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(sin_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SinGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(cos,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::CosKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(cos_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::CosGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(square,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SquareKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(square_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SquareGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(atan,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::AtanKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(atan_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::AtanGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(ceil,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::CeilKernel,
                          double,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(swish,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SwishKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(swish_raw,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SwishRawKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(swish_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SwishGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(rsqrt,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::RsqrtKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(rsqrt_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::RsqrtGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(floor,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::FloorKernel,
                          float,
                          phi::dtype::float16) {}
