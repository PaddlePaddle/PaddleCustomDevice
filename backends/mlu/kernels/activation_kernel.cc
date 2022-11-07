/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include "kernels/funcs/mlu_baseop.h"

namespace custom_kernel {

template <typename T, typename Context>
void ActivationKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      float alpha,
                      cnnlActivationMode_t act_mode,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlActivationDesc act_desc(act_mode, alpha);
  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);

  MLUCnnl::Active(dev_ctx,
                  act_desc.get(),
                  input_desc.get(),
                  GetBasePtr(&x),
                  output_desc.get(),
                  GetBasePtr(out));
}

template <typename T, typename Context>
void ActivationGradKernelV1(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            const phi::DenseTensor& dout,
                            float alpha,
                            cnnlActivationMode_t act_mode,
                            phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnlTensorDesc dx_desc(*dx);
  MLUCnnlActivationDesc act_desc(act_mode, alpha);
  MLUCnnl::ActiveGrad(dev_ctx,
                      act_desc.get(),
                      nullptr,
                      nullptr,
                      nullptr,
                      nullptr,
                      dout_desc.get(),
                      GetBasePtr(&dout),
                      x_desc.get(),
                      GetBasePtr(&x),
                      dx_desc.get(),
                      GetBasePtr(dx));
}

template <typename T, typename Context>
void ActivationGradKernelV2(const Context& dev_ctx,
                            const phi::DenseTensor& out,
                            const phi::DenseTensor& dout,
                            cnnlActivationMode_t act_mode,
                            phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  MLUCnnlTensorDesc out_desc(out);
  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnlTensorDesc dx_desc(*dx);
  MLUCnnlActivationDesc act_desc(act_mode, 1.0);
  MLUCnnl::ActiveGrad(dev_ctx,
                      act_desc.get(),
                      nullptr,
                      nullptr,
                      out_desc.get(),
                      GetBasePtr(&out),
                      dout_desc.get(),
                      GetBasePtr(&dout),
                      nullptr,
                      nullptr,
                      dx_desc.get(),
                      GetBasePtr(dx));
}

template <typename T, typename Context>
void ActivationGradKernelV3(const Context& dev_ctx,
                            const phi::DenseTensor& out,
                            const phi::DenseTensor& dout,
                            cnnlActivationMode_t act_mode,
                            phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  MLUCnnlTensorDesc out_desc(out);
  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnlTensorDesc dx_desc(*dx);
  MLUCnnlActivationDesc act_desc(act_mode, 1.0);
  MLUCnnl::ActiveGrad(dev_ctx,
                      act_desc.get(),
                      nullptr,
                      nullptr,
                      nullptr,
                      nullptr,
                      dout_desc.get(),
                      GetBasePtr(&dout),
                      out_desc.get(),
                      GetBasePtr(&out),
                      dx_desc.get(),
                      GetBasePtr(dx));
}

template <typename T, typename Context>
void ReluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  ActivationKernel<T, Context>(dev_ctx, x, 1.0, CNNL_ACTIVATION_RELU, out);
}

template <typename T, typename Context>
void ReluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  ActivationGradKernelV3<T, Context>(
      dev_ctx, out, dout, CNNL_ACTIVATION_RELU, dx);
}

template <typename T, typename Context>
void Relu6RawKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 float attr,
                 phi::DenseTensor* out) {
  ActivationKernel<T, Context>(dev_ctx, x, 1.0, CNNL_ACTIVATION_RELU6, out);
}

template <typename T, typename Context>
void Relu6Kernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  Relu6RawKernel<T, Context>(dev_ctx, x, 6.0, out);
}

template <typename T, typename Context>
void Relu6GradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& out,
                     const phi::DenseTensor& dout,
                     float attr,
                     phi::DenseTensor* dx) {
  ActivationGradKernelV3<T, Context>(
      dev_ctx, out, dout, CNNL_ACTIVATION_RELU6, dx);
}

template <typename T, typename Context>
void SigmoidKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  ActivationKernel<T, Context>(dev_ctx, x, 1.0, CNNL_ACTIVATION_SIGMOID, out);
}

template <typename T, typename Context>
void SigmoidGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& dout,
                       phi::DenseTensor* dx) {
  ActivationGradKernelV2<T, Context>(
      dev_ctx, out, dout, CNNL_ACTIVATION_SIGMOID, dx);
}

template <typename T, typename Context>
void TanhKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  ActivationKernel<T, Context>(dev_ctx, x, 1.0, CNNL_ACTIVATION_TANH, out);
}

template <typename T, typename Context>
void TanhGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& out_grad,
                    phi::DenseTensor* x_grad) {
  ActivationGradKernelV2<T, Context>(
      dev_ctx, out, out_grad, CNNL_ACTIVATION_TANH, x_grad);
}

template <typename T, typename Context>
void LeakyReluKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     float alpha,
                     phi::DenseTensor* out) {
  ActivationKernel<T, Context>(
      dev_ctx, x, alpha, CNNL_ACTIVATION_LEAKYRELU, out);
}

template <typename T, typename Context>
void LeakyReluGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& dout,
                         float alpha,
                         phi::DenseTensor* dx) {
  ActivationGradKernelV1<T, Context>(
      dev_ctx, x, dout, alpha, CNNL_ACTIVATION_LEAKYRELU, dx);
}

template <typename T, typename Context>
void GeluKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                bool approximate,
                phi::DenseTensor* out) {
  ActivationKernel<T, Context>(dev_ctx, x, 1.0, CNNL_ACTIVATION_GELU, out);
}

template <typename T, typename Context>
void GeluGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& out_grad,
                    bool approximate,
                    phi::DenseTensor* x_grad) {
  ActivationGradKernelV1<T, Context>(
      dev_ctx, x, out_grad, 1.0, CNNL_ACTIVATION_GELU, x_grad);
}

template <typename T, typename Context>
void ReciprocalKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnl::Reciprocal(
      dev_ctx, x_desc.get(), GetBasePtr(&x), out_desc.get(), GetBasePtr(out));
}

template <typename T, typename Context>
void ReciprocalGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& out,
                          const phi::DenseTensor& dout,
                          phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  Tensor square_out;
  square_out.Resize(out.dims());
  dev_ctx.template Alloc<T>(&square_out);
  MLUCnnlTensorDesc out_desc(out);
  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnlTensorDesc dx_desc(*dx);
  MLUCnnlTensorDesc square_out_desc(square_out);
  MLUCnnl::Square(dev_ctx,
                  out_desc.get(),
                  GetBasePtr(&out),
                  square_out_desc.get(),
                  GetBasePtr(&square_out));
  cnnlOpTensorDesc_t op_tensor_op = CNNL_OP_TENSOR_MUL;
  cnnlDataType_t op_tensor_comp_type = CNNL_DTYPE_FLOAT;
  cnnlNanPropagation_t op_tensor_nan_opt = CNNL_NOT_PROPAGATE_NAN;
  MLUCnnlOpTensorDesc op_tensor_desc(
      op_tensor_op, op_tensor_comp_type, op_tensor_nan_opt);
  float alpha1_float = -1;
  float alpha2_float = 1;
  float beta_float = 0;
  MLUCnnl::OpTensor(dev_ctx,
                    op_tensor_desc.get(),
                    dout_desc.get(),
                    GetBasePtr(&dout),
                    square_out_desc.get(),
                    GetBasePtr(&square_out),
                    dx_desc.get(),
                    GetBasePtr(dx),
                    op_tensor_comp_type,
                    alpha1_float,
                    alpha2_float,
                    beta_float);
}

template <typename T, typename Context>
void SqrtKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);

  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
  MLUCnnl::Sqrt(dev_ctx,
                prefer,
                input_desc.get(),
                GetBasePtr(&x),
                output_desc.get(),
                GetBasePtr(out));
}

template <typename T, typename Context>
void SqrtGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out,
                    const phi::DenseTensor& dout,
                    phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  MLUCnnlTensorDesc data_desc(out);
  MLUCnnl::SqrtGrad(dev_ctx,
                    data_desc.get(),
                    GetBasePtr(&out),
                    GetBasePtr(&dout),
                    GetBasePtr(dx));
}

// CNNL_LOG_E = 0,
// CNNL_LOG_2 = 1,
// CNNL_LOG_10 = 2,
template <typename T, typename Context>
void LogMLUKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  cnnlLogBase_t log_base,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);
  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;

  MLUCnnl::Log(dev_ctx,
               prefer,
               log_base,
               input_desc.get(),
               GetBasePtr(&x),
               output_desc.get(),
               GetBasePtr(out));
}

template <typename T, typename Context>
void LogKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  LogMLUKernel<T, Context>(dev_ctx, x, CNNL_LOG_E, out);
}

template <typename T, typename Context>
void Log2Kernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DenseTensor* out) {
  LogMLUKernel<T, Context>(dev_ctx, x, CNNL_LOG_2, out);
}

template <typename T, typename Context>
void Log10Kernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  LogMLUKernel<T, Context>(dev_ctx, x, CNNL_LOG_10, out);
}

template <typename T, typename Context>
void ExpKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);
  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;

  MLUCnnl::Exp(dev_ctx,
               prefer,
               input_desc.get(),
               GetBasePtr(&x),
               output_desc.get(),
               GetBasePtr(out));
}

template <typename T, typename Context>
void ExpGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& out,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnlTensorDesc out_desc(out);
  MLUCnnlTensorDesc dx_desc(*dx);

  MLUCnnlOpTensorDesc op_tensor_desc(
      CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);

  MLUCnnl::OpTensor(dev_ctx,
                    op_tensor_desc.get(),
                    dout_desc.get(),
                    GetBasePtr(&dout),
                    out_desc.get(),
                    GetBasePtr(&out),
                    dx_desc.get(),
                    GetBasePtr(dx),
                    ToCnnlDataType<T>());
}

template <typename T, typename Context>
void HardSwishRawKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     float threshold,
                     float scale,
                     float offset,
                     phi::DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      threshold,
      6.0f,
      phi::errors::External("Not support threshold [%f] in MLU", threshold));
  PADDLE_ENFORCE_EQ(
      scale,
      6.0f,
      phi::errors::External("Not support scale [%f] in MLU", scale));
  PADDLE_ENFORCE_EQ(
      offset,
      3.0f,
      phi::errors::External("Not support offset [%f] in MLU", offset));

  dev_ctx.template Alloc<T>(out);
  MLUCnnlActivationDesc act_desc(CNNL_ACTIVATION_HARDSWISH,
                                 1.0f /*ceof useless*/);
  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);

  MLUCnnl::Active(dev_ctx,
                  act_desc.get(),
                  input_desc.get(),
                  GetBasePtr(&x),
                  output_desc.get(),
                  GetBasePtr(out));
}

template <typename T, typename Context>
void HardSwishKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     phi::DenseTensor* out) {
HardSwishRawKernel<T, Context>(dev_ctx, x, 6, 6, 3, out);
}

template <typename T, typename Context>
void HardSwishGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& dout,
                         float threshold,
                         float scale,
                         float offset,
                         phi::DenseTensor* dx) {
  PADDLE_ENFORCE_EQ(
      threshold,
      6.0f,
      phi::errors::External("Not support threshold [%f] in MLU", threshold));
  PADDLE_ENFORCE_EQ(
      scale,
      6.0f,
      phi::errors::External("Not support scale [%f] in MLU", scale));
  PADDLE_ENFORCE_EQ(
      offset,
      3.0f,
      phi::errors::External("Not support offset [%f] in MLU", offset));

  dev_ctx.template Alloc<T>(dx);

  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnlTensorDesc dx_desc(*dx);
  MLUCnnlActivationDesc act_desc(CNNL_ACTIVATION_HARDSWISH,
                                 1.0f /*ceof useless*/);
  MLUCnnl::ActiveGrad(dev_ctx,
                      act_desc.get(),
                      nullptr,
                      nullptr,
                      nullptr,
                      nullptr,
                      dout_desc.get(),
                      GetBasePtr(&dout),
                      x_desc.get(),
                      GetBasePtr(&x),
                      dx_desc.get(),
                      GetBasePtr(dx));
}

template <typename T, typename Context>
void HardSigmoidKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       float slope,
                       float offset,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlActivationDesc act_desc(CNNL_ACTIVATION_HARDSIGMOID,
                                 1.0f /*ceof useless*/,
                                 1.0f /*sliced_dim useless*/,
                                 slope,
                                 offset);
  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);

  MLUCnnl::Active(dev_ctx,
                  act_desc.get(),
                  input_desc.get(),
                  GetBasePtr(&x),
                  output_desc.get(),
                  GetBasePtr(out));
}

template <typename T, typename Context>
void HardSigmoidGradKernel(const Context& dev_ctx,
                           const phi::DenseTensor& out,
                           const phi::DenseTensor& dout,
                           float slope,
                           float offset,
                           phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  MLUCnnlActivationDesc act_desc(CNNL_ACTIVATION_HARDSIGMOID,
                                 1.0f /*ceof useless*/,
                                 1.0f /*sliced_dim useless*/,
                                 slope,
                                 offset);
  MLUCnnlTensorDesc out_desc(out);
  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnlTensorDesc dx_desc(*dx);
  MLUCnnl::ActiveGrad(dev_ctx,
                      act_desc.get(),
                      nullptr,
                      nullptr,
                      out_desc.get(),
                      GetBasePtr(&out),
                      dout_desc.get(),
                      GetBasePtr(&dout),
                      nullptr,
                      nullptr,
                      dx_desc.get(),
                      GetBasePtr(dx));
}

template <typename T, typename Context>
void FloorKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);
  MLUCnnl::Floor(dev_ctx,
                 input_desc.get(),
                 GetBasePtr(&x),
                 output_desc.get(),
                 GetBasePtr(out));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(relu,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::ReluKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::ReluGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu6,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::Relu6Kernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu6_raw,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::Relu6RawKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(relu6_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::Relu6GradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(leaky_relu,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::LeakyReluKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(leaky_relu_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::LeakyReluGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(reciprocal,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::ReciprocalKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(reciprocal_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::ReciprocalGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(sqrt,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::SqrtKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(sqrt_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::SqrtGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(log,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::LogKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(log2,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::Log2Kernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(log10,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::Log10Kernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gelu,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::GeluKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gelu_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::GeluGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tanh,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::TanhKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tanh_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::TanhGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(exp,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::ExpKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(exp_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::ExpGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hard_swish,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::HardSwishKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hard_swish_raw,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::HardSwishRawKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hard_swish_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::HardSwishGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hard_sigmoid,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::HardSigmoidKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hard_sigmoid_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::HardSigmoidGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(floor,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::FloorKernel,
                          float,
                          phi::dtype::float16) {}
