// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include <iostream>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void InstanceNormKernel(const Context &dev_ctx,
                        const phi::DenseTensor &x,
                        const paddle::optional<phi::DenseTensor> &scale,
                        const paddle::optional<phi::DenseTensor> &bias,
                        float epsilon_f,
                        phi::DenseTensor *y,
                        phi::DenseTensor *saved_mean,
                        phi::DenseTensor *saved_variance) {
  // This OP only support NCHW for paddlepaddle.
  VLOG(4) << "CALL SDAA InstanceNormKernel.";

  auto x_dims = x.dims();

  PADDLE_ENFORCE_EQ((x_dims.size() == 4UL || x_dims.size() == 3UL),
                    true,
                    phi::errors::InvalidArgument(
                        "The input tensor X's dimension size must equal to 3 "
                        "or 4 for instance_norm op. "
                        " But got X's shape = [%s], X's dimension size = [%d].",
                        x_dims.to_str(),
                        x_dims.size()));

  dev_ctx.template Alloc<T>(y);

  using AccT = typename sdaa_ops::MPTypeTrait<T>::Type;
  double epsilon = static_cast<double>(epsilon_f);
  float alpha = 1.f, beta = 0.f;
  double exponentialAverageFactor = 1.0;
  int N, C, H, W, D;
  sdaa_ops::ExtractNCWHD(x_dims, phi::DataLayout::kNCHW, &N, &C, &H, &W, &D);

  PADDLE_ENFORCE_LT(C,
                    3840,
                    phi::errors::InvalidArgument(
                        "tecodnnInstanceNormalizationForwardTraining kernel "
                        "only support channel number < 3840, but got %d.",
                        C));

  std::vector<int> x_NHWC_dims = {N, H, W, C};
  std::vector<int> scale_bias_dims = {1, 1, 1, C};
  std::vector<int> mean_variance_dims = {N, 1, 1, C};

  phi::DenseTensor x_NHWC, y_NHWC;
  x_NHWC.Resize(phi::make_ddim(x_NHWC_dims));
  y_NHWC.Resize(phi::make_ddim(x_NHWC_dims));
  dev_ctx.template Alloc<T>(&x_NHWC);
  dev_ctx.template Alloc<T>(&y_NHWC);

  phi::DenseTensor scale_tensor, bias_tensor, saved_mean_tmp,
      saved_variance_tmp;
  if (scale) {
    scale_tensor = scale.get();
  } else {
    scale_tensor.Resize(phi::make_ddim(scale_bias_dims));
    dev_ctx.template Alloc<AccT>(&scale_tensor);
    sdaa_ops::doFillTensor<AccT>(
        dev_ctx, static_cast<AccT>(1.0), scale_tensor.dtype(), &scale_tensor);
  }

  if (bias) {
    bias_tensor = bias.get();
  } else {
    bias_tensor.Resize(phi::make_ddim(scale_bias_dims));
    dev_ctx.template Alloc<AccT>(&bias_tensor);
    sdaa_ops::doFillTensor<AccT>(
        dev_ctx, static_cast<AccT>(0.0), bias_tensor.dtype(), &bias_tensor);
  }

  if (saved_mean) {
    dev_ctx.template Alloc<AccT>(saved_mean);
    saved_mean_tmp = *saved_mean;
  } else {
    saved_mean_tmp.Resize(phi::make_ddim(mean_variance_dims));
    dev_ctx.template Alloc<AccT>(&saved_mean_tmp);
  }

  if (saved_variance) {
    dev_ctx.template Alloc<AccT>(saved_variance);
    saved_variance_tmp = *saved_variance;
  } else {
    saved_variance_tmp.Resize(phi::make_ddim(mean_variance_dims));
    dev_ctx.template Alloc<AccT>(&saved_variance_tmp);
  }

  // tecodnnInstanceNormalizationForwardTraining also calculate moving_mean and
  // moving_variance.
  phi::DenseTensor moving_mean, moving_var;
  moving_mean.Resize(phi::make_ddim(scale_bias_dims));
  dev_ctx.template Alloc<AccT>(&moving_mean);
  moving_var.Resize(phi::make_ddim(scale_bias_dims));
  dev_ctx.template Alloc<AccT>(&moving_var);

  sdaa_ops::doTransformTensor(dev_ctx, x, Convert_TF::NCHW2NHWC, &x_NHWC);

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

  tecodnnTensorDescriptor_t x_y_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_NHWC_dims, x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t scale_bias_Desc = sdaa_ops::GetTecodnnTensorDesc(
      scale_bias_dims, scale_tensor.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t mean_var_Desc = sdaa_ops::GetTecodnnTensorDesc(
      mean_variance_dims, saved_mean_tmp.dtype(), TensorFormat::NHWC);

  // NOTE: tecodnnInstanceNormalizationForwardTraining() returns the variance,
  // but in CPU or GPU, instance_norm returns the inverse of the standard
  // deviation for saved_variance.
  TECODNN_CHECK(
      tecodnnInstanceNormalizationForwardTraining(tecodnnHandle,
                                                  &alpha,
                                                  &beta,
                                                  epsilon,
                                                  exponentialAverageFactor,
                                                  x_y_Desc,
                                                  x_NHWC.data(),
                                                  scale_bias_Desc,
                                                  scale_tensor.data(),
                                                  bias_tensor.data(),
                                                  scale_bias_Desc,
                                                  moving_mean.data(),
                                                  moving_var.data(),
                                                  mean_var_Desc,
                                                  saved_mean_tmp.data(),
                                                  saved_variance_tmp.data(),
                                                  x_y_Desc,
                                                  y_NHWC.data()));

  sdaa_ops::doTransformTensor(dev_ctx, y_NHWC, Convert_TF::NHWC2NCHW, y);

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_y_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(scale_bias_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(mean_var_Desc));
}

template <typename T, typename Context>
void InstanceNormGradKernel(const Context &dev_ctx,
                            const phi::DenseTensor &x,
                            const paddle::optional<phi::DenseTensor> &scale,
                            const paddle::optional<phi::DenseTensor> &bias
                                UNUSED,
                            const phi::DenseTensor &saved_mean,
                            const phi::DenseTensor &saved_variance,
                            const phi::DenseTensor &d_y,
                            float epsilon_f,
                            phi::DenseTensor *d_x,
                            phi::DenseTensor *d_scale,
                            phi::DenseTensor *d_bias) {
  VLOG(4) << "CALL SDAA InstanceNormGradKernel.";

  auto x_dims = x.dims();

  PADDLE_ENFORCE_EQ((x_dims.size() == 4UL || x_dims.size() == 3UL),
                    true,
                    phi::errors::InvalidArgument(
                        "The input tensor X's dimension size must equal to 3 "
                        "or 4 for instance_norm_grad op. "
                        " But got X's shape = [%s], X's dimension size = [%d].",
                        x_dims.to_str(),
                        x_dims.size()));

  using AccT = typename sdaa_ops::MPTypeTrait<T>::Type;
  double epsilon = static_cast<double>(epsilon_f);
  float alpha = 1.f, beta = 0.f;
  int N, C, H, W, D;
  sdaa_ops::ExtractNCWHD(x_dims, phi::DataLayout::kNCHW, &N, &C, &H, &W, &D);

  PADDLE_ENFORCE_LT(C,
                    4096,
                    phi::errors::InvalidArgument(
                        "tecodnnInstanceNormalizationBackward kernel "
                        "only support channel number < 4096, but got %d.",
                        C));

  std::vector<int> x_NHWC_dims = {N, H, W, C};
  std::vector<int> scale_bias_dims = {1, 1, 1, C};
  std::vector<int> mean_variance_dims = {N, 1, 1, C};

  phi::DenseTensor x_NHWC, dy_NHWC, dx_NHWC;
  x_NHWC.Resize(phi::make_ddim(x_NHWC_dims));
  dy_NHWC.Resize(phi::make_ddim(x_NHWC_dims));
  dx_NHWC.Resize(phi::make_ddim(x_NHWC_dims));
  dev_ctx.template Alloc<T>(&x_NHWC);
  dev_ctx.template Alloc<T>(&dy_NHWC);
  dev_ctx.template Alloc<T>(&dx_NHWC);

  phi::DenseTensor scale_tensor, d_scale_temp, d_bias_temp;

  if (scale) {
    scale_tensor = scale.get();
  } else {
    scale_tensor.Resize(phi::make_ddim(scale_bias_dims));
    dev_ctx.template Alloc<AccT>(&scale_tensor);
    sdaa_ops::doFillTensor<AccT>(
        dev_ctx, static_cast<AccT>(1.0), scale_tensor.dtype(), &scale_tensor);
  }

  dev_ctx.template Alloc<T>(d_x);

  if (d_scale) {
    dev_ctx.template Alloc<AccT>(d_scale);
    d_scale_temp = *d_scale;
  } else {
    d_scale_temp.Resize(phi::make_ddim(scale_bias_dims));
    dev_ctx.template Alloc<AccT>(&d_scale_temp);
  }

  if (d_bias) {
    dev_ctx.template Alloc<AccT>(d_bias);
    d_bias_temp = *d_bias;
  } else {
    d_bias_temp.Resize(phi::make_ddim(scale_bias_dims));
    dev_ctx.template Alloc<AccT>(&d_bias_temp);
  }

  sdaa_ops::doTransformTensor(dev_ctx, x, Convert_TF::NCHW2NHWC, &x_NHWC);
  sdaa_ops::doTransformTensor(dev_ctx, d_y, Convert_TF::NCHW2NHWC, &dy_NHWC);

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

  tecodnnTensorDescriptor_t x_y_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_NHWC_dims, x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t scale_bias_Desc = sdaa_ops::GetTecodnnTensorDesc(
      scale_bias_dims, scale_tensor.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t mean_var_Desc = sdaa_ops::GetTecodnnTensorDesc(
      mean_variance_dims, d_scale_temp.dtype(), TensorFormat::NHWC);

  TECODNN_CHECK(tecodnnInstanceNormalizationBackward(tecodnnHandle,
                                                     &alpha,
                                                     &beta,
                                                     epsilon,
                                                     &alpha,
                                                     &beta,
                                                     x_y_Desc,
                                                     x_NHWC.data(),
                                                     x_y_Desc,
                                                     dy_NHWC.data(),
                                                     mean_var_Desc,
                                                     saved_mean.data(),
                                                     saved_variance.data(),
                                                     scale_bias_Desc,
                                                     scale_tensor.data(),
                                                     scale_bias_Desc,
                                                     d_scale_temp.data(),
                                                     d_bias_temp.data(),
                                                     x_y_Desc,
                                                     dx_NHWC.data()));

  sdaa_ops::doTransformTensor(dev_ctx, dx_NHWC, Convert_TF::NHWC2NCHW, d_x);

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_y_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(scale_bias_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(mean_var_Desc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(instance_norm,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::InstanceNormKernel,
                          float,
                          phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}

PD_REGISTER_PLUGIN_KERNEL(instance_norm_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::InstanceNormGradKernel,
                          float,
                          phi::dtype::float16) {}
