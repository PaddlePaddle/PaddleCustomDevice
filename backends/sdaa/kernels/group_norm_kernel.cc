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

inline bool CheckDNNSupport(const phi::DataType input_dtype,
                            const int input_channel,
                            const int groups) {
  if (input_dtype == phi::DataType::FLOAT32) {
    if (input_channel % 16 == 0 && input_channel > 6656) return false;
    if (input_channel % 16 != 0 && input_channel > 9216) return false;
  }

  if (input_dtype == phi::DataType::FLOAT16) {
    if (input_channel % 16 == 0 && input_channel > 9216) return false;
    if (input_channel % 16 != 0 && input_channel > 13824) return false;
  }

  return true;
}

template <typename T, typename Context>
void GroupNormKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const paddle::optional<phi::DenseTensor>& scale,
                     const paddle::optional<phi::DenseTensor>& bias,
                     float epsilon,
                     int groups,
                     const std::string& data_layout_str,
                     phi::DenseTensor* y,
                     phi::DenseTensor* mean,
                     phi::DenseTensor* var) {
  VLOG(4) << "CALL SDAA GroupNormKernel.";

  auto x_dims = x.dims();
  PADDLE_ENFORCE_EQ((x_dims.size() == 4UL || x_dims.size() == 3UL),
                    true,
                    phi::errors::InvalidArgument(
                        "The input tensor X's dimension size must equal to 3 "
                        "or 4 for group_norm op. "
                        " But got X's shape = [%s], X's dimension size = [%d].",
                        x_dims.to_str(),
                        x_dims.size()));

  const phi::DataLayout data_layout =
      common::StringToDataLayout(data_layout_str);
  using AccT = typename sdaa_ops::MPTypeTrait<T>::Type;
  int N, C, H, W, D;
  sdaa_ops::ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

  bool dnn_support = CheckDNNSupport(x.dtype(), C, groups);
  PADDLE_ENFORCE_EQ(dnn_support,
                    true,
                    phi::errors::InvalidArgument(
                        "SDAA not support this case which x dtype is %s, input "
                        "channel is %d, groups is %d for group norm op.",
                        x.dtype(),
                        C,
                        groups));

  std::vector<int> in_out_dims, scale_bias_dims, mean_var_dims;
  TensorFormat tf;
  if (data_layout == phi::DataLayout::kNCHW) {
    in_out_dims = {N, C, H, W};
    scale_bias_dims = {1, C, 1, 1};
    mean_var_dims = {N, groups, 1, 1};
    tf = TensorFormat::NCHW;
  } else {
    in_out_dims = {N, H, W, C};
    scale_bias_dims = {1, 1, 1, C};
    mean_var_dims = {N, 1, 1, groups};
    tf = TensorFormat::NHWC;
  }

  dev_ctx.template Alloc<T>(y);

  phi::DenseTensor scale_tensor, bias_tensor, mean_tensor, inv_var_tensor;

  if (scale) {
    scale_tensor = scale.get();
  } else {
    scale_tensor.Resize(phi::make_ddim(scale_bias_dims));
    dev_ctx.template Alloc<T>(&scale_tensor);
    sdaa_ops::doFillTensor<T>(
        dev_ctx, static_cast<T>(1.), scale_tensor.dtype(), &scale_tensor);
  }

  if (bias) {
    bias_tensor = bias.get();
  } else {
    bias_tensor.Resize(phi::make_ddim(scale_bias_dims));
    dev_ctx.template Alloc<T>(&bias_tensor);
    sdaa_ops::doFillTensor<T>(
        dev_ctx, static_cast<T>(0.), bias_tensor.dtype(), &bias_tensor);
  }

  if (mean) {
    dev_ctx.template Alloc<AccT>(mean);
    mean_tensor = *mean;
  } else {
    mean_tensor.Resize(phi::make_ddim(mean_var_dims));
    dev_ctx.template Alloc<AccT>(&mean_tensor);
  }

  if (var) {
    dev_ctx.template Alloc<AccT>(var);
  }

  inv_var_tensor.Resize(phi::make_ddim(mean_var_dims));
  dev_ctx.template Alloc<AccT>(&inv_var_tensor);

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

  tecodnnTensorDescriptor_t x_Desc =
      sdaa_ops::GetTecodnnTensorDesc(in_out_dims, x.dtype(), tf);
  tecodnnTensorDescriptor_t scale_bias_Desc =
      sdaa_ops::GetTecodnnTensorDesc(scale_bias_dims, scale_tensor.dtype(), tf);
  tecodnnTensorDescriptor_t y_Desc =
      sdaa_ops::GetTecodnnTensorDesc(in_out_dims, y->dtype(), tf);
  tecodnnTensorDescriptor_t mean_var_Desc =
      sdaa_ops::GetTecodnnTensorDesc(mean_var_dims, mean_tensor.dtype(), tf);

  TECODNN_CHECK(tecodnnGroupNormForward(tecodnnHandle,
                                        groups,
                                        epsilon,
                                        x_Desc,
                                        x.data(),
                                        scale_bias_Desc,
                                        scale_tensor.data(),
                                        scale_bias_Desc,
                                        bias_tensor.data(),
                                        y_Desc,
                                        y->data(),
                                        mean_var_Desc,
                                        mean_tensor.data(),
                                        mean_var_Desc,
                                        inv_var_tensor.data()));

  // The tecodnnGroupNormForward outputs is the inverse of the standard
  // deviation, but the group_norm OP outputs is variance. This operation can be
  // canceled when group_norm_grad support.
  if (var) {
    sdaa_ops::doUnaryOpTensor(
        dev_ctx, inv_var_tensor, 1.0, UnaryOpMode::SQUARE, var);
    sdaa_ops::doUnaryOpTensor(
        dev_ctx, *var, 1.0, UnaryOpMode::RDIV, &inv_var_tensor);
    sdaa_ops::doUnaryOpTensor(
        dev_ctx, inv_var_tensor, epsilon, UnaryOpMode::SUB_A, var);
  }

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(scale_bias_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(mean_var_Desc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(group_norm,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::GroupNormKernel,
                          float,
                          phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}
