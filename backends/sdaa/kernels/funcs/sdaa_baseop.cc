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

#include "kernels/funcs/sdaa_baseop.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <utility>

#include "kernels/funcs/elementwise_functor.h"
#include "paddle/phi/core/enforce.h"
#include "sdcops.h"  // NOLINT

namespace custom_kernel {

namespace sdaa_ops {

struct NCHWValue {
  int N, C, H, W;
};

NCHWValue GetNCHWValue(const std::vector<int> dims, tecodnnTensorFormat_t tf) {
  int N = dims[0], C = dims[1], H = dims[2], W = dims[3];
  switch (tf) {
    case TECODNN_TENSOR_NCHW:
      N = dims[0];
      C = dims[1];
      H = dims[2];
      W = dims[3];
      break;
    case TECODNN_TENSOR_NHWC:
      N = dims[0];
      C = dims[3];
      H = dims[1];
      W = dims[2];
      break;
    case TECODNN_TENSOR_CHWN:
      N = dims[3];
      C = dims[0];
      H = dims[1];
      W = dims[2];
      break;
    case TECODNN_TENSOR_NWHC:
      N = dims[0];
      C = dims[3];
      H = dims[2];
      W = dims[1];
      break;
    default:
      break;
  }
  return {N, C, H, W};
}

std::pair<TensorFormat, TensorFormat> GetTecodnnConvertTensorFormat(
    const Convert_TF convert_tf) {
  auto iter = tecodnnTensorFormatMap.find(convert_tf);
  if (iter != tecodnnTensorFormatMap.end()) {
    return iter->second;
  }
  PADDLE_THROW(phi::errors::InvalidArgument(
      "Not support tensor format convert of SDAA Device."));
}

tecodnnActivationMode_t GetTecodnnActivationMode(
    const ActivationMode activation_mode) {
  auto iter = tecodnnActivationModeMap.find(activation_mode);
  if (iter != tecodnnActivationModeMap.end()) {
    return iter->second;
  }
  PADDLE_THROW(phi::errors::InvalidArgument(
      "Not support activation mode of SDAA Device."));
}

tecodnnNanPropagation_t GetTecodnnNanPropagation(
    const NanPropagation nan_propagate) {
  auto iter = tecodnnNanPropagationMap.find(nan_propagate);
  if (iter != tecodnnNanPropagationMap.end()) {
    return iter->second;
  }
  PADDLE_THROW(phi::errors::InvalidArgument(
      "Not support propagate nan mode of SDAA Device."));
}

tecodnnUnaryOpsMode_t GetTecodnnUnaryOpsMode(const UnaryOpMode unaryOpMode) {
  auto iter = tecodnnUnaryOpsModeMap.find(unaryOpMode);
  if (iter != tecodnnUnaryOpsModeMap.end()) {
    return iter->second;
  }
  PADDLE_THROW(phi::errors::InvalidArgument(
      "Not support unary ops mode of SDAA Device."));
}

tecodnnOpTensorOp_t GetTecodnnOpTensorMode(const OpTensorMode opTensorMode) {
  auto iter = tecodnnOpTensorModeMap.find(opTensorMode);
  if (iter != tecodnnOpTensorModeMap.end()) {
    return iter->second;
  }
  PADDLE_THROW(phi::errors::InvalidArgument(
      "Not support op tensor mode of SDAA Device."));
}

Convert_TF GetTransposeTensorFormat(const std::vector<int>& dim_premute) {
  auto iter = TransposeModeMap.find(dim_premute);
  if (iter != TransposeModeMap.end()) {
    return iter->second;
  }
  phi::DDim not_support_dim = phi::make_ddim(dim_premute);
  PADDLE_THROW(phi::errors::InvalidArgument(
      "Not support transpose mode %s of SDAA Device.", not_support_dim));
}

tecodnnTensorDescriptor_t GetTecodnnTensorDesc(const std::vector<int>& dims,
                                               const DataType& dtype,
                                               TensorFormat tf) {
  tecodnnDataType_t dt = ToTecodnnDataType(dtype);
  tecodnnTensorDescriptor_t Desc;
  TECODNN_CHECK(tecodnnCreateTensorDescriptor(&Desc));

  auto tmp_dims = dims;
  if (dims.empty()) {
    tmp_dims.push_back(1);
  }

  if (tf != TensorFormat::Undefined && tmp_dims.size() <= 4) {
    tecodnnTensorFormat_t t_f = GetTecodnnTF(tf);

    std::vector<int> dimensions(4, 1);
    int index = 3;
    for (int i = tmp_dims.size() - 1; i >= 0; i--) {
      dimensions[index--] = tmp_dims[i];
    }

    int N, C, H, W;
    NCHWValue getNCHWvalue = GetNCHWValue(dimensions, t_f);
    N = getNCHWvalue.N;
    C = getNCHWvalue.C;
    H = getNCHWvalue.H;
    W = getNCHWvalue.W;
    TECODNN_CHECK(tecodnnSetTensor4dDescriptor(Desc, t_f, dt, N, C, H, W));
  } else {
    int dims_arr[kSliceMaxNum];
    PADDLE_ENFORCE_LE(tmp_dims.size(),
                      kSliceMaxNum,
                      phi::errors::InvalidArgument(
                          "The max ND Descriptor dims size is %d, but got %d.",
                          kSliceMaxNum,
                          tmp_dims.size()));
    std::copy(tmp_dims.begin(), tmp_dims.end(), dims_arr);
    TECODNN_CHECK(tecodnnSetTensorNdDescriptor(
        Desc, dt, tmp_dims.size(), dims_arr, NULL));
  }
  return Desc;
}

tecodnnTensorDescriptor_t GetTecodnnBoolTensorDesc(const std::vector<int>& dims,
                                                   TensorFormat tf) {
  tecodnnDataType_t dt = TECODNN_DATA_BOOL;
  tecodnnTensorDescriptor_t Desc;
  TECODNN_CHECK(tecodnnCreateTensorDescriptor(&Desc));

  auto tmp_dims = dims;
  if (dims.empty()) {
    tmp_dims.push_back(1);
  }

  if (tf != TensorFormat::Undefined && tmp_dims.size() <= 4) {
    tecodnnTensorFormat_t t_f = GetTecodnnTF(tf);

    std::vector<int> dimensions(4, 1);
    int index = 3;
    for (int i = tmp_dims.size() - 1; i >= 0; i--) {
      dimensions[index--] = tmp_dims[i];
    }

    int N, C, H, W;
    NCHWValue getNCHWvalue = GetNCHWValue(dimensions, t_f);
    N = getNCHWvalue.N;
    C = getNCHWvalue.C;
    H = getNCHWvalue.H;
    W = getNCHWvalue.W;

    TECODNN_CHECK(tecodnnSetTensor4dDescriptor(Desc, t_f, dt, N, C, H, W));
  } else {
    int dims_arr[kSliceMaxNum];
    PADDLE_ENFORCE_LE(tmp_dims.size(),
                      kSliceMaxNum,
                      phi::errors::InvalidArgument(
                          "The max ND Descriptor dims size is %d, but got %d.",
                          kSliceMaxNum,
                          tmp_dims.size()));
    std::copy(tmp_dims.begin(), tmp_dims.end(), dims_arr);
    TECODNN_CHECK(tecodnnSetTensorNdDescriptor(
        Desc, dt, tmp_dims.size(), dims_arr, NULL));
  }
  return Desc;
}

void doTransformTensor(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       Convert_TF convert_tf,
                       phi::DenseTensor* y) {
  VLOG(4) << "call tecodnn transform tensor";
  phi::DDim x_d;
  if (x.storage_properties_initialized()) {
    auto storages = x.storage_properties<SDAAStorageProperties>();
    x_d = storages.storage_dims;  // CHWN
  } else {
    x_d = x.dims();
  }
  std::pair<TensorFormat, TensorFormat> in_out_tf =
      GetTecodnnConvertTensorFormat(convert_tf);

  std::vector<int> x_dimensions = phi::vectorize<int>(x_d);
  std::vector<int> y_dimensions = phi::vectorize<int>(y->dims());

  PADDLE_ENFORCE_EQ(
      x_dimensions.size(),
      4,
      phi::errors::InvalidArgument("The input tensor dimension size must be 4."
                                   "But got %d.",
                                   x_dimensions.size()));

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

  tecodnnTensorDescriptor_t x_Desc =
      GetTecodnnTensorDesc(x_dimensions, x.dtype(), in_out_tf.first);
  tecodnnTensorDescriptor_t y_Desc =
      GetTecodnnTensorDesc(y_dimensions, y->dtype(), in_out_tf.second);

  float alpha = 1.0, beta = 0.0;
  TECODNN_CHECK(tecodnnTransformTensor(
      tecodnnHandle, &alpha, x_Desc, x.data(), &beta, y_Desc, y->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
}

void doCastTensor(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  phi::DenseTensor* y) {
  VLOG(4) << "call tecodnn cast tensor";
  phi::DDim x_d;
  if (x.storage_properties_initialized()) {
    auto storages = x.storage_properties<SDAAStorageProperties>();
    x_d = storages.storage_dims;  // CHWN
    doAddStorageProperties(dev_ctx, y, storages);
  } else {
    x_d = x.dims();
  }

  tecodnnDataType_t in_dt = ToTecodnnDataType(x.dtype());
  tecodnnDataType_t out_dt = ToTecodnnDataType(y->dtype());

  if (in_dt == out_dt) {
    phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, y);
    return;
  }

  std::vector<int> x_dims = phi::vectorize<int>(x_d);
  std::vector<int> x_dimensions(4, 1);
  int dims_merge = 1;
  if (x_dims.size() > 4) {
    for (int i = 0; i < x_dims.size() - 3; i++) {
      dims_merge *= x_dims[i];
    }
    x_dimensions[0] = dims_merge;
    std::copy(x_dims.end() - 3, x_dims.end(), x_dimensions.end() - 3);
  } else {
    std::copy(x_dims.begin(), x_dims.end(), x_dimensions.begin());
  }

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc, y_Desc;

  if (x.dtype() == DataType::BOOL) {
    x_Desc = GetTecodnnBoolTensorDesc(x_dimensions, TensorFormat::NCHW);
  } else {
    x_Desc = GetTecodnnTensorDesc(x_dimensions, x.dtype(), TensorFormat::NCHW);
  }

  if (y->dtype() == DataType::BOOL) {
    y_Desc = GetTecodnnBoolTensorDesc(x_dimensions, TensorFormat::NCHW);
  } else {
    y_Desc = GetTecodnnTensorDesc(x_dimensions, y->dtype(), TensorFormat::NCHW);
  }

  float alpha = 1.0, beta = 0.0;

  TECODNN_CHECK(tecodnnTransformTensor(
      tecodnnHandle, &alpha, x_Desc, x.data(), &beta, y_Desc, y->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
}

void doAddTensor(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 float alpha,
                 float beta,
                 phi::DenseTensor* out) {
  VLOG(4) << "call tecodnn add tensor";
  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> x_dimensions(4, 1);
  int dims_merge = 1;
  if (x_dims.size() > 4) {
    for (int i = 0; i < x_dims.size() - 3; i++) {
      dims_merge *= x_dims[i];
    }
    x_dimensions[0] = dims_merge;
    std::copy(x_dims.end() - 3, x_dims.end(), x_dimensions.end() - 3);
  } else {
    std::copy(x_dims.begin(), x_dims.end(), x_dimensions.begin());
  }

  tecodnnTensorDescriptor_t x_Desc =
      GetTecodnnTensorDesc(x_dimensions, x.dtype(), TensorFormat::NHWC);

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

  TECODNN_CHECK(tecodnnAddTensor(
      tecodnnHandle, &alpha, x_Desc, x.data(), &beta, x_Desc, out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
}

void doActivationForward(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         double factor,
                         ActivationMode activation_mode,
                         NanPropagation nan_propagate,
                         phi::DenseTensor* out) {
  VLOG(4) << "call tecodnn activation forward";

  int N = 1, C = x.numel(), H = 1, W = 1;
  std::vector<int> dims = {N, H, W, C};

  tecodnnDataType_t dt = ToTecodnnDataType(x.dtype());

  tecodnnTensorDescriptor_t t_Desc =
      GetTecodnnTensorDesc(dims, x.dtype(), TensorFormat::NHWC);
  tecodnnActivationDescriptor_t a_Desc;

  tecodnnActivationMode_t mode = GetTecodnnActivationMode(activation_mode);
  tecodnnNanPropagation_t nan_prop = GetTecodnnNanPropagation(nan_propagate);

  tecodnnHandle_t tecodnnHandle = custom_kernel::GetHandleFromCTX(dev_ctx);
  TECODNN_CHECK(tecodnnCreateActivationDescriptor(&a_Desc));
  TECODNN_CHECK(tecodnnSetActivationDescriptor(a_Desc, mode, nan_prop, factor));

  const float alpha = 1.0f, beta = 0.0f;
  TECODNN_CHECK(tecodnnActivationForward(tecodnnHandle,
                                         a_Desc,
                                         &alpha,
                                         t_Desc,
                                         x.data(),
                                         &beta,
                                         t_Desc,
                                         out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(t_Desc));
  TECODNN_CHECK(tecodnnDestroyActivationDescriptor(a_Desc));
}

void doActivationBackward(const Context& dev_ctx,
                          const phi::DenseTensor& out,
                          const phi::DenseTensor& dout,
                          double factor,
                          ActivationMode activation_mode,
                          NanPropagation nan_propagate,
                          phi::DenseTensor* dx) {
  VLOG(4) << "call tecodnn activation backward";

  int N = 1, C = out.numel(), H = 1, W = 1;
  std::vector<int> dims = {N, H, W, C};

  tecodnnDataType_t dt = ToTecodnnDataType(out.dtype());

  tecodnnTensorDescriptor_t t_Desc =
      GetTecodnnTensorDesc(dims, out.dtype(), TensorFormat::NHWC);
  tecodnnActivationDescriptor_t a_Desc;

  tecodnnActivationMode_t mode = GetTecodnnActivationMode(activation_mode);
  tecodnnNanPropagation_t nan_prop = GetTecodnnNanPropagation(nan_propagate);

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  TECODNN_CHECK(tecodnnCreateActivationDescriptor(&a_Desc));
  TECODNN_CHECK(tecodnnSetActivationDescriptor(a_Desc, mode, nan_prop, factor));

  const float alpha = 1.0f, beta = 0.0f;
  TECODNN_CHECK(tecodnnActivationBackward(tecodnnHandle,
                                          a_Desc,
                                          &alpha,
                                          t_Desc,
                                          out.data(),
                                          t_Desc,
                                          dout.data(),
                                          t_Desc,
                                          out.data(),
                                          &beta,
                                          t_Desc,
                                          dx->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(t_Desc));
  TECODNN_CHECK(tecodnnDestroyActivationDescriptor(a_Desc));
}

void doUnaryOpTensor(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     float alpha,
                     UnaryOpMode unaryOpMode,
                     phi::DenseTensor* out) {
  VLOG(4) << "call tecodnn unary op";

  int N = 1, C = x.numel(), H = 1, W = 1;
  std::vector<int> dims = {N, H, W, C};

  tecodnnDataType_t dt = ToTecodnnDataType(x.dtype());

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

  tecodnnTensorDescriptor_t Desc =
      GetTecodnnTensorDesc(dims, x.dtype(), TensorFormat::NHWC);

  tecodnnUnaryOpsMode_t mode = GetTecodnnUnaryOpsMode(unaryOpMode);

  TECODNN_CHECK(tecodnnUnaryOps(
      tecodnnHandle, mode, &alpha, Desc, x.data(), Desc, out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(Desc));
}

void doScaleTensor(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   float scale,
                   float bias,
                   bool inplace,
                   bool bias_flag,
                   phi::DenseTensor* out) {
  VLOG(4) << "call tecodnn scale tensor";

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());

  if ((x.dtype() == phi::DataType::INT32 ||
       x.dtype() == phi::DataType::INT64) &&
      bias_flag) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "tecodnn scale op not support adding bias when "
        "dtype is int64 or int32"));
  }

  union data_type {
    int64_t data_int64;
    int32_t data_int32;
  };
  data_type scale_dt;

  void* scale_ptr = nullptr;
  scale_ptr = &scale;
  // if x dtype is int64, the scale dtype must be int64,
  // if x dtype is int32, the scale dtype must be int32 for tecodnnScaleTensor()
  if (x.dtype() == phi::DataType::INT64) {
    scale_dt.data_int64 = static_cast<int64_t>(scale);
    scale_ptr = &scale_dt.data_int64;
  } else if (x.dtype() == phi::DataType::INT32) {
    scale_dt.data_int32 = static_cast<int32_t>(scale);
    scale_ptr = &scale_dt.data_int32;
  }

  tecodnnDataType_t dt = ToTecodnnDataType(x.dtype());

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

  tecodnnTensorDescriptor_t x_Desc =
      GetTecodnnTensorDesc(x_dims, x.dtype(), TensorFormat::Undefined);
  bool bias_after_scale_dnn = true;
  TECODNN_CHECK(tecodnnCustomScaleWithBias(tecodnnHandle,
                                           &scale,
                                           &bias,
                                           bias_after_scale_dnn,
                                           x_Desc,
                                           x.data(),
                                           x_Desc,
                                           out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
}

void doNegTensor(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* out) {
  VLOG(4) << "call tecodnn neg tensor";

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());

  std::vector<int> x_dimensions(4, 1);
  int dims_merge = 1;
  if (x_dims.size() > 4) {
    for (int i = 0; i < x_dims.size() - 3; i++) {
      dims_merge *= x_dims[i];
    }
    x_dimensions[0] = dims_merge;
    std::copy(x_dims.end() - 3, x_dims.end(), x_dimensions.end() - 3);
  } else {
    std::copy(x_dims.begin(), x_dims.end(), x_dimensions.begin());
  }

  tecodnnDataType_t dt = ToTecodnnDataType(x.dtype());

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

  tecodnnTensorDescriptor_t Desc =
      GetTecodnnTensorDesc(x_dimensions, x.dtype(), TensorFormat::NHWC);

  TECODNN_CHECK(
      tecodnnNegTensor(tecodnnHandle, Desc, x.data(), Desc, out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(Desc));
}

void doCompareTensor(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     CompareType tct,
                     phi::DenseTensor* out) {
  VLOG(4) << "call tecodnn compare tensor";

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> y_dims = phi::vectorize<int>(y.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  PADDLE_ENFORCE_LE(
      x_dims.size(),
      4,
      phi::errors::InvalidArgument(
          "The input of x dimension size need "
          "less than or equal to 4. But got x dimension size is %d.",
          x_dims.size()));
  PADDLE_ENFORCE_LE(
      y_dims.size(),
      4,
      phi::errors::InvalidArgument(
          "The input of y dimension size need "
          "less than or equal to 4. But got y dimension size is %d.",
          y_dims.size()));

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc =
      GetTecodnnTensorDesc(x_dims, x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t y_Desc =
      GetTecodnnTensorDesc(y_dims, y.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t out_Desc;
  if (tct == CompareType::GreaterThan || tct == CompareType::LessThan) {
    out_Desc = GetTecodnnBoolTensorDesc(out_dims, TensorFormat::NHWC);
  } else {
    out_Desc = GetTecodnnTensorDesc(out_dims, out->dtype(), TensorFormat::NHWC);
  }

  switch (tct) {
    case CompareType::Equal:
      TECODNN_CHECK(tecodnnTensorEqual(tecodnnHandle,
                                       x_Desc,
                                       x.data(),
                                       y_Desc,
                                       y.data(),
                                       out_Desc,
                                       out->data()));
      break;
    case CompareType::GreaterThan:
      TECODNN_CHECK(tecodnnTensorGreater(tecodnnHandle,
                                         x_Desc,
                                         x.data(),
                                         y_Desc,
                                         y.data(),
                                         out_Desc,
                                         out->data()));
      break;
    case CompareType::LessThan:
      TECODNN_CHECK(tecodnnTensorLess(tecodnnHandle,
                                      x_Desc,
                                      x.data(),
                                      y_Desc,
                                      y.data(),
                                      out_Desc,
                                      out->data()));
      break;
    case CompareType::GreaterEqual:
      TECODNN_CHECK(tecodnnTensorEqualGreater(tecodnnHandle,
                                              x_Desc,
                                              x.data(),
                                              y_Desc,
                                              y.data(),
                                              out_Desc,
                                              out->data()));
      break;
    case CompareType::LessEqual:
      TECODNN_CHECK(tecodnnTensorEqualLess(tecodnnHandle,
                                           x_Desc,
                                           x.data(),
                                           y_Desc,
                                           y.data(),
                                           out_Desc,
                                           out->data()));
      break;
    case CompareType::NotEqual:
      TECODNN_CHECK(tecodnnTensorNotEqual(tecodnnHandle,
                                          x_Desc,
                                          x.data(),
                                          y_Desc,
                                          y.data(),
                                          out_Desc,
                                          out->data()));
      break;
    default:
      break;
  }

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}

void doOpTensor(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::DenseTensor& y,
                OpTensorMode opTensorMode,
                phi::DenseTensor* out) {
  VLOG(4) << "call tecodnn Op Tensor";

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> y_dims = phi::vectorize<int>(y.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc =
      GetTecodnnTensorDesc(x_dims, x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t y_Desc =
      GetTecodnnTensorDesc(y_dims, y.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t out_Desc =
      GetTecodnnTensorDesc(out_dims, out->dtype(), TensorFormat::NHWC);

  tecodnnOpTensorOp_t mode = GetTecodnnOpTensorMode(opTensorMode);
  tecodnnDataType_t dt = ToTecodnnDataType(x.dtype());
  tecodnnNanPropagation_t nan_prop =
      GetTecodnnNanPropagation(NanPropagation::not_propagate_nan);

  tecodnnOpTensorDescriptor_t opTensor_Desc;
  TECODNN_CHECK(tecodnnCreateOpTensorDescriptor(&opTensor_Desc));

  TECODNN_CHECK(
      tecodnnSetOpTensorDescriptor(opTensor_Desc, mode, dt, nan_prop));

  const float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;

  TECODNN_CHECK(tecodnnOpTensor(tecodnnHandle,
                                opTensor_Desc,
                                &alpha1,
                                x_Desc,
                                x.data(),
                                &alpha2,
                                y_Desc,
                                y.data(),
                                &beta,
                                out_Desc,
                                out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
  TECODNN_CHECK(tecodnnDestroyOpTensorDescriptor(opTensor_Desc));
}

inline void doReduceTensor(tecodnnHandle_t handle,
                           const void* x,
                           const std::vector<int>& x_dims,
                           void* y,
                           const std::vector<int>& y_dims,
                           const DataType& dtype,
                           void* indices,
                           size_t indices_size,
                           void* workspace,
                           size_t workspace_size,
                           tecodnnReduceTensorDescriptor_t reduceDesc) {
  float alpha = 1.0, beta = 0.0;
  auto x_desc = GetTecodnnTensorDesc(x_dims, dtype, TensorFormat::NHWC);
  auto y_desc = GetTecodnnTensorDesc(y_dims, dtype, TensorFormat::NHWC);
  TECODNN_CHECK(tecodnnReduceTensor(handle,
                                    reduceDesc,
                                    indices,
                                    indices_size,
                                    workspace,
                                    workspace_size,
                                    &alpha,
                                    x_desc,
                                    x,
                                    &beta,
                                    y_desc,
                                    y));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_desc));
}

void doReduceTensorImpl(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const std::vector<int64_t>& reduce_dims,
                        tecodnnReduceTensorOp_t op,
                        tecodnnNanPropagation_t nan_prop,
                        tecodnnReduceTensorIndices_t indices_op,
                        tecodnnIndicesType_t indices_type,
                        phi::DenseTensor* y) {
  if (reduce_dims.size() == 0) {
    if (x.data() == y->data()) {
      return;
    }
    auto y_dims = y->dims();
    phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, y);
    y->Resize(y_dims);  // copy would change dims
    return;
  }

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnReduceTensorDescriptor_t reduce_desc;
  TECODNN_CHECK(tecodnnCreateReduceTensorDescriptor(&reduce_desc));
  TECODNN_CHECK(tecodnnSetReduceTensorDescriptor(reduce_desc,
                                                 op,
                                                 ToTecodnnDataType(x.dtype()),
                                                 nan_prop,
                                                 indices_op,
                                                 indices_type));
  phi::DenseTensor workspace;
  workspace.Resize({static_cast<int64_t>(sizeof(float) * x.numel())});
  dev_ctx.Alloc(&workspace, DataType::INT8);
  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  if (x_dims.size() <= 4) {
    std::vector<int> y_dims{x_dims};
    for (auto&& i : reduce_dims) {
      y_dims[i] = 1;
    }
    phi::DenseTensor y_temp;
    // Reduce is not an inplace op
    if (x.data() == y->data()) {
      y_temp.Resize(y->dims());
      dev_ctx.Alloc(&y_temp, y->dtype());
    } else {
      y_temp = *y;
    }
    doReduceTensor(tecodnnHandle,
                   x.data(),
                   x_dims,
                   y_temp.data(),
                   y_dims,
                   x.dtype(),
                   nullptr,
                   0,
                   workspace.data(),
                   workspace.numel(),
                   reduce_desc);
    TECODNN_CHECK(tecodnnDestroyReduceTensorDescriptor(reduce_desc));
    phi::Copy(dev_ctx, y_temp, dev_ctx.GetPlace(), false, y);
    return;
  }

  std::vector<int> reduce_dims_int;
  for (auto&& i : reduce_dims) {
    reduce_dims_int.push_back(i);
  }
  std::vector<int> ref_dims, ref_reduce_dims;
  foldNonReduceDims(x_dims, reduce_dims_int, &ref_dims, &ref_reduce_dims);

  phi::DenseTensor temp_input, temp_output;
  TensorCopy(dev_ctx, x, false, &temp_input);

  temp_output.Resize(x.dims());
  dev_ctx.Alloc(&temp_output, x.dtype());

  auto temp_input_device = temp_input.data();
  auto temp_output_device = temp_output.data();

  std::vector<bool> reduce_dims_sparse(ref_dims.size(), false);
  for (size_t i = 0; i < ref_reduce_dims.size(); i++) {
    reduce_dims_sparse[ref_reduce_dims[i]] = true;
  }
  std::vector<int> temp_dims(4), temp_out_dims(4);
  for (int i = ref_dims.size(); i >= 0; i -= 2) {
    if (i <= 4) {
      std::vector<int> y_dims{ref_dims};
      for (auto&& i : ref_reduce_dims) {
        y_dims[i] = 1;
      }
      doReduceTensor(tecodnnHandle,
                     temp_input_device,
                     ref_dims,
                     y->data(),
                     y_dims,
                     x.dtype(),
                     nullptr,
                     0,
                     workspace.data(),
                     workspace.numel(),
                     reduce_desc);
      TECODNN_CHECK(tecodnnDestroyReduceTensorDescriptor(reduce_desc));
      return;
    }
    // fold dims to the 4 dimensions, keep last 3 dimensions
    temp_dims[0] = accumulate(
        ref_dims.begin(), ref_dims.end() - 3, 1, std::multiplies<int>());
    temp_out_dims[0] = temp_dims[0];

    // move the last 3 dimensions
    for (size_t j = 1; j < 4; j++) {
      size_t order = i - 4 + j;
      temp_dims[j] = ref_dims[order];
      temp_out_dims[j] = temp_dims[j];
      if (reduce_dims_sparse[order]) {
        ref_dims[order] = 1;
        temp_out_dims[j] = 1;
      }
    }
    doReduceTensor(tecodnnHandle,
                   temp_input_device,
                   temp_dims,
                   temp_output_device,
                   temp_out_dims,
                   x.dtype(),
                   nullptr,
                   0,
                   workspace.data(),
                   workspace.numel(),
                   reduce_desc);
    // cut the tail for reduced data
    auto end = accumulate(
        ref_dims.end() - 3, ref_dims.end(), 1, std::multiplies<int>());
    ref_dims.erase(ref_dims.end() - 2, ref_dims.end());
    ref_dims.back() = end;
    reduce_dims_sparse.erase(reduce_dims_sparse.end() - 2,
                             reduce_dims_sparse.end());
    reduce_dims_sparse.back() = false;
    std::swap(temp_input_device, temp_output_device);
    auto iter = std::remove_if(ref_reduce_dims.begin(),
                               ref_reduce_dims.end(),
                               [i](int& dim) { return dim >= i - 3; });
    ref_reduce_dims.erase(iter, ref_reduce_dims.end());
  }
  TECODNN_CHECK(tecodnnDestroyReduceTensorDescriptor(reduce_desc));
}

inline tecodnnIndicesType_t ToTecodnnIndiceDataType(const DataType& dtype) {
  tecodnnIndicesType_t dt = TECODNN_32BIT_INDICES;
  switch (dtype) {
    case DataType::FLOAT16:
    case DataType::INT16:
      dt = TECODNN_16BIT_INDICES;
      break;
    case DataType::FLOAT32:
    case DataType::INT32:
      dt = TECODNN_32BIT_INDICES;
      break;
    case DataType::INT8:
      dt = TECODNN_8BIT_INDICES;
      break;
    case DataType::FLOAT64:
    case DataType::INT64:
      dt = TECODNN_64BIT_INDICES;
      break;
    default:
      break;
  }
  return dt;
}

// temp function to mitigate int64 problem, when int64 is adapted change the
// impl function to this function
void doReduceTensor(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const std::vector<int64_t>& reduce_dims,
                    tecodnnReduceTensorOp_t op,
                    tecodnnNanPropagation_t nan_prop,
                    tecodnnReduceTensorIndices_t indices_op,
                    tecodnnIndicesType_t indices_type,
                    phi::DenseTensor* y) {
  // The reason for removing the dtype cast expect float16
  // is to avoid the loss of precision due to the dtype cast.
  if (x.dtype() == DataType::FLOAT16) {
    phi::DenseTensor x_temp, y_temp;
    x_temp.Resize(x.dims());
    dev_ctx.Alloc(&x_temp, DataType::FLOAT32);
    sdaa_ops::doCastTensor(dev_ctx, x, &x_temp);
    y_temp.Resize(y->dims());
    dev_ctx.Alloc(&y_temp, DataType::FLOAT32);
    doReduceTensorImpl(dev_ctx,
                       x_temp,
                       reduce_dims,
                       op,
                       nan_prop,
                       indices_op,
                       indices_type,
                       &y_temp);
    sdaa_ops::doCastTensor(dev_ctx, y_temp, y);
  } else {
    doReduceTensorImpl(
        dev_ctx, x, reduce_dims, op, nan_prop, indices_op, indices_type, y);
  }
}

void doMeanTensor(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const std::vector<int64_t>& reduce_dims,
                  phi::DenseTensor* y) {
  doReduceTensor(dev_ctx,
                 x,
                 reduce_dims,
                 TECODNN_REDUCE_TENSOR_AVG,
                 TECODNN_NOT_PROPAGATE_NAN,
                 TECODNN_REDUCE_TENSOR_NO_INDICES,
                 ToTecodnnIndiceDataType(x.dtype()),
                 y);
}

void doSumTensor(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const std::vector<int64_t>& reduce_dims,
                 phi::DenseTensor* y) {
  doReduceTensor(dev_ctx,
                 x,
                 reduce_dims,
                 TECODNN_REDUCE_TENSOR_ADD,
                 TECODNN_NOT_PROPAGATE_NAN,
                 TECODNN_REDUCE_TENSOR_NO_INDICES,
                 ToTecodnnIndiceDataType(x.dtype()),
                 y);
}

void doProdTensor(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const std::vector<int64_t>& reduce_dims,
                  phi::DenseTensor* y) {
  doReduceTensor(dev_ctx,
                 x,
                 reduce_dims,
                 TECODNN_REDUCE_TENSOR_MUL,
                 TECODNN_NOT_PROPAGATE_NAN,
                 TECODNN_REDUCE_TENSOR_NO_INDICES,
                 ToTecodnnIndiceDataType(x.dtype()),
                 y);
}

void doMinTensor(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const std::vector<int64_t>& reduce_dims,
                 phi::DenseTensor* y) {
  doReduceTensor(dev_ctx,
                 x,
                 reduce_dims,
                 TECODNN_REDUCE_TENSOR_MIN,
                 TECODNN_NOT_PROPAGATE_NAN,
                 TECODNN_REDUCE_TENSOR_NO_INDICES,
                 ToTecodnnIndiceDataType(x.dtype()),
                 y);
}

void doMaxTensor(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const std::vector<int64_t>& reduce_dims,
                 phi::DenseTensor* y) {
  doReduceTensor(dev_ctx,
                 x,
                 reduce_dims,
                 TECODNN_REDUCE_TENSOR_MAX,
                 TECODNN_NOT_PROPAGATE_NAN,
                 TECODNN_REDUCE_TENSOR_NO_INDICES,
                 ToTecodnnIndiceDataType(x.dtype()),
                 y);
}

template <typename T>
void doElementWise(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   int axis,
                   T mode,
                   funcs::ElementwiseFunc tecodnnElementwiseFunctor,
                   phi::DenseTensor* out) {
  std::vector<int> x_expanded_dims, y_expanded_dims;
  custom_kernel::broadcastDims(
      x.dims(), y.dims(), axis, &x_expanded_dims, &y_expanded_dims);

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

  auto x_dims = x_expanded_dims;
  auto y_dims = y_expanded_dims;
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  auto x_desc =
      GetTecodnnTensorDesc(x_dims, x.dtype(), TensorFormat::Undefined);
  auto y_desc =
      GetTecodnnTensorDesc(y_dims, x.dtype(), TensorFormat::Undefined);
  auto out_desc =
      GetTecodnnTensorDesc(out_dims, x.dtype(), TensorFormat::Undefined);

  TECODNN_CHECK(tecodnnElementwiseFunctor(tecodnnHandle,
                                          x_desc,
                                          x.data(),
                                          y_desc,
                                          y.data(),
                                          out_desc,
                                          out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_desc));
}

void doElementAdd(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out) {
  doElementWise(dev_ctx, x, y, axis, BINARY_ADD, tecodnnAddTensorEx, out);
}

void doElementSub(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out) {
  doElementWise(dev_ctx, x, y, axis, BINARY_SUB, tecodnnSubTensorEx, out);
}

void doElementMul(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out) {
  doElementWise(dev_ctx, x, y, axis, BINARY_MUL, tecodnnMulTensorEx, out);
}

void doElementDiv(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out) {
  doElementWise(dev_ctx, x, y, axis, BINARY_DIV, tecodnnDivTensorEx, out);
}

void doReciprocalTensor(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        phi::DenseTensor* out) {
  VLOG(4) << "call tecodnn reciprocal op.";

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> x_dimensions(4, 1);
  int dims_merge = 1;
  if (x_dims.size() > 4) {
    for (int i = 0; i < x_dims.size() - 3; i++) {
      dims_merge *= x_dims[i];
    }
    x_dimensions[0] = dims_merge;
    std::copy(x_dims.end() - 3, x_dims.end(), x_dimensions.end() - 3);
  } else {
    std::copy(x_dims.begin(), x_dims.end(), x_dimensions.begin());
  }

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc =
      GetTecodnnTensorDesc(x_dimensions, x.dtype(), TensorFormat::NHWC);

  phi::DenseTensor x_(x);

  TECODNN_CHECK(tecodnnReciprocalTensor(
      tecodnnHandle, x_Desc, x_.data(), x_Desc, out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
}

void doSoftmaxForward(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      int axis,
                      bool high_precision,
                      phi::DenseTensor* out) {
  VLOG(4) << "call tecodnn softmaxforward op";

  if (axis < 0) {
    axis += x.dims().size();
  }
  PADDLE_ENFORCE_EQ(
      axis,
      x.dims().size() - 1,
      phi::errors::InvalidArgument(
          "sdaa only support softmax on last dimension"
          "But recieved: softmax dimension is %d, last dimension is [%s]",
          axis,
          x.dims().size() - 1));

  int N = 1, C = 1, H = 1, W = 1;
  for (int64_t i = 0; i < axis; ++i) N *= x.dims()[i];
  for (int64_t i = axis; i < x.dims().size(); ++i) C *= x.dims()[i];

  std::vector<int> dims = {N, C, H, W};

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t Desc =
      GetTecodnnTensorDesc(dims, x.dtype(), TensorFormat::NHWC);

  const float alpha = 1.0f, beta = 0.0f;

  tecodnnSoftmaxAlgorithm_t softmax_algo = TECODNN_SOFTMAX_ACCURATE;
  if (high_precision) softmax_algo = TECODNN_SOFTMAX_ACCURATE_HIGH_PRECISION;

  TECODNN_CHECK(tecodnnSoftmaxForward(tecodnnHandle,
                                      softmax_algo,
                                      TECODNN_SOFTMAX_MODE_INSTANCE,
                                      &alpha,
                                      Desc,
                                      x.data(),
                                      &beta,
                                      Desc,
                                      out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(Desc));
}

void doSoftmaxBackward(const Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& dout,
                       int axis,
                       bool high_precision,
                       phi::DenseTensor* dx) {
  VLOG(4) << "call tecodnn softmaxbackward op";

  if (axis < 0) {
    axis += out.dims().size();
  }
  PADDLE_ENFORCE_EQ(
      axis,
      out.dims().size() - 1,
      phi::errors::InvalidArgument(
          "sdaa only support softmax on last dimension"
          "But recieved: softmax dimension is %d, last dimension is [%s]",
          axis,
          out.dims().size() - 1));

  int N = 1, C = 1, H = 1, W = 1;
  for (int64_t i = 0; i < axis; ++i) N *= out.dims()[i];
  for (int64_t i = axis; i < out.dims().size(); ++i) C *= out.dims()[i];

  std::vector<int> dims = {N, C, H, W};

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t Desc =
      GetTecodnnTensorDesc(dims, out.dtype(), TensorFormat::NHWC);

  const float alpha = 1.0f, beta = 0.0f;

  tecodnnSoftmaxAlgorithm_t softmax_algo = TECODNN_SOFTMAX_ACCURATE;
  if (high_precision) softmax_algo = TECODNN_SOFTMAX_ACCURATE_HIGH_PRECISION;

  TECODNN_CHECK(tecodnnSoftmaxBackward(tecodnnHandle,
                                       softmax_algo,
                                       TECODNN_SOFTMAX_MODE_INSTANCE,
                                       &alpha,
                                       Desc,
                                       out.data(),
                                       Desc,
                                       dout.data(),
                                       &beta,
                                       Desc,
                                       dx->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(Desc));
}

/*This function has not benn tested.*/
void doLogSoftmaxForward(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         int axis,
                         phi::DenseTensor* out) {
  VLOG(4) << "call tecodnn softmaxforward op";

  if (axis < 0) {
    axis += x.dims().size();
  }
  PADDLE_ENFORCE_EQ(
      axis,
      x.dims().size() - 1,
      phi::errors::InvalidArgument(
          "sdaa only support softmax on last dimension"
          "But recieved: softmax dimension is %d, last dimension is [%s]",
          axis,
          x.dims().size() - 1));

  int N = 1, C = 1, H = 1, W = 1;
  for (int64_t i = 0; i < axis; ++i) N *= x.dims()[i];
  for (int64_t i = axis; i < x.dims().size(); ++i) C *= x.dims()[i];

  std::vector<int> dims = {N, C, H, W};

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t Desc =
      GetTecodnnTensorDesc(dims, x.dtype(), TensorFormat::NHWC);

  const float alpha = 1.0f, beta = 0.0f;

  TECODNN_CHECK(tecodnnSoftmaxForward(tecodnnHandle,
                                      TECODNN_SOFTMAX_LOG,
                                      TECODNN_SOFTMAX_MODE_INSTANCE,
                                      &alpha,
                                      Desc,
                                      x.data(),
                                      &beta,
                                      Desc,
                                      out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(Desc));
}

/*This function has not benn tested.*/
void doLogSoftmaxBackward(const Context& dev_ctx,
                          const phi::DenseTensor& out,
                          const phi::DenseTensor& dout,
                          int axis,
                          phi::DenseTensor* dx) {
  VLOG(4) << "call tecodnn softmaxbackward op";

  if (axis < 0) {
    axis += out.dims().size();
  }
  PADDLE_ENFORCE_EQ(
      axis,
      out.dims().size() - 1,
      phi::errors::InvalidArgument(
          "sdaa only support softmax on last dimension"
          "But recieved: softmax dimension is %d, last dimension is [%s]",
          axis,
          out.dims().size() - 1));

  int N = 1, C = 1, H = 1, W = 1;
  for (int64_t i = 0; i < axis; ++i) N *= out.dims()[i];
  for (int64_t i = axis; i < out.dims().size(); ++i) C *= out.dims()[i];

  std::vector<int> dims = {N, C, H, W};

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t Desc =
      GetTecodnnTensorDesc(dims, out.dtype(), TensorFormat::NHWC);

  const float alpha = 1.0f, beta = 0.0f;

  TECODNN_CHECK(tecodnnSoftmaxBackward(tecodnnHandle,
                                       TECODNN_SOFTMAX_LOG,
                                       TECODNN_SOFTMAX_MODE_INSTANCE,
                                       &alpha,
                                       Desc,
                                       out.data(),
                                       Desc,
                                       dout.data(),
                                       &beta,
                                       Desc,
                                       dx->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(Desc));
}

/*This function only tests NCHW2NHWC.*/
phi::DDim doDimPermute(const phi::DenseTensor& x, Convert_TF convert_tf) {
  std::vector<int> dim_permute;
  switch (convert_tf) {
    case Convert_TF::NCHW2NHWC:
      dim_permute = {0, 2, 3, 1};
      break;
    case Convert_TF::NCHW2CHWN:
      dim_permute = {1, 2, 3, 0};
      break;
    case Convert_TF::NHWC2NCHW:
      dim_permute = {0, 3, 1, 2};
      break;
    case Convert_TF::NHWC2CHWN:
      dim_permute = {3, 1, 2, 0};
      break;
    case Convert_TF::CHWN2NCHW:
      dim_permute = {3, 0, 1, 2};
      break;
    case Convert_TF::CHWN2NHWC:
      dim_permute = {3, 1, 2, 0};
    default:
      break;
  }

  std::vector<int> out_format_dim =
      phi::vectorize<int>(x.dims().transpose(dim_permute));
  phi::DDim out_dim = phi::make_ddim(out_format_dim);

  return out_dim;
}

template <typename T>
void doSliceTensor(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const std::vector<T>& axes,
                   const std::vector<T>& starts,
                   const std::vector<T>& ends,
                   const std::vector<T>& strides,
                   const std::vector<int64_t>& decrease_axis,
                   phi::DenseTensor* out) {
  VLOG(4) << "call tecodnn slice tensor op";

  int axes_num = axes.size();
  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  // support Zero-Dim.
  if (!out_dims.size()) {
    out_dims.emplace_back(1);
  }

  if (x_dims.size() != out_dims.size()) {
    for (int i = 0; i < decrease_axis.size(); i++) {
      out_dims.insert(out_dims.begin() + decrease_axis[i], 1);
    }
  }

  PADDLE_ENFORCE_LE(
      axes_num,
      kSliceMaxNum,
      phi::errors::InvalidArgument(
          "The max slice axes num is %d, but got %d.", kSliceMaxNum, axes_num));
  int axes_arr[kSliceMaxNum];
  std::copy(axes.begin(), axes.end(), axes_arr);
  if (x_dims.size() < 4) {
    int add_factor = 4 - x_dims.size();
    for (int i = 0; i < axes_num; i++) {
      axes_arr[i] += add_factor;
    }
  }

  int starts_arr[kSliceMaxNum];
  std::copy(starts.begin(), starts.end(), starts_arr);
  int ends_arr[kSliceMaxNum];
  std::copy(ends.begin(), ends.end(), ends_arr);
  int strides_arr[kSliceMaxNum];
  std::copy(strides.begin(), strides.end(), strides_arr);

  int* axes_ptr = &axes_arr[0];
  int* starts_ptr = &starts_arr[0];
  int* ends_ptr = &ends_arr[0];
  int* strides_ptr = &strides_arr[0];

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc =
      GetTecodnnTensorDesc(x_dims, x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t out_Desc =
      GetTecodnnTensorDesc(out_dims, out->dtype(), TensorFormat::NHWC);

  TECODNN_CHECK(tecodnnSlice(tecodnnHandle,
                             axes_num,
                             axes_ptr,
                             starts_ptr,
                             ends_ptr,
                             strides_ptr,
                             x_Desc,
                             x.data(),
                             out_Desc,
                             out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}
template decltype(doSliceTensor<int64_t>) doSliceTensor<int64_t>;
template decltype(doSliceTensor<int>) doSliceTensor<int>;

void Padding(const Context& dev_ctx,
             const phi::DenseTensor& x,
             const std::vector<std::vector<int>>& paddings,
             const std::vector<int>& x_dims,
             const std::vector<int>& out_dims,
             phi::DenseTensor* out) {
  tecodnnTensorTransformDescriptor_t tensorTranDesc;
  tecodnnFoldingDirection_t direction = TECODNN_TRANSFORM_FOLD;
  TECODNN_CHECK(tecodnnCreateTensorTransformDescriptor(&tensorTranDesc));

  constexpr unsigned int nbDims = 4;
  int PadBeforeArr[nbDims] = {
      paddings[0][0], paddings[0][3], paddings[0][1], paddings[0][2]};
  int PadAfterArr[nbDims] = {
      paddings[1][0], paddings[1][3], paddings[1][1], paddings[1][2]};
  unsigned int foldA[4] = {1, 1, 1, 1};
  float alpha = 1.0, beta = 0.0;

  TECODNN_CHECK(tecodnnSetTensorTransformDescriptor(tensorTranDesc,
                                                    nbDims,
                                                    TECODNN_TENSOR_NHWC,
                                                    PadBeforeArr,
                                                    PadAfterArr,
                                                    foldA,
                                                    direction));

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc =
      GetTecodnnTensorDesc(x_dims, x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t out_Desc =
      GetTecodnnTensorDesc(out_dims, out->dtype(), TensorFormat::NHWC);

  TECODNN_CHECK(tecodnnTransformTensorEx(tecodnnHandle,
                                         tensorTranDesc,
                                         &alpha,
                                         x_Desc,
                                         x.data(),
                                         &beta,
                                         out_Desc,
                                         out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorTransformDescriptor(tensorTranDesc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}

void doPaddingTensor(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const std::vector<std::vector<int>>& Paddings,
                     phi::DenseTensor* out) {
  VLOG(4) << "call tecodnn padding tensor";

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  std::vector<std::vector<int>> paddings(2, std::vector<int>(x_dims.size()));
  std::copy(Paddings[0].begin(), Paddings[0].end(), paddings[0].begin());
  std::copy(Paddings[1].begin(), Paddings[1].end(), paddings[1].begin());

  PADDLE_ENFORCE_EQ(paddings.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "The vector of paddings size must be equal to 2."
                        "But got %d.",
                        paddings.size()));

  PADDLE_ENFORCE_EQ(
      paddings[0].size(),
      paddings[1].size(),
      phi::errors::InvalidArgument(
          "The size of before padding size must be equal to "
          "after padding size. But got before padding size is %d, "
          "after padding size is %d.",
          paddings[0].size(),
          paddings[1].size()));

  PADDLE_ENFORCE_EQ(
      paddings[0].size(),
      x_dims.size(),
      phi::errors::InvalidArgument(
          "The size of padding size must be equal to the input tensor "
          "dimension size. But got padding size is %d, the input tensor"
          "dimension size is %d.",
          paddings[0].size(),
          x_dims.size()));

  int Dims = paddings[0].size();
  if (Dims <= 4) {
    // because GetTecodnnTensorDesc() fills 1 in front, 0 is also filled in
    // front
    for (int i = 0; i < 4 - Dims; i++) {
      paddings[0].insert(paddings[0].begin(), 0);
      paddings[1].insert(paddings[1].begin(), 0);
    }

    Padding(dev_ctx, x, paddings, x_dims, out_dims, out);
  } else {
    for (int i = 0; i < Dims; i++) {
      if (!(paddings[0][i] == 0 && paddings[1][i] == 0)) {
        int x_dim_before_axis = std::accumulate(
            x_dims.begin(), x_dims.begin() + i, 1, std::multiplies<int>());
        int x_dim_after_axis = std::accumulate(
            x_dims.begin() + i + 1, x_dims.end(), 1, std::multiplies<int>());
        std::vector<int> x_dim_tmp = {
            1, x_dim_before_axis, x_dims[i], x_dim_after_axis};

        int out_dim_before_axis = std::accumulate(
            out_dims.begin(), out_dims.begin() + i, 1, std::multiplies<int>());
        int out_dim_after_axis = std::accumulate(
            x_dims.begin() + i + 1, x_dims.end(), 1, std::multiplies<int>());
        std::vector<int> out_dim_tmp = {
            1, out_dim_before_axis, out_dims[i], out_dim_after_axis};

        std::vector<int> PadBefore = {0, 0, paddings[0][i], 0};
        std::vector<int> PadAfter = {0, 0, paddings[1][i], 0};
        std::vector<std::vector<int>> paddings_tmp = {std::move(PadBefore),
                                                      std::move(PadAfter)};

        Padding(dev_ctx, x, paddings_tmp, x_dim_tmp, out_dim_tmp, out);
      }
    }
  }
}

static bool IsAscPerm(const std::vector<int>& perm) {
  int st = 0;
  for (const int x : perm) {
    if (x != st++) return false;
  }
  return true;
}

paddle::optional<std::tuple<phi::DDim, phi::DDim>> TryDDimFusion(
    const phi::DDim& x_dims,
    const phi::DDim& out_dims,
    const std::vector<int>& axis) {
  auto iter = TransposeModeMap.find(axis);
  if (iter == TransposeModeMap.end()) {
    return paddle::none;
  }

  std::vector<int64_t> x_dimension, out_dimension;

  if (x_dims.size() == 2) {
    x_dimension = {1, x_dims[0], x_dims[1], 1};
    out_dimension = {1, out_dims[0], out_dims[1], 1};
  } else if (x_dims.size() == 3) {
    std::vector<int> unique_axis = {0, 2, 1};
    if (axis == unique_axis) {
      x_dimension = {x_dims[0], x_dims[1], x_dims[2], 1};
      out_dimension = {out_dims[0], out_dims[1], out_dims[2], 1};
    } else {
      x_dimension = {1, x_dims[0], x_dims[1], x_dims[2]};
      out_dimension = {1, out_dims[0], out_dims[1], out_dims[2]};
    }
  } else if (x_dims.size() == 4) {
    x_dimension = phi::vectorize(x_dims);
    out_dimension = phi::vectorize(out_dims);
  } else if (x_dims.size() == 5) {
    std::vector<int> axis_01342 = {0, 1, 3, 4, 2};
    std::vector<int> axis_23401 = {2, 3, 4, 0, 1};
    std::vector<int> axis_01423 = {0, 1, 4, 2, 3};
    std::vector<int> axis_42301 = {4, 2, 3, 0, 1};
    std::vector<int> axis_01324 = {0, 1, 3, 2, 4};

    std::vector<int> axis_03412 = {0, 3, 4, 1, 2};
    std::vector<int> axis_12340 = {1, 2, 3, 4, 0};
    std::vector<int> axis_04123 = {0, 4, 1, 2, 3};
    std::vector<int> axis_41230 = {4, 1, 2, 3, 0};
    std::vector<int> axis_03124 = {0, 3, 1, 2, 4};
    std::vector<int> axis_40123 = {4, 0, 1, 2, 3};

    std::vector<int> axis_02341 = {0, 2, 3, 4, 1};
    std::vector<int> axis_02314 = {0, 2, 3, 1, 4};

    std::vector<int> axis_34120 = {3, 4, 1, 2, 0};
    std::vector<int> axis_02134 = {0, 2, 1, 3, 4};
    std::vector<int> axis_34012 = {3, 4, 0, 1, 2};

    if (axis == axis_01342 || axis == axis_01423 ||
        axis == axis_01324) {  // dim0, dim1
      x_dimension = {x_dims[0] * x_dims[1], x_dims[2], x_dims[3], x_dims[4]};
      out_dimension = {
          out_dims[0] * out_dims[1], out_dims[2], out_dims[3], out_dims[4]};
    } else if (axis == axis_23401 || axis == axis_42301) {
      x_dimension = {x_dims[0] * x_dims[1], x_dims[2], x_dims[3], x_dims[4]};
      out_dimension = {
          out_dims[0], out_dims[1], out_dims[2], out_dims[3] * out_dims[4]};
    } else if (axis == axis_04123 || axis == axis_03124 ||
               axis == axis_40123) {  // dim1, dim2
      x_dimension = {x_dims[0], x_dims[1] * x_dims[2], x_dims[3], x_dims[4]};
      out_dimension = {
          out_dims[0], out_dims[1], out_dims[2] * out_dims[3], out_dims[4]};
    } else if (axis == axis_03412) {
      x_dimension = {x_dims[0], x_dims[1] * x_dims[2], x_dims[3], x_dims[4]};
      out_dimension = {
          out_dims[0], out_dims[1], out_dims[2], out_dims[3] * out_dims[4]};
    } else if (axis == axis_12340) {
      x_dimension = {x_dims[0], x_dims[1] * x_dims[2], x_dims[3], x_dims[4]};
      out_dimension = {
          out_dims[0] * out_dims[1], out_dims[2], out_dims[3], out_dims[4]};
    } else if (axis == axis_41230) {
      x_dimension = {x_dims[0], x_dims[1] * x_dims[2], x_dims[3], x_dims[4]};
      out_dimension = {
          out_dims[0], out_dims[1] * out_dims[2], out_dims[3], out_dims[4]};
    } else if (axis == axis_02314 || axis == axis_02341) {  // dim2, dim3
      x_dimension = {x_dims[0], x_dims[1], x_dims[2] * x_dims[3], x_dims[4]};
      out_dimension = {
          out_dims[0], out_dims[1] * out_dims[2], out_dims[3], out_dims[4]};
    } else if (axis == axis_34120 || axis == axis_34012) {  // dim3, dim4
      x_dimension = {x_dims[0], x_dims[1], x_dims[2], x_dims[3] * x_dims[4]};
      out_dimension = {
          out_dims[0] * out_dims[1], out_dims[2], out_dims[3], out_dims[4]};
    } else if (axis == axis_02134) {
      x_dimension = {x_dims[0], x_dims[1], x_dims[2], x_dims[3] * x_dims[4]};
      out_dimension = {
          out_dims[0], out_dims[1], out_dims[2], out_dims[3] * out_dims[4]};
    }
  }
  return paddle::make_optional(std::make_tuple(phi::make_ddim(x_dimension),
                                               phi::make_ddim(out_dimension)));
}

void doTransposeTensor(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const std::vector<int>& axis,
                       phi::DenseTensor* out) {
  VLOG(4) << "call tecodnn transpose tensor";

  // FIXME(huangzhen): not sure whether the efficiency of the
  // dnn for incremental perm is higher than that of copy, so
  // keep it
  if (IsAscPerm(axis)) {
    phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    return;
  }

  // argument constraints
  PADDLE_ENFORCE_LT(
      axis.size(),
      8,
      phi::errors::InvalidArgument(
          "The max number of dimensions supported by tecodnn is 7."));

  auto x_out_dims = TryDDimFusion(x.dims(), out->dims(), axis);

  if (x_out_dims) {
    phi::DDim& fused_x_dims = std::get<0>(x_out_dims.get());
    phi::DDim& fused_out_dimension = std::get<1>(x_out_dims.get());
    Convert_TF convert_tf = GetTransposeTensorFormat(axis);
    std::pair<TensorFormat, TensorFormat> in_out_tf =
        GetTecodnnConvertTensorFormat(convert_tf);

    tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

    tecodnnTensorDescriptor_t x_Desc = GetTecodnnTensorDesc(
        phi::vectorize<int>(fused_x_dims), x.dtype(), in_out_tf.first);
    tecodnnTensorDescriptor_t y_Desc =
        GetTecodnnTensorDesc(phi::vectorize<int>(fused_out_dimension),
                             out->dtype(),
                             in_out_tf.second);

    float alpha = 1.0, beta = 0.0;
    TECODNN_CHECK(tecodnnTransformTensor(
        tecodnnHandle, &alpha, x_Desc, x.data(), &beta, y_Desc, out->data()));

    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));

    return;
  }

  tecodnnTensorDescriptor_t input_desc = GetTecodnnTensorDesc(
      phi::vectorize<int>(x.dims()), x.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t output_desc = GetTecodnnTensorDesc(
      phi::vectorize<int>(out->dims()), out->dtype(), TensorFormat::Undefined);

  tecodnnHandle_t tecodnn_handler = GetHandleFromCTX(dev_ctx);
  TECODNN_CHECK(tecodnnTranspose(tecodnn_handler,
                                 axis.data(),
                                 input_desc,
                                 x.data(),
                                 output_desc,
                                 out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(input_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(output_desc));
}

void doLogicTensor(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const std::vector<int64_t>& axis_reduce,
                   TensorLogicType TLT,
                   phi::DenseTensor* out) {
  VLOG(4) << "call tecodnn logic tensor";
  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  PADDLE_ENFORCE_LE(axis_reduce.size(),
                    1,
                    phi::errors::InvalidArgument(
                        "tecodnn lib do not support dims as list or tuple"));

  int x_ndim = x_dims.size();

  phi::DenseTensor x_int;
  x_int.Resize(x.dims());
  dev_ctx.Alloc(&x_int, DataType::INT32);

  phi::DenseTensor out_int;
  out_int.Resize(out->dims());
  dev_ctx.Alloc(&out_int, DataType::INT32);

  doCastTensor(dev_ctx, x, &x_int);

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_int_Desc =
      GetTecodnnTensorDesc(x_dims, x_int.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t out_int_Desc =
      GetTecodnnTensorDesc(out_dims, out_int.dtype(), TensorFormat::Undefined);

  if (axis_reduce.empty()) {
    if (TLT == TensorLogicType::all) {
      TECODNN_CHECK(tecodnnTensorAll(tecodnnHandle,
                                     NULL,
                                     x_int_Desc,
                                     x_int.data(),
                                     out_int_Desc,
                                     out_int.data()));
    } else if (TLT == TensorLogicType::any) {
      TECODNN_CHECK(tecodnnTensorAny(tecodnnHandle,
                                     NULL,
                                     x_int_Desc,
                                     x_int.data(),
                                     out_int_Desc,
                                     out_int.data()));
    }
  } else {
    int axis = axis_reduce[0] > 0 ? axis_reduce[0] : axis_reduce[0] + x_ndim;
    if (TLT == TensorLogicType::all) {
      TECODNN_CHECK(tecodnnTensorAll(tecodnnHandle,
                                     &axis,
                                     x_int_Desc,
                                     x_int.data(),
                                     out_int_Desc,
                                     out_int.data()));
    } else if (TLT == TensorLogicType::any) {
      TECODNN_CHECK(tecodnnTensorAny(tecodnnHandle,
                                     &axis,
                                     x_int_Desc,
                                     x_int.data(),
                                     out_int_Desc,
                                     out_int.data()));
    }
  }

  doCastTensor(dev_ctx, out_int, out);

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_int_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_int_Desc));
}

void doConcatTensor(const Context& dev_ctx,
                    const std::vector<const phi::DenseTensor*>& x,
                    int axis,
                    phi::DenseTensor* out) {
  VLOG(4) << "tecodnn concat tensor called";

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

  if (x.size() == 1) {
    // memcpy is most efficient in single concat situation
    phi::Copy(dev_ctx, *x[0], dev_ctx.GetPlace(), false, out);
    return;
  }

  std::vector<void*> input_ptr;
  std::vector<tecodnnTensorDescriptor_t> Desc;
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  for (int i = 0; i < x.size(); i++) {
    std::vector<int> every_x_dims = phi::vectorize<int>(x[i]->dims());
    void* every_x_ptr = const_cast<void*>(x[i]->data());

    input_ptr.push_back(every_x_ptr);
    tecodnnTensorDescriptor_t every_x_Desc = sdaa_ops::GetTecodnnTensorDesc(
        every_x_dims, x[i]->dtype(), TensorFormat::Undefined);
    Desc.push_back(every_x_Desc);
  }

  size_t extra_input_size;
  TECODNN_CHECK(
      tecodnnGetConcatExtraInputSize(Desc.data(), x.size(), &extra_input_size));

  int sizeworkspaceInBytes = input_ptr.size() * sizeof(void*);
  int kPaddingSize = 128;
  int64_t paddingSize =
      (static_cast<int64_t>(extra_input_size) + kPaddingSize * 2) /
      kPaddingSize * kPaddingSize;
  int64_t hostInputSize = paddingSize + sizeworkspaceInBytes;

  std::vector<int8_t> host_input(hostInputSize);

  TECODNN_CHECK(
      tecodnnConcatInitExtraInput(Desc.data(), x.size(), host_input.data()));
  memcpy(
      host_input.data() + paddingSize, input_ptr.data(), sizeworkspaceInBytes);

  tecodnnTensorDescriptor_t out_Desc = sdaa_ops::GetTecodnnTensorDesc(
      out_dims, out->dtype(), TensorFormat::Undefined);

  phi::DenseTensor tmp;
  tmp.Resize({hostInputSize});
  dev_ctx.Alloc(&tmp, phi::DataType::INT8);
  AsyncMemCpyH2D(nullptr,
                 static_cast<C_Stream>(dev_ctx.stream()),
                 tmp.data(),
                 host_input.data(),
                 hostInputSize);

  TECODNN_CHECK(
      tecodnnConcat(tecodnnHandle,
                    axis,
                    x.size(),
                    Desc.data(),
                    reinterpret_cast<void**>(tmp.data<int8_t>() + paddingSize),
                    out_Desc,
                    out->data(),
                    tmp.data(),
                    extra_input_size));

  for (int i = 0; i < Desc.size(); i++) {
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(Desc[i]));
  }
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}

void doScatterTensor(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& index,
                     const phi::DenseTensor& updates,
                     bool overwrite,
                     phi::DenseTensor* out) {
  VLOG(4) << "tecodnn scatter tensor called";

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> index_dims = phi::vectorize<int>(index.dims());
  std::vector<int> updates_dims = phi::vectorize<int>(updates.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  // support 0D tensor.
  if (index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(
        index_dims[1],
        1,
        phi::errors::InvalidArgument("index_dims[1] should be 1 when "
                                     "index_dims.size() =2 in scatter_op."
                                     "But received value is [%d]",
                                     index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        index_dims.size() == 1 || index_dims.size() == 0,
        true,
        phi::errors::InvalidArgument("index_dims.size() should be 0, 1 or 2 in "
                                     "scatter_op. But received value is [%d]",
                                     index_dims.size()));
  }
  if (index_dims.size() != 0) {
    // check updates shape and out shape should match
    for (int i = 1; i < updates_dims.size(); i++)
      PADDLE_ENFORCE_EQ(
          updates_dims[i],
          out_dims[i],
          phi::errors::InvalidArgument(
              "The dimensions of the source tensor and target tensor should"
              " match, but received source tensor's %d-th dimension is %d,"
              "target tensor's %d-th dimension is %d.",
              i,
              updates_dims[i],
              i,
              out_dims[i]));
  } else {
    // if index.dims().size() == 0, need to check whether
    // updates.dims().size()
    // == x.dims().size()
    if (x_dims.size() != updates_dims.size())
      updates_dims.insert(updates_dims.begin(), 1);
  }

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnScatterMode_t ScatterMode = TECODNN_SCATTER_ACCUMULATE;

  if (overwrite) {
    ScatterMode = TECODNN_SCATTER_OVERWRITE;
  }

  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t updates_Desc = sdaa_ops::GetTecodnnTensorDesc(
      updates_dims, updates.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t index_Desc = sdaa_ops::GetTecodnnTensorDesc(
      index_dims, index.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t out_Desc = sdaa_ops::GetTecodnnTensorDesc(
      out_dims, out->dtype(), TensorFormat::Undefined);

  TECODNN_CHECK(tecodnnScatter(tecodnnHandle,
                               ScatterMode,
                               x_Desc,
                               x.data(),
                               updates_Desc,
                               updates.data(),
                               index_Desc,
                               index.data(),
                               out_Desc,
                               out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(updates_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(index_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}

void doSplitTensor(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int axis,
                   std::vector<phi::DenseTensor*> outs) {
  VLOG(4) << "tecodnn split tensor called";

  if (outs.size() == 1) {
    // memcpy is most efficient in single split situation
    phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, outs[0]);
    return;
  }

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

  std::vector<void*> outs_ptr;
  std::vector<tecodnnTensorDescriptor_t> outs_Desc;
  std::vector<int> x_dims = phi::vectorize<int>(x.dims());

  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);

  for (int i = 0; i < outs.size(); i++) {
    std::vector<int> every_out_dims = phi::vectorize<int>(outs[i]->dims());
    void* every_out_ptr = outs[i]->data();

    outs_ptr.push_back(every_out_ptr);
    tecodnnTensorDescriptor_t every_out_Desc = sdaa_ops::GetTecodnnTensorDesc(
        every_out_dims, outs[i]->dtype(), TensorFormat::Undefined);
    outs_Desc.push_back(every_out_Desc);
  }

  size_t extra_out_size;
  TECODNN_CHECK(tecodnnGetSplitExtraInputSize(
      outs_Desc.data(), outs.size(), &extra_out_size));

  int sizeworkspaceInBytes = outs_ptr.size() * sizeof(void*);
  int kPaddingSize = 128;
  int64_t paddingSize =
      (static_cast<int64_t>(extra_out_size) + kPaddingSize * 2) / kPaddingSize *
      kPaddingSize;
  int64_t hostOutputSize = paddingSize + sizeworkspaceInBytes;

  std::vector<int8_t> host_output(hostOutputSize);

  TECODNN_CHECK(tecodnnSplitInitExtraInput(
      outs_Desc.data(), outs.size(), host_output.data()));
  memcpy(
      host_output.data() + paddingSize, outs_ptr.data(), sizeworkspaceInBytes);

  phi::DenseTensor tmp;
  tmp.Resize(phi::make_ddim({hostOutputSize}));
  dev_ctx.Alloc(&tmp, phi::DataType::INT8);
  AsyncMemCpyH2D(nullptr,
                 static_cast<C_Stream>(dev_ctx.stream()),
                 tmp.data(),
                 host_output.data(),
                 hostOutputSize);

  TECODNN_CHECK(
      tecodnnSplit(tecodnnHandle,
                   axis,
                   outs.size(),
                   x_Desc,
                   const_cast<void*>(x.data()),
                   outs_Desc.data(),
                   reinterpret_cast<void**>(tmp.data<int8_t>() + paddingSize),
                   tmp.data(),
                   extra_out_size));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  for (int i = 0; i < outs_Desc.size(); i++) {
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(outs_Desc[i]));
  }
}

void doExpandTensor(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    phi::DenseTensor* out) {
  VLOG(4) << "tecodnn expand tensor called.";

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t out_Desc = sdaa_ops::GetTecodnnTensorDesc(
      out_dims, out->dtype(), TensorFormat::Undefined);

  TECODNN_CHECK(
      tecodnnExpand(tecodnnHandle, x_Desc, x.data(), out_Desc, out->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}

void doNearestInterpolateForward(const Context& dev_ctx,
                                 const phi::DenseTensor& x,
                                 const float ratio_w,
                                 const float ratio_h,
                                 const float ratio_d,
                                 const bool align_corners,
                                 phi::DenseTensor* out) {
  // Note: data layout of doNearestInterpolateForward only support NHWC
  VLOG(4) << "tecodnn nearest interpolate forward called";

  PADDLE_ENFORCE_EQ(
      align_corners,
      false,
      phi::errors::Unimplemented("sdaa only support `align_corners=false`"));

  tecodnnHandle_t tecodnn_handle = GetHandleFromCTX(dev_ctx);

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  if (x_dims.size() == 4) {
    // 4-D Tensor
    tecodnnTensorDescriptor_t x_desc =
        sdaa_ops::GetTecodnnTensorDesc(x_dims, x.dtype(), TensorFormat::NHWC);
    tecodnnTensorDescriptor_t out_desc = sdaa_ops::GetTecodnnTensorDesc(
        out_dims, out->dtype(), TensorFormat::NHWC);

    TECODNN_CHECK(tecodnnUpSample(tecodnn_handle,
                                  ratio_h,
                                  ratio_w,
                                  x_desc,
                                  x.data(),
                                  out_desc,
                                  out->data()));

    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_desc));
  } else if (x_dims.size() == 5) {
    // TODO(zhangrb): for future
    // 5-D Tensor

    // tecodnn unimplement
    PADDLE_THROW(phi::errors::Unimplemented(
        "Nearest interpolate of 5-D tensor on %s is not supported.",
        dev_ctx.GetPlace()));

    PADDLE_ENFORCE_GT(
        ratio_d,
        0,
        phi::errors::InvalidArgument("sdaa only support ratio depth > 0. "
                                     "But recieved: ratio depth is %d",
                                     ratio_d));
  }
}

void doNearestInterpolateBackward(const Context& dev_ctx,
                                  const phi::DenseTensor& out,
                                  const float ratio_w,
                                  const float ratio_h,
                                  const float ratio_d,
                                  const bool align_corners,
                                  phi::DenseTensor* dx) {
  // Note: data layout of doNearestInterpolateBackward only support NHWC
  VLOG(4) << "tecodnn nearest interpolate backward called";

  tecodnnHandle_t tecodnn_handle = GetHandleFromCTX(dev_ctx);

  std::vector<int> out_dims = phi::vectorize<int>(out.dims());
  std::vector<int> dx_dims = phi::vectorize<int>(dx->dims());

  tecodnnAlignCorners_t tecodnn_align_corners =
      align_corners ? TECODNN_ALIGN_CORNERS : TECODNN_NOT_ALIGN_CORNERS;

  if (out_dims.size() == 4) {
    // 4-D Tensor
    tecodnnTensorDescriptor_t out_desc = sdaa_ops::GetTecodnnTensorDesc(
        out_dims, out.dtype(), TensorFormat::NHWC);
    tecodnnTensorDescriptor_t dx_desc = sdaa_ops::GetTecodnnTensorDesc(
        dx_dims, dx->dtype(), TensorFormat::NHWC);

    // tecodnn scale base on size when scale factors is [-1,-1]
    const float scale_factors[2] = {-1, -1};
    TECODNN_CHECK(tecodnnNearestInterpBackward(tecodnn_handle,
                                               tecodnn_align_corners,
                                               scale_factors,
                                               out_desc,
                                               out.data(),
                                               dx_desc,
                                               dx->data()));

    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(dx_desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_desc));
  } else if (out_dims.size() == 5) {
    // 5-D Tensor
    // TODO(zhangrb): for future

    // tecodnn unimplement
    PADDLE_THROW(phi::errors::Unimplemented(
        "Nearest interpolate of 5-D tensor on %s is not supported.",
        dev_ctx.GetPlace()));

    PADDLE_ENFORCE_GT(
        ratio_d,
        0,
        phi::errors::InvalidArgument("sdaa only support ratio depth > 0. But "
                                     "recieved: ratio depth is %d",
                                     ratio_d));
  }
}

void doBitwiseBinaryOpTensor(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::DenseTensor& y,
                             BitwiseOpType bitwiseType,
                             phi::DenseTensor* out) {
  VLOG(4) << "tecodnn bitwise op tensor called.";

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> y_dims = phi::vectorize<int>(y.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc =
      GetTecodnnTensorDesc(x_dims, x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t y_Desc =
      GetTecodnnTensorDesc(y_dims, y.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t out_Desc =
      GetTecodnnTensorDesc(out_dims, out->dtype(), TensorFormat::NHWC);

  switch (bitwiseType) {
    case BitwiseOpType::And:
      TECODNN_CHECK(tecodnnBitwiseAndTensor(tecodnnHandle,
                                            x_Desc,
                                            x.data(),
                                            y_Desc,
                                            y.data(),
                                            out_Desc,
                                            out->data()));
      break;
    case BitwiseOpType::Or:
      TECODNN_CHECK(tecodnnBitwiseOrTensor(tecodnnHandle,
                                           x_Desc,
                                           x.data(),
                                           y_Desc,
                                           y.data(),
                                           out_Desc,
                                           out->data()));
      break;
    case BitwiseOpType::Xor:
      TECODNN_CHECK(tecodnnBitwiseXorTensor(tecodnnHandle,
                                            x_Desc,
                                            x.data(),
                                            y_Desc,
                                            y.data(),
                                            out_Desc,
                                            out->data()));
      break;
    default:
      break;
  }

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}

void doBitwiseUnaryOpTensor(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            BitwiseOpType bitwiseType,
                            phi::DenseTensor* out) {
  VLOG(4) << "tecodnn bitwise op tensor called.";

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc =
      GetTecodnnTensorDesc(x_dims, x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t out_Desc =
      GetTecodnnTensorDesc(out_dims, out->dtype(), TensorFormat::NHWC);

  switch (bitwiseType) {
    case BitwiseOpType::Not:
      TECODNN_CHECK(tecodnnBitwiseNotTensor(
          tecodnnHandle, x_Desc, x.data(), out_Desc, out->data()));
      break;
    default:
      break;
  }

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}
void doLogicalOpTensor(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       LogicalOpType logicaltype,
                       phi::DenseTensor* out) {
  VLOG(4) << "tecodnn logical op tensor called.";

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> y_dims = phi::vectorize<int>(y.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  phi::DenseTensor out_int;
  out_int.Resize(out->dims());
  dev_ctx.Alloc(&out_int, DataType::INT32);

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc, y_Desc, out_Desc;

  if (logicaltype == LogicalOpType::And && x.dtype() == phi::DataType::BOOL) {
    x_Desc = GetTecodnnBoolTensorDesc(x_dims, TensorFormat::NHWC);
    y_Desc = GetTecodnnBoolTensorDesc(y_dims, TensorFormat::NHWC);
    out_Desc = GetTecodnnBoolTensorDesc(out_dims, TensorFormat::NHWC);
    TECODNN_CHECK(tecodnnLogicalAndTensor(tecodnnHandle,
                                          x_Desc,
                                          x.data(),
                                          y_Desc,
                                          y.data(),
                                          out_Desc,
                                          out->data()));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
    return;
  }

  x_Desc = GetTecodnnTensorDesc(x_dims, x.dtype(), TensorFormat::NHWC);
  y_Desc = GetTecodnnTensorDesc(y_dims, y.dtype(), TensorFormat::NHWC);
  out_Desc =
      GetTecodnnTensorDesc(out_dims, out_int.dtype(), TensorFormat::NHWC);

  switch (logicaltype) {
    case LogicalOpType::And:
      TECODNN_CHECK(tecodnnLogicalAndTensor(tecodnnHandle,
                                            x_Desc,
                                            x.data(),
                                            y_Desc,
                                            y.data(),
                                            out_Desc,
                                            out_int.data()));
      break;
    case LogicalOpType::Or:
      TECODNN_CHECK(tecodnnLogicalOrTensor(tecodnnHandle,
                                           x_Desc,
                                           x.data(),
                                           y_Desc,
                                           y.data(),
                                           out_Desc,
                                           out_int.data()));
      break;
    case LogicalOpType::Xor:
      TECODNN_CHECK(tecodnnLogicalXorTensor(tecodnnHandle,
                                            x_Desc,
                                            x.data(),
                                            y_Desc,
                                            y.data(),
                                            out_Desc,
                                            out_int.data()));
      break;
    default:
      break;
  }

  sdaa_ops::doCastTensor(dev_ctx, out_int, out);

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}

void doLogicalNotOpTensor(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          phi::DenseTensor* out) {
  VLOG(4) << "tecodnn Logical Notop tensor called.";

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  phi::DenseTensor out_int;
  out_int.Resize(out->dims());
  dev_ctx.Alloc(&out_int, DataType::INT32);

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc =
      GetTecodnnTensorDesc(x_dims, x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t out_Desc =
      GetTecodnnTensorDesc(out_dims, out_int.dtype(), TensorFormat::NHWC);

  TECODNN_CHECK(tecodnnLogicalNotTensor(
      tecodnnHandle, x_Desc, x.data(), out_Desc, out_int.data()));

  sdaa_ops::doCastTensor(dev_ctx, out_int, out);

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}

void doIsnanOp(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  VLOG(4) << "tecodnn isnan op called";

  // basic settings
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  std::vector<int> dims = phi::vectorize<int>(x.dims());

  tecodnnTensorDescriptor_t Desc_input =
      sdaa_ops::GetTecodnnTensorDesc(dims, x.dtype(), TensorFormat::NHWC);

  tecodnnTensorDescriptor_t Desc_output = sdaa_ops::GetTecodnnTensorDesc(
      dims, phi::DataType::BOOL, TensorFormat::NHWC);

  // Isnan op
  TECODNN_CHECK(tecodnnIsnan(
      tecodnnHandle, Desc_input, x.data(), Desc_output, out->data()));
  // destroy descriptors
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(Desc_input));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(Desc_output));
}

int64_t doAddStorageProperties(
    const Context& dev_ctx,
    phi::DenseTensor* tensor,
    SDAAStorageProperties& storage_properties) {  // NOLINT
  PADDLE_ENFORCE(
      tensor->valid(),
      phi::errors::InvalidArgument(
          "The input tensor of PrepareTensorWithFormat must be valid."));

  PADDLE_ENFORCE_EQ(
      tensor->storage_properties_initialized(),
      false,
      phi::errors::InvalidArgument(
          "Before storage_properties, The input tensor's storage Properties "
          "must be not intiialized."));
  int64_t requested_size = phi::product(storage_properties.storage_dims);
  auto sdaa_properties =
      std::make_unique<SDAAStorageProperties>(storage_properties);
  tensor->set_storage_properties(std::move(sdaa_properties));

  return requested_size;
}

void swapTensorData(const Context& dev_ctx,
                    const phi::DenseTensor& in,
                    SDAAStorageProperties& storage_properties) {  // NOLINT
  Convert_TF tf;
  switch (storage_properties.storage_format) {
    case StoragePropertiesCHWN:
      tf = Convert_TF::NCHW2CHWN;
      break;
    default:
      PADDLE_THROW(phi::errors::InvalidArgument("invaild storage format"));
      break;
  }
  phi::DenseTensor* temp_in = const_cast<phi::DenseTensor*>(&in);
  phi::DenseTensor trans_in;
  phi::DenseTensorMeta meta_in = {in.dtype(), doDimPermute(in, tf)};
  trans_in.set_meta(meta_in);
  dev_ctx.Alloc(&trans_in, in.dtype());
  doTransformTensor(dev_ctx, in, tf, &trans_in);  // CHWN
  AsyncMemCpyD2D(nullptr,
                 static_cast<C_Stream>(dev_ctx.stream()),
                 temp_in->data(),
                 trans_in.data(),
                 phi::SizeOf(in.dtype()) * trans_in.numel());
  doAddStorageProperties(dev_ctx, temp_in, storage_properties);
}

std::vector<int64_t> GetReduceDimAxis(const phi::DDim& in,
                                      const phi::DDim& out,
                                      int axis) {
  axis =
      (axis == -1 ? std::abs(static_cast<int>(out.size() - in.size())) : axis);
  std::vector<int64_t> dims;
  for (int i = 0; i < axis; ++i) {
    dims.push_back(i);
  }
  for (int i = 0; i < in.size(); ++i) {
    if (out[i + axis] != in[i]) {
      dims.push_back(i + axis);
    }
  }
  for (int i = axis + in.size(); i < out.size(); ++i) {
    dims.push_back(i);
  }
  return dims;
}

void BatchNormFunc(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& mean,
                   const phi::DenseTensor& variance,
                   const phi::DenseTensor& scale,
                   const phi::DenseTensor& bias,
                   float momentum,
                   float epsilon,
                   bool training,
                   const std::string& data_layout_str,
                   phi::DenseTensor* y,
                   phi::DenseTensor* mean_out,
                   phi::DenseTensor* variance_out,
                   phi::DenseTensor* saved_mean,
                   phi::DenseTensor* saved_variance) {
  // check arguments
  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      (x_dims.size() == 4UL || x_dims.size() == 3UL || x_dims.size() == 2UL),
      true,
      phi::errors::InvalidArgument(
          "The input tensor X's dimension must equal to 2, 3 or 4. "
          " But got X's shape = [%s], X's dimension = [%d].",
          x_dims.to_str(),
          x_dims.size()));

  PADDLE_ENFORCE_GT(
      epsilon,
      0.,
      phi::errors::InvalidArgument("epsilon should be greater than zero. "
                                   "But received epsilon = %f",
                                   static_cast<float>(epsilon)));

  phi::DataLayout data_layout = common::StringToDataLayout(data_layout_str);
  bool need_trans = data_layout == phi::DataLayout::kNCHW;

  int N, H, W, C, D;
  sdaa_ops::ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

  std::vector<int> sbmv_dims = {1, 1, 1, C};
  const float alpha = 1.0f, beta = 0.0f;

  // since the tecodnnBatchNormForward func only supports 4-D tensor,
  // when tensor dims=3, a dimensional complement is required.
  phi::DenseTensor x_temp(x), y_temp(*y);
  if (x_dims.size() < 4) {
    if (need_trans) {
      x_temp.Resize(phi::make_ddim({N, C, H, W}));
      y_temp.Resize(phi::make_ddim({N, C, H, W}));
    } else {
      x_temp.Resize(phi::make_ddim({N, H, W, C}));
      y_temp.Resize(phi::make_ddim({N, H, W, C}));
    }
  }

  phi::DenseTensor x_NHWC, y_NHWC;
  phi::DDim x_NHWC_dims, y_NHWC_dims;

  if (need_trans) {
    x_NHWC_dims = sdaa_ops::doDimPermute(x_temp, Convert_TF::NCHW2NHWC);
    y_NHWC_dims = sdaa_ops::doDimPermute(y_temp, Convert_TF::NCHW2NHWC);
    x_NHWC.Resize(x_NHWC_dims);
    y_NHWC.Resize(y_NHWC_dims);
    dev_ctx.Alloc(&x_NHWC, x.dtype());
    dev_ctx.Alloc(&y_NHWC, y->dtype());

    sdaa_ops::doTransformTensor(
        dev_ctx, x_temp, Convert_TF::NCHW2NHWC, &x_NHWC);
  } else {
    x_NHWC = x_temp;
    y_NHWC = y_temp;
  }

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnBatchNormMode_t bnMode = TECODNN_BATCHNORM_SPATIAL;

  tecodnnTensorDescriptor_t x_NHWC_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(x_NHWC.dims()), x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t y_NHWC_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(y_NHWC.dims()), y->dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t sbmv_NHWC_Desc = sdaa_ops::GetTecodnnTensorDesc(
      sbmv_dims, mean.dtype(), TensorFormat::NHWC);

  if (training) {
    TECODNN_CHECK(
        tecodnnBatchNormalizationForwardTraining(tecodnnHandle,
                                                 bnMode,
                                                 &alpha,
                                                 &beta,
                                                 x_NHWC_Desc,
                                                 x_NHWC.data(),
                                                 y_NHWC_Desc,
                                                 y_NHWC.data(),
                                                 sbmv_NHWC_Desc,
                                                 scale.data(),
                                                 bias.data(),
                                                 momentum,
                                                 mean_out->data(),
                                                 variance_out->data(),
                                                 epsilon,
                                                 saved_mean->data(),
                                                 saved_variance->data()));
  } else {
    TECODNN_CHECK(
        tecodnnBatchNormalizationForwardInference(tecodnnHandle,
                                                  bnMode,
                                                  &alpha,
                                                  &beta,
                                                  x_NHWC_Desc,
                                                  x_NHWC.data(),
                                                  y_NHWC_Desc,
                                                  y_NHWC.data(),
                                                  sbmv_NHWC_Desc,
                                                  scale.data(),
                                                  bias.data(),
                                                  mean_out->data(),
                                                  variance_out->data(),
                                                  epsilon));
  }

  if (need_trans) {
    sdaa_ops::doTransformTensor(
        dev_ctx, y_NHWC, Convert_TF::NHWC2NCHW, &y_temp);
  }

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_NHWC_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_NHWC_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(sbmv_NHWC_Desc));
}

void doMemsetTensor(const Context& dev_ctx,
                    const int value,
                    phi::DenseTensor* tensor) {
  tecodnnHandle_t handle = GetHandleFromCTX(dev_ctx);
  TECODNN_CHECK(tecodnnMemset(handle,
                              tensor->data(),
                              value,
                              tensor->numel() * phi::SizeOf(tensor->dtype())));
}

template <typename T>
void doScatterNdAdd(const Context& ctx,
                    const phi::DenseTensor& index,
                    const phi::DenseTensor& updates,
                    phi::DenseTensor* out) {
  const auto& index_type = index.dtype();

  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    phi::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s], but "
                        "desires to be [%s] or [%s].",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));

  auto updates_dims = phi::vectorize<int>(updates.dims());
  auto index_dims = phi::vectorize<int>(index.dims());
  auto out_dims = phi::vectorize<int>(out->dims());

  auto update_size = phi::SizeOf(updates.dtype()) * updates.numel();
  auto out_size = phi::SizeOf(out->dtype()) * out->numel();
  auto index_size = phi::SizeOf(index_type) * index.numel();

  PADDLE_ENFORCE_EQ(
      220 * 1024 >= update_size + index_size + (out_size + 31) / 32,
      true,
      phi::errors::InvalidArgument(
          "The sum of data size (updates size + index size + out size / 32 ) "
          "large than 220KB is not support on %s.",
          ctx.GetPlace()));

  sdaaStream_t custom_stream = GetStreamFromCTX(ctx);
  if (index_type == phi::DataType::INT64) {
    TCUS_CHECK((sdcops::scatter_nd_add<T, int64_t>)(updates.data(),
                                                    index.data(),
                                                    out->data(),
                                                    updates_dims.data(),
                                                    index_dims.data(),
                                                    out_dims.data(),
                                                    updates_dims.size(),
                                                    index_dims.size(),
                                                    out_dims.size(),
                                                    custom_stream));
  } else if (index_type == phi::DataType::INT32) {
    TCUS_CHECK((sdcops::scatter_nd_add<T, int>)(updates.data(),
                                                index.data(),
                                                out->data(),
                                                updates_dims.data(),
                                                index_dims.data(),
                                                out_dims.data(),
                                                updates_dims.size(),
                                                index_dims.size(),
                                                out_dims.size(),
                                                custom_stream));
  }
}

template void doScatterNdAdd<float>(const Context& ctx,
                                    const phi::DenseTensor& index,
                                    const phi::DenseTensor& updates,
                                    phi::DenseTensor* out);

void GetReduceDimReduceAll(const std::vector<int>& axis_dims,
                           int input_dims_size,
                           bool reduce_all,
                           std::vector<int>* reduce_dims) {
  if (reduce_all) {
    for (int i = 0; i < input_dims_size; ++i) {
      (*reduce_dims).push_back(i);
    }
  } else {
    for (auto e : axis_dims) {
      PADDLE_ENFORCE_LT(e,
                        input_dims_size,
                        phi::errors::InvalidArgument(
                            "ReduceOp: invalid axis, when x_dims is %d, "
                            "axis[i] should less than x_dims, but got %d.",
                            input_dims_size,
                            e));
      (*reduce_dims).push_back(e >= 0 ? e : e + input_dims_size);
    }
  }
}

}  // namespace sdaa_ops
}  // namespace custom_kernel
