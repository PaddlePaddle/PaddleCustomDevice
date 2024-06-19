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
}  // namespace sdaa_ops
}  // namespace custom_kernel
