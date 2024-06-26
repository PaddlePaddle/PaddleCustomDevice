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

#pragma once
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "kernels/funcs/sdaa_baseop.h"
#include "kernels/funcs/sdaa_funcs.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/extension.h"  // self-defined kernel dependency
#include "runtime/runtime.h"
#include "tecodnn.h"  // NOLINT

namespace custom_kernel {

enum class checkformat {
  LessE4D = 0,
  Equal4D = 1,
};
template <typename T = int>
inline void checkpadding(int* h, int* w, std::vector<T> paddings) {
  if (paddings.size() == 2) {
    *h = paddings[0];
    *w = paddings[1];
  } else if (paddings.size() == 4) {
    if (paddings[0] != paddings[1]) {
      PADDLE_ENFORCE_EQ(
          paddings[0],
          paddings[1],
          phi::errors::InvalidArgument("padding_height_top size should be the "
                                       "same  as padding_height_bottom "
                                       "But recieved: padding_height_top size "
                                       "is %d, padding_height_bottom is [%s]",
                                       paddings[0],
                                       paddings[1]));
    } else if (paddings[2] != paddings[3]) {
      PADDLE_ENFORCE_EQ(
          paddings[2],
          paddings[3],
          phi::errors::InvalidArgument("padding_width_left size should be the "
                                       "same  as padding_width_right "
                                       "But recieved: padding_width_left size "
                                       "is %d, padding_width_right is [%s]",
                                       paddings[2],
                                       paddings[3]));
    } else {
      *h = paddings[0];
      *w = paddings[2];
    }
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(("invalid padding para")));
  }
}

template <typename T = int, int D>
inline phi::DenseTensor build_dummy_tensor(const Context& dev_ctx,
                                           phi::DataType dtype,
                                           phi::Dim<D> dims) {
  phi::DDim input_dims(dims);
  phi::DenseTensor input_;
  phi::DenseTensorMeta input_meta = {dtype, input_dims};
  input_.set_meta(input_meta);
  dev_ctx.template Alloc<T>(&input_);
  return input_;
}

template <typename T = int>
inline void UpdatePaddingAndDilation(std::vector<T>* paddings,
                                     std::vector<T>* dilation,
                                     const std::string padding_algorithm,
                                     const phi::DDim data_dims,
                                     const std::vector<T>& strides,
                                     const std::vector<T>& ksize) {
  auto data_shape = phi::vectorize<T>(data_dims);
  if (static_cast<int>(paddings->size()) == data_dims.size()) {
    for (int i = 0; i < data_dims.size(); ++i) {
      T copy_pad = *(paddings->begin() + 2 * i);
      paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
    }
  } else {
    PADDLE_ENFORCE_EQ(
        data_dims.size() * 2,
        paddings->size(),
        phi::errors::InvalidArgument(
            "Attribute padding's size should be the same or twice as the "
            "input's dimension. "
            "But recieved: padding's size is %d, padding is [%s]; input's "
            "dimension is %d, input's shape is [%s].",
            paddings->size(),
            phi::make_ddim(*paddings),
            data_dims.size(),
            data_dims));
  }

  if (padding_algorithm == "SAME") {
    for (int i = 0; i < data_dims.size(); ++i) {
      T out_size = (data_dims[i] + strides[i] - 1) / strides[i];
      T pad_sum =
          std::max((out_size - 1) * strides[i] + ksize[i] - data_shape[i],
                   static_cast<T>(0));
      T pad_0 = pad_sum / 2;
      T pad_1 = pad_sum - pad_0;
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;
      // dilation
      *(dilation->begin() + i) = 1;
    }
  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }
}

inline void check_paddings(int* h, int* w, const std::vector<int>& paddings) {
  if (paddings.size() == 2) {
    *h = paddings[0];
    *w = paddings[1];
  } else if (paddings.size() == 4) {
    if (paddings[0] != paddings[1]) {
      PADDLE_ENFORCE_EQ(
          paddings[0],
          paddings[1],
          phi::errors::InvalidArgument("padding_height_top size should be the "
                                       "same  as padding_height_bottom "
                                       "But recieved: padding_height_top size "
                                       "is %d, padding_height_bottom is [%s]",
                                       paddings[0],
                                       paddings[1]));
    } else if (paddings[2] != paddings[3]) {
      PADDLE_ENFORCE_EQ(
          paddings[2],
          paddings[3],
          phi::errors::InvalidArgument("padding_width_left size should be the "
                                       "same  as padding_width_right "
                                       "But recieved: padding_width_left size "
                                       "is %d, padding_width_right is [%s]",
                                       paddings[2],
                                       paddings[3]));
    } else {
      *h = paddings[0];
      *w = paddings[2];
    }
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(("invalid padding para")));
  }
}

inline void checkdims(const phi::DenseTensor* input,
                      checkformat check_f,
                      std::string kernel_name) {
  switch (check_f) {
    case checkformat::LessE4D:
      PADDLE_ENFORCE_LE(input->dims().size(),
                        4,
                        phi::errors::InvalidArgument(
                            "tecodnn %s do not support tensor larger than 4d"
                            "But recieved: input dims size is %d",
                            kernel_name,
                            input->dims().size()));
      break;
    case checkformat::Equal4D:
      PADDLE_ENFORCE_EQ(
          input->dims().size(),
          4,
          phi::errors::InvalidArgument("tecodnn %s do not support 5D tensor"
                                       "But recieved: input dims size is %d",
                                       kernel_name,
                                       input->dims().size()));
      break;
    default:
      PADDLE_THROW(
          phi::errors::InvalidArgument("invalid args for %s", kernel_name));
  }
}

template <typename T>
void Gen_Tecodnn_Out(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     phi::DenseTensor* out,
                     bool if_nchw) {
  phi::DDim out_dims;
  if (if_nchw) {
    out_dims = sdaa_ops::doDimPermute(x, Convert_TF::NCHW2NHWC);
  } else {
    out_dims = x.dims();
  }
  phi::DenseTensorMeta out_meta = {x.dtype(), out_dims};
  out->set_meta(out_meta);
  dev_ctx.template Alloc<T>(out);
}

template <typename T>
bool Trans_Xy_Tensor_in(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        phi::DenseTensor* out,
                        bool is_NCHW = true,
                        bool is_depthwise_conv = false) {
  phi::DDim out_dims;
  if (is_NCHW) {
    out_dims = sdaa_ops::doDimPermute(x, Convert_TF::NCHW2NHWC);
  } else {
    out_dims = x.dims();
  }
  if (is_depthwise_conv) {
    if (is_NCHW) {
      out->Resize(out_dims);
      dev_ctx.template Alloc<T>(out);
      sdaa_ops::doTransformTensor(dev_ctx, x, Convert_TF::NCHW2NHWC, out);
      return false;
    }
    return true;
  }
  phi::DenseTensorMeta out_meta = {phi::DataType::FLOAT16, out_dims};
  out->set_meta(out_meta);
  dev_ctx.template Alloc<phi::dtype::float16>(out);

  if (std::is_same<T, phi::dtype::float16>::value && !is_NCHW) {
    VLOG(4) << "do not need trans operation";
    return true;
  }
  if (is_NCHW) {
    sdaa_ops::doTransformTensor(dev_ctx, x, Convert_TF::NCHW2NHWC, out);
  } else {
    sdaa_ops::doCastTensor(dev_ctx, x, out);
  }
  return false;
}

template <typename T>
void Trans_Xy_Tensor_out(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         phi::DenseTensor* out,
                         bool is_NCHW = true) {
  dev_ctx.template Alloc<T>(out);
  // input -> input_nchw
  if (std::is_same<T, phi::dtype::float16>::value && !is_NCHW) {
    VLOG(4) << "do not need trans operation";
  } else if (std::is_same<T, phi::dtype::float16>::value && is_NCHW) {
    sdaa_ops::doTransformTensor(dev_ctx, x, Convert_TF::NHWC2NCHW, out);
  } else if (std::is_same<T, float>::value && !is_NCHW) {
    sdaa_ops::doCastTensor(dev_ctx, x, out);
  } else if (std::is_same<T, float>::value && is_NCHW) {
    // input -> input_float
    phi::DenseTensor in_float;
    phi::DenseTensorMeta out_meta = {phi::DataType::FLOAT32, x.dims()};
    in_float.set_meta(out_meta);
    dev_ctx.template Alloc<float>(&in_float);
    sdaa_ops::doCastTensor(dev_ctx, x, &in_float);
    // input_half -> input_half_nhwc
    sdaa_ops::doTransformTensor(dev_ctx, in_float, Convert_TF::NHWC2NCHW, out);
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument("Not support tensor format"));
  }
}
template <typename T>
void doConv2dForward(const Context& dev_ctx,
                     const phi::DenseTensor& in_x_NHWC_HALF,
                     const phi::DenseTensor& filter_CHWN_HALF,
                     const phi::DDim& filter_dims,
                     phi::DenseTensor* out,
                     int* padA,
                     int* filterStrideA,
                     int* upscaleA,
                     int groups,
                     int Nd) {
  // tecodnn plugin
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  phi::DDim in_dims = in_x_NHWC_HALF.dims();
  phi::DDim out_dims = out->dims();

  tecodnnFilterDescriptor_t filterDesc;
  tecodnnConvolutionDescriptor_t convDesc;
  tecodnnConvolutionFwdAlgo_t algo = TECODNN_CONVOLUTION_FWD_ALGO_0;
  tecodnnTensorDescriptor_t x_Desc, y_Desc;
  tecodnnDataType_t dt = sdaa_ops::ToTecodnnDataType(filter_CHWN_HALF.dtype());
  TECODNN_CHECK(tecodnnCreateFilterDescriptor(&filterDesc));
  TECODNN_CHECK(tecodnnCreateConvolutionDescriptor(&convDesc));
  TECODNN_CHECK(tecodnnSetFilter4dDescriptor(filterDesc,
                                             dt,
                                             TECODNN_TENSOR_CHWN,
                                             filter_dims[3],
                                             filter_dims[0],
                                             filter_dims[1],
                                             filter_dims[2]));
  x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(in_dims), in_x_NHWC_HALF.dtype(), TensorFormat::NHWC);
  y_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(out_dims), out->dtype(), TensorFormat::NHWC);
  TECODNN_CHECK(tecodnnSetConvolutionGroupCount(convDesc, groups));
  TECODNN_CHECK(tecodnnSetConvolution2dDescriptor(convDesc,
                                                  padA[0],
                                                  padA[1],
                                                  filterStrideA[0],
                                                  filterStrideA[1],
                                                  upscaleA[0],
                                                  upscaleA[1],
                                                  TECODNN_CROSS_CORRELATION,
                                                  TECODNN_DATA_FLOAT));

  TECODNN_CHECK(
      tecodnnSetConvolutionMathType(convDesc, TECODNN_TENSOR_ACC_MATH));
  size_t workSpaceSizeInBytes = 0;

  TECODNN_CHECK(
      tecodnnGetConvolutionForwardWorkspaceSize(tecodnnHandle,
                                                x_Desc,
                                                filterDesc,
                                                convDesc,
                                                y_Desc,
                                                algo,
                                                &workSpaceSizeInBytes));
  phi::DenseTensor workspace;
  if (workSpaceSizeInBytes != 0)
    workspace.Resize({static_cast<int64_t>(workSpaceSizeInBytes)});
  dev_ctx.Alloc(&workspace, DataType::INT8);
  const float alpha = 1.0f, beta = 0.0f;
  TECODNN_CHECK(tecodnnConvolutionForward(tecodnnHandle,
                                          &alpha,
                                          x_Desc,
                                          in_x_NHWC_HALF.data(),
                                          filterDesc,
                                          filter_CHWN_HALF.data(),
                                          convDesc,
                                          algo,
                                          workspace.data(),
                                          workSpaceSizeInBytes,
                                          &beta,
                                          y_Desc,
                                          out->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
  TECODNN_CHECK(tecodnnDestroyFilterDescriptor(filterDesc));
  TECODNN_CHECK(tecodnnDestroyConvolutionDescriptor(convDesc));
}

template <typename T, typename Context>
void ConvKernel(const Context& dev_ctx,
                int Nd,
                const phi::DenseTensor& input,
                const phi::DenseTensor& filter_t,
                const std::vector<int>& strides_t,
                const std::vector<int>& paddings_t,
                const std::string& padding_algorithm,
                const std::vector<int>& dilations_t,
                int groups,
                bool is_depthwise_conv,
                const std::string& data_format,
                phi::DenseTensor* output) {
  // HIGH PERFORMANCE CONV

  if (isEnvEnable("HIGH_PERFORMANCE_CONV") &&
      (!filter_t.storage_properties_initialized())) {
    VLOG(1) << "PERFORMANCE tecodnn conv impl called first time";
    // auto filter_properties = std::make_unique<SDAAStorageProperties>();
    SDAAStorageProperties filter_properties;
    phi::DDim out_dims =
        sdaa_ops::doDimPermute(filter_t, Convert_TF::NCHW2CHWN);
    filter_properties.storage_format = 0;
    filter_properties.storage_dims = out_dims;
    sdaa_ops::swapTensorData(dev_ctx, filter_t, filter_properties);
  }

  VLOG(1) << "filter_t.storage_properties_initialized: "
          << filter_t.storage_properties_initialized();
  phi::DDim filter_dims;
  phi::DDim filter_data_dims;
  if (filter_t.storage_properties_initialized()) {
    auto storages = filter_t.storage_properties<SDAAStorageProperties>();
    filter_dims = storages.storage_dims;  // CHWN
    filter_data_dims = phi::slice_ddim(filter_dims, 1, 3);
  } else {
    filter_dims = filter_t.dims();  // NCHW
    filter_data_dims = phi::slice_ddim(filter_dims, 2, filter_dims.size());
  }
  VLOG(1) << "real filter dims " << filter_dims;

  dev_ctx.template Alloc<T>(output);
  checkdims(&input, checkformat::Equal4D, "conv2d");
  const bool channel_last = data_format == "NHWC";  // default: 0
  phi::DDim in_dims = input.dims();
  phi::DDim in_data_dims;
  if (channel_last) {  // NHWC
    in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  } else {  // NCHW
    in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  }

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  auto strides = strides_t;
  auto paddings = paddings_t;
  auto dilations = dilations_t;
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);
  int padH;
  int padW;
  check_paddings(&padH, &padW, paddings);
  int padA[2] = {padH, padW};
  int filterStrideA[2] = {strides_t[0], strides_t[1]};
  int upscaleA[2] = {dilations[0], dilations[1]};
  bool is_NCHW = !channel_last;

  phi::DDim filter_chwn_dim;
  phi::DenseTensor input_nhwc_half;

  bool flag = Trans_Xy_Tensor_in<T>(
      dev_ctx, input, &input_nhwc_half, is_NCHW, is_depthwise_conv);
  if (flag) {
    input_nhwc_half = input;
  }

  phi::DenseTensor filter_chwn_half;
  if (!filter_t.storage_properties_initialized()) {
    phi::DDim out_dims =
        sdaa_ops::doDimPermute(filter_t, Convert_TF::NCHW2CHWN);
    phi::DenseTensorMeta out_meta;
    if (is_depthwise_conv) {
      out_meta = {filter_t.dtype(), out_dims};
    } else {
      out_meta = {phi::DataType::FLOAT16, out_dims};
    }

    filter_chwn_half.set_meta(out_meta);
    if (is_depthwise_conv) {
      dev_ctx.template Alloc<T>(&filter_chwn_half);
    } else {
      dev_ctx.template Alloc<phi::dtype::float16>(&filter_chwn_half);
    }
    sdaa_ops::doTransformTensor(
        dev_ctx, filter_t, Convert_TF::NCHW2CHWN, &filter_chwn_half);
    filter_chwn_dim = filter_chwn_half.dims();
  } else {
    if (is_depthwise_conv || (std::is_same<T, phi::dtype::float16>::value)) {
      filter_chwn_half = filter_t;
    } else {
      phi::DenseTensorMeta out_meta = {phi::DataType::FLOAT16, filter_dims};
      filter_chwn_half.set_meta(out_meta);
      dev_ctx.template Alloc<phi::dtype::float16>(&filter_chwn_half);
      sdaa_ops::doCastTensor(dev_ctx, filter_t, &filter_chwn_half);
    }
    filter_chwn_dim = filter_dims;
  }

  dev_ctx.template Alloc<T>(output);
  if (!is_NCHW) {  // NHWC
    doConv2dForward<T>(dev_ctx,
                       input_nhwc_half,
                       filter_chwn_half,
                       filter_chwn_dim,
                       output,
                       padA,
                       filterStrideA,
                       upscaleA,
                       groups,
                       Nd);

  } else {  // NCHW
    phi::DenseTensor out_nhwc;
    phi::DDim out_dims = sdaa_ops::doDimPermute(*output, Convert_TF::NCHW2NHWC);
    phi::DenseTensorMeta out_meta = {output->dtype(), out_dims};
    out_nhwc.set_meta(out_meta);
    dev_ctx.template Alloc<T>(&out_nhwc);
    doConv2dForward<T>(dev_ctx,
                       input_nhwc_half,
                       filter_chwn_half,
                       filter_chwn_dim,
                       &out_nhwc,
                       padA,
                       filterStrideA,
                       upscaleA,
                       groups,
                       Nd);
    sdaa_ops::doTransformTensor(
        dev_ctx, out_nhwc, Convert_TF::NHWC2NCHW, output);
  }

  // compute Conv2dForward
}

inline void doConv2dBackwardFilter(
    const Context& dev_ctx,
    const phi::DenseTensor& input_NHWC_HALF,
    const phi::DenseTensor& output_grad_NHWC_HALF,
    phi::DenseTensor* filter_grad_CHWN,
    const phi::DDim& filter_dims_chwn,
    int* padA,
    int* filterStrideA,
    int* upscaleA,
    int groups,
    int Nd) {
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc, dy_Desc;
  tecodnnFilterDescriptor_t filterDesc;
  tecodnnConvolutionDescriptor_t convDesc;
  tecodnnConvolutionBwdFilterAlgo_t BF_algo =
      TECODNN_CONVOLUTION_BWD_FILTER_ALGO_0;
  TECODNN_CHECK(tecodnnCreateFilterDescriptor(&filterDesc));
  TECODNN_CHECK(tecodnnCreateConvolutionDescriptor(&convDesc));
  x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(input_NHWC_HALF.dims()),
      input_NHWC_HALF.dtype(),
      TensorFormat::NHWC);
  dy_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(output_grad_NHWC_HALF.dims()),
      output_grad_NHWC_HALF.dtype(),
      TensorFormat::NHWC);
  tecodnnDataType_t dt = sdaa_ops::ToTecodnnDataType(filter_grad_CHWN->dtype());

  TECODNN_CHECK(tecodnnSetFilter4dDescriptor(filterDesc,
                                             dt,
                                             TECODNN_TENSOR_CHWN,
                                             filter_dims_chwn[3],
                                             filter_dims_chwn[0],
                                             filter_dims_chwn[1],
                                             filter_dims_chwn[2]));

  TECODNN_CHECK(tecodnnSetConvolutionGroupCount(convDesc, groups));
  size_t workSpaceSizeInBytes = 0;

  TECODNN_CHECK(tecodnnSetConvolution2dDescriptor(convDesc,
                                                  padA[0],
                                                  padA[1],
                                                  filterStrideA[0],
                                                  filterStrideA[1],
                                                  upscaleA[0],
                                                  upscaleA[1],
                                                  TECODNN_CROSS_CORRELATION,
                                                  TECODNN_DATA_FLOAT));
  TECODNN_CHECK(
      tecodnnSetConvolutionMathType(convDesc, TECODNN_TENSOR_ACC_MATH));
  TECODNN_CHECK(
      tecodnnGetConvolutionBackwardFilterWorkspaceSize(tecodnnHandle,
                                                       x_Desc,
                                                       dy_Desc,
                                                       convDesc,
                                                       filterDesc,
                                                       BF_algo,
                                                       &workSpaceSizeInBytes));
  phi::DenseTensor workspace;
  if (workSpaceSizeInBytes != 0)
    workspace.Resize({static_cast<int64_t>(workSpaceSizeInBytes)});
  dev_ctx.Alloc(&workspace, DataType::INT8);
  const float alpha = 1.0f, beta = 0.0f;
  TECODNN_CHECK(tecodnnConvolutionBackwardFilter(tecodnnHandle,
                                                 &alpha,
                                                 x_Desc,
                                                 input_NHWC_HALF.data(),
                                                 dy_Desc,
                                                 output_grad_NHWC_HALF.data(),
                                                 convDesc,
                                                 BF_algo,
                                                 workspace.data(),
                                                 workSpaceSizeInBytes,
                                                 &beta,
                                                 filterDesc,
                                                 filter_grad_CHWN->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(dy_Desc));
  TECODNN_CHECK(tecodnnDestroyFilterDescriptor(filterDesc));
  TECODNN_CHECK(tecodnnDestroyConvolutionDescriptor(convDesc));
}

inline void doConv2dBackwardData(const Context& dev_ctx,
                                 const phi::DenseTensor& filter_CHWN_HALF,
                                 const phi::DDim& filter_dims_chwn,
                                 const phi::DenseTensor& output_grad_NHWC_HALF,
                                 phi::DenseTensor* input_grad_NHWC,
                                 int* padA,
                                 int* filterStrideA,
                                 int* upscaleA,
                                 int groups,
                                 int Nd) {
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t dy_Desc;
  tecodnnTensorDescriptor_t dx_Desc;
  tecodnnFilterDescriptor_t filterDesc;
  tecodnnConvolutionDescriptor_t convDesc;
  tecodnnDataType_t dt = sdaa_ops::ToTecodnnDataType(filter_CHWN_HALF.dtype());
  tecodnnConvolutionBwdDataAlgo_t BD_algo = TECODNN_CONVOLUTION_BWD_DATA_ALGO_0;
  TECODNN_CHECK(tecodnnCreateFilterDescriptor(&filterDesc));
  TECODNN_CHECK(tecodnnCreateConvolutionDescriptor(&convDesc));
  dx_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(input_grad_NHWC->dims()),
      input_grad_NHWC->dtype(),
      TensorFormat::NHWC);
  dy_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(output_grad_NHWC_HALF.dims()),
      output_grad_NHWC_HALF.dtype(),
      TensorFormat::NHWC);
  const float alpha = 1.0f, beta = 0.0f;
  tecodnnSetFilter4dDescriptor(filterDesc,
                               dt,
                               TECODNN_TENSOR_CHWN,
                               filter_dims_chwn[3],
                               filter_dims_chwn[0],
                               filter_dims_chwn[1],
                               filter_dims_chwn[2]);
  TECODNN_CHECK(tecodnnSetConvolutionGroupCount(convDesc, groups));
  TECODNN_CHECK(tecodnnSetConvolution2dDescriptor(convDesc,
                                                  padA[0],
                                                  padA[1],
                                                  filterStrideA[0],
                                                  filterStrideA[1],
                                                  upscaleA[0],
                                                  upscaleA[1],
                                                  TECODNN_CROSS_CORRELATION,
                                                  TECODNN_DATA_FLOAT));

  TECODNN_CHECK(
      tecodnnSetConvolutionMathType(convDesc, TECODNN_TENSOR_ACC_MATH));
  size_t workSpaceSizeInBytes = 0;
  // TODO(tecodnn): param error unsupport cases!!!
  TECODNN_CHECK(
      tecodnnGetConvolutionBackwardDataWorkspaceSize(tecodnnHandle,
                                                     filterDesc,
                                                     dy_Desc,
                                                     convDesc,
                                                     dx_Desc,
                                                     BD_algo,
                                                     &workSpaceSizeInBytes));
  phi::DenseTensor workspace;
  if (workSpaceSizeInBytes != 0)
    workspace.Resize({static_cast<int64_t>(workSpaceSizeInBytes)});
  dev_ctx.Alloc(&workspace, DataType::INT8);
  TECODNN_CHECK(tecodnnConvolutionBackwardData(tecodnnHandle,
                                               &alpha,
                                               filterDesc,
                                               filter_CHWN_HALF.data(),
                                               dy_Desc,
                                               output_grad_NHWC_HALF.data(),
                                               convDesc,
                                               BD_algo,
                                               workspace.data(),
                                               workSpaceSizeInBytes,
                                               &beta,
                                               dx_Desc,
                                               input_grad_NHWC->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(dx_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(dy_Desc));
  TECODNN_CHECK(tecodnnDestroyFilterDescriptor(filterDesc));
  TECODNN_CHECK(tecodnnDestroyConvolutionDescriptor(convDesc));
}

template <typename T, typename Context>
void ConvBackwardKernel(const Context& dev_ctx,
                        int Nd,
                        const phi::DenseTensor& input,
                        const phi::DenseTensor& filter,
                        const phi::DenseTensor& output_grad,
                        const std::vector<int>& strides_t,
                        const std::vector<int>& paddings_t,
                        const std::string& padding_algorithm,
                        const std::vector<int>& dilations_t,
                        int groups,
                        bool is_depthwise_conv,
                        const std::string& data_format,
                        phi::DenseTensor* input_grad,
                        phi::DenseTensor* filter_grad) {
  phi::DDim filter_data_dims;
  phi::DDim filter_dims;

  if (isEnvEnable("HIGH_PERFORMANCE_CONV") &&
      (!filter.storage_properties_initialized())) {
    VLOG(1) << "PERFORMANCE tecodnn conv_transpose impl called first time";
    // auto filter_properties = std::make_unique<SDAAStorageProperties>();
    SDAAStorageProperties filter_properties;
    phi::DDim out_dims = sdaa_ops::doDimPermute(filter, Convert_TF::NCHW2CHWN);
    filter_properties.storage_format = 0;
    filter_properties.storage_dims = out_dims;
    sdaa_ops::swapTensorData(dev_ctx, filter, filter_properties);
  }

  VLOG(1) << "filter.storage_properties_initialized: "
          << filter.storage_properties_initialized();

  if (filter.storage_properties_initialized()) {
    auto storages = filter.storage_properties<SDAAStorageProperties>();
    filter_dims = storages.storage_dims;  // CHWN
    filter_data_dims = phi::slice_ddim(filter_dims, 1, 3);
  } else {
    filter_dims = filter.dims();
    filter_data_dims = phi::slice_ddim(filter_dims, 2, filter_dims.size());
  }
  VLOG(4) << "conv backward called" << filter_dims;
  VLOG(4) << "filter.storage_properties_initialized "
          << filter.storage_properties_initialized() << std::endl;
  phi::DenseTensor filter_chwn_half;
  phi::DDim filter_chwn_dim;
  if (!filter.storage_properties_initialized()) {
    phi::DDim out_dims = sdaa_ops::doDimPermute(filter, Convert_TF::NCHW2CHWN);
    phi::DenseTensorMeta out_meta;
    if (is_depthwise_conv) {
      out_meta = {filter.dtype(), out_dims};
    } else {
      out_meta = {phi::DataType::FLOAT16, out_dims};
    }
    filter_chwn_half.set_meta(out_meta);
    if (is_depthwise_conv) {
      dev_ctx.template Alloc<T>(&filter_chwn_half);
    } else {
      dev_ctx.template Alloc<phi::dtype::float16>(&filter_chwn_half);
    }
    sdaa_ops::doTransformTensor(
        dev_ctx, filter, Convert_TF::NCHW2CHWN, &filter_chwn_half);
    filter_chwn_dim = filter_chwn_half.dims();
  } else {
    if (is_depthwise_conv || (std::is_same<T, phi::dtype::float16>::value)) {
      filter_chwn_half = filter;
    } else {
      phi::DenseTensorMeta out_meta = {phi::DataType::FLOAT16, filter_dims};
      filter_chwn_half.set_meta(out_meta);
      dev_ctx.template Alloc<phi::dtype::float16>(&filter_chwn_half);
      sdaa_ops::doCastTensor(dev_ctx, filter, &filter_chwn_half);
    }
    filter_chwn_dim = filter_dims;
  }
  VLOG(4) << "filter_chwn_dim" << filter_chwn_dim;

  auto strides = strides_t;
  auto paddings = paddings_t;
  auto dilations = dilations_t;
  const bool channel_last = data_format == "NHWC";  // default: 0
  PADDLE_ENFORCE_EQ(
      input.dims().size(),
      4,
      phi::errors::InvalidArgument("tecodnn not support 5D tensor"
                                   "But recieved: input dims size is %d",
                                   input.dims().size()));
  // update padding and dilation
  auto in_dims = input.dims();

  phi::DDim in_data_dims;
  if (channel_last) {  // NHWC
    in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  } else {  // NCHW
    in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  }

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);
  int padH = 0;
  int padW = 0;
  check_paddings(&padH, &padW, paddings);
  int padA[2] = {padH, padW};
  int filterStrideA[2] = {strides[0], strides[1]};
  int upscaleA[2] = {dilations[0], dilations[1]};
  bool is_NCHW = !channel_last;

  if (filter_grad) {
    phi::DenseTensor input_nhwc_half;
    phi::DenseTensor output_grad_nhwc_half;
    dev_ctx.template Alloc<T>(filter_grad);
    if (Trans_Xy_Tensor_in<T>(
            dev_ctx, input, &input_nhwc_half, is_NCHW, is_depthwise_conv))
      input_nhwc_half = input;
    if (Trans_Xy_Tensor_in<T>(dev_ctx,
                              output_grad,
                              &output_grad_nhwc_half,
                              is_NCHW,
                              is_depthwise_conv))
      output_grad_nhwc_half = output_grad;

    if (filter.storage_properties_initialized()) {
      SDAAStorageProperties filter_grad_properties =
          filter.storage_properties<SDAAStorageProperties>();
      sdaa_ops::doAddStorageProperties(
          dev_ctx, filter_grad, filter_grad_properties);
      doConv2dBackwardFilter(dev_ctx,
                             input_nhwc_half,
                             output_grad_nhwc_half,
                             filter_grad,
                             filter_chwn_dim,
                             padA,
                             filterStrideA,
                             upscaleA,
                             groups,
                             Nd);
    } else {
      // output: filter_grad
      phi::DenseTensor filter_grad_chwn;
      phi::DenseTensorMeta out_meta = {
          filter_grad->dtype(),
          sdaa_ops::doDimPermute(*filter_grad, Convert_TF::NCHW2CHWN)};
      filter_grad_chwn.set_meta(out_meta);
      dev_ctx.template Alloc<T>(&filter_grad_chwn);
      // compute doConv2dBackwardFilter
      doConv2dBackwardFilter(dev_ctx,
                             input_nhwc_half,
                             output_grad_nhwc_half,
                             &filter_grad_chwn,
                             filter_chwn_dim,
                             padA,
                             filterStrideA,
                             upscaleA,
                             groups,
                             Nd);
      sdaa_ops::doTransformTensor(
          dev_ctx, filter_grad_chwn, Convert_TF::CHWN2NCHW, filter_grad);
    }
  }
  if (input_grad) {
    VLOG(4) << "input_grad compute";
    dev_ctx.template Alloc<T>(input_grad);
    phi::DenseTensor output_grad_nhwc_half;
    if (Trans_Xy_Tensor_in<T>(dev_ctx,
                              output_grad,
                              &output_grad_nhwc_half,
                              is_NCHW,
                              is_depthwise_conv))
      output_grad_nhwc_half = output_grad;

    if (!is_NCHW) {  // NHWC
      doConv2dBackwardData(dev_ctx,
                           filter_chwn_half,
                           filter_chwn_dim,
                           output_grad_nhwc_half,
                           input_grad,
                           padA,
                           filterStrideA,
                           upscaleA,
                           groups,
                           Nd);
    } else {  // NCHW
      phi::DenseTensor input_grad_nhwc;
      Gen_Tecodnn_Out<T>(dev_ctx, *input_grad, &input_grad_nhwc, is_NCHW);
      doConv2dBackwardData(dev_ctx,
                           filter_chwn_half,
                           filter_chwn_dim,
                           output_grad_nhwc_half,
                           &input_grad_nhwc,
                           padA,
                           filterStrideA,
                           upscaleA,
                           groups,
                           Nd);
      sdaa_ops::doTransformTensor(
          dev_ctx, input_grad_nhwc, Convert_TF::NHWC2NCHW, input_grad);
    }
  }
}

}  // namespace custom_kernel

// namespace custom_kernel
