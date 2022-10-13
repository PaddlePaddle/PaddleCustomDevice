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

#include "kernels/funcs/conv_utils.h"

namespace custom_kernel {

template <typename T, typename Context>
void Conv2dKernel(const Context& dev_ctx,
                  const phi::DenseTensor& input,
                  const phi::DenseTensor& filter,
                  const std::vector<int>& strides_t,
                  const std::vector<int>& paddings_t,
                  const std::string& padding_algorithm,
                  const std::vector<int>& dilations_t,
                  int groups,
                  const std::string& data_format,
                  phi::DenseTensor* output) {
  dev_ctx.template Alloc<T>(output);
  auto strides = strides_t;
  auto paddings = paddings_t;
  auto dilations = dilations_t;

  const bool channel_last = data_format == "NHWC";
  // update padding and dilation
  auto in_dims = input.dims();
  auto filter_dims = filter.dims();
  auto in_dims_size = in_dims.size();
  phi::DDim in_data_dims;
  phi::DDim filter_data_dims;

  if (channel_last) {
    in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  } else {
    in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  }
  filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());
  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  Tensor input_tensor;
  Tensor output_tensor;
  const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};
  if (channel_last) {
    input_tensor = input;
    output_tensor = *output;
  } else {
    // transpose input from NCHW to NHWC
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nhwc,
                              &input,
                              &input_tensor,
                              true /*need_reshape_or_alloc*/);
    auto output_dims = output->dims();
    output_tensor.Resize(
        {output_dims[0], output_dims[2], output_dims[3], output_dims[1]});
    dev_ctx.template Alloc<T>(&output_tensor);
  }

  // transpose filter from MCHW to MHWC
  Tensor trans_filter;
  TransposeFromMLUTensor<T>(dev_ctx,
                            perm_to_nhwc,
                            &filter,
                            &trans_filter,
                            true /*need_reshape_or_alloc*/);

  cnnlTensorLayout_t data_layout = CNNL_LAYOUT_NHWC;
  MLUCnnlTensorDesc input_desc(
      input_tensor, data_layout, ToCnnlDataType(input_tensor.dtype()));
  MLUCnnlTensorDesc filter_desc(
      trans_filter, data_layout, ToCnnlDataType(trans_filter.dtype()));
  MLUCnnlTensorDesc output_desc(
      output_tensor, data_layout, ToCnnlDataType(output_tensor.dtype()));

  MLUCnnlConvolutionDesc conv_desc(in_dims_size,
                                   paddings.data(),
                                   strides.data(),
                                   dilations.data(),
                                   groups,
                                   ToCnnlDataType<T>());

  MLUCnnl::ConvolutionForward(dev_ctx,
                              conv_desc.get(),
                              nullptr /*alpha*/,
                              nullptr /*beta*/,
                              nullptr /*bias_desc*/,
                              nullptr /*bias_ptr*/,
                              input_desc.get(),
                              GetBasePtr(&input_tensor),
                              filter_desc.get(),
                              GetBasePtr(&trans_filter),
                              output_desc.get(),
                              GetBasePtr(&output_tensor));

  if (!channel_last) {
    // transpose output from NHWC to NCHW
    const std::vector<int> perm_to_nchw = {0, 3, 1, 2};
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nchw,
                              &output_tensor,
                              output,
                              false /*need_reshape_or_alloc*/);
  }
}

template <typename T, typename Context>
void Conv2dGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& input,
                      const phi::DenseTensor& filter,
                      const phi::DenseTensor& output_grad,
                      const std::vector<int>& strides_t,
                      const std::vector<int>& paddings_t,
                      const std::string& padding_algorithm,
                      const std::vector<int>& dilations_t,
                      int groups,
                      const std::string& data_format,
                      phi::DenseTensor* input_grad,
                      phi::DenseTensor* filter_grad) {
  auto strides = strides_t;
  auto paddings = paddings_t;
  auto dilations = dilations_t;
  const bool channel_last = data_format == "NHWC";
  // update padding and dilation
  auto in_dims = input.dims();
  auto filter_dims = filter.dims();
  auto in_dims_size = in_dims.size();
  phi::DDim in_data_dims;
  phi::DDim filter_data_dims;

  if (channel_last) {
    in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  } else {
    in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  }
  filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  Tensor input_tensor;
  Tensor output_grad_tensor;
  const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};
  const std::vector<int> perm_to_nchw = {0, 3, 1, 2};
  if (channel_last) {
    input_tensor = input;
    output_grad_tensor = output_grad;
  } else {
    // transpose input and output_grad from NCHW to NHWC
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nhwc,
                              &input,
                              &input_tensor,
                              true /*need_reshape_or_alloc*/);
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nhwc,
                              &output_grad,
                              &output_grad_tensor,
                              true /*need_reshape_or_alloc*/);
  }

  if (filter_grad) {
    dev_ctx.template Alloc<T>(filter_grad);

    auto filter_grad_dims = filter_grad->dims();
    Tensor temp_filter_grad;
    temp_filter_grad.Resize({filter_grad_dims[0],
                             filter_grad_dims[2],
                             filter_grad_dims[3],
                             filter_grad_dims[1]});
    dev_ctx.template Alloc<T>(&temp_filter_grad);

    cnnlDataType_t tensor_dtype = ToCnnlDataType<T>();
    cnnlTensorLayout_t data_layout = CNNL_LAYOUT_NHWC;
    MLUCnnlTensorDesc input_desc(input_tensor, data_layout, tensor_dtype);
    MLUCnnlTensorDesc out_grad_desc(
        output_grad_tensor, data_layout, tensor_dtype);
    MLUCnnlTensorDesc temp_filter_grad_desc(
        temp_filter_grad, data_layout, tensor_dtype);

    MLUCnnlConvolutionDesc conv_desc(in_dims_size,
                                     paddings.data(),
                                     strides.data(),
                                     dilations.data(),
                                     groups,
                                     tensor_dtype);

    MLUCnnl::ConvBackpropFilter(dev_ctx,
                                conv_desc.get(),
                                input_desc.get(),
                                GetBasePtr(&input_tensor),
                                out_grad_desc.get(),
                                GetBasePtr(&output_grad_tensor),
                                temp_filter_grad_desc.get(),
                                GetBasePtr(&temp_filter_grad));

    // transpose filter_grad from MHWC to MCHW
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nchw,
                              &temp_filter_grad,
                              filter_grad,
                              false /*need_reshape_or_alloc*/);
  }
  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);

    Tensor input_grad_tensor;
    if (channel_last) {
      input_grad_tensor = *input_grad;
    } else {
      auto input_grad_dims = input_grad->dims();
      input_grad_tensor.Resize({input_grad_dims[0],
                                input_grad_dims[2],
                                input_grad_dims[3],
                                input_grad_dims[1]});
      dev_ctx.template Alloc<T>(&input_grad_tensor);
    }

    // transpose filter from MCHW to MHWC
    Tensor trans_filter;
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nhwc,
                              &filter,
                              &trans_filter,
                              true /*need_reshape_or_alloc*/);

    cnnlDataType_t tensor_dtype = ToCnnlDataType<T>();
    cnnlTensorLayout_t data_layout = CNNL_LAYOUT_NHWC;
    MLUCnnlTensorDesc filter_desc(trans_filter, data_layout, tensor_dtype);
    MLUCnnlTensorDesc out_grad_desc(
        output_grad_tensor, data_layout, tensor_dtype);
    MLUCnnlTensorDesc in_grad_desc(
        input_grad_tensor, data_layout, tensor_dtype);

    MLUCnnlConvolutionDesc conv_desc(in_dims_size,
                                     paddings.data(),
                                     strides.data(),
                                     dilations.data(),
                                     groups,
                                     tensor_dtype);

    MLUCnnl::ConvBackpropInput(dev_ctx,
                               conv_desc.get(),
                               filter_desc.get(),
                               GetBasePtr(&trans_filter),
                               out_grad_desc.get(),
                               GetBasePtr(&output_grad_tensor),
                               in_grad_desc.get(),
                               GetBasePtr(&input_grad_tensor));

    if (!channel_last) {
      // transpose input_grad from NHWC to NCHW
      TransposeFromMLUTensor<T>(dev_ctx,
                                perm_to_nchw,
                                &input_grad_tensor,
                                input_grad,
                                false /*need_reshape_or_alloc*/);
    }
  }
}

template <typename T, typename Context>
void DepthwiseConv2dKernel(const Context& dev_ctx,
                           const phi::DenseTensor& input,
                           const phi::DenseTensor& filter,
                           const std::vector<int>& stride,
                           const std::vector<int>& paddings_in,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations_in,
                           const std::string& data_format,
                           bool use_addto,
                           int workspace_size_MB,
                           bool exhaustive_search,
                           bool fuse_relu,
                           phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  std::vector<int> strides = stride;
  std::vector<int> paddings = paddings_in;
  std::vector<int> dilations = dilations_in;
  const bool channel_last = data_format == "NHWC";

  // update padding and dilation
  auto in_dims = input.dims();
  auto filter_dims = filter.dims();
  auto in_dims_size = in_dims.size();
  phi::DDim in_data_dims;
  phi::DDim filter_data_dims;

  if (channel_last) {
    in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  } else {
    in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  }
  filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());
  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  Tensor input_tensor;
  Tensor output_tensor;
  const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};
  if (channel_last) {
    groups = in_dims[3];
    input_tensor = input;
    output_tensor = *out;
  } else {
    // transpose input from NCHW to NHWC
    groups = in_dims[1];
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nhwc,
                              &input,
                              &input_tensor,
                              true /*need_reshape_or_alloc*/);
    auto output_dims = out->dims();
    output_tensor.Resize(
        {output_dims[0], output_dims[2], output_dims[3], output_dims[1]});
    dev_ctx.template Alloc<T>(&output_tensor);
  }

  // transpose filter from MCHW to MHWC
  Tensor trans_filter;
  TransposeFromMLUTensor<T>(dev_ctx,
                            perm_to_nhwc,
                            &filter,
                            &trans_filter,
                            true /*need_reshape_or_alloc*/);

  cnnlTensorLayout_t data_layout = CNNL_LAYOUT_NHWC;
  MLUCnnlTensorDesc input_desc(
      input_tensor, data_layout, ToCnnlDataType(input_tensor.dtype()));
  MLUCnnlTensorDesc filter_desc(
      trans_filter, data_layout, ToCnnlDataType(trans_filter.dtype()));
  MLUCnnlTensorDesc output_desc(
      output_tensor, data_layout, ToCnnlDataType(output_tensor.dtype()));

  MLUCnnlConvolutionDesc conv_desc(in_dims_size,
                                   paddings.data(),
                                   strides.data(),
                                   dilations.data(),
                                   groups,
                                   ToCnnlDataType<T>());

  MLUCnnl::ConvolutionForward(dev_ctx,
                              conv_desc.get(),
                              nullptr /*alpha*/,
                              nullptr /*beta*/,
                              nullptr /*bias_desc*/,
                              nullptr /*bias_ptr*/,
                              input_desc.get(),
                              GetBasePtr(&input_tensor),
                              filter_desc.get(),
                              GetBasePtr(&trans_filter),
                              output_desc.get(),
                              GetBasePtr(&output_tensor));

  if (!channel_last) {
    // transpose out from NHWC to NCHW
    const std::vector<int> perm_to_nchw = {0, 3, 1, 2};
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nchw,
                              &output_tensor,
                              out,
                              false /*need_reshape_or_alloc*/);
  }
}

template <typename T, typename Context>
void DepthwiseConv2dGradKernel(const Context& dev_ctx,
                               const phi::DenseTensor& input,
                               const phi::DenseTensor& filter,
                               const phi::DenseTensor& out_grad,
                               const std::vector<int>& stride,
                               const std::vector<int>& paddings_in,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations_in,
                               const std::string& data_format,
                               bool use_addto,
                               int workspace_size_MB,
                               bool exhaustive_search,
                               bool fuse_relu,
                               phi::DenseTensor* input_grad,
                               phi::DenseTensor* filter_grad) {
  std::vector<int> strides = stride;
  std::vector<int> paddings = paddings_in;
  std::vector<int> dilations = dilations_in;
  const bool channel_last = data_format == "NHWC";

  // update padding and dilation
  auto in_dims = input.dims();
  auto filter_dims = filter.dims();
  auto in_dims_size = in_dims.size();
  phi::DDim in_data_dims;
  phi::DDim filter_data_dims;

  if (channel_last) {
    in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  } else {
    in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  }
  filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  Tensor input_tensor;
  Tensor output_grad_tensor;
  const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};
  const std::vector<int> perm_to_nchw = {0, 3, 1, 2};
  const std::vector<int> perm_hwcm_to_mchw = {3, 2, 0, 1};
  const std::vector<int> perm_mchw_to_hwcm = {2, 3, 1, 0};
  if (channel_last) {
    input_tensor = input;
    output_grad_tensor = out_grad;
    groups = in_dims[3];
  } else {
    groups = in_dims[1];
    // transpose input and out_grad from NCHW to NHWC
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nhwc,
                              &input,
                              &input_tensor,
                              true /*need_reshape_or_alloc*/);
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nhwc,
                              &out_grad,
                              &output_grad_tensor,
                              true /*need_reshape_or_alloc*/);
  }

  if (filter_grad) {
    dev_ctx.template Alloc<T>(filter_grad);
    auto filter_grad_dims = filter_grad->dims();
    Tensor temp_filter_grad;
    // Details about setting diff_w hwcn for better performance, see the CNNL
    // documentation.
    temp_filter_grad.Resize({filter_grad_dims[perm_mchw_to_hwcm[0]],
                             filter_grad_dims[perm_mchw_to_hwcm[1]],
                             filter_grad_dims[perm_mchw_to_hwcm[2]],
                             filter_grad_dims[perm_mchw_to_hwcm[3]]});
    dev_ctx.template Alloc<T>(&temp_filter_grad);

    cnnlDataType_t tensor_dtype = ToCnnlDataType<T>();
    cnnlTensorLayout_t data_layout = CNNL_LAYOUT_NHWC;
    MLUCnnlTensorDesc input_desc(input_tensor, data_layout, tensor_dtype);
    MLUCnnlTensorDesc out_grad_desc(
        output_grad_tensor, data_layout, tensor_dtype);
    MLUCnnlTensorDesc temp_filter_grad_desc(
        temp_filter_grad, CNNL_LAYOUT_HWCN, tensor_dtype);

    MLUCnnlConvolutionDesc conv_desc(in_dims_size,
                                     paddings.data(),
                                     strides.data(),
                                     dilations.data(),
                                     groups,
                                     tensor_dtype);

    MLUCnnl::ConvBackpropFilter(dev_ctx,
                                conv_desc.get(),
                                input_desc.get(),
                                GetBasePtr(&input_tensor),
                                out_grad_desc.get(),
                                GetBasePtr(&output_grad_tensor),
                                temp_filter_grad_desc.get(),
                                GetBasePtr(&temp_filter_grad));

    // transpose filter_grad from HWCM to MCHW
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_hwcm_to_mchw,
                              &temp_filter_grad,
                              filter_grad,
                              false /*need_reshape_or_alloc*/);
  }
  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);

    Tensor input_grad_tensor;
    if (channel_last) {
      input_grad_tensor = *input_grad;
    } else {
      auto input_grad_dims = input_grad->dims();
      input_grad_tensor.Resize({input_grad_dims[0],
                                input_grad_dims[2],
                                input_grad_dims[3],
                                input_grad_dims[1]});
      dev_ctx.template Alloc<T>(&input_grad_tensor);
    }

    // transpose filter from MCHW to MHWC
    Tensor trans_filter;
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nhwc,
                              &filter,
                              &trans_filter,
                              true /*need_reshape_or_alloc*/);

    cnnlDataType_t tensor_dtype = ToCnnlDataType<T>();
    cnnlTensorLayout_t data_layout = CNNL_LAYOUT_NHWC;
    MLUCnnlTensorDesc filter_desc(trans_filter, data_layout, tensor_dtype);
    MLUCnnlTensorDesc out_grad_desc(
        output_grad_tensor, data_layout, tensor_dtype);
    MLUCnnlTensorDesc in_grad_desc(
        input_grad_tensor, data_layout, tensor_dtype);

    MLUCnnlConvolutionDesc conv_desc(in_dims_size,
                                     paddings.data(),
                                     strides.data(),
                                     dilations.data(),
                                     groups,
                                     tensor_dtype);

    MLUCnnl::ConvBackpropInput(dev_ctx,
                               conv_desc.get(),
                               filter_desc.get(),
                               GetBasePtr(&trans_filter),
                               out_grad_desc.get(),
                               GetBasePtr(&output_grad_tensor),
                               in_grad_desc.get(),
                               GetBasePtr(&input_grad_tensor));

    if (!channel_last) {
      // transpose input_grad from NHWC to NCHW
      TransposeFromMLUTensor<T>(dev_ctx,
                                perm_to_nchw,
                                &input_grad_tensor,
                                input_grad,
                                false /*need_reshape_or_alloc*/);
    }
  }
}

// template <typename T, typename Context>
// void Conv3dKernel(const Context& dev_ctx,
//                   const phi::DenseTensor& input,
//                   const phi::DenseTensor& filter,
//                   const std::vector<int>& strides,
//                   const std::vector<int>& padding,
//                   const std::string& padding_algorithm,
//                   int groups,
//                   const std::vector<int>& dilation,
//                   const std::string& data_format,
//                   bool use_addto,
//                   int workspace_size_MB,
//                   bool exhaustive_search,
//                   phi::DenseTensor* out) {

// }

// template <typename T, typename Context>
// void Conv3dGradKernel(const Context& dev_ctx,
//                       const phi::DenseTensor& input,
//                       const phi::DenseTensor& filter,
//                       const phi::DenseTensor& out_grad,
//                       const std::vector<int>& strides,
//                       const std::vector<int>& padding,
//                       const std::string& padding_algorithm,
//                       int groups,
//                       const std::vector<int>& dilation,
//                       const std::string& data_format,
//                       bool use_addto,
//                       int workspace_size_MB,
//                       bool exhaustive_search,
//                       phi::DenseTensor* input_grad,
//                       phi::DenseTensor* filter_grad) {

// }

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(conv2d,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv2d_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(depthwise_conv2d,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::DepthwiseConv2dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(depthwise_conv2d_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::DepthwiseConv2dGradKernel,
                          float,
                          phi::dtype::float16) {}

// PD_REGISTER_PLUGIN_KERNEL(conv3d,
//                           CustomMLU,
//                           ALL_LAYOUT,
//                           custom_kernel::Conv3dKernel,
//                           float,
//                           phi::dtype::float16) {}

// PD_REGISTER_PLUGIN_KERNEL(conv3d_grad,
//                           CustomMLU,
//                           ALL_LAYOUT,
//                           custom_kernel::Conv3dGradKernel,
//                           float,
//                           phi::dtype::float16) {}
