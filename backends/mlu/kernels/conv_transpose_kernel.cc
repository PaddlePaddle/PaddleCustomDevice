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
void Conv2dTransposeKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const phi::DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& padding,
                           const std::vector<int>& out_padding,
                           const std::vector<int>& output_size,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilation,
                           const std::string& data_format,
                           phi::DenseTensor* out) {
  auto paddings = padding;
  auto dilations = dilation;
  auto output_padding = out_padding;
  dev_ctx.template Alloc<T>(out);
  // check dimension
  const bool channel_last = data_format == "NHWC";
  auto in_dims = x.dims();
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
    input_tensor = x;
    output_tensor = *out;
  } else {
    // transpose x from NCHW to NHWC
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nhwc,
                              &x,
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

  // construct MLU attr
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

  MLUCnnl::ConvBackpropInput(dev_ctx,
                             conv_desc.get(),
                             filter_desc.get(),
                             GetBasePtr(&trans_filter),
                             input_desc.get(),
                             GetBasePtr(&input_tensor),
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
void Conv2dTransposeGradKernel(const Context& dev_ctx,
                               const phi::DenseTensor& x,
                               const phi::DenseTensor& filter,
                               const phi::DenseTensor& dout,
                               const std::vector<int>& strides,
                               const std::vector<int>& padding,
                               const std::vector<int>& output_padding,
                               const std::vector<int>& output_size,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilation,
                               const std::string& data_format,
                               phi::DenseTensor* dx,
                               phi::DenseTensor* dfilter) {
  auto paddings = padding;
  auto dilations = dilation;
  if ((!dx) && (!dfilter)) return;

  const phi::DataLayout data_layout =
      paddle::framework::StringToDataLayout(data_format);
  auto in_dims = x.dims();
  auto filter_dims = filter.dims();
  auto in_dims_size = in_dims.size();

  const bool channel_last = (data_layout == DataLayout::kNHWC);

  phi::DDim in_data_dims;
  if (channel_last) {
    in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  } else {
    in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  }
  phi::DDim filter_data_dims =
      phi::slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  Tensor input_tensor;
  Tensor output_grad_tensor;

  const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};
  if (channel_last) {
    input_tensor = x;
    output_grad_tensor = dout;
  } else {
    // transpose x from NCHW to NHWC
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nhwc,
                              &x,
                              &input_tensor,
                              true /*need_reshape_or_alloc*/);
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nhwc,
                              &dout,
                              &output_grad_tensor,
                              true /*need_reshape_or_alloc*/);
  }

  // transpose filter from MCHW to MHWC
  Tensor trans_filter;
  TransposeFromMLUTensor<T>(dev_ctx,
                            perm_to_nhwc,
                            &filter,
                            &trans_filter,
                            true /*need_reshape_or_alloc*/);

  // MLU descs
  cnnlTensorLayout_t data_layout_mlu = CNNL_LAYOUT_NHWC;
  MLUCnnlTensorDesc input_desc(
      input_tensor, data_layout_mlu, ToCnnlDataType(input_tensor.dtype()));
  MLUCnnlTensorDesc trans_filter_desc(
      trans_filter, data_layout_mlu, ToCnnlDataType(trans_filter.dtype()));
  MLUCnnlTensorDesc output_grad_desc(
      output_grad_tensor,
      data_layout_mlu,
      ToCnnlDataType(output_grad_tensor.dtype()));
  MLUCnnlConvolutionDesc conv_desc(in_dims_size,
                                   paddings.data(),
                                   strides.data(),
                                   dilations.data(),
                                   groups,
                                   ToCnnlDataType<T>());

  if (dfilter) {
    dev_ctx.template Alloc<T>(dfilter);
    Tensor filter_grad_tensor;
    // dfilter always MCHW
    // filter_grad_tensor always MHWC
    auto filter_grad_dims = dfilter->dims();
    filter_grad_tensor.Resize({filter_grad_dims[0],
                               filter_grad_dims[2],
                               filter_grad_dims[3],
                               filter_grad_dims[1]});
    dev_ctx.template Alloc<T>(&filter_grad_tensor);

    MLUCnnlTensorDesc filter_grad_desc(
        filter_grad_tensor,
        data_layout_mlu,
        ToCnnlDataType(filter_grad_tensor.dtype()));

    MLUCnnl::ConvBackpropFilter(dev_ctx,
                                conv_desc.get(),
                                output_grad_desc.get(),
                                GetBasePtr(&dout),
                                input_desc.get(),
                                GetBasePtr(&input_tensor),
                                filter_grad_desc.get(),
                                GetBasePtr(&filter_grad_tensor));
    // transpose output from MHWC to MCHW
    const std::vector<int> perm_to_mchw = {0, 3, 1, 2};
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_mchw,
                              &filter_grad_tensor,
                              dfilter,
                              false /*need_reshape_or_alloc*/);
  }

  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    Tensor input_grad_tensor;
    if (channel_last) {
      input_grad_tensor = *dx;
    } else {
      auto input_grad_dims = dx->dims();
      input_grad_tensor.Resize({input_grad_dims[0],
                                input_grad_dims[2],
                                input_grad_dims[3],
                                input_grad_dims[1]});
      dev_ctx.template Alloc<T>(&input_grad_tensor);
    }

    MLUCnnlTensorDesc input_grad_desc(
        input_grad_tensor,
        data_layout_mlu,
        ToCnnlDataType(input_grad_tensor.dtype()));

    MLUCnnl::ConvolutionForward(dev_ctx,
                                conv_desc.get(),
                                nullptr /*alpha*/,
                                nullptr /*beta*/,
                                nullptr /*bias_desc*/,
                                nullptr /*bias_ptr*/,
                                output_grad_desc.get(),
                                GetBasePtr(&output_grad_tensor),
                                trans_filter_desc.get(),
                                GetBasePtr(&trans_filter),
                                input_grad_desc.get(),
                                GetBasePtr(&input_grad_tensor));
    if (!channel_last) {
      // transpose output from NHWC to NCHW
      const std::vector<int> perm_to_nchw = {0, 3, 1, 2};
      TransposeFromMLUTensor<T>(dev_ctx,
                                perm_to_nchw,
                                &input_grad_tensor,
                                dx,
                                false /*need_reshape_or_alloc*/);
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(conv2d_transpose,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dTransposeKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv2d_transpose_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dTransposeGradKernel,
                          float,
                          phi::dtype::float16) {}
