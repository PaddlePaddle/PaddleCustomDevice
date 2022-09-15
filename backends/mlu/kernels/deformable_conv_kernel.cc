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

#include "kernels/funcs/mlu_baseop.h"

namespace custom_kernel {

template <typename T, typename Context>
void DeformableConvKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& offset,
                          const phi::DenseTensor& filter,
                          const paddle::optional<phi::DenseTensor>& mask,
                          const std::vector<int>& strides,
                          const std::vector<int>& paddings,
                          const std::vector<int>& dilations,
                          int deformable_groups,
                          int groups,
                          int im2col_step,
                          phi::DenseTensor* out) {
  // TODO(fwg): Remove this check when cnnl fix the bug that groups > 1.
  PADDLE_ENFORCE_EQ(
      groups == 1,
      true,
      phi::errors::InvalidArgument(
          "MLU deformable_conv kernel only support groups == 1, but get %d.",
          groups));

  dev_ctx.template Alloc<T>(out);
  auto mask_tensor = mask.get();
  // transform paddings from {h, w} to {top, bottom, left, right}.
  const std::vector<int> trans_paddings{
      paddings[0], paddings[0], paddings[1], paddings[1]};
  MLUCnnlDCNDesc dcn_desc(x.dims().size(),
                          trans_paddings.data(),
                          strides.data(),
                          dilations.data(),
                          deformable_groups,
                          groups,
                          im2col_step);

  const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};
  Tensor trans_input;
  TransposeFromMLUTensor<T>(
      dev_ctx, perm_to_nhwc, &x, &trans_input, true /*need_reshape_or_alloc*/);

  Tensor trans_offset;
  TransposeFromMLUTensor<T>(dev_ctx,
                            perm_to_nhwc,
                            &offset,
                            &trans_offset,
                            true /*need_reshape_or_alloc*/);

  Tensor trans_mask;
  TransposeFromMLUTensor<T>(dev_ctx,
                            perm_to_nhwc,
                            &mask_tensor,
                            &trans_mask,
                            true /*need_reshape_or_alloc*/);

  Tensor trans_filter;
  TransposeFromMLUTensor<T>(dev_ctx,
                            perm_to_nhwc,
                            &filter,
                            &trans_filter,
                            true /*need_reshape_or_alloc*/);

  Tensor tmp_output;
  auto output_dims = out->dims();
  tmp_output.Resize(
      {output_dims[0], output_dims[2], output_dims[3], output_dims[1]});
  dev_ctx.template Alloc<T>(&tmp_output);

  cnnlTensorLayout_t data_layout = CNNL_LAYOUT_NHWC;
  MLUCnnlTensorDesc input_desc(
      trans_input, data_layout, ToCnnlDataType(trans_input.dtype()));
  MLUCnnlTensorDesc offset_desc(
      trans_offset, data_layout, ToCnnlDataType(trans_offset.dtype()));
  MLUCnnlTensorDesc mask_desc(
      trans_mask, data_layout, ToCnnlDataType(trans_mask.dtype()));
  MLUCnnlTensorDesc filter_desc(
      trans_filter, data_layout, ToCnnlDataType(trans_filter.dtype()));
  MLUCnnlTensorDesc output_desc(
      tmp_output, data_layout, ToCnnlDataType(tmp_output.dtype()));
  MLUCnnl::DCNForward(dev_ctx,
                      dcn_desc.get(),
                      input_desc.get(),
                      GetBasePtr(&trans_input),
                      offset_desc.get(),
                      GetBasePtr(&trans_offset),
                      mask_desc.get(),
                      GetBasePtr(&trans_mask),
                      filter_desc.get(),
                      GetBasePtr(&trans_filter),
                      nullptr,
                      nullptr,
                      output_desc.get(),
                      GetBasePtr(&tmp_output));

  const std::vector<int> perm_to_nchw = {0, 3, 1, 2};
  TransposeFromMLUTensor<T>(
      dev_ctx, perm_to_nchw, &tmp_output, out, false /*need_reshape_or_alloc*/);
}

template <typename T, typename Context>
void DeformableConvGradKernel(const Context& dev_ctx,
                              const phi::DenseTensor& x,
                              const phi::DenseTensor& offset,
                              const phi::DenseTensor& filter,
                              const paddle::optional<phi::DenseTensor>& mask,
                              const phi::DenseTensor& out_grad,
                              const std::vector<int>& strides,
                              const std::vector<int>& paddings,
                              const std::vector<int>& dilations,
                              int deformable_groups,
                              int groups,
                              int im2col_step,
                              phi::DenseTensor* dx,
                              phi::DenseTensor* offset_grad,
                              phi::DenseTensor* filter_grad,
                              phi::DenseTensor* mask_grad) {
  // TODO(fwg): Remove this check when cnnl fix the bug that groups > 1.
  PADDLE_ENFORCE_EQ(groups == 1,
                    true,
                    phi::errors::InvalidArgument(
                        "MLU deformable_conv_grad kernel only support groups "
                        "== 1, but get %d.",
                        groups));
  auto mask_tensor = mask.get();

  // transform paddings from {h, w} to {top, bottom, left, right}.
  const std::vector<int> trans_paddings{
      paddings[0], paddings[0], paddings[1], paddings[1]};
  MLUCnnlDCNDesc dcn_desc(x.dims().size(),
                          trans_paddings.data(),
                          strides.data(),
                          dilations.data(),
                          deformable_groups,
                          groups,
                          im2col_step);

  Tensor tmp_input_grad;
  auto input_dims = x.dims();
  tmp_input_grad.Resize(
      {input_dims[0], input_dims[2], input_dims[3], input_dims[1]});
  dev_ctx.template Alloc<T>(&tmp_input_grad);

  Tensor tmp_filter_grad;
  auto filter_dims = filter.dims();
  tmp_filter_grad.Resize(
      {filter_dims[0], filter_dims[2], filter_dims[3], filter_dims[1]});
  dev_ctx.template Alloc<T>(&tmp_filter_grad);

  Tensor tmp_offset_grad;
  auto offset_dims = offset.dims();
  tmp_offset_grad.Resize(
      {offset_dims[0], offset_dims[2], offset_dims[3], offset_dims[1]});
  dev_ctx.template Alloc<T>(&tmp_offset_grad);

  Tensor tmp_mask_grad;
  auto mask_dims = mask_tensor.dims();
  tmp_mask_grad.Resize(
      {mask_dims[0], mask_dims[2], mask_dims[3], mask_dims[1]});
  dev_ctx.template Alloc<T>(&tmp_mask_grad);

  const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};
  Tensor trans_output_grad;
  TransposeFromMLUTensor<T>(dev_ctx,
                            perm_to_nhwc,
                            &out_grad,
                            &trans_output_grad,
                            true /*need_reshape_or_alloc*/);

  Tensor trans_input;
  TransposeFromMLUTensor<T>(
      dev_ctx, perm_to_nhwc, &x, &trans_input, true /*need_reshape_or_alloc*/);

  Tensor trans_offset;
  TransposeFromMLUTensor<T>(dev_ctx,
                            perm_to_nhwc,
                            &offset,
                            &trans_offset,
                            true /*need_reshape_or_alloc*/);

  Tensor trans_mask;
  TransposeFromMLUTensor<T>(dev_ctx,
                            perm_to_nhwc,
                            &mask_tensor,
                            &trans_mask,
                            true /*need_reshape_or_alloc*/);

  Tensor trans_filter;
  TransposeFromMLUTensor<T>(dev_ctx,
                            perm_to_nhwc,
                            &filter,
                            &trans_filter,
                            true /*need_reshape_or_alloc*/);

  cnnlTensorLayout_t data_layout = CNNL_LAYOUT_NHWC;
  MLUCnnlTensorDesc output_grad_desc(trans_output_grad,
                                     data_layout,
                                     ToCnnlDataType(trans_output_grad.dtype()));
  MLUCnnlTensorDesc input_desc(
      trans_input, data_layout, ToCnnlDataType(trans_input.dtype()));
  MLUCnnlTensorDesc offset_desc(
      trans_offset, data_layout, ToCnnlDataType(trans_offset.dtype()));
  MLUCnnlTensorDesc mask_desc(
      trans_mask, data_layout, ToCnnlDataType(trans_mask.dtype()));
  MLUCnnlTensorDesc filter_desc(
      trans_filter, data_layout, ToCnnlDataType(trans_filter.dtype()));

  MLUCnnl::DCNBackwardData(dev_ctx,
                           dcn_desc.get(),
                           input_desc.get(),
                           GetBasePtr(&trans_input),
                           offset_desc.get(),
                           GetBasePtr(&trans_offset),
                           mask_desc.get(),
                           GetBasePtr(&trans_mask),
                           filter_desc.get(),
                           GetBasePtr(&trans_filter),
                           output_grad_desc.get(),
                           GetBasePtr(&trans_output_grad),
                           input_desc.get(),
                           GetBasePtr(&tmp_input_grad),
                           offset_desc.get(),
                           GetBasePtr(&tmp_offset_grad),
                           mask_desc.get(),
                           GetBasePtr(&tmp_mask_grad));

  MLUCnnl::DCNBackwardWeight(dev_ctx,
                             dcn_desc.get(),
                             input_desc.get(),
                             GetBasePtr(&trans_input),
                             offset_desc.get(),
                             GetBasePtr(&trans_offset),
                             mask_desc.get(),
                             GetBasePtr(&trans_mask),
                             output_grad_desc.get(),
                             GetBasePtr(&trans_output_grad),
                             filter_desc.get(),
                             GetBasePtr(&tmp_filter_grad),
                             nullptr,
                             nullptr);

  const std::vector<int> perm_to_nchw = {0, 3, 1, 2};
  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nchw,
                              &tmp_input_grad,
                              dx,
                              false /*need_reshape_or_alloc*/);
  }

  if (filter_grad) {
    dev_ctx.template Alloc<T>(filter_grad);
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nchw,
                              &tmp_filter_grad,
                              filter_grad,
                              false /*need_reshape_or_alloc*/);
  }

  if (offset_grad) {
    dev_ctx.template Alloc<T>(offset_grad);
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nchw,
                              &tmp_offset_grad,
                              offset_grad,
                              false /*need_reshape_or_alloc*/);
  }

  if (mask_grad) {
    dev_ctx.template Alloc<T>(mask_grad);
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nchw,
                              &tmp_mask_grad,
                              mask_grad,
                              false /*need_reshape_or_alloc*/);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(deformable_conv,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::DeformableConvKernel,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(deformable_conv_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::DeformableConvGradKernel,
                          float) {}
