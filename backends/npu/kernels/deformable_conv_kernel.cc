// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "kernels/funcs/string_helper.h"
namespace custom_kernel {
template <typename Context, typename T>
static void TranposeNPU(const Context& dev_ctx,
                        const aclrtStream& stream,
                        std::vector<int64_t>* perm,
                        const phi::DenseTensor& in,
                        phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  NpuOpRunner runner;
  runner.SetType("Transpose")
      .AddInput(in)
      .AddInput(dev_ctx, std::move(*perm))
      .AddOutput(*out)
      .Run(stream);
}

template <typename T, typename Context>
static void StridedSliceNPU(const Context& dev_ctx,
                            const aclrtStream& stream,
                            const phi::DenseTensor& in,
                            const std::vector<int32_t>& begin,
                            const std::vector<int32_t>& end,
                            const std::vector<int32_t>& strides,
                            phi::DenseTensor* out,
                            int begin_mask = 0,
                            int end_mask = 0,
                            int ellipsis_mask = 0,
                            int new_axis_mask = 0,
                            int shrink_axis_mask = 0) {
  NpuOpRunner runner;
  runner.SetType("StridedSlice")
      .AddInput(in)
      .AddInput(dev_ctx, std::move(begin))
      .AddInput(dev_ctx, std::move(end))
      .AddInput(dev_ctx, std::move(strides))
      .AddAttr("begin_mask", begin_mask)
      .AddAttr("end_mask", end_mask)
      .AddAttr("ellipsis_mask", ellipsis_mask)
      .AddAttr("new_axis_mask", new_axis_mask)
      .AddAttr("shrink_axis_mask", shrink_axis_mask)
      .AddOutput(*out)
      .Run(stream);
}

template <typename T, typename Context>
void GetDeformableOffsetsOutput(const Context& dev_ctx,
                                const aclrtStream& stream,
                                const phi::DenseTensor& x,
                                const phi::DenseTensor& offset,
                                const phi::DenseTensor& filter,
                                const phi::DenseTensor& mask,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                const std::vector<int>& dilations,
                                int deformable_groups,
                                int groups,
                                std::vector<phi::DenseTensor>* out) {
  /* 1. transform offsets data arrangement:
   *
   * NNAdapter:     yyyyy          ->      Ascend:   xxxxx
   *                xxxxx                            xxxxx
   *                yyyyy                            xxxxx
   *                xxxxx                            yyyyy
   *                yyyyy                            yyyyy
   *                xxxxx                            yyyyy
   *
   *
   * 2. concat offset and mask : [N, 3 * deformable_groups * h_f * h_w, h_in,
   * w_in]
   */
  // 1.1 get strided_slice_x
  auto offset_channel = offset.dims()[1];
  phi::DenseTensor strided_slice_x, strided_slice_y;
  if (offset_channel % 2) {
    std::vector<int32_t> strided_slice_x_dims = {offset.dims()[0],
                                                 offset_channel / 2,
                                                 offset.dims()[2],
                                                 offset.dims()[3]};
    phi::DenseTensorMeta meta_x = {offset.dtype(),
                                   phi::make_ddim(strided_slice_x_dims)};
    strided_slice_x.set_meta(meta_x);
    dev_ctx.template Alloc<T>(&strided_slice_x);
    std::vector<int32_t> strided_slice_y_dims = {offset.dims()[0],
                                                 offset_channel / 2 + 1,
                                                 offset.dims()[2],
                                                 offset.dims()[3]};
    phi::DenseTensorMeta meta_y = {offset.dtype(),
                                   phi::make_ddim(strided_slice_y_dims)};
    strided_slice_y.set_meta(meta_y);
    dev_ctx.template Alloc<T>(&strided_slice_y);
  } else {
    std::vector<int32_t> strided_slice_dims = {offset.dims()[0],
                                               offset_channel / 2,
                                               offset.dims()[2],
                                               offset.dims()[3]};
    phi::DenseTensorMeta meta = {offset.dtype(),
                                 phi::make_ddim(strided_slice_dims)};
    strided_slice_x.set_meta(meta);
    strided_slice_y.set_meta(meta);
    dev_ctx.template Alloc<T>(&strided_slice_x);
    dev_ctx.template Alloc<T>(&strided_slice_y);
  }
  std::vector<int32_t> begin_v{0, 1, 0, 0};
  std::vector<int32_t> end_v{
      offset.dims()[0], offset_channel, offset.dims()[2], offset.dims()[3]};
  std::vector<int32_t> strides_v{1, 2, 1, 1};
  StridedSliceNPU<T, Context>(
      dev_ctx, stream, offset, begin_v, end_v, strides_v, &strided_slice_x);

  // 1.2 get strided_slice_y
  begin_v[1] = 0;
  end_v[1] = offset_channel - 1;
  StridedSliceNPU<T, Context>(
      dev_ctx, stream, offset, begin_v, end_v, strides_v, &strided_slice_y);

  // concat
  std::vector<int> offset_all_dims_vec{
      static_cast<int>(x.dims()[0]),
      static_cast<int>((offset.dims()[1] + mask.dims()[1])),
      static_cast<int>(offset.dims()[2]),
      static_cast<int>(offset.dims()[3])};
  //   offset_all.Resize(phi::make_ddim(offset_all_dims_vec));
  phi::DenseTensor offset_all;
  phi::DenseTensorMeta offset_all_meta = {offset.dtype(),
                                          phi::make_ddim(offset_all_dims_vec),
                                          phi::DataLayout::NCHW};
  offset_all.set_meta(offset_all_meta);
  dev_ctx.template Alloc<T>(&offset_all);

  std::vector<phi::DenseTensor> concat_inputs;
  concat_inputs.push_back(strided_slice_x);
  concat_inputs.push_back(strided_slice_y);
  concat_inputs.push_back(mask);
  std::vector<std::string> concat_names{
      "concat_dim", "strided_slice_x", "strided_slice_y", "mask"};
  NpuOpRunner concat_runner;
  concat_runner.SetType("Concat")
      .AddInput(dev_ctx, std::vector<int>(1, 1))
      .AddInputs(concat_inputs)
      .AddOutput(offset_all)
      .AddAttr("N", static_cast<int>(concat_inputs.size()))
      .AddInputNames(concat_names);
  concat_runner.Run(stream);

  (*out)[0] = offset_all;

  std::vector<int> trans_strides{1, strides[0], strides[1], 1};
  std::vector<int> trans_paddings{
      paddings[0], paddings[0], paddings[1], paddings[1]};
  std::vector<int> trans_dilations{1, dilations[0], dilations[1], 1};
  std::vector<int> kernel_sizes{filter.dims()[3], filter.dims()[2]};

  phi::DenseTensor tmp_output;
  // nhwc
  std::vector<int> tmp_dims = {x.dims()[0],
                               offset_all.dims()[2] * kernel_sizes[0],
                               offset_all.dims()[3] * kernel_sizes[1],
                               x.dims()[1]};
  phi::DenseTensorMeta tmp_meta = {
      x.dtype(), phi::make_ddim(tmp_dims), phi::DataLayout::NHWC};
  tmp_output.set_meta(tmp_meta);
  dev_ctx.template Alloc<T>(&tmp_output);

  phi::DenseTensor x_nhwc, offset_nhwc;
  std::vector<int64_t> perm_nhwc = {0, 2, 3, 1};
  // nchw -> nhwc
  phi::DDim x_nhwc_dims =
      phi::make_ddim({x.dims()[0], x.dims()[2], x.dims()[3], x.dims()[1]});
  phi::DenseTensorMeta x_nhwc_meta = {
      x.dtype(), x_nhwc_dims, phi::DataLayout::NHWC};
  x_nhwc.set_meta(x_nhwc_meta);
  TranposeNPU<Context, T>(dev_ctx, stream, &perm_nhwc, x, &x_nhwc);

  phi::DDim offset_nhwc_dims = phi::make_ddim({offset_all.dims()[0],
                                               offset_all.dims()[2],
                                               offset_all.dims()[3],
                                               offset_all.dims()[1]});
  phi::DenseTensorMeta offset_nhwc_meta = {
      offset_all.dtype(), offset_nhwc_dims, phi::DataLayout::NHWC};
  offset_nhwc.set_meta(offset_nhwc_meta);
  TranposeNPU<Context, T>(
      dev_ctx, stream, &perm_nhwc, offset_all, &offset_nhwc);

  std::string dataFormat = "NHWC";

  NpuOpRunner strideslicerunner;
  strideslicerunner.SetType("DeformableOffsets")
      .AddInput(x_nhwc)
      .AddInput(offset_nhwc)
      .AddAttr("ksize", kernel_sizes)
      .AddAttr("strides", trans_strides)
      .AddAttr("pads", trans_paddings)
      .AddAttr("dilations", trans_dilations)
      .AddAttr("deformable_groups", deformable_groups)
      .AddAttr("modulated", true)
      .AddAttr("data_format", dataFormat)
      .AddOutput(tmp_output)
      .Run(stream);

  // nhwc -> nchw
  std::vector<int64_t> perm_nchw = {0, 3, 1, 2};
  std::vector<int> out_nchw_dims = {tmp_output.dims()[0],
                                    tmp_output.dims()[3],
                                    tmp_output.dims()[1],
                                    tmp_output.dims()[2]};
  phi::DenseTensorMeta out_nchw_meta = {tmp_output.dtype(),
                                        phi::make_ddim(out_nchw_dims)};
  phi::DenseTensor deformableOffsetsOutput;
  deformableOffsetsOutput.set_meta(out_nchw_meta);
  dev_ctx.template Alloc<T>(&deformableOffsetsOutput);
  TranposeNPU<Context, T>(
      dev_ctx, stream, &perm_nchw, tmp_output, &deformableOffsetsOutput);
  (*out)[1] = deformableOffsetsOutput;
}

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
  /**
   * fuse deformable_offset and conv2d as deformable_conv2d
   * [input] [offsets]  [mask]
   *    \             \ /
   *     \             |
   *      \          concat
   *       \         /
   *        \       /
   * deformable_offset(input) [filter]  [bias]
   *                       \     |     /
   *                           \   /
   *                           conv2d
   *                             |
   *                      deformable_conv2d
   */
  auto stream = dev_ctx.stream();
  dev_ctx.template Alloc<T>(out);
  auto mask_t = *mask.get_ptr();

  std::vector<phi::DenseTensor> offset_out(2);
  GetDeformableOffsetsOutput<T, Context>(dev_ctx,
                                         stream,
                                         x,
                                         offset,
                                         filter,
                                         mask_t,
                                         strides,
                                         paddings,
                                         dilations,
                                         deformable_groups,
                                         groups,
                                         &offset_out);
  phi::DenseTensor deformableOffsetsOutput = offset_out[1];

  std::vector<int> conv2d_strides{1, 1, filter.dims()[2], filter.dims()[3]};
  std::vector<int> conv2d_paddings{0, 0, 0, 0};
  std::vector<int> conv2d_dilations{1, 1, 1, 1};

  NpuOpRunner runner_conv2d;
  runner_conv2d.SetType("Conv2D")
      .AddInput(deformableOffsetsOutput)
      .AddInput(filter)
      .AddOutput(*out)
      .AddAttrs({{"strides", conv2d_strides}})
      .AddAttrs({{"pads", conv2d_paddings}})
      .AddAttrs({{"dilations", conv2d_dilations}})
      .AddAttrs({{"groups", groups}})
      .AddAttrs({{"data_format", "NCHW"}})
      .Run(stream);
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
  auto mask_t = *mask.get_ptr();
  auto stream = dev_ctx.stream();
  dev_ctx.template Alloc<T>(dx);
  dev_ctx.template Alloc<T>(offset_grad);
  dev_ctx.template Alloc<T>(filter_grad);
  dev_ctx.template Alloc<T>(mask_grad);
  std::vector<int> conv2d_strides{1, 1, filter.dims()[2], filter.dims()[3]};
  std::vector<int> conv2d_paddings{0, 0, 0, 0};
  std::vector<int> conv2d_dilations{1, 1, 1, 1};

  // get deformableOffsetsOutput
  std::vector<phi::DenseTensor> offset_out(2);
  GetDeformableOffsetsOutput<T, Context>(dev_ctx,
                                         stream,
                                         x,
                                         offset,
                                         filter,
                                         mask_t,
                                         strides,
                                         paddings,
                                         dilations,
                                         deformable_groups,
                                         groups,
                                         &offset_out);
  phi::DenseTensor offset_all = offset_out[0];
  phi::DenseTensor deformableOffsetsOutput = offset_out[1];

  phi::DenseTensor deformableOffsetsBackwardInput;
  phi::DenseTensorMeta meta = {deformableOffsetsOutput.dtype(),
                               deformableOffsetsOutput.dims()};
  deformableOffsetsBackwardInput.set_meta(meta);
  dev_ctx.template Alloc<T>(&deformableOffsetsBackwardInput);

  // get deformableOffsetsBackwardInput and filter_grad
  std::string dataFormat = "NCHW";
  NpuOpRunner runner_conv2d;
  runner_conv2d.SetType("Conv2DBackpropInput")
      .AddInput(dev_ctx, phi::vectorize<int>(deformableOffsetsOutput.dims()))
      .AddInput(filter)
      .AddInput(out_grad)
      .AddOutput(deformableOffsetsBackwardInput)
      .AddAttrs({{"strides", conv2d_strides}})
      .AddAttrs({{"pads", conv2d_paddings}})
      .AddAttrs({{"dilations", conv2d_dilations}})
      .AddAttrs({{"groups", groups}})
      .AddAttrs({{"data_format", dataFormat}})
      .Run(stream);

  NpuOpRunner runner_conv2d_1;
  runner_conv2d_1.SetType("Conv2DBackpropFilter")
      .AddInput(deformableOffsetsOutput)
      .AddInput(dev_ctx, phi::vectorize<int>(filter.dims()))
      .AddInput(out_grad)
      .AddOutput(*filter_grad)
      .AddAttrs({{"strides", conv2d_strides}})
      .AddAttrs({{"pads", conv2d_paddings}})
      .AddAttrs({{"dilations", conv2d_dilations}})
      .AddAttrs({{"groups", groups}})
      .AddAttrs({{"data_format", dataFormat}})
      .Run(stream);

  // DeformableOffsetsGrad only support NHWC. Transform NCHW -> NHWC.
  std::vector<int> trans_strides{1, strides[0], strides[1], 1};
  std::vector<int> trans_paddings{
      paddings[0], paddings[0], paddings[1], paddings[1]};
  std::vector<int> trans_dilations{1, dilations[0], dilations[1], 1};

  phi::DenseTensor offset_input_nhwc, x_nhwc, offset_all_nhwc;
  phi::DenseTensor dx_nhwc, offset_grad_nhwc;
  std::vector<int64_t> perm_nhwc = {0, 2, 3, 1};
  // transform offset_grad
  std::vector<int> offset_input_nhwc_dims = {
      deformableOffsetsBackwardInput.dims()[0],
      deformableOffsetsBackwardInput.dims()[2],
      deformableOffsetsBackwardInput.dims()[3],
      deformableOffsetsBackwardInput.dims()[1]};
  phi::DenseTensorMeta offset_input_nhwc_meta = {
      deformableOffsetsBackwardInput.dtype(),
      phi::make_ddim(offset_input_nhwc_dims),
      phi::DataLayout::NHWC};
  offset_input_nhwc.set_meta(offset_input_nhwc_meta);
  TranposeNPU<Context, T>(dev_ctx,
                          stream,
                          &perm_nhwc,
                          deformableOffsetsBackwardInput,
                          &offset_input_nhwc);

  // transform x to x_nhwc
  phi::DDim x_nhwc_dims =
      phi::make_ddim({x.dims()[0], x.dims()[2], x.dims()[3], x.dims()[1]});
  phi::DenseTensorMeta x_nhwc_meta = {
      x.dtype(), x_nhwc_dims, phi::DataLayout::NHWC};
  x_nhwc.set_meta(x_nhwc_meta);
  TranposeNPU<Context, T>(dev_ctx, stream, &perm_nhwc, x, &x_nhwc);

  // transform offset_all to offset_all_nhwc
  phi::DDim offset_nhwc_dims = phi::make_ddim({offset_all.dims()[0],
                                               offset_all.dims()[2],
                                               offset_all.dims()[3],
                                               offset_all.dims()[1]});
  phi::DenseTensorMeta offset_nhwc_meta = {
      offset_all.dtype(), offset_nhwc_dims, phi::DataLayout::NHWC};
  offset_all_nhwc.set_meta(offset_nhwc_meta);
  TranposeNPU<Context, T>(
      dev_ctx, stream, &perm_nhwc, offset_all, &offset_all_nhwc);

  // transform dx to dx_nhwc
  std::vector<int> dx_nhwc_dims = {
      dx->dims()[0], dx->dims()[2], dx->dims()[3], dx->dims()[1]};
  phi::DenseTensorMeta dx_nhwc_meta = {
      dx->dtype(), phi::make_ddim(dx_nhwc_dims), phi::DataLayout::NHWC};
  dx_nhwc.set_meta(dx_nhwc_meta);
  dev_ctx.template Alloc<T>(&dx_nhwc);

  // transform offset_grad to offset_grad_nhwc
  std::vector<int> offset_grad_nhwc_dims = {offset_all.dims()[0],
                                            offset_all.dims()[2],
                                            offset_all.dims()[3],
                                            offset_all.dims()[1]};
  phi::DenseTensorMeta offset_grad_nhwc_meta = {
      offset_grad->dtype(),
      phi::make_ddim(offset_grad_nhwc_dims),
      phi::DataLayout::NHWC};
  offset_grad_nhwc.set_meta(offset_grad_nhwc_meta);
  dev_ctx.template Alloc<T>(&offset_grad_nhwc);

  std::vector<int> kernel_sizes{filter.dims()[3], filter.dims()[2]};
  std::string dataFormat_nhwc = "NHWC";
  NpuOpRunner runner_deformable_conv;
  runner_deformable_conv.SetType("DeformableOffsetsGrad")
      .AddInput(offset_input_nhwc)
      .AddInput(x_nhwc)
      .AddInput(offset_all_nhwc)
      .AddAttrs({{"strides", trans_strides}})
      .AddAttrs({{"pads", trans_paddings}})
      .AddAttrs({{"ksize", kernel_sizes}})
      .AddAttrs({{"dilations", trans_dilations}})
      .AddAttrs({{"data_format", dataFormat_nhwc}})
      .AddAttrs({{"deformable_groups", deformable_groups}})
      .AddAttrs({{"modulated", true}})
      .AddOutput(dx_nhwc)
      .AddOutput(offset_grad_nhwc)
      .Run(stream);

  dev_ctx.Wait();
  // transform dx_nhwc, offset_grad_nhwc to NCHW format.
  std::vector<int64_t> perm_nchw = {0, 3, 1, 2};
  TranposeNPU<Context, T>(dev_ctx, stream, &perm_nchw, dx_nhwc, dx);
  phi::DenseTensor tmp_offset_grad;
  phi::DenseTensorMeta tmp_offset_grad_meta = {
      offset.dtype(), offset_all.dims(), phi::DataLayout::NCHW};
  tmp_offset_grad.set_meta(tmp_offset_grad_meta);
  dev_ctx.template Alloc<T>(&tmp_offset_grad);
  TranposeNPU<Context, T>(
      dev_ctx, stream, &perm_nchw, offset_grad_nhwc, &tmp_offset_grad);

  auto offset_channel = offset.dims()[1];
  // split grad_offset to offset_grad and mask_grad
  phi::DenseTensor tmp_offset;
  phi::DenseTensorMeta tmp_offset_meta = {offset.dtype(), offset.dims()};
  tmp_offset.set_meta(tmp_offset_meta);
  dev_ctx.template Alloc<T>(&tmp_offset);

  std::vector<int32_t> begin_v{0, 0, 0, 0};
  std::vector<int32_t> end_v{
      offset.dims()[0], offset_channel, offset.dims()[2], offset.dims()[3]};
  std::vector<int32_t> strides_v{1, 1, 1, 1};
  StridedSliceNPU<T, Context>(
      dev_ctx, stream, tmp_offset_grad, begin_v, end_v, strides_v, &tmp_offset);

  begin_v[1] = offset_channel;
  end_v[1] = offset_all.dims()[1];
  StridedSliceNPU<T, Context>(
      dev_ctx, stream, tmp_offset_grad, begin_v, end_v, strides_v, mask_grad);

  // transform offset
  /*
   * Ascend:       xxxxx          ->      Paddle:   yyyyy
   *               xxxxx                            xxxxx
   *               xxxxx                            yyyyy
   *               yyyyy                            xxxxx
   *               yyyyy                            yyyyy
   *               yyyyy                            xxxxx
   */
  std::vector<int32_t> indices;
  auto start_x = 0;
  auto start_y = offset_channel / 2;
  while (start_x < (offset_channel / 2) && start_y < offset_channel) {
    indices.push_back(static_cast<int32_t>(start_y));
    indices.push_back(static_cast<int32_t>(start_x));
    start_x += 1;
    start_y += 1;
  }
  if (start_y < offset_channel) {
    indices.push_back(start_y);
  }

  NpuOpRunner gather_runner;
  gather_runner.SetType("GatherV2")
      .AddInput(tmp_offset)
      .AddInput(dev_ctx, std::move(indices))
      .AddInput(dev_ctx, std::vector<int32_t>({1}))
      .AddOutput(*offset_grad)
      .Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(deformable_conv,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::DeformableConvKernel,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(deformable_conv_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::DeformableConvGradKernel,
                          float) {}
