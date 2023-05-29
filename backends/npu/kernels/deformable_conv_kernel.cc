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
  phi::DenseTensor offset_all;
  auto mask_t = *mask.get_ptr();

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

  // strided_slice_x
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
  std::vector<int64_t> begin_v{0, 1, 0, 0};
  std::vector<int64_t> end_v{
      offset.dims()[0], offset_channel, offset.dims()[2], offset.dims()[3]};
  std::vector<int64_t> strides_v{1, 2, 1, 1};
  NpuOpRunner runner1;
  runner1.SetType("StridedSlice")
      .AddInput(offset)
      .AddInput(dev_ctx, std::move(begin_v))
      .AddInput(dev_ctx, std::move(end_v))
      .AddInput(dev_ctx, std::move(strides_v))
      .AddAttr("begin_mask", 0)
      .AddAttr("end_mask", 0)
      .AddAttr("ellipsis_mask", 0)
      .AddAttr("new_axis_mask", 0)
      .AddAttr("shrink_axis_mask", 0)
      .AddOutput(strided_slice_x)
      .Run(stream);
  // strided_slice_y
  std::vector<int64_t> begin_v1{0, 0, 0, 0};
  std::vector<int64_t> end_v1{
      offset.dims()[0], offset_channel - 1, offset.dims()[2], offset.dims()[3]};
  NpuOpRunner runner2;
  runner2.SetType("StridedSlice")
      .AddInput(offset)
      .AddInput(dev_ctx, std::move(begin_v1))
      .AddInput(dev_ctx, std::move(end_v1))
      .AddInput(dev_ctx, std::move(strides_v))
      .AddAttr("begin_mask", 0)
      .AddAttr("end_mask", 0)
      .AddAttr("ellipsis_mask", 0)
      .AddAttr("new_axis_mask", 0)
      .AddAttr("shrink_axis_mask", 0)
      .AddOutput(strided_slice_y)
      .Run(stream);
  // concat
  std::vector<int> offset_all_dims_vec{
      static_cast<int>(x.dims()[0]),
      static_cast<int>((offset.dims()[1] + mask_t.dims()[1])),
      static_cast<int>(x.dims()[2]),
      static_cast<int>(x.dims()[3])};
  offset_all.Resize(phi::make_ddim(offset_all_dims_vec));
  dev_ctx.template Alloc<T>(&offset_all);

  std::vector<phi::DenseTensor> concat_inputs;
  concat_inputs.push_back(strided_slice_x);
  concat_inputs.push_back(strided_slice_y);
  concat_inputs.push_back(mask_t);
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
      out->dtype(), phi::make_ddim(tmp_dims), phi::DataLayout::NHWC};
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

  phi::DenseTensor deformableOffsetsOutput;
  // nhwc -> nchw
  std::vector<int64_t> perm_nchw = {0, 3, 1, 2};
  std::vector<int> out_nchw_dims = {tmp_output.dims()[0],
                                    tmp_output.dims()[3],
                                    tmp_output.dims()[1],
                                    tmp_output.dims()[2]};
  phi::DenseTensorMeta out_nchw_meta = {tmp_output.dtype(),
                                        phi::make_ddim(out_nchw_dims)};
  deformableOffsetsOutput.set_meta(out_nchw_meta);
  dev_ctx.template Alloc<T>(&deformableOffsetsOutput);
  NpuOpRunner runner;
  runner.SetType("Transpose")
      .AddInput(tmp_output)
      .AddInput(dev_ctx, std::move(perm_nchw))
      .AddOutput(deformableOffsetsOutput)
      .Run(stream);

  std::vector<int> conv2d_strides{1, 1, filter.dims()[3], filter.dims()[2]};
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

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(deformable_conv,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::DeformableConvKernel,
                          float) {}
