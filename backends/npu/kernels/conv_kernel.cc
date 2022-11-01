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

#include "kernels/funcs/conv_util.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename Context>
static void CastToFP16(const Context& dev_ctx,
                       const phi::DenseTensor& in,
                       phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();
  dev_ctx.template Alloc<phi::dtype::float16>(out);
  NpuOpRunner runner;
  runner.SetType("Cast")
      .AddInput(in)
      .AddOutput(*out)
      .AddAttr("dst_type", ACL_FLOAT16)
      .Run(stream);
}

template <typename Context>
static void CastToFP32(const Context& dev_ctx,
                       const phi::DenseTensor& in,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<float>(out);
  auto stream = dev_ctx.stream();
  NpuOpRunner runner;
  runner.SetType("Cast")
      .AddInput(in)
      .AddOutput(*out)
      .AddAttr("dst_type", ACL_FLOAT)
      .Run(stream);
}

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
  phi::DDim in_data_dims;
  phi::DDim filter_data_dims;

  if (channel_last) {
    in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  } else {
    in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  }
  filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  custom_kernel::UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  std::vector<int> strides_vec(4, 1);
  std::vector<int> dilations_vec(4, 1);

  phi::DenseTensor input_tensor(input), output_tensor(*output);
  if (channel_last) {
    phi::DenseTensorMeta input_meta = {
        input.dtype(), input.dims(), phi::DataLayout::kNHWC};
    phi::DenseTensorMeta output_meta = {
        output->dtype(), output->dims(), phi::DataLayout::kNHWC};
    input_tensor.set_meta(input_meta);
    output_tensor.set_meta(output_meta);
    strides_vec[1] = strides[0];
    strides_vec[2] = strides[1];
    dilations_vec[1] = dilations[0];
    dilations_vec[2] = dilations[1];
  } else {
    strides_vec[2] = strides[0];
    strides_vec[3] = strides[1];
    dilations_vec[2] = dilations[0];
    dilations_vec[3] = dilations[1];
  }

  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("Conv2D",
                                   {input_tensor, filter},
                                   {output_tensor},
                                   {{"strides", strides_vec},
                                    {"pads", paddings},
                                    {"dilations", dilations_vec},
                                    {"groups", groups},
                                    {"data_format", data_format}});
  runner.Run(stream);
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
  phi::DDim in_data_dims;
  phi::DDim filter_data_dims;

  if (channel_last) {
    in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  } else {
    in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  }
  filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  custom_kernel::UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  std::vector<int> strides_vec(4, 1);
  std::vector<int> dilations_vec(4, 1);

  phi::DenseTensor input_tensor(input), output_grad_tensor(output_grad);
  if (channel_last) {
    phi::DenseTensorMeta input_meta = {
        input.dtype(), input.dims(), phi::DataLayout::kNHWC};
    phi::DenseTensorMeta output_grad_meta = {
        output_grad.dtype(), output_grad.dims(), phi::DataLayout::kNHWC};
    input_tensor.set_meta(input_meta);
    output_grad_tensor.set_meta(output_grad_meta);
    strides_vec[1] = strides[0];
    strides_vec[2] = strides[1];
    dilations_vec[1] = dilations[0];
    dilations_vec[2] = dilations[1];
  } else {
    strides_vec[2] = strides[0];
    strides_vec[3] = strides[1];
    dilations_vec[2] = dilations[0];
    dilations_vec[3] = dilations[1];
  }

  auto stream = dev_ctx.stream();
  if (filter_grad) {
    dev_ctx.template Alloc<T>(filter_grad);
    std::vector<int> filter_shape_vec = phi::vectorize<int>(filter.dims());

    phi::DenseTensor filter_grad_fp32;
    phi::DenseTensorMeta filter_grad_fp32_meta = {phi::DataType::FLOAT32,
                                                  filter_grad->dims()};
    filter_grad_fp32.set_meta(filter_grad_fp32_meta);

    if (input.dtype() == phi::DataType::FLOAT16) {
      CastToFP32<Context>(dev_ctx, *filter_grad, &filter_grad_fp32);
    } else {
      filter_grad_fp32 = *filter_grad;
    }
    const auto& runner = NpuOpRunner("Conv2DBackpropFilterD",
                                     {input_tensor, output_grad_tensor},
                                     {filter_grad_fp32},
                                     {{"filter_size", filter_shape_vec},
                                      {"strides", strides_vec},
                                      {"pads", paddings},
                                      {"dilations", dilations_vec},
                                      {"groups", groups},
                                      {"data_format", data_format}});
    runner.Run(stream);

    if (input.dtype() == phi::DataType::FLOAT16) {
      CastToFP16<Context>(dev_ctx, filter_grad_fp32, filter_grad);
    }
  }
  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    std::vector<int> input_shape_vec = phi::vectorize<int>(input.dims());

    phi::DenseTensor input_grad_tensor(*input_grad);
    if (channel_last) {
      phi::DenseTensorMeta input_grad_meta = {
          input_grad->dtype(), input_grad->dims(), phi::DataLayout::kNHWC};
      input_grad_tensor.set_meta(input_grad_meta);
    }
    const auto& runner = NpuOpRunner("Conv2DBackpropInputD",
                                     {filter, output_grad_tensor},
                                     {input_grad_tensor},
                                     {{"input_size", input_shape_vec},
                                      {"strides", strides_vec},
                                      {"pads", paddings},
                                      {"dilations", dilations_vec},
                                      {"groups", groups},
                                      {"data_format", data_format}});
    runner.Run(stream);
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

  std::vector<int> padding = paddings_in;
  std::vector<int> dilation = dilations_in;

  const bool channel_last = data_format == "NHWC";
  if (channel_last) {
    PADDLE_ENFORCE_EQ(
        out->dims()[out->dims().size() - 1],
        input.dims()[input.dims().size() - 1],
        phi::errors::InvalidArgument(
            "ShapeError: The output channels must be equal to the "
            "input channels. But receivced output channel number is %d "
            "and input channel number is %d",
            out->dims()[out->dims().size() - 1],
            input.dims()[input.dims().size() - 1]));
  } else {
    PADDLE_ENFORCE_EQ(
        out->dims()[1],
        input.dims()[1],
        phi::errors::InvalidArgument(
            "ShapeError: The output channels must be equal to the "
            "input channels. But receivced output channel number is %d "
            "and input channel number is %d",
            out->dims()[1],
            input.dims()[1]));
  }

  auto in_dims = input.dims();
  auto filter_dims = filter.dims();
  phi::DDim in_data_dims;
  phi::DDim filter_data_dims;

  if (channel_last) {
    in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  } else {
    in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  }
  filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  custom_kernel::UpdatePaddingAndDilation(
      &padding, &dilation, padding_algorithm, in_data_dims, stride, ksize);

  std::vector<int> strides(4, 1);
  std::vector<int> dilations(4, 1);

  phi::DenseTensor input_tensor(input), output_tensor(*out);

  if (channel_last) {
    phi::DenseTensorMeta input_meta = {
        input.dtype(), input.dims(), phi::DataLayout::kNHWC};
    phi::DenseTensorMeta output_meta = {
        out->dtype(), out->dims(), phi::DataLayout::kNHWC};
    input_tensor.set_meta(input_meta);
    output_tensor.set_meta(output_meta);
    strides[1] = stride[0];
    strides[2] = stride[1];
    dilations[1] = dilation[0];
    dilations[2] = dilation[1];
  } else {
    strides[2] = stride[0];
    strides[3] = stride[1];
    dilations[2] = dilation[0];
    dilations[3] = dilation[1];
  }

  auto stream = dev_ctx.stream();

  // Transform filter (n, 1, h, w) --> (1, n, h, w)
  phi::DenseTensor transformed_filter;
  phi::DenseTensorMeta meta = {
      filter.dtype(),
      {filter.dims()[1], filter.dims()[0], filter.dims()[2], filter.dims()[3]}};

  transformed_filter.set_meta(meta);
  dev_ctx.template Alloc<T>(&transformed_filter);
  std::vector<int> perm = {1, 0, 2, 3};
  const auto& runner_trans = NpuOpRunner(
      "TransposeD", {filter}, {transformed_filter}, {{"perm", perm}});
  runner_trans.Run(stream);

  const auto& runner = NpuOpRunner("DepthwiseConv2D",
                                   {input_tensor, transformed_filter},
                                   {output_tensor},
                                   {{"strides", strides},
                                    {"dilations", dilations},
                                    {"pads", padding},
                                    {"data_format", data_format}});
  runner.Run(stream);
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
  std::vector<int> padding = paddings_in;
  std::vector<int> dilation = dilations_in;
  const bool channel_last = data_format == "NHWC";
  // update padding and dilation
  auto in_dims = input.dims();
  auto filter_dims = filter.dims();
  phi::DDim in_data_dims;
  phi::DDim filter_data_dims;

  if (channel_last) {
    in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  } else {
    in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  }
  filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  custom_kernel::UpdatePaddingAndDilation(
      &padding, &dilation, padding_algorithm, in_data_dims, stride, ksize);

  auto stream = dev_ctx.stream();

  // Transform filter (n, 1, h, w) --> (1, n, h, w)
  phi::DenseTensor transformed_filter;
  phi::DenseTensorMeta meta = {
      filter.dtype(),
      {filter.dims()[1], filter.dims()[0], filter.dims()[2], filter.dims()[3]}};

  transformed_filter.set_meta(meta);
  dev_ctx.template Alloc<T>(&transformed_filter);
  std::vector<int> perm = {1, 0, 2, 3};
  const auto& runner_trans = NpuOpRunner(
      "TransposeD", {filter}, {transformed_filter}, {{"perm", perm}});
  runner_trans.Run(stream);

  // construct NPU attr
  std::vector<int> strides(4, 1);
  std::vector<int> dilations(4, 1);

  phi::DenseTensor input_tensor(input), output_grad_tensor(out_grad);
  if (channel_last) {
    phi::DenseTensorMeta input_meta = {
        input.dtype(), input.dims(), phi::DataLayout::kNHWC};
    phi::DenseTensorMeta output_grad_meta = {
        out_grad.dtype(), out_grad.dims(), phi::DataLayout::kNHWC};
    input_tensor.set_meta(input_meta);
    output_grad_tensor.set_meta(output_grad_meta);
    strides[1] = stride[0];
    strides[2] = stride[1];
    dilations[1] = dilation[0];
    dilations[2] = dilation[1];
  } else {
    strides[2] = stride[0];
    strides[3] = stride[1];
    dilations[2] = dilation[0];
    dilations[3] = dilation[1];
  }

  if (filter_grad) {
    dev_ctx.template Alloc<T>(filter_grad);

    PADDLE_ENFORCE_EQ(
        (dilations[2] == 1 && dilations[3] == 1),
        true,
        phi::errors::InvalidArgument(
            "dilation_h and dilation_w in DepthwiseConv2DBackpropFilterD "
            "must be equal to 1, but got dilation_h %d, dilation_w %d",
            dilation[2],
            dilation[3]));

    NpuOpRunner runner;
    runner.SetType("DepthwiseConv2DBackpropFilterD")
        .AddInput(input_tensor)
        .AddInput(output_grad_tensor)
        .AddOutput(*filter_grad)
        .AddAttr("filter_size", phi::vectorize(transformed_filter.dims()))
        .AddAttr("strides", strides)
        .AddAttr("dilations", dilations)
        .AddAttr("pads", padding)
        .AddAttr("data_format", data_format)
        .Run(stream);
  }
  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);

    phi::DenseTensor input_grad_tensor(*input_grad);
    if (channel_last) {
      phi::DenseTensorMeta input_grad_meta = {
          input_grad->dtype(), input_grad->dims(), phi::DataLayout::kNHWC};
      input_grad_tensor.set_meta(input_grad_meta);
    }
    NpuOpRunner runner;
    runner.SetType("DepthwiseConv2DBackpropInputD")
        .AddInput(transformed_filter)
        .AddInput(output_grad_tensor)
        .AddOutput(input_grad_tensor)
        .AddAttr("input_size", phi::vectorize(input.dims()))
        .AddAttr("strides", strides)
        .AddAttr("dilations", dilations)
        .AddAttr("pads", padding)
        .AddAttr("data_format", data_format)
        .Run(stream);
  }
}

template <typename T, typename Context>
void Conv3dKernel(const Context& dev_ctx,
                  const phi::DenseTensor& input,
                  const phi::DenseTensor& filter,
                  const std::vector<int>& strides,
                  const std::vector<int>& padding,
                  const std::string& padding_algorithm,
                  int groups,
                  const std::vector<int>& dilation,
                  const std::string& data_format,
                  bool use_addto,
                  int workspace_size_MB,
                  bool exhaustive_search,
                  phi::DenseTensor* out) {
  auto paddings = padding;
  auto dilations = dilation;

  PADDLE_ENFORCE_EQ(data_format,
                    "NCDHW",
                    phi::errors::Unimplemented(
                        "the data_format must be NCDHW in "
                        "the npu kernel of conv3d, but got data_format "
                        "= [%s]",
                        data_format));

  PADDLE_ENFORCE_EQ(
      groups,
      1,
      phi::errors::Unimplemented("the groups must be 1 in "
                                 "the npu kernel of conv3d, but got groups "
                                 "= [%d]",
                                 groups));

  dev_ctx.template Alloc<T>(out);

  phi::DenseTensor input_tensor(input);
  phi::DenseTensor filter_tensor(filter);
  phi::DenseTensor output_tensor(*out);

  phi::DenseTensorMeta input_meta = {
      input_tensor.dtype(), input_tensor.dims(), phi::DataLayout::kNCDHW};
  input_tensor.set_meta(input_meta);

  phi::DenseTensorMeta filter_meta = {
      filter_tensor.dtype(), filter_tensor.dims(), phi::DataLayout::kNCDHW};
  filter_tensor.set_meta(filter_meta);

  phi::DenseTensorMeta output_meta = {
      output_tensor.dtype(), output_tensor.dims(), phi::DataLayout::kNCDHW};
  output_tensor.set_meta(output_meta);

  // update padding and dilation
  auto in_dims = input.dims();
  auto filter_dims = filter.dims();
  phi::DDim in_data_dims;
  phi::DDim filter_data_dims;

  in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  custom_kernel::UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  std::vector<int> strides_vec(5, 1);
  std::vector<int> dilations_vec(5, 1);

  strides_vec[2] = strides[0];
  strides_vec[3] = strides[1];
  strides_vec[4] = strides[2];
  dilations_vec[2] = dilations[0];
  dilations_vec[3] = dilations[1];
  dilations_vec[4] = dilations[2];

  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner("Conv3D",
                                   {input_tensor, filter_tensor},
                                   {output_tensor},
                                   {{"strides", strides_vec},
                                    {"pads", paddings},
                                    {"dilations", dilations_vec},
                                    {"groups", groups},
                                    {"data_format", data_format}});
  runner.Run(stream);
}

template <typename T, typename Context>
void Conv3dGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& input,
                      const phi::DenseTensor& filter,
                      const phi::DenseTensor& out_grad,
                      const std::vector<int>& strides,
                      const std::vector<int>& padding,
                      const std::string& padding_algorithm,
                      int groups,
                      const std::vector<int>& dilation,
                      const std::string& data_format,
                      bool use_addto,
                      int workspace_size_MB,
                      bool exhaustive_search,
                      phi::DenseTensor* input_grad,
                      phi::DenseTensor* filter_grad) {
  auto paddings = padding;
  auto dilations = dilation;

  PADDLE_ENFORCE_EQ(data_format,
                    "NCDHW",
                    phi::errors::Unimplemented(
                        "the data_format must be NCDHW in "
                        "the npu kernel of conv3d, but got data_format "
                        "= [%s]",
                        data_format));

  PADDLE_ENFORCE_EQ(
      groups,
      1,
      phi::errors::Unimplemented("the groups must be 1 in "
                                 "the npu kernel of conv3d, but got groups "
                                 "= [%d]",
                                 groups));

  phi::DenseTensor input_tensor(input);
  phi::DenseTensor filter_tensor(filter);
  phi::DenseTensor output_grad_tensor(out_grad);

  phi::DenseTensorMeta input_meta = {
      input_tensor.dtype(), input_tensor.dims(), phi::DataLayout::kNCDHW};
  input_tensor.set_meta(input_meta);

  phi::DenseTensorMeta filter_meta = {
      filter_tensor.dtype(), filter_tensor.dims(), phi::DataLayout::kNCDHW};
  filter_tensor.set_meta(filter_meta);

  phi::DenseTensorMeta output_meta = {output_grad_tensor.dtype(),
                                      output_grad_tensor.dims(),
                                      phi::DataLayout::kNCDHW};
  output_grad_tensor.set_meta(output_meta);

  // update padding and dilation
  auto in_dims = input.dims();
  auto filter_dims = filter.dims();
  phi::DDim in_data_dims;
  phi::DDim filter_data_dims;

  in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  custom_kernel::UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  std::vector<int> strides_vec(5, 1);
  std::vector<int> dilations_vec(5, 1);

  strides_vec[2] = strides[0];
  strides_vec[3] = strides[1];
  strides_vec[4] = strides[2];
  dilations_vec[2] = dilations[0];
  dilations_vec[3] = dilations[1];
  dilations_vec[4] = dilations[2];

  auto stream = dev_ctx.stream();

  if (filter_grad) {
    dev_ctx.template Alloc<T>(filter_grad);
    std::vector<int> filter_shape_vec = phi::vectorize<int>(filter.dims());

    phi::DenseTensor filter_grad_tensor(*filter_grad);
    phi::DenseTensorMeta filter_grad_meta = {filter_grad_tensor.dtype(),
                                             filter_grad_tensor.dims(),
                                             phi::DataLayout::kNCDHW};
    filter_grad_tensor.set_meta(filter_grad_meta);

    const auto& runner = NpuOpRunner("Conv3DBackpropFilterD",
                                     {input_tensor, output_grad_tensor},
                                     {filter_grad_tensor},
                                     {{"filter_size", filter_shape_vec},
                                      {"strides", strides_vec},
                                      {"pads", paddings},
                                      {"dilations", dilations_vec},
                                      {"groups", groups},
                                      {"data_format", data_format}});
    runner.Run(stream);
  }

  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    std::vector<int> input_shape_vec = phi::vectorize<int>(input.dims());

    phi::DenseTensor input_grad_tensor(*input_grad);
    phi::DenseTensorMeta input_grad_meta = {input_grad_tensor.dtype(),
                                            input_grad_tensor.dims(),
                                            phi::DataLayout::kNCDHW};
    input_grad_tensor.set_meta(input_grad_meta);

    const auto& runner = NpuOpRunner("Conv3DBackpropInputD",
                                     {filter_tensor, output_grad_tensor},
                                     {input_grad_tensor},
                                     {{"input_size", input_shape_vec},
                                      {"strides", strides_vec},
                                      {"pads", paddings},
                                      {"dilations", dilations_vec},
                                      {"groups", groups},
                                      {"data_format", data_format}});
    runner.Run(stream);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(conv2d,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv2d_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(depthwise_conv2d,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::DepthwiseConv2dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(depthwise_conv2d_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::DepthwiseConv2dGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv3d,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::Conv3dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv3d_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::Conv3dGradKernel,
                          float,
                          phi::dtype::float16) {}
