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

#include "paddle/utils/optional.h"

namespace custom_kernel {

template <typename T = int>
inline void UpdatePaddingAndDilation(std::vector<T>* paddings,
                                     std::vector<T>* dilation,
                                     const std::string padding_algorithm,
                                     const phi::DDim data_dims,
                                     const std::vector<T>& strides,
                                     const std::vector<T>& ksize) {
  // set padding size == data_dims.size() * 2
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

  // when padding_algorithm is "VALID" or "SAME"
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

template <typename T, typename Context>
void Conv2dKernel(const Context& dev_ctx,
                  const phi::DenseTensor& input,
                  const phi::DenseTensor& filter,
                  const std::vector<int>& strides_t,
                  const std::vector<int>& paddings_t,
                  const std::string& padding_algorithm,
                  int groups,
                  const std::vector<int>& dilations_t,
                  const std::string& data_format,
                  bool use_addto,
                  int workspace_size_MB,
                  bool exhaustive_search,
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
  UpdatePaddingAndDilation(
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
                      int groups,
                      const std::vector<int>& dilations_t,
                      const std::string& data_format,
                      bool use_addto,
                      int workspace_size_MB,
                      bool exhaustive_search,
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
  UpdatePaddingAndDilation(
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

    const auto& runner = NpuOpRunner("Conv2DBackpropFilterD",
                                     {input_tensor, output_grad_tensor},
                                     {*filter_grad},
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
void DepthwiseConvKernel(const Context& dev_ctx,
                         const phi::DenseTensor& input,
                         const phi::DenseTensor& filter,
                         const std::vector<int>& strides_t,
                         const std::vector<int>& paddings_t,
                         const std::string& padding_algorithm,
                         int groups,
                         const std::vector<int>& dilations_t,
                         const std::string& data_format,
                         bool use_addto,
                         int workspace_size_MB,
                         bool exhaustive_search,
                         bool fuse_relu,
                         phi::DenseTensor* output) {
  dev_ctx.template Alloc<T>(output);
  auto stream = dev_ctx.stream();

  auto stride = strides_t;
  auto padding = paddings_t;
  auto dilation = dilations_t;

  const bool channel_last = data_format == "NHWC";
  if (channel_last) {
    PADDLE_ENFORCE_EQ(
        output->dims()[output->dims().size() - 1],
        input.dims()[input.dims().size() - 1],
        phi::errors::InvalidArgument(
            "ShapeError: The output channels must be equal to the "
            "input channels. But receivced output channel number is %d "
            "and input channel number is %d",
            output->dims()[output->dims().size() - 1],
            input.dims()[input.dims().size() - 1]));
  } else {
    PADDLE_ENFORCE_EQ(
        output->dims()[1],
        input.dims()[1],
        phi::errors::InvalidArgument(
            "ShapeError: The output channels must be equal to the "
            "input channels. But receivced output channel number is %d "
            "and input channel number is %d",
            output->dims()[1],
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
  UpdatePaddingAndDilation(
      &padding, &dilation, padding_algorithm, in_data_dims, stride, ksize);

  std::vector<int> strides(4, 1);
  std::vector<int> dilations(4, 1);

  phi::DenseTensor input_tensor(input), output_tensor(*output);
  if (channel_last) {
    phi::DenseTensorMeta input_tensor_meta = {
        input.dtype(), input.dims(), phi::DataLayout::kNHWC};
    phi::DenseTensorMeta output_tensor_meta = {
        output->dtype(), output->dims(), phi::DataLayout::kNHWC};
    input_tensor.set_meta(input_tensor_meta);
    output_tensor.set_meta(output_tensor_meta);
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

  // Transform filter (n, 1, h, w) --> (1, n, h, w)
  phi::DenseTensorMeta transformed_filter_meta = {
      filter.dtype(),
      {filter.dims()[1], filter.dims()[0], filter.dims()[2], filter.dims()[3]}};
  phi::DenseTensor transformed_filter;
  transformed_filter.set_meta(transformed_filter_meta);
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
void DepthwiseConvGradKernel(const Context& dev_ctx,
                             const phi::DenseTensor& input,
                             const phi::DenseTensor& filter,
                             const phi::DenseTensor& output_grad,
                             const std::vector<int>& strides_t,
                             const std::vector<int>& paddings_t,
                             const std::string& padding_algorithm,
                             int groups,
                             const std::vector<int>& dilations_t,
                             const std::string& data_format,
                             bool use_addto,
                             int workspace_size_MB,
                             bool exhaustive_search,
                             bool fuse_relu,
                             phi::DenseTensor* input_grad,
                             phi::DenseTensor* filter_grad) {
  auto stream = dev_ctx.stream();

  auto stride = strides_t;
  auto padding = paddings_t;
  auto dilation = dilations_t;
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
  UpdatePaddingAndDilation(
      &padding, &dilation, padding_algorithm, in_data_dims, stride, ksize);

  // Transform filter (n, 1, h, w) --> (1, n, h, w)
  phi::DenseTensorMeta transformed_filter_meta = {
      filter.dtype(),
      {filter.dims()[1], filter.dims()[0], filter.dims()[2], filter.dims()[3]}};
  phi::DenseTensor transformed_filter;
  transformed_filter.set_meta(transformed_filter_meta);
  dev_ctx.template Alloc<T>(&transformed_filter);
  std::vector<int> perm = {1, 0, 2, 3};
  const auto& runner_trans = NpuOpRunner(
      "TransposeD", {filter}, {transformed_filter}, {{"perm", perm}});
  runner_trans.Run(stream);

  // construct NPU attr
  std::vector<int> strides(4, 1);
  std::vector<int> dilations(4, 1);

  phi::DenseTensor input_tensor(input), output_grad_tensor(output_grad);
  if (channel_last) {
    phi::DenseTensorMeta input_tensor_meta = {
        input.dtype(), input.dims(), phi::DataLayout::kNHWC};
    phi::DenseTensorMeta output_grad_tensor_meta = {
        output_grad.dtype(), output_grad.dims(), phi::DataLayout::kNHWC};
    input_tensor.set_meta(input_tensor_meta);
    output_grad_tensor.set_meta(output_grad_tensor_meta);
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
      phi::DenseTensorMeta input_grad_tensor_meta = {
          input_grad->dtype(), input_grad->dims(), phi::DataLayout::kNHWC};
      input_grad_tensor.set_meta(input_grad_tensor_meta);
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

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    conv2d, ascend, ALL_LAYOUT, custom_kernel::Conv2dKernel, float) {}
PD_REGISTER_PLUGIN_KERNEL(
    conv2d_grad, ascend, ALL_LAYOUT, custom_kernel::Conv2dGradKernel, float) {}

PD_REGISTER_PLUGIN_KERNEL(depthwise_conv2d,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::DepthwiseConvKernel,
                          float,
                          phi::dtype::float16) {}
PD_REGISTER_PLUGIN_KERNEL(depthwise_conv2d_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::DepthwiseConvGradKernel,
                          float,
                          phi::dtype::float16) {}
