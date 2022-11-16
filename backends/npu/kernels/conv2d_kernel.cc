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
  VLOG(1)<< "------------------------------Conv2dKernel--------------------------------------------";
  VLOG(1)<< "0 - Conv2dKernel - Input: input" << OpPreparation::DebugString(input);
  VLOG(1)<< "0 - Conv2dKernel - Input: filter" << OpPreparation::DebugString(filter);
  VLOG(1)<< "0 - Conv2dKernel - Output: output" << OpPreparation::DebugString(*output);

  auto requested_size = OpPreparation::PrepareTensorWithFormat(*output, ACL_FORMAT_NC1HWC0);
  dev_ctx.template Alloc<T>(output, requested_size * paddle::experimental::SizeOf(output->dtype()));

  VLOG(1)<< "1 - Conv2dKernel - Input: input" << OpPreparation::DebugString(input);
  VLOG(1)<< "1 - Conv2dKernel - Input: filter" << OpPreparation::DebugString(filter);
  VLOG(1)<< "1 - Conv2dKernel - Output: output" << OpPreparation::DebugString(*output);

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


  VLOG(1)<< "2 - Conv2dKernel - Input: input" << OpPreparation::DebugString(input_tensor);
  VLOG(1)<< "2 - Conv2dKernel - Input: filter" << OpPreparation::DebugString(filter);
  VLOG(1)<< "2 - Conv2dKernel - Output: output" << OpPreparation::DebugString(output_tensor);

  auto stream = dev_ctx.stream();

  NpuOpRunner runner_conv2d;
  runner_conv2d.SetType("Conv2D")
      .AddInput(input_tensor)
      .AddInput(filter)
      .AddOutput(output_tensor)
      .AddAttrs({{"strides", strides_vec}})
      .AddAttrs({{"pads", paddings}})
      .AddAttrs({{"dilations", dilations_vec}})
      .AddAttrs({{"groups", groups}})
      .AddAttrs({{"data_format", data_format}})
      .Run(stream);
}

template <typename T, typename Context>
void Conv2DGradKernel(const Context& dev_ctx,
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
  VLOG(1)<< "-------------------------------Conv2DGradKernel-------------------------------------------";
  VLOG(1)<< "0 - Conv2DGradKernel - Input: input" << OpPreparation::DebugString(input);
  VLOG(1)<< "0 - Conv2DGradKernel - Input: filter" << OpPreparation::DebugString(filter);
  VLOG(1)<< "0 - Conv2DGradKernel - Input: output_grad" << OpPreparation::DebugString(output_grad);


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
    VLOG(1)<< "0 - Conv2DGradKernel - Output: filter_grad" << OpPreparation::DebugString(*filter_grad);
    if (groups > 1) {
      dev_ctx.template Alloc<T>(filter_grad);
    } else {
      auto requested_size = OpPreparation::PrepareTensorWithFormat(*filter_grad, ACL_FORMAT_FRACTAL_Z);
      dev_ctx.template Alloc<T>(filter_grad, requested_size * paddle::experimental::SizeOf(filter_grad->dtype()));
    }
    VLOG(1)<< "1 - Conv2DGradKernel - Output: filter_grad" << OpPreparation::DebugString(*filter_grad);

    NpuOpRunner runner_filter;
    runner_filter.SetType("Conv2DBackpropFilter")
        .AddInput(input_tensor)
        .AddInput(dev_ctx, phi::vectorize<int>(filter.dims()))
        .AddInput(output_grad_tensor)
        .AddOutput(*filter_grad)
        .AddAttrs({{"strides", strides_vec}})
        .AddAttrs({{"pads", paddings}})
        .AddAttrs({{"dilations", dilations_vec}})
        .AddAttrs({{"groups", groups}})
        .AddAttrs({{"data_format", data_format}})
        .Run(stream);
  }

  if (input_grad) {
    VLOG(1)<< "0 - Conv2DGradKernel - Output: input_grad" << OpPreparation::DebugString(*input_grad);
    
    auto requested_size = OpPreparation::PrepareTensorWithFormat(*input_grad, ACL_FORMAT_NC1HWC0);
    dev_ctx.template Alloc<T>(input_grad, requested_size * paddle::experimental::SizeOf(input_grad->dtype()));

    VLOG(1)<< "1 - Conv2DGradKernel - Output: input_grad" << OpPreparation::DebugString(*input_grad);

    NpuOpRunner runner_filter;
    runner_filter.SetType("Conv2DBackpropInput")
        .AddInput(dev_ctx, phi::vectorize<int>(input_tensor.dims()))
        .AddInput(filter)
        .AddInput(output_grad_tensor)
        .AddOutput(*input_grad)
        .AddAttrs({{"strides", strides_vec}})
        .AddAttrs({{"pads", paddings}})
        .AddAttrs({{"dilations", dilations_vec}})
        .AddAttrs({{"groups", groups}})
        .AddAttrs({{"data_format", data_format}})
        .Run(stream);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(conv2d,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv2d_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::Conv2DGradKernel,
                          float,
                          phi::dtype::float16) {}
