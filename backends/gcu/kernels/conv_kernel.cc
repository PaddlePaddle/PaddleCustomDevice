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

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void GcuConvKernel(const Context& dev_ctx,
                   const phi::DenseTensor& input,
                   const phi::DenseTensor& filter,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   const std::string& padding_algorithm,
                   int groups,
                   const std::vector<int>& dilations,
                   const std::string& data_format,
                   phi::DenseTensor* out,
                   const std::string& op_type) {
  dev_ctx.template Alloc<T>(out);

  TensorNameMap input_names;
  input_names["Input"] = {"input"};
  input_names["Filter"] = {"filter"};

  TensorValueMap inputs;
  inputs["Input"] = {const_cast<DenseTensor*>(&input)};
  inputs["Filter"] = {const_cast<DenseTensor*>(&filter)};

  TensorNameMap output_names;
  output_names["Output"] = {"out"};

  TensorValueMap outputs;
  outputs["Output"] = {out};

  GcuAttributeMap attrs;
  attrs["strides"] = strides;
  attrs["paddings"] = paddings;
  attrs["padding_algorithm"] = padding_algorithm;
  attrs["dilations"] = dilations;
  attrs["groups"] = groups;
  attrs["data_format"] = data_format;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);
}

template <typename T, typename Context>
void GcuConvGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& input,
                       const phi::DenseTensor& filter,
                       const phi::DenseTensor& out_grad,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       const std::string& padding_algorithm,
                       int groups,
                       const std::vector<int>& dilations,
                       const std::string& data_format,
                       phi::DenseTensor* input_grad,
                       phi::DenseTensor* filter_grad,
                       const std::string& op_type) {
  TensorNameMap input_names;
  input_names["Input"] = {"input"};
  input_names["Filter"] = {"filter"};
  input_names[GradVarName("Output")] = {"out_grad"};

  TensorValueMap inputs;
  inputs["Input"] = {const_cast<DenseTensor*>(&input)};
  inputs["Filter"] = {const_cast<DenseTensor*>(&filter)};
  inputs[GradVarName("Output")] = {const_cast<DenseTensor*>(&out_grad)};

  TensorNameMap output_names;
  TensorValueMap outputs;
  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    output_names[GradVarName("Input")] = {"input_grad"};
    outputs[GradVarName("Input")] = {input_grad};
  }
  if (filter_grad) {
    dev_ctx.template Alloc<T>(filter_grad);
    output_names[GradVarName("Filter")] = {"filter_grad"};
    outputs[GradVarName("Filter")] = {filter_grad};
  }

  GcuAttributeMap attrs;
  attrs["strides"] = strides;
  attrs["paddings"] = paddings;
  attrs["padding_algorithm"] = padding_algorithm;
  attrs["dilations"] = dilations;
  attrs["groups"] = groups;
  attrs["data_format"] = data_format;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);
}

template <typename T, typename Context>
void Conv2dKernel(const Context& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& filter,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string& padding_algorithm,
                  const std::vector<int>& dilations,
                  int groups,
                  const std::string& data_format,
                  DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("conv2d");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuConvKernel<T, Context>(dev_ctx,
                              input,
                              filter,
                              strides,
                              paddings,
                              padding_algorithm,
                              groups,
                              dilations,
                              data_format,
                              out,
                              "conv2d");
  }
}

template <typename T, typename Context>
void Conv2DGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& input,
                      const phi::DenseTensor& filter,
                      const phi::DenseTensor& output_grad,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      const std::string& padding_algorithm,
                      const std::vector<int>& dilations,
                      int groups,
                      const std::string& data_format,
                      phi::DenseTensor* input_grad,
                      phi::DenseTensor* filter_grad) {
  PADDLE_GCU_KERNEL_TRACE("conv2d_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuConvGradKernel<T, Context>(dev_ctx,
                                  input,
                                  filter,
                                  output_grad,
                                  strides,
                                  paddings,
                                  padding_algorithm,
                                  groups,
                                  dilations,
                                  data_format,
                                  input_grad,
                                  filter_grad,
                                  "conv2d_grad");
  }
}

template <typename T, typename Context>
void DepthwiseConv2dKernel(const Context& dev_ctx,
                           const phi::DenseTensor& input,
                           const phi::DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format,
                           phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("depthwise_conv2d");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuConvKernel<T, Context>(dev_ctx,
                              input,
                              filter,
                              strides,
                              paddings,
                              padding_algorithm,
                              groups,
                              dilations,
                              data_format,
                              out,
                              "depthwise_conv2d");
  }
}

template <typename T, typename Context>
void DepthwiseConv2dGradKernel(const Context& dev_ctx,
                               const phi::DenseTensor& input,
                               const phi::DenseTensor& filter,
                               const phi::DenseTensor& out_grad,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format,
                               phi::DenseTensor* input_grad,
                               phi::DenseTensor* filter_grad) {
  PADDLE_GCU_KERNEL_TRACE("depthwise_conv2d_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuConvGradKernel<T, Context>(dev_ctx,
                                  input,
                                  filter,
                                  out_grad,
                                  strides,
                                  paddings,
                                  padding_algorithm,
                                  groups,
                                  dilations,
                                  data_format,
                                  input_grad,
                                  filter_grad,
                                  "depthwise_conv2d_grad");
  }
}

template <typename T, typename Context>
void Conv3dKernel(const Context& dev_ctx,
                  const phi::DenseTensor& input,
                  const phi::DenseTensor& filter,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string& padding_algorithm,
                  int groups,
                  const std::vector<int>& dilations,
                  const std::string& data_format,
                  phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("conv3d");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuConvKernel<T, Context>(dev_ctx,
                              input,
                              filter,
                              strides,
                              paddings,
                              padding_algorithm,
                              groups,
                              dilations,
                              data_format,
                              out,
                              "conv3d");
  }
}

template <typename T, typename Context>
void Conv3dGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& input,
                      const phi::DenseTensor& filter,
                      const phi::DenseTensor& out_grad,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      const std::string& padding_algorithm,
                      int groups,
                      const std::vector<int>& dilations,
                      const std::string& data_format,
                      phi::DenseTensor* input_grad,
                      phi::DenseTensor* filter_grad) {
  PADDLE_GCU_KERNEL_TRACE("conv3d_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    GcuConvGradKernel<T, Context>(dev_ctx,
                                  input,
                                  filter,
                                  out_grad,
                                  strides,
                                  paddings,
                                  padding_algorithm,
                                  groups,
                                  dilations,
                                  data_format,
                                  input_grad,
                                  filter_grad,
                                  "conv3d_grad");
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(depthwise_conv2d,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::DepthwiseConv2dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(depthwise_conv2d_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::DepthwiseConv2dGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv3d,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv3dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv3d_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv3dGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv2d,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv2d_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv2DGradKernel,
                          float,
                          phi::dtype::float16) {}
