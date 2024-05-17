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
void ConvTransposeRawKernel(const std::string& conv_type,
                            const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& filter,
                            const std::vector<int>& strides,
                            const std::vector<int>& paddings,
                            const std::vector<int>& output_padding,
                            const phi::IntArray& output_size,
                            const std::string& padding_algorithm,
                            int groups,
                            const std::vector<int>& dilations,
                            const std::string& data_format,
                            DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  TensorNameMap input_names;
  input_names["Input"] = {"input"};
  input_names["Filter"] = {"filter"};

  TensorValueMap inputs;
  inputs["Input"] = {const_cast<DenseTensor*>(&x)};
  inputs["Filter"] = {const_cast<DenseTensor*>(&filter)};

  TensorNameMap output_names;
  output_names["Out"] = {"out"};

  TensorValueMap outputs;
  outputs["Out"] = {out};

  std::vector<int> output_size_list = GetIntList(output_size.GetData());

  GcuAttributeMap attrs;
  attrs["strides"] = strides;
  attrs["paddings"] = paddings;
  attrs["output_padding"] = output_padding;
  attrs["output_size"] = output_size_list;
  attrs["padding_algorithm"] = padding_algorithm;
  attrs["groups"] = groups;
  attrs["dilations"] = dilations;
  attrs["data_format"] = data_format;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, conv_type, dev_ctx);
}

template <typename T, typename Context>
void ConvTransposeGradRawKernel(const std::string& conv_grad_type,
                                const Context& dev_ctx,
                                const DenseTensor& x,
                                const DenseTensor& filter,
                                const DenseTensor& dout,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                const std::vector<int>& output_padding,
                                const phi::IntArray& output_size,
                                const std::string& padding_algorithm,
                                int groups,
                                const std::vector<int>& dilations,
                                const std::string& data_format,
                                DenseTensor* dx,
                                DenseTensor* dfilter) {
  dev_ctx.template Alloc<T>(dx);
  dev_ctx.template Alloc<T>(dfilter);

  TensorNameMap input_names;
  input_names["Input"] = {"input"};
  input_names["Filter"] = {"filter"};
  input_names[GradVarName("Output")] = {"dout"};

  TensorValueMap inputs;
  inputs["Input"] = {const_cast<DenseTensor*>(&x)};
  inputs["Filter"] = {const_cast<DenseTensor*>(&filter)};
  inputs[GradVarName("Output")] = {const_cast<DenseTensor*>(&dout)};

  TensorNameMap output_names;
  output_names[GradVarName("Input")] = {"dx"};
  output_names[GradVarName("Filter")] = {"dfilter"};

  TensorValueMap outputs;
  outputs[GradVarName("Input")] = {dx};
  outputs[GradVarName("Filter")] = {dfilter};

  std::vector<int> output_size_list = GetIntList(output_size.GetData());

  GcuAttributeMap attrs;
  attrs["strides"] = strides;
  attrs["paddings"] = paddings;
  attrs["output_padding"] = output_padding;
  attrs["output_size"] = output_size_list;
  attrs["padding_algorithm"] = padding_algorithm;
  attrs["groups"] = groups;
  attrs["dilations"] = dilations;
  attrs["data_format"] = data_format;

  GcuRunner(input_names,
            inputs,
            output_names,
            outputs,
            attrs,
            conv_grad_type,
            dev_ctx);
}

template <typename T, typename Context>
void Conv2dTransposeKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& output_padding,
                           const phi::IntArray& output_size,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format,
                           DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("conv2d_transpose");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ConvTransposeRawKernel<T, Context>("conv2d_transpose",
                                       dev_ctx,
                                       x,
                                       filter,
                                       strides,
                                       paddings,
                                       output_padding,
                                       output_size,
                                       padding_algorithm,
                                       groups,
                                       dilations,
                                       data_format,
                                       out);
  }
}

template <typename T, typename Context>
void Conv2dTransposeGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& filter,
                               const DenseTensor& dout,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::vector<int>& output_padding,
                               const phi::IntArray& output_size,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format,
                               DenseTensor* dx,
                               DenseTensor* dfilter) {
  PADDLE_GCU_KERNEL_TRACE("conv2d_transpose_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ConvTransposeGradRawKernel<T, Context>("conv2d_transpose_grad",
                                           dev_ctx,
                                           x,
                                           filter,
                                           dout,
                                           strides,
                                           paddings,
                                           output_padding,
                                           output_size,
                                           padding_algorithm,
                                           groups,
                                           dilations,
                                           data_format,
                                           dx,
                                           dfilter);
  }
}

template <typename T, typename Context>
void Conv3dTransposeKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& output_padding,
                           const std::vector<int>& output_size,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format,
                           DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("conv3d_transpose");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ConvTransposeRawKernel<T, Context>("conv3d_transpose",
                                       dev_ctx,
                                       x,
                                       filter,
                                       strides,
                                       paddings,
                                       output_padding,
                                       output_size,
                                       padding_algorithm,
                                       groups,
                                       dilations,
                                       data_format,
                                       out);
  }
}

template <typename T, typename Context>
void Conv3dTransposeGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& filter,
                               const DenseTensor& dout,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::vector<int>& output_padding,
                               const std::vector<int>& output_size,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format,
                               DenseTensor* dx,
                               DenseTensor* dfilter) {
  PADDLE_GCU_KERNEL_TRACE("conv3d_transpose_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    ConvTransposeGradRawKernel<T, Context>("conv3d_transpose_grad",
                                           dev_ctx,
                                           x,
                                           filter,
                                           dout,
                                           strides,
                                           paddings,
                                           output_padding,
                                           output_size,
                                           padding_algorithm,
                                           groups,
                                           dilations,
                                           data_format,
                                           dx,
                                           dfilter);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(conv2d_transpose,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dTransposeKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv2d_transpose_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dTransposeGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv3d_transpose,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv3dTransposeKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv3d_transpose_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv3dTransposeGradKernel,
                          float,
                          phi::dtype::float16) {}
