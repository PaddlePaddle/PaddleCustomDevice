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

#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {
template <typename T, typename Context>
void InterpolateKernel(
    const std::string& Interpolated_type,
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  ctx.template Alloc<T>(output);

  TensorNameMap input_names;
  input_names["X"] = {"x"};

  TensorValueMap inputs;
  inputs["X"] = {const_cast<DenseTensor*>(&x)};

  if (out_size) {
    input_names["OutSize"] = {"out_size"};
    inputs["OutSize"] = {const_cast<DenseTensor*>(out_size.get_ptr())};
  }
  if (scale_tensor) {
    input_names["Scale"] = {"scale"};
    inputs["Scale"] = {const_cast<DenseTensor*>(scale_tensor.get_ptr())};
  }
  if (size_tensor) {
    auto tensors = size_tensor.get();
    std::vector<std::string> in_names;
    in_names.reserve(tensors.size());
    std::vector<phi::DenseTensor*> in_tensors;
    in_tensors.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
      in_names.emplace_back(std::string("size_tensor_") + std::to_string(i));
      in_tensors.emplace_back(const_cast<DenseTensor*>(tensors[i]));
    }
    input_names["SizeTensor"] = in_names;
    inputs["SizeTensor"] = in_tensors;
  }

  TensorNameMap output_names;
  output_names["Output"] = {"output"};

  TensorValueMap outputs;
  outputs["Output"] = {output};

  GcuAttributeMap attrs;
  attrs["data_layout"] = data_layout;
  attrs["out_d"] = out_d;
  attrs["out_h"] = out_h;
  attrs["out_w"] = out_w;
  attrs["scale"] = scale;
  attrs["interp_method"] = interp_method;
  attrs["align_corners"] = align_corners;
  attrs["align_mode"] = align_mode;

  GcuRunner(input_names,
            inputs,
            output_names,
            outputs,
            attrs,
            Interpolated_type,
            ctx);
}

template <typename T, typename Context>
void InterpolateGradKernel(
    const std::string& Interpolated_grad_type,
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* x_grad) {
  ctx.template Alloc<T>(x_grad);

  TensorNameMap input_names;
  input_names["X"] = {"x"};
  input_names[GradVarName("Out")] = {"out_grad"};

  TensorValueMap inputs;
  inputs["X"] = {const_cast<DenseTensor*>(&x)};
  inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};

  if (out_size) {
    input_names["OutSize"] = {"out_size"};
    inputs["OutSize"] = {const_cast<DenseTensor*>(out_size.get_ptr())};
  }
  if (scale_tensor) {
    input_names["Scale"] = {"scale"};
    inputs["Scale"] = {const_cast<DenseTensor*>(scale_tensor.get_ptr())};
  }
  if (size_tensor) {
    auto tensors = size_tensor.get();
    std::vector<std::string> in_names;
    in_names.reserve(tensors.size());
    std::vector<phi::DenseTensor*> in_tensors;
    in_tensors.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
      in_names.emplace_back(std::string("size_tensor_") + std::to_string(i));
      in_tensors.emplace_back(const_cast<DenseTensor*>(tensors[i]));
    }
    input_names["SizeTensor"] = in_names;
    inputs["SizeTensor"] = in_tensors;
  }

  TensorNameMap output_names;
  output_names[GradVarName("X")] = {"x_grad"};

  TensorValueMap outputs;
  outputs[GradVarName("X")] = {x_grad};

  GcuAttributeMap attrs;
  attrs["data_layout"] = data_layout;
  attrs["out_d"] = out_d;
  attrs["out_h"] = out_h;
  attrs["out_w"] = out_w;
  attrs["scale"] = scale;
  attrs["interp_method"] = interp_method;
  attrs["align_corners"] = align_corners;
  attrs["align_mode"] = align_mode;

  GcuRunner(input_names,
            inputs,
            output_names,
            outputs,
            attrs,
            Interpolated_grad_type,
            ctx);
}

template <typename T, typename Context>
void BilinearInterpKernel(
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  InterpolateKernel<T, Context>("bilinear_interp_v2",
                                ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale,
                                interp_method,
                                align_corners,
                                align_mode,
                                output);
}

template <typename T, typename Context>
void BilinearInterpGradKernel(
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* x_grad) {
  InterpolateGradKernel<T, Context>("bilinear_interp_v2_grad",
                                    ctx,
                                    x,
                                    out_size,
                                    size_tensor,
                                    scale_tensor,
                                    out_grad,
                                    data_layout,
                                    out_d,
                                    out_h,
                                    out_w,
                                    scale,
                                    interp_method,
                                    align_corners,
                                    align_mode,
                                    x_grad);
}

template <typename T, typename Context>
void NearestInterpKernel(
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* output) {
  InterpolateKernel<T, Context>("nearest_interp_v2",
                                ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale,
                                interp_method,
                                align_corners,
                                align_mode,
                                output);
}

template <typename T, typename Context>
void NearestInterpGradKernel(
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    DenseTensor* x_grad) {
  InterpolateGradKernel<T, Context>("nearest_interp_v2_grad",
                                    ctx,
                                    x,
                                    out_size,
                                    size_tensor,
                                    scale_tensor,
                                    out_grad,
                                    data_layout,
                                    out_d,
                                    out_h,
                                    out_w,
                                    scale,
                                    interp_method,
                                    align_corners,
                                    align_mode,
                                    x_grad);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(bilinear_interp,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::BilinearInterpKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(bilinear_interp_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::BilinearInterpGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(nearest_interp,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::NearestInterpKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(nearest_interp_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::NearestInterpGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}
