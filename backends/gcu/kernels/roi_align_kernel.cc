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
void RoiAlignKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& boxes,
                    const paddle::optional<phi::DenseTensor>& boxes_num,
                    int pooled_height,
                    int pooled_width,
                    float spatial_scale,
                    int sampling_ratio,
                    bool aligned,
                    phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("roi_align");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    //   PADDLE_ENFORCE_EQ(
    //       aligned, false,
    //       phi::errors::InvalidArgument(
    //           "GCU only support Aligned attribute equaled to False"));
    dev_ctx.template Alloc<T>(out);

    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["ROIs"] = {"rois"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["ROIs"] = {const_cast<DenseTensor*>(&boxes)};

    if (boxes_num) {
      input_names["RoisNum"] = {"rois_num"};
      inputs["RoisNum"] = {const_cast<DenseTensor*>(boxes_num.get_ptr())};
    }

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["pooled_height"] = pooled_height;
    attrs["pooled_width"] = pooled_width;
    attrs["sampling_ratio"] = sampling_ratio;
    attrs["spatial_scale"] = spatial_scale;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "roi_align",
              dev_ctx);
  }
}

template <typename T, typename Context>
void RoiAlignGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& boxes,
                        const paddle::optional<phi::DenseTensor>& boxes_num,
                        const phi::DenseTensor& out_grad,
                        int pooled_height,
                        int pooled_width,
                        float spatial_scale,
                        int sampling_ratio,
                        bool aligned,
                        phi::DenseTensor* dx) {
  PADDLE_GCU_KERNEL_TRACE("roi_align_grad");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    //   PADDLE_ENFORCE_EQ(
    //     aligned, false,
    //     phi::errors::InvalidArgument(
    //         "GCU only support Aligned attribute equaled to False"));
    if (!dx) {
      return;
    }
    dev_ctx.template Alloc<T>(dx);

    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["ROIs"] = {"rois"};
    input_names[GradVarName("Out")] = {"out_grad"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["ROIs"] = {const_cast<DenseTensor*>(&boxes)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};

    if (boxes_num) {
      input_names["RoisNum"] = {"rois_num"};
      inputs["RoisNum"] = {const_cast<DenseTensor*>(boxes_num.get_ptr())};
    }

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"dx"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {dx};

    GcuAttributeMap attrs;
    attrs["pooled_height"] = pooled_height;
    attrs["pooled_width"] = pooled_width;
    attrs["sampling_ratio"] = sampling_ratio;
    attrs["spatial_scale"] = spatial_scale;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "roi_align_grad",
              dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    roi_align, gcu, ALL_LAYOUT, custom_kernel::RoiAlignKernel, float, double) {}

PD_REGISTER_PLUGIN_KERNEL(roi_align_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::RoiAlignGradKernel,
                          float,
                          double) {}
