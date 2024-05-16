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
void YoloBoxKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& img_size,
                   const std::vector<int>& anchors,
                   int class_num,
                   float conf_thresh,
                   int downsample_ratio,
                   bool clip_bbox,
                   float scale_x_y,
                   bool iou_aware,
                   float iou_aware_factor,
                   DenseTensor* boxes,
                   DenseTensor* scores) {
  PADDLE_GCU_KERNEL_TRACE("yolo_box");
  dev_ctx.template Alloc<T>(boxes);
  dev_ctx.template Alloc<T>(scores);
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["ImgSize"] = {"img_size"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["ImgSize"] = {const_cast<DenseTensor*>(&img_size)};

    TensorNameMap output_names;
    output_names["Boxes"] = {"boxes"};
    output_names["Scores"] = {"scores"};

    TensorValueMap outputs;
    outputs["Boxes"] = {boxes};
    outputs["Scores"] = {scores};

    GcuAttributeMap attrs;
    attrs["class_num"] = class_num;
    attrs["anchors"] = anchors;
    attrs["conf_thresh"] = conf_thresh;
    attrs["downsample_ratio"] = downsample_ratio;
    attrs["clip_bbox"] = clip_bbox;
    attrs["scale_x_y"] = scale_x_y;
    attrs["iou_aware"] = iou_aware;
    attrs["iou_aware_factor"] = iou_aware_factor;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "yolo_box", dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    yolo_box, gcu, ALL_LAYOUT, custom_kernel::YoloBoxKernel, float, double) {}
