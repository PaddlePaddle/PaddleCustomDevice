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
  dev_ctx.template Alloc<T>(out);

  auto roi_end_mode = 0;
  PADDLE_ENFORCE_EQ(
      aligned,
      false,
      phi::errors::InvalidArgument(
          "ROIAlignNPU only support Aligned attribute equaled to False"));

  NPUAttributeMap attr_boxes = {{"spatial_scale", spatial_scale},
                                {"pooled_height", pooled_height},
                                {"pooled_width", pooled_width},
                                {"sample_num", sampling_ratio},
                                {"roi_end_mode", roi_end_mode}};

  auto stream = dev_ctx.stream();

  // Combine boxes_num with boxes to get new boxes
  // change boxes_num's datatype & resize
  int dtype = static_cast<int>(
      ConvertToNpuDtype(phi::DenseTensorMeta::DataType::FLOAT32));
  NPUAttributeMap attr_cast = {{"dst_type", dtype}};
  phi::DenseTensor boxes_num_fp;
  phi::DenseTensorMeta boxes_num_fp_meta = {
      boxes.dtype(), phi::make_ddim({boxes.dims()[0], 1})};
  boxes_num_fp.set_meta(boxes_num_fp_meta);
  dev_ctx.template Alloc<T>(&boxes_num_fp);

  const auto& runner_c =
      NpuOpRunner("Cast", {*boxes_num}, {boxes_num_fp}, attr_cast);
  runner_c.Run(stream);

  // concate to make (N, 5)
  std::vector<phi::DenseTensor> x_list;
  x_list.push_back(boxes_num_fp);
  x_list.push_back(boxes);
  auto axis = 1;
  // output of concate
  phi::DenseTensor boxes_N5;
  phi::DenseTensorMeta boxes_N5_meta = {boxes.dtype(),
                                        phi::make_ddim({boxes.dims()[0], 5})};
  boxes_N5.set_meta(boxes_N5_meta);
  dev_ctx.template Alloc<T>(&boxes_N5);

  // attribute of concate
  auto EleNum = 2;
  NPUAttributeMap attr_concat = {{"N", EleNum}, {"concat_dim", axis}};

  NpuOpRunner runner0;
  runner0.SetType("ConcatD")
      .AddInputs(x_list)
      .AddOutput(boxes_N5)
      .AddInputNames({"x0", "x1"})
      .AddAttrs(attr_concat);
  runner0.Run(stream);

  const auto& runner =
      NpuOpRunner("ROIAlign", {x, boxes_N5}, {*out}, attr_boxes);
  runner.Run(stream);
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
  auto in_dims = x.dims();

  int rois_num = boxes.dims()[0];

  auto stream = dev_ctx.stream();

  if (!dx) {
    return;
  }
  dev_ctx.template Alloc<T>(dx);

  PADDLE_ENFORCE_EQ(
      aligned,
      false,
      phi::errors::InvalidArgument(
          "ROIAlignGradNPU only support Aligned attribute equaled to False"));

  PADDLE_ENFORCE_EQ(
      boxes.dtype(),
      phi::DenseTensorMeta::DataType::FLOAT32,
      phi::errors::InvalidArgument(
          "ROIAlignGradNPU only support ROIs type equaled to FP32."));

  // Cast boxes_num to fp32 tensor
  phi::DenseTensor boxes_N5;
  boxes_N5.Resize({rois_num, 5});
  dev_ctx.template Alloc<T>(&boxes_N5);
  phi::DenseTensor boxes_num_fp;
  boxes_num_fp.Resize(boxes_num->dims());
  dev_ctx.template Alloc<T>(&boxes_num_fp);

  int nputype_fp32 = static_cast<int>(
      ConvertToNpuDtype(phi::DenseTensorMeta::DataType::FLOAT32));
  const auto& runner_cast = NpuOpRunner(
      "Cast", {*boxes_num}, {boxes_num_fp}, {{"dst_type", nputype_fp32}});
  runner_cast.Run(stream);
  boxes_num_fp.Resize({rois_num, 1});

  // Combine *ROIsNum with ROIs to get new ROIs
  std::vector<phi::DenseTensor> x_list;
  x_list.push_back(boxes_num_fp);
  x_list.push_back(boxes);
  const auto& runner_concat = NpuOpRunner(
      "ConcatD", {x_list}, {boxes_N5}, {{"N", 2}, {"concat_dim", 1}});
  runner_concat.Run(stream);

  //  If CANN version code is less than 504, by analysis, in order to match cpu
  //  grad version, rois[:,3:5] should substrate 1 before call ascend grad
  //  function
#if (CANN_VERSION_CODE < 504000)
  std::vector<float> vec_dlt = {0, 0, 0, -1.0f, -1.0f};
  phi::DenseTensor tsr_dlt;
  tsr_dlt.Resize({5});
  dev_ctx.template Alloc<float>(&tsr_dlt);
  TensorFromVector<float>(dev_ctx, vec_dlt, dev_ctx, &tsr_dlt);
  dev_ctx.Wait();
  const auto& runner_add =
      NpuOpRunner("AddV2", {boxes_N5, tsr_dlt}, {boxes_N5}, {});
  runner_add.Run(stream);
#endif

  //  Call ascend RoiAlignGrad function
  int roi_end_mode = 0;
  const auto& runner_roi_align_grad =
      NpuOpRunner("ROIAlignGrad",
                  {out_grad, boxes_N5},
                  {*dx},
                  {{"xdiff_shape", phi::vectorize<int>(in_dims)},
                   {"pooled_width", pooled_width},
                   {"pooled_height", pooled_height},
                   {"spatial_scale", spatial_scale},
                   {"sample_num", sampling_ratio},
                   {"roi_end_mode", roi_end_mode}});
  runner_roi_align_grad.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(roi_align,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::RoiAlignKernel,
                          float,
                          double,
                          int) {}

PD_REGISTER_PLUGIN_KERNEL(roi_align_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::RoiAlignGradKernel,
                          float,
                          double,
                          int) {}
