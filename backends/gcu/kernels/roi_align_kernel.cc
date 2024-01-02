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

#include "common/common.h"
#include "common/utils.h"
#include "kernels/common_ops/common_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_name_list.h"
#include "kernels/funcs/gcu_op_runner.h"

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
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "roi_align", roi_align);
    PADDLE_ENFORCE_EQ(
        boxes.dims().at(1),
        4,
        phi::errors::NotFound("boxes must have shape as Tensor[K, 4]"));

    auto boxes_batch = boxes.dims().at(0);
    auto channels = x.dims().at(1);
    auto input_batch = x.dims().at(0);

    int boxes_batch_size;
    std::vector<int32_t> roi_batch_id_data((boxes.dims()[0]));
    int batch_size = x.dims()[0];
    if (boxes_num) {
      boxes_batch_size = boxes_num->numel();
      PADDLE_ENFORCE_EQ(
          boxes_batch_size,
          batch_size,
          phi::errors::InvalidArgument(
              "The batch size of rois and the batch size of images "
              " must be the same. But received the batch size of rois is %d, "
              "and the batch size of images is %d",
              boxes_batch_size,
              batch_size));

      std::vector<int> boxes_num_data;
      TensorToVector(dev_ctx, *boxes_num, dev_ctx, &boxes_num_data);
      int start = 0;
      // transfrom boxes_num to roi_batch_id_data
      for (int n = 0; n < boxes_batch_size; ++n) {
        for (int i = start; i < start + boxes_num_data[n]; ++i) {
          roi_batch_id_data[i] = static_cast<float>(n);
        }
        start += boxes_num_data[n];
      }
    } else {
      auto lod = boxes.lod();
      PADDLE_ENFORCE_EQ(
          lod.empty(),
          false,
          phi::errors::InvalidArgument("Input(ROIs) Tensor of ROIAlignOp "
                                       "does not contain LoD information."));
      auto boxes_lod = lod.back();
      boxes_batch_size = boxes_lod.size() - 1;
      PADDLE_ENFORCE_EQ(
          boxes_batch_size,
          batch_size,
          phi::errors::InvalidArgument("The boxes_batch_size and imgs "
                                       "batch_size must be the same. But "
                                       "received boxes_batch_size = %d, "
                                       "batch_size = %d",
                                       boxes_batch_size,
                                       batch_size));
      int boxes_num_with_lod = boxes_lod[boxes_batch_size];
      PADDLE_ENFORCE_EQ(
          boxes.dims()[0],
          boxes_num_with_lod,
          phi::errors::InvalidArgument(
              "The actual number of rois and the number of rois "
              "provided from Input(RoIsLoD) in RoIAlign must be the same."
              " But received actual number of rois is %d, and the number "
              "of rois from RoIsLoD is %d",
              boxes.dims()[0],
              boxes_num_with_lod));
      for (int n = 0; n < boxes_batch_size; ++n) {
        for (std::size_t i = boxes_lod[n]; i < boxes_lod[n + 1]; ++i) {
          roi_batch_id_data[i] = n;
        }
      }
    }
    phi::DenseTensor batch_indices;
    TensorFromVector<int32_t>(
        dev_ctx, roi_batch_id_data, dev_ctx, &batch_indices);

    auto tmp_x = x;
    if (x.dtype() == phi::DataType::FLOAT64) {
      tmp_x = cast(dev_ctx, x, phi::DataType::FLOAT32);
    }
    auto tmp_boxes = boxes;
    if (boxes.dtype() == phi::DataType::FLOAT64) {
      tmp_boxes = cast(dev_ctx, boxes, phi::DataType::FLOAT32);
    }
    auto x_gcu = GetHlirTensor(tmp_x);
    auto boxes_gcu = GetHlirTensor(tmp_boxes);
    auto batch_indices_gcu = GetHlirTensor(batch_indices);

    std::vector<int64_t> output_shape;
    output_shape = {boxes_batch, channels, pooled_height, pooled_width};
    out->Resize(phi::make_ddim(output_shape));
    dev_ctx.template Alloc<T>(out);

    if (out->numel() > 0) {
      phi::DenseTensor tmp_out = *out;
      if (tmp_out.dtype() == phi::DataType::FLOAT64) {
        auto tmp = EmptyTensor(dev_ctx, phi::DataType::FLOAT32, tmp_out.dims());
        dev_ctx.template Alloc(&tmp, tmp.dtype());
        tmp_out = tmp;
      }
      auto out_gcu = GetHlirTensor(tmp_out);

      hlir::DispatchParam params;
      params.inputs = {x_gcu, boxes_gcu, batch_indices_gcu};
      params.outputs = {out_gcu};
      params.metadata.setValue("output_height",
                               static_cast<int64_t>(pooled_height));
      params.metadata.setValue("output_width",
                               static_cast<int64_t>(pooled_width));
      params.metadata.setValue(
          "sampling_ratio",
          static_cast<int64_t>((sampling_ratio >= 0 ? sampling_ratio : 0)));
      params.metadata.setValue("spatial_scale", spatial_scale);
      params.metadata.setValue("mode", static_cast<int64_t>(0));
      params.metadata.setValue(
          "transformation_mode",
          aligned ? static_cast<int64_t>(0) : static_cast<int64_t>(1));
      params.stream = static_cast<topsStream_t>(dev_ctx.stream());
      AOTOPS_DEBUG(kRoiAlign, params);

      GCUOPS_TRACE_START(roi_align);
      auto func_ptr = GetOpFuncPtr(kRoiAlign, params);
      if (func_ptr) {
        auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
        PADDLE_ENFORCE(
            pass,
            phi::errors::InvalidArgument("dispatch %s failed!", kRoiAlign));
      } else {
        PADDLE_ENFORCE(false,
                       phi::errors::InvalidArgument("not find aot func for %s",
                                                    kRoiAlign));
      }
      FreeDispatchParam(params);
      GcuOpStreamSync(dev_ctx);
      GCUOPS_TRACE_END(roi_align);
      if (out->dtype() == phi::DataType::FLOAT64) {
        cast(dev_ctx, tmp_out, phi::DataType::FLOAT64, out);
      }

      PADDLE_GCU_KERNEL_END("roi_align", roi_align);
    }
  } else {
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
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "roi_align_grad", roi_align_grad);
    dx->Resize(x.dims());
    dev_ctx.template Alloc<T>(dx);

    int boxes_batch_size;
    int batch_size = x.dims()[0];
    std::vector<int32_t> box_batch_id_data((boxes.dims()[0]));

    if (boxes_num) {
      boxes_batch_size = boxes_num->numel();
      std::vector<int> boxes_num_data;
      TensorToVector(dev_ctx, *boxes_num, dev_ctx, &boxes_num_data);
      int start = 0;
      // transfrom boxes_num to box_batch_id_data
      for (int n = 0; n < boxes_batch_size; ++n) {
        for (int i = start; i < start + boxes_num_data[n]; ++i) {
          box_batch_id_data[i] = static_cast<int32_t>(n);
        }
        start += boxes_num_data[n];
      }
    } else {
      auto boxes_lod = boxes.lod().back();
      boxes_batch_size = boxes_lod.size() - 1;
      for (int n = 0; n < boxes_batch_size; ++n) {
        for (std::size_t i = boxes_lod[n]; i < boxes_lod[n + 1]; ++i) {
          box_batch_id_data[i] = n;
        }
      }
    }
    phi::DenseTensor batch_indices;
    TensorFromVector<int32_t>(
        dev_ctx, box_batch_id_data, dev_ctx, &batch_indices);

    // handle possibly empty gradients
    if (dx->numel() > 0 && out_grad.numel() > 0) {
      auto grad_gcu = GetHlirTensor(out_grad);
      auto boxes_gcu = GetHlirTensor(boxes);
      auto batch_indices_gcu = GetHlirTensor(batch_indices);
      auto dx_gcu = GetHlirTensor(*dx);

      hlir::DispatchParam params;
      params.inputs = {grad_gcu, boxes_gcu, batch_indices_gcu};
      params.outputs = {dx_gcu};
      params.metadata.setValue("spatial_scale", spatial_scale);
      params.metadata.setValue("pooled_height",
                               static_cast<int64_t>(pooled_height));
      params.metadata.setValue("pooled_width",
                               static_cast<int64_t>(pooled_width));
      params.metadata.setValue("batch_size", x.dims().at(0));
      params.metadata.setValue("channels", x.dims().at(1));
      params.metadata.setValue("height", x.dims().at(2));
      params.metadata.setValue("width", x.dims().at(3));
      params.metadata.setValue(
          "sampling_ratio",
          static_cast<int64_t>((sampling_ratio >= 0 ? sampling_ratio : 0)));
      params.metadata.setValue("aligned", aligned);
      params.stream = static_cast<topsStream_t>(dev_ctx.stream());
      AOTOPS_DEBUG(kRoiAlignBackward, params);

      GCUOPS_TRACE_START(roi_align_backward);
      auto func_ptr = GetOpFuncPtr(kRoiAlignBackward, params);
      if (func_ptr) {
        auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
        PADDLE_ENFORCE(pass,
                       phi::errors::InvalidArgument("dispatch %s failed!",
                                                    kRoiAlignBackward));
      } else {
        PADDLE_ENFORCE(false,
                       phi::errors::InvalidArgument("not find aot func for %s",
                                                    kRoiAlignBackward));
      }
      FreeDispatchParam(params);
      GCUOPS_TRACE_END(roi_align_backward);
      GcuOpStreamSync(dev_ctx);
    }
    PADDLE_GCU_KERNEL_END("roi_align_grad", roi_align_grad);
  } else {
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
