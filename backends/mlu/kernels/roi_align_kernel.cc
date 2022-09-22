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


#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"
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
    phi::DenseTensor out_tensor(*out);
    phi::DenseTensorMeta out_meta= {out->dtype(), out->dims(),phi::DataLayout::kNHWC};
    out_tensor.set_meta(out_meta);
    const auto& in_dims = x.dims();
    int batch_size = in_dims[0];
    int rois_num = boxes.dims()[0];

    if (rois_num == 0) return;
    std::vector<int> roi_batch_id_list(rois_num);
    int rois_batch_size = 0;
    if (boxes_num.is_initialized()) {
      rois_batch_size = boxes_num->numel();
      PADDLE_ENFORCE_EQ(
          rois_batch_size,
          batch_size,
          phi::errors::InvalidArgument(
              "The batch size of rois and the batch size of images "
              " must be the same. But received the batch size of rois is %d, "
              "and the batch size of images is %d",
              rois_batch_size,
              batch_size));
      std::vector<int> rois_num_list(rois_batch_size);

      MemCpyD2H(nullptr,
                rois_num_list.data(),
                boxes_num->data<int>(),
                sizeof(int) * rois_batch_size);
      
      int last_idx = 0;
      for (int i = 0; i < rois_batch_size; i++) {
        int end_idx = last_idx + rois_num_list[i];
        for (int j = last_idx; j < end_idx; j++) {
          roi_batch_id_list[j] = i;
        }
        last_idx = end_idx;
      }
    } else {
      auto lod = boxes.lod();
      PADDLE_ENFORCE_EQ(lod.empty(),
                        false,
                        phi::errors::InvalidArgument(
                            "Input(ROIs) Tensor of ROIAlignOp "
                            "does not contain LoD information."));
      auto rois_lod = lod.back();
      rois_batch_size = rois_lod.size() - 1;
      PADDLE_ENFORCE_EQ(rois_batch_size,
                        batch_size,
                        phi::errors::InvalidArgument(
                            "The rois_batch_size and imgs "
                            "batch_size must be the same. But received "
                            "rois_batch_size = %d, "
                            "batch_size = %d",
                            rois_batch_size,
                            batch_size));
      int rois_num_with_lod = rois_lod[rois_batch_size];
      PADDLE_ENFORCE_EQ(
          rois_num,
          rois_num_with_lod,
          phi::errors::InvalidArgument(
              "The actual number of rois and the number of rois "
              "provided from Input(RoIsLoD) in RoIAlign must be the same."
              " But received actual number of rois is %d, and the number "
              "of rois from RoIsLoD is %d",
              rois_num,
              rois_num_with_lod));
      for (int i = 0; i < rois_batch_size; i++) {
        int start_idx = rois_lod[i];
        int end_idx = rois_lod[i + 1];
        for (int j = start_idx; j < end_idx; j++) {
          roi_batch_id_list[j] = i;
        }
      }
    }

    // only support float32 for now
    Tensor rois_cpu;
    TensorCopy(dev_ctx, boxes, true, &rois_cpu, phi::CPUPlace());
    T* rois_cpu_ptr = dev_ctx.template HostAlloc<T>(&rois_cpu);
    
    // boxes; [batch_idx, x1, y1, x2, y2]
    Tensor boxes_cpu;
    Tensor boxes_mlu;
    boxes_cpu.Resize({rois_num, 5});
    boxes_mlu.Resize({rois_num, 5});
   
    T* boxes_cpu_ptr = dev_ctx.template HostAlloc<float>(&boxes_cpu);
    dev_ctx.template Alloc<float>(&boxes_mlu);
    for (int i = 0; i < rois_num; ++i) {
      boxes_cpu_ptr[i * 5 + 0] = static_cast<T>(roi_batch_id_list[i]);
      boxes_cpu_ptr[i * 5 + 1] = rois_cpu_ptr[i * 4 + 0];
      boxes_cpu_ptr[i * 5 + 2] = rois_cpu_ptr[i * 4 + 1];
      boxes_cpu_ptr[i * 5 + 3] = rois_cpu_ptr[i * 4 + 2];
      boxes_cpu_ptr[i * 5 + 4] = rois_cpu_ptr[i * 4 + 3];
    }

    // copy boxes_cpu to boxes_mlu
    TensorCopy(dev_ctx, boxes_cpu, true, &boxes_mlu,phi::CustomPlace());

    const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};
    const std::vector<int> perm_to_nchw = {0, 3, 1, 2};
    
    Tensor input_nhwc;
    dev_ctx.template Alloc<T>(&input_nhwc);
    
    Tensor output_nhwc;
    TransposeFromMLUTensor<T>(
        dev_ctx, perm_to_nhwc, &x, &input_nhwc, true /*need_reshape_or_alloc*/);
    auto output_dims = out_tensor.dims();
    output_nhwc.Resize({output_dims[0], output_dims[2], output_dims[3], output_dims[1]});
    
    dev_ctx.template Alloc<T>(&output_nhwc);

    MLUCnnlTensorDesc input_desc(
        input_nhwc, CNNL_LAYOUT_NHWC, ToCnnlDataType(input_nhwc.dtype()));
    MLUCnnlTensorDesc boxes_desc(boxes_mlu);
    MLUCnnlTensorDesc out_desc(
        output_nhwc, CNNL_LAYOUT_NHWC, ToCnnlDataType(output_nhwc.dtype()));
    MLUCnnl::RoiAlign(dev_ctx,
                      pooled_height,
                      pooled_width,
                      sampling_ratio,
                      spatial_scale,
                      aligned,
                      input_desc.get(),
                      GetBasePtr(&input_nhwc),
                      boxes_desc.get(),
                      GetBasePtr(&boxes_mlu),
                      out_desc.get(),
                      GetBasePtr(&output_nhwc));
    TransposeFromMLUTensor<T>(
        dev_ctx, perm_to_nchw, &output_nhwc, &out_tensor, false /*need_reshape_or_alloc*/);
  
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
    int rois_num = boxes.dims()[0];
    if (!dx) {
      return;
    }
    dev_ctx.template Alloc<T>(dx);

    std::vector<int> roi_batch_id_list(rois_num);
    int rois_batch_size = 0;
  
    if (boxes_num.is_initialized()) {
      rois_batch_size = boxes_num->numel();
      std::vector<int> rois_num_list(rois_batch_size);
      MemCpyD2H(nullptr,
              rois_num_list.data(),
              boxes_num->data<int>(),
              sizeof(int) * rois_batch_size);
      
      int last_idx = 0;
      for (int i = 0; i < rois_batch_size; i++) {
        int end_idx = last_idx + rois_num_list[i];
        for (int j = last_idx; j < end_idx; j++) {
          roi_batch_id_list[j] = i;
        }
        last_idx = end_idx;
      }
    } else {
      auto rois_lod = boxes.lod().back();
      rois_batch_size = rois_lod.size() - 1;
      for (int i = 0; i < rois_batch_size; i++) {
        int start_idx = rois_lod[i];
        int end_idx = rois_lod[i + 1];
        for (int j = start_idx; j < end_idx; j++) {
          roi_batch_id_list[j] = i;
        }
      }
    }

    Tensor rois_cpu;
    rois_cpu.Resize({rois_num, 4});
    dev_ctx.template Alloc<T>(&rois_cpu);
    TensorCopy(dev_ctx, boxes, true, &rois_cpu, phi::CPUPlace());
    T* rois_cpu_ptr = dev_ctx.template HostAlloc<T>(&rois_cpu);

    // boxes; [batch_idx, x1, y1, x2, y2]
    Tensor boxes_cpu;
    Tensor boxes_mlu;
    boxes_cpu.Resize({rois_num, 5});
    boxes_mlu.Resize({rois_num, 5});
    T* boxes_cpu_ptr = dev_ctx.template HostAlloc<T>(&boxes_cpu);
    dev_ctx.template Alloc<T>(&boxes_mlu);
    for (int i = 0; i < rois_num; ++i) {
      boxes_cpu_ptr[i * 5 + 0] = static_cast<T>(roi_batch_id_list[i]);
      boxes_cpu_ptr[i * 5 + 1] = rois_cpu_ptr[i * 4 + 0];
      boxes_cpu_ptr[i * 5 + 2] = rois_cpu_ptr[i * 4 + 1];
      boxes_cpu_ptr[i * 5 + 3] = rois_cpu_ptr[i * 4 + 2];
      boxes_cpu_ptr[i * 5 + 4] = rois_cpu_ptr[i * 4 + 3];
    }

    // copy boxes_cpu to boxes_mlu
    TensorCopy(dev_ctx, boxes_cpu, true, &boxes_mlu,phi::CustomPlace());

    const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};
    const std::vector<int> perm_to_nchw = {0, 3, 1, 2};

    Tensor grads_nhwc;
    dev_ctx.template Alloc<T>(&grads_nhwc);

    Tensor grads_image_nhwc;
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nhwc,
                              &out_grad,
                              &grads_nhwc,
                              true /*need_reshape_or_alloc*/);
    auto grads_image_dims = dx->dims();
    grads_image_nhwc.Resize({grads_image_dims[0],
                            grads_image_dims[2],
                            grads_image_dims[3],
                            grads_image_dims[1]});
    
    dev_ctx.template Alloc<T>(&grads_image_nhwc);

    MLUCnnlTensorDesc grads_desc(
        grads_nhwc, CNNL_LAYOUT_NHWC, ToCnnlDataType(grads_nhwc.dtype()));
    MLUCnnlTensorDesc boxes_desc(boxes_mlu);
    MLUCnnlTensorDesc grads_image_desc(
        grads_image_nhwc,
        CNNL_LAYOUT_NHWC,
        ToCnnlDataType(grads_image_nhwc.dtype()));
    MLUCnnl::RoiAlignBackward(dev_ctx,
                              sampling_ratio,
                              spatial_scale,
                              aligned,
                              grads_desc.get(),
                              GetBasePtr(&grads_nhwc),
                              boxes_desc.get(),
                              GetBasePtr(&boxes_mlu),
                              grads_image_desc.get(),
                              GetBasePtr(&grads_image_nhwc));
    
    TransposeFromMLUTensor<T>(dev_ctx,
                              perm_to_nchw,
                              &grads_image_nhwc,
                              dx,
                              false /*need_reshape_or_alloc*/);
  
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(roi_align,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::RoiAlignKernel,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(roi_align_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::RoiAlignGradKernel,
                          float) {}
