/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {
template <typename T, typename Context>
void YoloBoxKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& img_size,
                   const std::vector<int>& anchors,
                   int class_num,
                   float conf_thresh,
                   int downsample_ratio,
                   bool clip_bbox,
                   float scale_x_y,
                   bool iou_aware,
                   float iou_aware_factor,
                   phi::DenseTensor* boxes,
                   phi::DenseTensor* scores){
    int anchor_num = anchors.size() / 2;
    int64_t size = anchors.size();
    auto dim_x = x.dims();
    int n = dim_x[0];
    int s = anchor_num;
    int h = dim_x[2];
    int w = dim_x[3];

    // The output of mluOpYoloBox: A 4-D tensor with shape [N, anchor_num, 4,
    // H*W], the coordinates of boxes, and a 4-D tensor with shape [N,
    // anchor_num, :attr:`class_num`, H*W], the classification scores of boxes.
    std::vector<int64_t> boxes_dim_mluops({n, s, 4, h * w});
    std::vector<int64_t> scores_dim_mluops({n, s, class_num, h * w});

    // In Paddle framework: A 3-D tensor with shape [N, M, 4], the coordinates
    // of boxes, and a 3-D tensor with shape [N, M, :attr:`class_num`], the
    // classification scores of boxes.
    std::vector<int64_t> boxes_out_dim({n, s, h * w, 4});
    std::vector<int64_t> scores_out_dim({n, s, h * w, class_num});

    MLUOpTensorDesc boxes_trans_desc_mluops(
        4, boxes_dim_mluops.data(), ToMluOpDataType<T>());
    MLUCnnlTensorDesc boxes_trans_desc_cnnl(
        4, boxes_dim_mluops.data(), ToCnnlDataType<T>());
    MLUOpTensorDesc scores_trans_desc_mluops(
        4, scores_dim_mluops.data(), ToMluOpDataType<T>());
    MLUCnnlTensorDesc scores_trans_desc_cnnl(
        4, scores_dim_mluops.data(), ToCnnlDataType<T>());

    dev_ctx.template Alloc<T>(boxes);
    dev_ctx.template Alloc<T>(scores);

    MLUOpTensorDesc x_desc(x, MLUOP_LAYOUT_ARRAY, ToMluOpDataType<T>());
    MLUOpTensorDesc img_size_desc(
        img_size, MLUOP_LAYOUT_ARRAY, ToMluOpDataType<int32_t>());
    Tensor anchors_temp;
    anchors_temp.Resize({size});
    custom_kernel::TensorFromVector(dev_ctx, anchors, dev_ctx, &anchors_temp);
    MLUOpTensorDesc anchors_desc(anchors_temp);
    MLUCnnlTensorDesc boxes_desc_cnnl(
        4, boxes_out_dim.data(), ToCnnlDataType<T>());
    MLUCnnlTensorDesc scores_desc_cnnl(
        4, scores_out_dim.data(), ToCnnlDataType<T>());


    MLUOP::OpYoloBox(dev_ctx,
                     x_desc.get(),
                     GetBasePtr(&x),
                     img_size_desc.get(),
                     GetBasePtr(&img_size),
                     anchors_desc.get(),
                     GetBasePtr(&anchors_temp),
                     class_num,
                     conf_thresh,
                     downsample_ratio,
                     clip_bbox,
                     scale_x_y,
                     iou_aware,
                     iou_aware_factor,
                     boxes_trans_desc_mluops.get(),
                     GetBasePtr(boxes),
                     scores_trans_desc_mluops.get(),
                     GetBasePtr(scores));
    const std::vector<int> perm = {0, 1, 3, 2};

    // transpose the boxes from [N, S, 4, H*W] to [N, S, H*W, 4]
    MLUCnnl::Transpose(dev_ctx,
                       perm,
                       4,
                       boxes_trans_desc_cnnl.get(),
                       GetBasePtr(boxes),
                       boxes_desc_cnnl.get(),
                       GetBasePtr(boxes));

    // transpose the scores from [N, S, class_num, H*W] to [N, S, H*W,
    // class_num]
    MLUCnnl::Transpose(dev_ctx,
                       perm,
                       4,
                       scores_trans_desc_cnnl.get(),
                       GetBasePtr(scores),
                       scores_desc_cnnl.get(),
                       GetBasePtr(scores));


}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(yolo_box,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::YoloBoxKernel,
                          float) {}