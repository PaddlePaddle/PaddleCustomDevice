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

#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void GenerateProposalsKernel(const Context& dev_ctx,
                             const phi::DenseTensor& scores,
                             const phi::DenseTensor& bbox_deltas,
                             const phi::DenseTensor& im_shape,
                             const phi::DenseTensor& anchors,
                             const phi::DenseTensor& variances,
                             int pre_nms_top_n,
                             int post_nms_top_n,
                             float nms_thresh,
                             float min_size,
                             float eta,
                             bool pixel_offset,
                             phi::DenseTensor* rpn_rois,
                             phi::DenseTensor* rpn_roi_probs,
                             phi::DenseTensor* rpn_rois_num) {
  PADDLE_ENFORCE_GE(eta,
                    1.,
                    phi::errors::InvalidArgument(
                        "Not support adaptive NMS. The attribute 'eta' "
                        "should not less than 1. But received eta=[%d]",
                        eta));

  auto scores_dim = scores.dims();
  int64_t num = scores_dim[0];
  int64_t c_score = scores_dim[1];
  int64_t h_score = scores_dim[2];
  int64_t w_score = scores_dim[3];

  rpn_rois->Resize({num * post_nms_top_n, 4});
  dev_ctx.template Alloc<T>(rpn_rois);
  rpn_roi_probs->Resize({num * post_nms_top_n, 1});
  dev_ctx.template Alloc<T>(rpn_roi_probs);
  Tensor rpn_roi_num_tmp;
  if (rpn_rois_num) {
    rpn_rois_num->Resize({num});
    dev_ctx.template Alloc<int>(rpn_rois_num);
    rpn_roi_num_tmp = *rpn_rois_num;
  } else {
    rpn_roi_num_tmp.Resize({num});
    dev_ctx.template Alloc<int>(&rpn_roi_num_tmp);
  }

  Tensor scores_swap, bbox_deltas_swap;
  const std::vector<int> perm = {0, 2, 3, 1};
  TransposeFromMLUTensor<T>(
      dev_ctx, perm, &scores, &scores_swap, true /*need_reshape_or_alloc*/);
  TransposeFromMLUTensor<T>(dev_ctx,
                            perm,
                            &bbox_deltas,
                            &bbox_deltas_swap,
                            true /*need_reshape_or_alloc*/);
  MLUOpTensorDesc scores_desc(scores_swap);
  MLUOpTensorDesc bbox_deltas_desc(bbox_deltas_swap);
  MLUOpTensorDesc im_shape_desc(im_shape);
  const int64_t anchor_var_shape_4D[4] = {h_score, w_score, c_score, 4};
  MLUOpTensorDesc anchors_desc(4, anchor_var_shape_4D, ToMluOpDataType<T>());
  MLUOpTensorDesc variances_desc(4, anchor_var_shape_4D, ToMluOpDataType<T>());
  MLUOpTensorDesc rpn_rois_desc(*rpn_rois);
  MLUOpTensorDesc rpn_rois_probs_desc(*rpn_roi_probs);
  MLUOpTensorDesc rpn_rois_num_desc(rpn_roi_num_tmp);

  Tensor rpn_rois_batch_size;
  rpn_rois_batch_size.Resize({1});
  dev_ctx.template Alloc<int>(&rpn_rois_batch_size);
  MLUOP::GenerateProposalsV2(dev_ctx,
                             pre_nms_top_n,
                             post_nms_top_n,
                             nms_thresh,
                             min_size,
                             eta,
                             pixel_offset,
                             scores_desc.get(),
                             GetBasePtr(&scores_swap),
                             bbox_deltas_desc.get(),
                             GetBasePtr(&bbox_deltas_swap),
                             im_shape_desc.get(),
                             GetBasePtr(&im_shape),
                             anchors_desc.get(),
                             GetBasePtr(&anchors),
                             variances_desc.get(),
                             GetBasePtr(&variances),
                             rpn_rois_desc.get(),
                             GetBasePtr(rpn_rois),
                             rpn_rois_probs_desc.get(),
                             GetBasePtr(rpn_roi_probs),
                             rpn_rois_num_desc.get(),
                             GetBasePtr(&rpn_roi_num_tmp),
                             GetBasePtr(&rpn_rois_batch_size));

  dev_ctx.Wait();
  std::vector<int> rpn_rois_batch_size_cpu;
  TensorToVector<int>(
      dev_ctx, rpn_rois_batch_size, dev_ctx, &rpn_rois_batch_size_cpu);

  int roi_num_final = rpn_rois_batch_size_cpu[0];
  rpn_rois->Resize({roi_num_final, 4});
  rpn_roi_probs->Resize({roi_num_final, 1});
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(generate_proposals,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::GenerateProposalsKernel,
                          float) {}
