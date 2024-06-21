// BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include <iostream>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"  // 自定义Kernel依赖头文件

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
  VLOG(4) << "Call Sdaa GenerateProposalkKernel";
  // N,C,H,W
  auto scores_dim = scores.dims();
  int64_t num = scores_dim[0];
  int64_t c_score = scores_dim[1];
  int64_t h_score = scores_dim[2];
  int64_t w_score = scores_dim[3];
  // [N*C*H*W,4] , Rois
  rpn_rois->Resize({bbox_deltas.numel() / 4, 4});
  dev_ctx.template Alloc<T>(rpn_rois);
  // [N,1] , Scores
  rpn_roi_probs->Resize({scores.numel(), 1});
  dev_ctx.template Alloc<T>(rpn_roi_probs);
  // [N,1]

  phi::DenseTensor rpn_roi_num_tmp;
  if (rpn_rois_num) {
    rpn_rois_num->Resize({num});
    dev_ctx.template Alloc<int>(rpn_rois_num);
    rpn_roi_num_tmp = *rpn_rois_num;
  } else {
    rpn_roi_num_tmp.Resize({num});
    dev_ctx.template Alloc<int>(&rpn_roi_num_tmp);
  }
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  // Scores [N,C,H,W] -> [N,H,W,C]

  phi::DenseTensor scores_swap, bbox_deltas_swap;
  std::vector<int> dims = {num, h_score, w_score, c_score};

  scores_swap.Resize(phi::make_ddim(dims));
  dev_ctx.Alloc(&scores_swap, scores.dtype());

  const std::vector<int> perm = {0, 2, 3, 1};
  sdaa_ops::doTransposeTensor(dev_ctx, scores, perm, &scores_swap);
  tecodnnTensorDescriptor_t score_Desc =
      sdaa_ops::GetTecodnnTensorDesc(dims, scores.dtype(), TensorFormat::NHWC);

  // bbox_deltas [N,C*4,H,W ] - > [N,H,W,C*4]
  std::vector<int> bbox_dims = {num, h_score, w_score, c_score * 4};

  bbox_deltas_swap.Resize(phi::make_ddim(bbox_dims));
  dev_ctx.Alloc(&bbox_deltas_swap, bbox_deltas.dtype());
  sdaa_ops::doTransposeTensor(dev_ctx, bbox_deltas, perm, &bbox_deltas_swap);
  tecodnnTensorDescriptor_t bbox_Desc = sdaa_ops::GetTecodnnTensorDesc(
      bbox_dims, bbox_deltas.dtype(), TensorFormat::NHWC);

  // img_shape [N,2]
  tecodnnTensorDescriptor_t img_Desc =
      sdaa_ops::GetTecodnnTensorDesc(phi::vectorize<int>(im_shape.dims()),
                                     im_shape.dtype(),
                                     TensorFormat::Undefined);
  // anchors [H,W,C,4]
  tecodnnTensorDescriptor_t anchor_Desc =
      sdaa_ops::GetTecodnnTensorDesc(phi::vectorize<int>(anchors.dims()),
                                     anchors.dtype(),
                                     TensorFormat::Undefined);
  // variances [H,W,C,4]
  tecodnnTensorDescriptor_t var_Desc =
      sdaa_ops::GetTecodnnTensorDesc(phi::vectorize<int>(variances.dims()),
                                     variances.dtype(),
                                     TensorFormat::Undefined);

  tecodnnTensorDescriptor_t rpn_Desc =
      sdaa_ops::GetTecodnnTensorDesc(phi::vectorize<int>(rpn_rois->dims()),
                                     scores.dtype(),
                                     TensorFormat::Undefined);

  tecodnnTensorDescriptor_t rpn_probsDesc =
      sdaa_ops::GetTecodnnTensorDesc(phi::vectorize<int>(rpn_roi_probs->dims()),
                                     scores.dtype(),
                                     TensorFormat::Undefined);

  tecodnnTensorDescriptor_t rpn_numDesc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(rpn_roi_num_tmp.dims()),
      DataType::INT32,
      TensorFormat::Undefined);

  // bool return_rois_num=false;
  // if (rpn_rois_num != nullptr) {
  //     return_rois_num=true;
  // }
  // return_rois_num should be true that rpn_roi_num_tmp have value
  bool return_rois_num = true;
  size_t workSpaceSizeInBytes = 0;
  TECODNN_CHECK(
      tecodnnGetGenerateProposalsWorkspaceSize(tecodnnHandle,
                                               pre_nms_top_n,
                                               post_nms_top_n,
                                               score_Desc,
                                               bbox_Desc,
                                               anchor_Desc,
                                               var_Desc,
                                               &workSpaceSizeInBytes));
  phi::DenseTensor workspace;
  int8_t* workspace_data =
      dev_ctx.template Alloc<int8_t>(&workspace, workSpaceSizeInBytes);

  TECODNN_CHECK(tecodnnGenerateProposals(tecodnnHandle,
                                         pre_nms_top_n,
                                         post_nms_top_n,
                                         nms_thresh,
                                         min_size,
                                         eta,
                                         pixel_offset,
                                         return_rois_num,
                                         score_Desc,
                                         scores_swap.data(),
                                         bbox_Desc,
                                         bbox_deltas_swap.data(),
                                         img_Desc,
                                         im_shape.data(),
                                         anchor_Desc,
                                         anchors.data(),
                                         var_Desc,
                                         variances.data(),
                                         rpn_Desc,
                                         rpn_rois->data(),
                                         rpn_probsDesc,
                                         rpn_roi_probs->data(),
                                         rpn_numDesc,
                                         rpn_roi_num_tmp.data(),
                                         workSpaceSizeInBytes,
                                         workspace_data));
  std::vector<int> vec;
  int64_t num_proposals = 0;
  TensorToVector(dev_ctx, rpn_roi_num_tmp, dev_ctx, &vec);
  for (int i = 0; i < vec.size(); i++) {
    num_proposals += vec[i];
  }

  rpn_rois->Resize(phi::make_ddim({num_proposals, 4}));
  rpn_roi_probs->Resize(phi::make_ddim({num_proposals, 1}));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(score_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(bbox_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(img_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(anchor_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(var_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(rpn_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(rpn_probsDesc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(rpn_numDesc));
}

}  // namespace custom_kernel
PD_REGISTER_PLUGIN_KERNEL(generate_proposals,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::GenerateProposalsKernel,
                          float,
                          double) {
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}
