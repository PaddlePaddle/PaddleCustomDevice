// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
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

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void MultiClassNMSKernel(const Context& dev_ctx,
                         const phi::DenseTensor& bboxes,
                         const phi::DenseTensor& scores,
                         const paddle::optional<phi::DenseTensor>& rois_num,
                         float score_threshold,
                         int nms_top_k,
                         int keep_top_k,
                         float nms_threshold,
                         bool normalized,
                         float nms_eta,
                         int background_label,
                         phi::DenseTensor* out,
                         phi::DenseTensor* index,
                         phi::DenseTensor* nms_rois_num) {
  VLOG(4) << "Call SDAA MultiClassNMSKernel";

  bool return_index = index != nullptr;
  bool has_roisnum = rois_num.get_ptr() != nullptr;
  auto score_dims = phi::vectorize<int>(scores.dims());
  auto score_size = score_dims.size();

  if (2 == score_size) {
    PADDLE_ENFORCE_EQ(
        rois_num.get_ptr() != nullptr,
        true,
        phi::errors::InvalidArgument(
            " when scores layout is [M, C], rois_num cannot be empty!"));
  }

  tecodnnHandle_t tecodnn_handle = GetHandleFromCTX(dev_ctx);

  std::vector<int> bboxes_dimensions = phi::vectorize<int>(bboxes.dims());

  tecodnnTensorDescriptor_t bboxes_desc = sdaa_ops::GetTecodnnTensorDesc(
      bboxes_dimensions, bboxes.dtype(), TensorFormat::Undefined);

  std::vector<int> scores_dimensions = phi::vectorize<int>(scores.dims());
  tecodnnTensorDescriptor_t scores_desc = sdaa_ops::GetTecodnnTensorDesc(
      scores_dimensions, scores.dtype(), TensorFormat::Undefined);

  tecodnnTensorDescriptor_t rois_num_desc;
  const void* rois_num_ptr;
  if (has_roisnum) {
    std::vector<int> rois_num_dimensions =
        phi::vectorize<int>(rois_num.get_ptr()->dims());

    rois_num_desc = sdaa_ops::GetTecodnnTensorDesc(rois_num_dimensions,
                                                   rois_num.get_ptr()->dtype(),
                                                   TensorFormat::Undefined);
    rois_num_ptr = rois_num.get_ptr()->data();
  } else {
    std::vector<int> rois_num_dimensions = {0};
    rois_num_desc = sdaa_ops::GetTecodnnTensorDesc(
        rois_num_dimensions, DataType::INT32, TensorFormat::Undefined);
    rois_num_ptr = nullptr;
  }

  size_t workspace_size;
  TECODNN_CHECK(tecodnnGetMulticlassNMSWorkspaceSize(tecodnn_handle,
                                                     nms_top_k,
                                                     bboxes_desc,
                                                     scores_desc,
                                                     rois_num_desc,
                                                     &workspace_size));
  phi::DenseTensor dev_workspace;
  dev_workspace.Resize(phi::make_ddim({static_cast<int64_t>(workspace_size)}));
  dev_ctx.Alloc(&dev_workspace, phi::DataType::INT8);

  // out and index have uncentain shapes, so in advance allocing max shape
  std::vector<int> out_dimensions_temp;
  std::vector<int> index_dimensions_temp;
  std::vector<int> nms_rois_num_dimensions;
  // when scores layout is [N, C, M], out layout is [N*C*M, 6]
  // when scores layout is [M, C], out layout is [M*C, 6]
  if (3 == score_size) {
    int N = score_dims[0];
    int C = score_dims[1];
    int M = score_dims[2];
    int No = N * C * M;
    out_dimensions_temp = {No, 6};
    index_dimensions_temp = {No, 1};
    nms_rois_num_dimensions = {N};
  } else {
    int M = score_dims[0];
    int C = score_dims[1];
    int No = M * C;
    out_dimensions_temp = {No, 6};
    index_dimensions_temp = {No, 1};

    auto rois_num_dims = phi::vectorize<int>(rois_num.get_ptr()->dims());
    nms_rois_num_dimensions = {rois_num_dims[0]};
  }

  tecodnnTensorDescriptor_t out_desc_temp = sdaa_ops::GetTecodnnTensorDesc(
      out_dimensions_temp, out->dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t index_desc_temp =
      return_index
          ? sdaa_ops::GetTecodnnTensorDesc(
                index_dimensions_temp, index->dtype(), TensorFormat::Undefined)
          : sdaa_ops::GetTecodnnTensorDesc(
                {0}, DataType::INT32, TensorFormat::Undefined);
  tecodnnTensorDescriptor_t nms_rois_num_desc = sdaa_ops::GetTecodnnTensorDesc(
      nms_rois_num_dimensions, DataType::INT32, TensorFormat::Undefined);

  phi::DenseTensor out_temp;
  out_temp.Resize(phi::make_ddim(out_dimensions_temp));
  dev_ctx.template Alloc<T>(&out_temp);

  phi::DenseTensor index_temp;
  // if index is nullptr
  if (return_index) {
    index_temp.Resize(phi::make_ddim(index_dimensions_temp));
    dev_ctx.template Alloc<int>(&index_temp);
  }

  phi::DenseTensor nms_rois_num_temp;
  nms_rois_num_temp.Resize(phi::make_ddim(nms_rois_num_dimensions));
  dev_ctx.template Alloc<int>(&nms_rois_num_temp);

  TECODNN_CHECK(tecodnnMultiClassNMS(tecodnn_handle,
                                     score_threshold,
                                     nms_threshold,
                                     nms_eta,
                                     nms_top_k,
                                     keep_top_k,
                                     background_label,
                                     normalized,
                                     bboxes_desc,
                                     bboxes.data(),
                                     scores_desc,
                                     scores.data(),
                                     rois_num_desc,
                                     rois_num_ptr,
                                     out_desc_temp,
                                     out_temp.data(),
                                     index_desc_temp,
                                     index_temp.data(),
                                     nms_rois_num_desc,
                                     nms_rois_num_temp.data(),
                                     dev_workspace.data(),
                                     workspace_size));

  // truncate output, judging the length of out by the sum of nms_rois_num
  // nms_rois_num data is on sdaa, copy from sdaa to CPU
  phi::DenseTensor nms_rois_num_cpu;
  auto cpu_place = phi::CPUPlace();
  auto custom_place = dev_ctx.GetPlace();
  phi::Copy(dev_ctx, nms_rois_num_temp, cpu_place, true, &nms_rois_num_cpu);
  int ele_length = 0;
  auto nms_rois_num_dims = phi::vectorize<int>(nms_rois_num_cpu.dims());
  const int* nms_rois_num_data = nms_rois_num_cpu.data<int>();
  for (int i = 0; i < nms_rois_num_dims[0]; ++i) {
    ele_length += nms_rois_num_data[i];
  }

  // alloc out
  std::vector<int> out_dimensions = {ele_length, 6};

  out->Resize({ele_length, 6});
  if (0 == ele_length) {
    // custom device and cpu use two different ZeroAlloctors, which make it will
    // get different shape in numpy in static graph: if out shape is [0, 6], in
    // numpy it will be [0,] in custom device and be [0, 6] in cpu
    // TODO(wuzp): what are the differences between the two ZeroAlloctors?
    dev_ctx.template HostAlloc<T>(out);
  } else {
    dev_ctx.template Alloc<T>(out);
  }

  if (ele_length) {
    const std::vector<int> axes = {0, 1};
    const std::vector<int> starts = {0, 0};
    const std::vector<int> ends = {ele_length, 6};
    const std::vector<int> strides(axes.size(), 1);
    const std::vector<int64_t> decreased_dims = {};
    sdaa_ops::doSliceTensor(
        dev_ctx, out_temp, axes, starts, ends, strides, decreased_dims, out);
  }

  if (return_index) {
    // alloc out
    index->Resize({ele_length, 1});
    if (ele_length == 0) {
      dev_ctx.template HostAlloc<int>(index);
    } else {
      dev_ctx.template Alloc<int>(index);
    }
    const std::vector<int> index_axes = {0, 1};
    const std::vector<int> index_starts = {0, 0};
    const std::vector<int> index_ends = {ele_length, 1};
    const std::vector<int> index_strides(index_axes.size(), 1);
    const std::vector<int64_t> index_decreased_dims = {};
    if (ele_length) {
      sdaa_ops::doSliceTensor(dev_ctx,
                              index_temp,
                              index_axes,
                              index_starts,
                              index_ends,
                              index_strides,
                              index_decreased_dims,
                              index);
    }
  }

  if (nms_rois_num != nullptr) {
    nms_rois_num->Resize(phi::make_ddim(nms_rois_num_dimensions));
    dev_ctx.template Alloc<int>(nms_rois_num);
    *nms_rois_num = nms_rois_num_temp;
  }

  // destroy descriptors
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(bboxes_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(scores_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(rois_num_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_desc_temp));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(index_desc_temp));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(nms_rois_num_desc));
}

}  // namespace custom_kernel

// fix(zhanggq): unregister this OP due to its poor performance
// PD_REGISTER_PLUGIN_KERNEL(multiclass_nms3,
//                           sdaa,
//                           ALL_LAYOUT,
//                           custom_kernel::MultiClassNMSKernel,
//                           float) {
//   kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
//   kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
// }
