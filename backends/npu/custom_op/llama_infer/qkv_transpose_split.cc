// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/extension.h"

std::vector<paddle::Tensor> QKVTransposeSplit(
    const paddle::Tensor& qkv,
    const paddle::Tensor& padding_offset,
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& input_ids,
    int num_head,
    int head_size) {
  std::vector<int64_t> qkv_shape = qkv.shape();
  const int bsz = seq_lens.shape()[0];
  const int max_seq_len =
      input_ids.shape()[1];  // max_seq_len_tensor.copy_to(paddle::CPUPlace(),
                             // false).data<int>()[0];

  int64_t fused_hidden_size = qkv.shape()[1];
  int kv_num_head = (fused_hidden_size - num_head * head_size) / head_size / 2;

  auto q_out = paddle::full(
      {bsz, num_head, max_seq_len, head_size}, 0, qkv.dtype(), qkv.place());
  auto k_out = paddle::full(
      {bsz, kv_num_head, max_seq_len, head_size}, 0, qkv.dtype(), qkv.place());
  auto v_out = paddle::full(
      {bsz, kv_num_head, max_seq_len, head_size}, 0, qkv.dtype(), qkv.place());
  return {q_out, k_out, v_out};
}

std::vector<std::vector<int64_t>> QKVTransposeSplitInferShape(
    const std::vector<int64_t>& qkv_shape,
    const std::vector<int64_t>& padding_offset_shape,
    const std::vector<int64_t>& seq_lens_shape,
    const std::vector<int64_t>& input_ids_shape,
    int num_head,
    int head_size) {
  int64_t bsz = seq_lens_shape[0];
  int64_t fused_hidden_size = qkv_shape[1];
  int kv_num_head = (fused_hidden_size - num_head * head_size) / head_size / 2;
  return {{bsz, num_head, -1, head_size},
          {bsz, kv_num_head, -1, head_size},
          {bsz, kv_num_head, -1, head_size}};
}

std::vector<paddle::DataType> QKVTransposeSplitInferDtype(
    const paddle::DataType& qkv_dtype,
    const paddle::DataType& padding_offset_dtype,
    const paddle::DataType& seq_lens_dtype,
    const paddle::DataType& input_ids_dtype) {
  return {qkv_dtype, qkv_dtype, qkv_dtype};
}

PD_BUILD_OP(qkv_transpose_split)
    .Inputs({"qkv", "padding_offset", "seq_lens", "input_ids"})
    .Outputs({"q_out", "k_out", "v_out"})
    .Attrs({"num_head: int", "head_size: int"})
    .SetKernelFn(PD_KERNEL(QKVTransposeSplit))
    .SetInferShapeFn(PD_INFER_SHAPE(QKVTransposeSplitInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(QKVTransposeSplitInferDtype));
