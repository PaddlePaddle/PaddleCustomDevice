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

#include "custom_op/custom_op_common.h"

std::vector<std::vector<int64_t>> FusedRotaryEmbeddingInferShape(
    std::vector<int64_t> query_shape,
    std::vector<int64_t> key_shape,
    std::vector<int64_t> cos_sin_table_shape,
    std::vector<int64_t> positions_shape,
    bool is_neox) {
  return {query_shape, key_shape};
}

std::vector<paddle::Tensor> FusedRotaryEmbeddingKernel(
    const paddle::Tensor& query,
    const paddle::Tensor& key,
    const paddle::Tensor& cos_sin_table,
    const paddle::Tensor& positions,
    bool is_neox) {
  PADDLE_GCU_KERNEL_TRACE("fused_rotary_embedding_gcu");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: fused_rotary_embedding_gcu";
  return custom_op_common::FusedRotaryEmbedding(
      query, key, cos_sin_table, positions, is_neox);
}

PD_BUILD_OP(fused_rotary_embedding_gcu)
    .Inputs({"query", "key", "cos_sin_table", "positions"})
    .Outputs({"query_out", "key_out"})
    .Attrs({"is_neox: bool"})
    .SetKernelFn(PD_KERNEL(FusedRotaryEmbeddingKernel))
    .SetInferShapeFn(
        PD_INFER_SHAPE(FusedRotaryEmbeddingInferShape));  // neccessary if the
                                                          // op has muti_inputs
