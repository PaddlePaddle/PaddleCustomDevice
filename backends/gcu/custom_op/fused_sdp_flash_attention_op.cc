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

std::vector<std::vector<int64_t>> FusedSdpFlashAttentionInferShape(
    std::vector<int64_t> q_shape,
    std::vector<int64_t> k_shape,
    std::vector<int64_t> v_shape) {
  return {q_shape};
}

std::vector<paddle::Tensor> FusedSdpFlashAttentionKernel(
    const paddle::Tensor& query,
    const paddle::Tensor& key,
    const paddle::Tensor& value,
    const paddle::optional<paddle::Tensor>& attn_mask,
    float dropout = 0.0,
    bool casual = false,
    bool is_test = true) {
  PADDLE_GCU_KERNEL_TRACE("fused_sdp_flash_attention_gcu");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: fused_sdp_flash_attention_gcu";
  return custom_op_common::FusedSdpFlashAttention(
      query, key, value, attn_mask, dropout, casual, is_test);
}

PD_BUILD_OP(fused_sdp_flash_attention_gcu)
    .Inputs({"query", "key", "value", paddle::Optional("attn_mask")})
    .Outputs({"attention_out"})
    .Attrs({"dropout: float", "causal: bool", "is_test: bool"})
    .SetKernelFn(PD_KERNEL(FusedSdpFlashAttentionKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(
        FusedSdpFlashAttentionInferShape));  // neccessary if the op has
                                             // muti_inputs
