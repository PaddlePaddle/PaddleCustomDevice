// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

std::vector<std::vector<int64_t>> MemEfficientAttentionInferShape(
    std::vector<int64_t> q_shape,
    std::vector<int64_t> k_shape,
    std::vector<int64_t> v_shape) {
  return {q_shape};
}

std::vector<paddle::DataType> MemEfficientAttentionInferDtype(
    const paddle::DataType& q_dtype,
    const paddle::DataType& k_dtype,
    const paddle::DataType& v_dtype) {
  return {q_dtype};
}

std::vector<paddle::Tensor> MemEfficientAttentionKernel(
    const paddle::Tensor& query,
    const paddle::Tensor& key,
    const paddle::Tensor& value,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const float dropout,
    const float softmax_scale,
    const int32_t mask_mode = 1,
    const std::vector<int>& seqlens = {},
    const bool casual = true) {
  PADDLE_GCU_KERNEL_TRACE("mem_efficient_attention_gcu");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: mem_efficient_attention_gcu";

  // mask_mode == 1 support dynamic block diag casual mask.
  // mask_mode == 4 support Variable-length attention.
  PADDLE_ENFORCE_EQ((mask_mode == 1 || mask_mode == 4),
                    true,
                    phi::errors::InvalidArgument(
                        "Only support (mask_mode == 1 or mask_mode == 4) now"));

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(query.place()));

  // [batch_size, q_seq_len, q_head_num, head_size]
  auto query_tensor = static_cast<const phi::DenseTensor*>(query.impl().get());
  // [batch_size, kv_seq_len, kv_head_num, head_size]
  auto key_tensor = static_cast<const phi::DenseTensor*>(key.impl().get());
  // [batch_size, kv_seq_len, kv_head_num, head_size]
  auto value_tensor = static_cast<const phi::DenseTensor*>(value.impl().get());
  // [batch_size, 1, target_seq_len, seq_len]
  phi::DenseTensor tmp;
  phi::DenseTensor* attn_mask_tensor = &tmp;
  if (attn_mask.is_initialized()) {
    attn_mask_tensor =
        static_cast<phi::DenseTensor*>(attn_mask.get().impl().get());
  }

  auto query_dims = query_tensor->dims();
  //   const double scale = 1.0f / std::sqrt(query_dims.at(3));

  // attention_out
  std::shared_ptr<phi::DenseTensor> attention_out =
      std::make_shared<phi::DenseTensor>();
  attention_out->Resize(query_dims);
  dev_ctx->Alloc(attention_out.get(), query_tensor->dtype());

  int sliding_window = 0;

  auto dropout_scalar = phi::Scalar(dropout);
  auto scale_scalar = phi::Scalar(softmax_scale);
  auto mask_mode_scalar = phi::Scalar(mask_mode);
  auto alibi_slopes = phi::DenseTensor();
  auto sliding_window_scalar = phi::Scalar(sliding_window);

  LAUNCH_TOPSATENOP(topsvllmMemEfficientAttentionV1,
                    (*dev_ctx),
                    *attention_out,
                    *query_tensor,
                    *key_tensor,
                    *value_tensor,
                    *attn_mask_tensor,
                    dropout_scalar,
                    scale_scalar,
                    seqlens,
                    mask_mode_scalar,
                    alibi_slopes,
                    sliding_window_scalar);

  return {paddle::Tensor(attention_out)};
}

PD_BUILD_OP(mem_efficient_attention_gcu)
    .Inputs({"query", "key", "value", paddle::Optional("attn_mask")})
    .Outputs({"attention_out"})
    .Attrs({
        "dropout: float",
        "softmax_scale: float",
        "mask_mode: int",
        "seqlens: std::vector<int>",
        "causal: bool",
    })
    .SetKernelFn(PD_KERNEL(MemEfficientAttentionKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(MemEfficientAttentionInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MemEfficientAttentionInferDtype));
