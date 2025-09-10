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

std::vector<std::vector<int64_t>> PagedAttentionInferShape(
    std::vector<int64_t> q_shape,
    std::vector<int64_t> k_cache_shape,
    std::vector<int64_t> v_cache_shape,
    std::vector<int64_t> block_tables_shape,
    std::vector<int64_t> context_lens_shape) {
  return {q_shape};
}

std::vector<paddle::DataType> PagedAttentionInferDtype(
    const paddle::DataType& q_dtype,
    const paddle::DataType& k_cache_dtype,
    const paddle::DataType& v_cache_dtype,
    const paddle::DataType& block_tables_dtype,
    const paddle::DataType& context_lens_dtype) {
  return {q_dtype};
}

std::vector<paddle::Tensor> PagedAttentionKernel(
    const paddle::Tensor& query,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& context_lens,
    const paddle::optional<paddle::Tensor>& alibi_slopes,
    const paddle::optional<paddle::Tensor>& out_scales,
    int64_t num_kv_heads,
    float scale,
    int64_t block_size,
    int64_t max_context_len,
    const std::string& kv_cache_dtype,
    const float k_scale = 1.0f,
    const float k_zero = 0.0f,
    const float v_scale = 1.0f,
    const float v_zero = 0.0f) {
  PADDLE_GCU_KERNEL_TRACE("paged_attention_gcu");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: paged_attention_gcu";

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(query.place()));

  //  * @note Limitation
  //  * 1. The mem stride of key_cache must be [num_kv_heads * head_size *
  //  block_size, head_size * block_size, x, head_size, 1].
  //  * 2. The mem stride of value_cache must be [num_kv_heads * head_size *
  //  block_size, head_size * block_size, 1, head_size].

  // [num_seqs, num_heads, head_size]
  auto query_tensor = static_cast<const phi::DenseTensor*>(query.impl().get());
  // [num_blocks, num_kv_heads, head_size/x, block_size, x]
  auto key_cache_tensor =
      static_cast<phi::DenseTensor*>(key_cache.impl().get());
  // [num_blocks, num_kv_heads, head_size, block_size]
  auto value_cache_tensor =
      static_cast<phi::DenseTensor*>(value_cache.impl().get());
  // [num_seqs, max_num_blocks_per_seq]
  auto block_tables_tensor =
      static_cast<const phi::DenseTensor*>(block_tables.impl().get());
  // [num_seqs]
  auto context_lens_tensor =
      static_cast<const phi::DenseTensor*>(context_lens.impl().get());

  //   auto key_cache_dims = key_cache_tensor->dims();
  //   int64_t num_blocks = key_cache_dims.at(0);
  // //   int64_t num_kv_heads = key_cache_dims.at(1);
  // //   int64_t block_size = key_cache_dims.at(3);
  //   int64_t x = key_cache_dims.at(4);
  //   auto value_cache_dims = value_cache_tensor->dims();
  //   int64_t head_size = key_cache_dims.at(0);

  //   auto key_cache_meta = key_cache_tensor->meta();
  //   std::vector<int64_t> key_cache_strides = {num_kv_heads * head_size *
  //   block_size, head_size * block_size, x, head_size, 1};
  //   key_cache_meta.strides = common::make_ddim(key_cache_strides);
  //   key_cache_tensor->set_meta(key_cache_meta);
  //   auto value_cache_meta = value_cache_tensor->meta();
  //   std::vector<int64_t> value_cache_strides = {num_kv_heads * head_size *
  //   block_size, head_size * block_size, 1, head_size};
  //   value_cache_meta.strides = common::make_ddim(value_cache_strides);
  //   value_cache_tensor->set_meta(value_cache_meta);

  phi::DenseTensor tmp;
  // [num_heads]
  phi::DenseTensor* alibi_slopes_tensor = &tmp;
  if (alibi_slopes.is_initialized()) {
    alibi_slopes_tensor =
        static_cast<phi::DenseTensor*>(alibi_slopes.get().impl().get());
  }

  // [1] or [num_heads * head_size]
  phi::DenseTensor* out_scales_tensor = &tmp;
  if (out_scales.is_initialized()) {
    out_scales_tensor =
        static_cast<phi::DenseTensor*>(out_scales.get().impl().get());
  }

  auto query_dims = query_tensor->dims();
  // attention_out
  std::shared_ptr<phi::DenseTensor> attention_out =
      std::make_shared<phi::DenseTensor>();
  attention_out->Resize(query_dims);
  dev_ctx->Alloc(attention_out.get(), query_tensor->dtype());

  const char* kv_dtype = kv_cache_dtype.data();
  phi::DenseTensor head_mapping_tensor;

  phi::Scalar scale_scalar(scale);
  phi::Scalar block_size_scalar(block_size);
  phi::Scalar max_context_len_scalar(max_context_len);

  phi::Scalar k_scale_scalar(k_scale);
  phi::Scalar k_zp_scalar(k_zero);
  phi::Scalar v_scale_scalar(v_scale);
  phi::Scalar v_zp_scalar(v_zero);

  LAUNCH_TOPSATENOP(topsvllmPagedAttentionV1,
                    (*dev_ctx),
                    *attention_out,
                    *query_tensor,
                    *key_cache_tensor,
                    *value_cache_tensor,
                    head_mapping_tensor,
                    scale_scalar,
                    *block_tables_tensor,
                    *context_lens_tensor,
                    block_size_scalar,
                    max_context_len_scalar,
                    *alibi_slopes_tensor,
                    kv_dtype,
                    k_scale_scalar,
                    k_zp_scalar,
                    v_scale_scalar,
                    v_zp_scalar,
                    *out_scales_tensor);

  return {paddle::Tensor(attention_out)};
}

PD_BUILD_OP(paged_attention_gcu)
    .Inputs({"query",
             "key_cache",
             "value_cache",
             "block_tables",
             "context_lens",
             paddle::Optional("alibi_slopes"),
             paddle::Optional("out_scales")})
    .Outputs({"attention_out"})
    .Attrs({
        "num_kv_heads: int64_t",
        "scale: float",
        "block_size: int64_t",
        "max_context_len: int64_t",
        "kv_cache_dtype: std::string",
        "k_scale: float",
        "k_zero: float",
        "v_scale: float",
        "v_zero: float",
    })
    .SetKernelFn(PD_KERNEL(PagedAttentionKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(PagedAttentionInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PagedAttentionInferDtype));
