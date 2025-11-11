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

std::vector<std::vector<int64_t>> FlashAttnVarLenInferShape(
    std::vector<int64_t> q_shape,
    std::vector<int64_t> k_shape,
    std::vector<int64_t> v_shape,
    std::vector<int64_t> cu_seqlens_q_shape,
    std::vector<int64_t> cu_seqlens_k_shape) {
  return {q_shape};
}

std::vector<paddle::DataType> FlashAttnVarLenInferDtype(
    const paddle::DataType& q_dtype,
    const paddle::DataType& k_dtype,
    const paddle::DataType& v_dtype,
    const paddle::DataType& cu_seqlens_q_dtype,
    const paddle::DataType& cu_seqlens_k_dtype) {
  return {q_dtype};
}

std::vector<paddle::Tensor> FlashAttnVarLenKernel(
    const paddle::Tensor& query,
    const paddle::Tensor& key,
    const paddle::Tensor& value,
    const paddle::Tensor& cu_seqlens_q,                    // b+1
    const paddle::optional<paddle::Tensor>& cu_seqlens_k,  // b+1
    const paddle::optional<paddle::Tensor>& seqused_k,     // b
    const paddle::optional<paddle::Tensor>& leftpad_k,     // b
    const paddle::optional<paddle::Tensor>& block_table,
    const paddle::optional<paddle::Tensor>& alibi_slopes,
    const int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool return_softmax) {
  PADDLE_GCU_KERNEL_TRACE("flash_attn_var_len_gcu");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: flash_attn_var_len_gcu";

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(query.place()));

  // [total_q, q_num_heads, head_size]
  auto query_tensor = static_cast<const phi::DenseTensor*>(query.impl().get());
  // [total_k, k_num_heads, head_size] or [num_blocks, block_size, k_num_heads,
  // head_size]
  auto key_tensor = static_cast<const phi::DenseTensor*>(key.impl().get());
  // [total_k, k_num_heads, head_size] or [num_blocks, block_size, k_num_heads,
  // head_size]
  auto value_tensor = static_cast<const phi::DenseTensor*>(value.impl().get());
  // [batch_size + 1]
  auto cu_seqlens_q_tensor =
      static_cast<const phi::DenseTensor*>(cu_seqlens_q.impl().get());

  phi::DenseTensor tmp;

  // [batch_size + 1]
  phi::DenseTensor* cu_seqlens_k_tensor = &tmp;
  if (cu_seqlens_k.is_initialized()) {
    cu_seqlens_k_tensor =
        static_cast<phi::DenseTensor*>(cu_seqlens_k.get().impl().get());
  }

  // [batch_size]
  phi::DenseTensor* seqused_k_tensor = &tmp;
  if (seqused_k.is_initialized()) {
    seqused_k_tensor =
        static_cast<phi::DenseTensor*>(seqused_k.get().impl().get());
  }

  // [batch_size]
  phi::DenseTensor* leftpad_k_tensor = &tmp;
  if (leftpad_k.is_initialized()) {
    leftpad_k_tensor =
        static_cast<phi::DenseTensor*>(leftpad_k.get().impl().get());
  }

  // [batch_size, max_num_blocks_per_seq]
  phi::DenseTensor* block_table_tensor = &tmp;
  if (block_table.is_initialized()) {
    block_table_tensor =
        static_cast<phi::DenseTensor*>(block_table.get().impl().get());
  }

  // [batch_size, max_num_blocks_per_seq]
  phi::DenseTensor* alibi_slopes_tensor = &tmp;
  if (alibi_slopes.is_initialized()) {
    alibi_slopes_tensor =
        static_cast<phi::DenseTensor*>(alibi_slopes.get().impl().get());
  }

  auto query_dims = query_tensor->dims();
  const int64_t batch_size = cu_seqlens_q_tensor->numel() - 1;
  const int64_t total_q = query_dims.at(0);
  const int64_t q_num_heads = query_dims.at(1);
  const int64_t head_size = query_dims.at(2);

  VLOG(3) << "This is an informational log message for flash_attn_var_len_gcu "
             "begin.";
  VLOG(3) << "[CUSTOM_KERNEL] [total_q, q_num_heads, head_size] is [" << total_q
          << ", " << q_num_heads << ", " << head_size << "]";
  VLOG(3) << "[CUSTOM_KERNEL] batch_size is " << batch_size;
  VLOG(3) << "[CUSTOM_KERNEL] max_seqlen_q is " << max_seqlen_q
          << ", max_seqlen_k is " << max_seqlen_k;

  // attention_out
  std::shared_ptr<phi::DenseTensor> attention_out =
      std::make_shared<phi::DenseTensor>();
  attention_out->Resize(query_dims);
  dev_ctx->Alloc(attention_out.get(), query_tensor->dtype());
  auto out_tensor = custom_kernel::CreateTopsatenTensor(*attention_out);

  // softmax_lse_out
  std::shared_ptr<phi::DenseTensor> softmax_lse_out =
      std::make_shared<phi::DenseTensor>();
  softmax_lse_out->Resize(phi::make_ddim({q_num_heads, total_q}));
  dev_ctx->Alloc(softmax_lse_out.get(), phi::DataType::FLOAT32);
  auto softmax_lse_tensor =
      custom_kernel::CreateTopsatenTensor(*softmax_lse_out);

  // p_out
  topsatenTensor p_tensor;

  // rng_state_out
  std::shared_ptr<phi::DenseTensor> rng_state_out =
      std::make_shared<phi::DenseTensor>();
  rng_state_out->Resize({2});
  dev_ctx->Alloc(rng_state_out.get(), phi::DataType::INT32);
  auto rng_state_tensor = custom_kernel::CreateTopsatenTensor(*rng_state_out);

  // Create topsaten tensor
  std::tuple<topsatenTensor, topsatenTensor, topsatenTensor, topsatenTensor>
      outputs(out_tensor, softmax_lse_tensor, p_tensor, rng_state_tensor);

  auto q_aten = custom_kernel::CreateTopsatenTensor(*query_tensor);
  auto k_aten = custom_kernel::CreateTopsatenTensor(*key_tensor);
  auto v_aten = custom_kernel::CreateTopsatenTensor(*value_tensor);
  auto cu_seqlens_q_aten =
      custom_kernel::CreateTopsatenTensor(*cu_seqlens_q_tensor);
  auto cu_seqlens_k_aten =
      custom_kernel::CreateTopsatenTensor(*cu_seqlens_k_tensor);
  auto seqused_k_aten = custom_kernel::CreateTopsatenTensor(*seqused_k_tensor);
  auto leftpad_k_aten = custom_kernel::CreateTopsatenTensor(*leftpad_k_tensor);
  auto block_table_aten =
      custom_kernel::CreateTopsatenTensor(*block_table_tensor);
  auto alibi_slopes_aten =
      custom_kernel::CreateTopsatenTensor(*alibi_slopes_tensor);

  topsatenScalar_t max_seqlen_q_scalar;
  max_seqlen_q_scalar.dtype = TOPSATEN_DATA_I32;
  max_seqlen_q_scalar.ival = max_seqlen_q;

  topsatenScalar_t max_seqlen_k_scalar;
  max_seqlen_k_scalar.dtype = TOPSATEN_DATA_I32;
  max_seqlen_k_scalar.ival = max_seqlen_k;

  topsatenScalar_t p_dropout_scalar;
  p_dropout_scalar.dtype = TOPSATEN_DATA_FP32;
  p_dropout_scalar.fval = p_dropout;

  topsatenScalar_t softmax_scale_scalar;
  softmax_scale_scalar.dtype = TOPSATEN_DATA_FP32;
  softmax_scale_scalar.fval = softmax_scale;

  topsatenScalar_t window_size_left_scalar;
  window_size_left_scalar.dtype = TOPSATEN_DATA_I32;
  window_size_left_scalar.ival = window_size_left;

  topsatenScalar_t window_size_right_scalar;
  window_size_right_scalar.dtype = TOPSATEN_DATA_I32;
  window_size_right_scalar.ival = window_size_right;

  topsatenScalar_t softcap_scalar;
  softcap_scalar.dtype = TOPSATEN_DATA_FP32;
  softcap_scalar.fval = softcap;

  auto gen = dev_ctx->GetGenerator();
  topsatenGenerator_t generator{gen->GetCurrentSeed(), 0};

  auto stream = static_cast<topsStream_t>(dev_ctx->stream());
  auto status = topsfa::topsfaFlashAttnVarlenFwd(outputs,
                                                 q_aten,
                                                 k_aten,
                                                 v_aten,
                                                 cu_seqlens_q_aten,
                                                 cu_seqlens_k_aten,
                                                 seqused_k_aten,
                                                 leftpad_k_aten,
                                                 block_table_aten,
                                                 alibi_slopes_aten,
                                                 max_seqlen_q_scalar,
                                                 max_seqlen_k_scalar,
                                                 p_dropout_scalar,
                                                 softmax_scale_scalar,
                                                 zero_tensors,
                                                 is_causal,
                                                 window_size_left_scalar,
                                                 window_size_right_scalar,
                                                 softcap_scalar,
                                                 return_softmax,
                                                 generator,
                                                 stream);
  PADDLE_ENFORCE_EQ(
      status,
      TOPSATEN_STATUS_SUCCESS,
      phi::errors::Fatal("Failed to call aten op "
                         "topsfa::topsfaFlashAttnVarlenFwd, get error: %d",
                         status));

  return {paddle::Tensor(attention_out)};
}

PD_BUILD_OP(flash_attn_var_len_gcu)
    .Inputs({"query",
             "key",
             "value",
             "cu_seqlens_q",
             paddle::Optional("cu_seqlens_k"),
             paddle::Optional("seqused_k"),
             paddle::Optional("leftpad_k"),
             paddle::Optional("block_table"),
             paddle::Optional("alibi_slopes")})
    .Outputs({"attention_out"})
    .Attrs({
        "max_seqlen_q: int",
        "max_seqlen_k: int",
        "p_dropout: float",
        "softmax_scale: float",
        "zero_tensors: bool",
        "is_causal: bool",
        "window_size_left: int",
        "window_size_right: int",
        "softcap: float",
        "return_softmax: bool",
    })
    .SetKernelFn(PD_KERNEL(FlashAttnVarLenKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(FlashAttnVarLenInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FlashAttnVarLenInferDtype));
