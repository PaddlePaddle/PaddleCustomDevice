// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <vector>

#include "kernels/funcs/npu_op_runner.h"
#include "paddle/extension.h"


// qkv_transpose_split
std::vector<std::vector<int64_t>> QkvTransposeSplitOpInferShape(
    const std::vector<int64_t>& input_ids_shape,
    const std::vector<int64_t>& padding_offset_shape,
    const std::vector<int64_t>& qkv_shape,
	  const std::vector<int64_t>& seq_len_shape) {

  std::vector<int64_t> out_shape = input_ids_shape;
  return {out_shape, out_shape, out_shape};
}

std::vector<paddle::Tensor> QkvTransposeSplitOp(
    const paddle::Tensor& input_ids,
    const paddle::Tensor& padding_offset,
    const paddle::Tensor& qkv,
    const paddle::Tensor& seq_len) {

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  return {paddle::Tensor(out_tensor), paddle::Tensor(out_tensor), paddle::Tensor(out_tensor)};
}


// write_cache_kv
std::vector<std::vector<int64_t>> WriteCacheKvOpInferShape(
    const std::vector<int64_t>& cache_kv_shape,
    const std::vector<int64_t>& input_k_shape,
    const std::vector<int64_t>& input_v_shape,
	const std::vector<int64_t>& sequence_lengths_shape) {

  std::vector<int64_t> out_shape = cache_kv_shape;
  return {out_shape};
}

std::vector<paddle::Tensor> WriteCacheKvOp(
    const paddle::Tensor& cache_kv,
    const paddle::Tensor& input_k,
    const paddle::Tensor& input_v,
    const paddle::Tensor& sequence_lengths) {

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  return {paddle::Tensor(out_tensor)};
}


// transpose_remove_padding
std::vector<std::vector<int64_t>> TransposeRemovePaddingOpInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& padding_offset_shape,
    const std::vector<int64_t>& iseq_lens_shape) {

  std::vector<int64_t> out_shape = input_shape;
  return {out_shape};
}

std::vector<paddle::Tensor> TransposeRemovePaddingOp(
    const paddle::Tensor& input,
    const paddle::Tensor& padding_offset,
    const paddle::Tensor& seq_lens) {

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  return {paddle::Tensor(out_tensor)};
}

// variable_length_memory_efficient_attention
std::vector<std::vector<int64_t>> VariablAttOpInferShape(
    const std::vector<int64_t>& key_shape,
    const std::vector<int64_t>& kv_seq_lens_shape,
    const std::vector<int64_t>& mask_shape,
	  const std::vector<int64_t>& query_shape,
    const std::vector<int64_t>& seq_lens_shape,
    const std::vector<int64_t>& value_shape) {

  std::vector<int64_t> out_shape = key_shape;
  return {out_shape};
}

std::vector<paddle::Tensor> VariablAttOp(
    const paddle::Tensor& key,
    const paddle::Tensor& kv_seq_lens,
    const paddle::Tensor& mask,
	  const paddle::Tensor& query,
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& value) {

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  return {paddle::Tensor(out_tensor)};
}

// encode_rotary_qk
std::vector<std::vector<int64_t>> EncodeRotaryQkOpInferShape(
    const std::vector<int64_t>& kv_shape,
    const std::vector<int64_t>& q_shape,
    const std::vector<int64_t>& rotary_emb_shape,
	  const std::vector<int64_t>& seq_lens_shape) {

  std::vector<int64_t> out_shape = kv_shape;
  return {out_shape, out_shape};
}

std::vector<paddle::Tensor> EncodeRotaryQkOp(
    const paddle::Tensor& kv,
    const paddle::Tensor& q,
    const paddle::Tensor& rotary_emb,
	  const paddle::Tensor& seq_lens) {

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  return {paddle::Tensor(out_tensor), paddle::Tensor(out_tensor)};
}

// llama_layer
int64_t out_num = 8;
std::vector<std::vector<int64_t>> LlamaLayerOpInferShape(
    const std::vector<int64_t>& shape_1,
    const std::vector<int64_t>& shape_2,
    const std::vector<int64_t>& shape_3,
    const std::vector<int64_t>& shape_4,
    const std::vector<int64_t>& shape_5,
    const std::vector<int64_t>& shape_6,
    const std::vector<int64_t>& shape_7,
    const std::vector<int64_t>& shape_8,
    const std::vector<int64_t>& shape_9,
    const std::vector<int64_t>& shape_10,
    const std::vector<int64_t>& shape_11,
    const std::vector<int64_t>& shape_12,
    const std::vector<int64_t>& shape_13,
    const std::vector<int64_t>& shape_14) {

  std::vector<int64_t> out_shape = shape_1;
  std::vector<std::vector<int64_t>> out_shapes(out_num, out_shape);
  return out_shapes;
}

std::vector<paddle::Tensor> LlamaLayerOp(
    const paddle::Tensor& tensor_1,
    const paddle::Tensor& tensor_2,
    const paddle::Tensor& tensor_3,
    const paddle::Tensor& tensor_4,
    const paddle::Tensor& tensor_5,
    const paddle::Tensor& tensor_6,
    const paddle::Tensor& tensor_7,
    const paddle::Tensor& tensor_8,
    const paddle::Tensor& tensor_9,
    const paddle::Tensor& tensor_10,
    const paddle::Tensor& tensor_11,
    const paddle::Tensor& tensor_12,
    const paddle::Tensor& tensor_13,
    const paddle::Tensor& tensor_14) {

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  std::vector<paddle::Tensor> out_tensors(out_num, paddle::Tensor(out_tensor));
  return out_tensors;
}

// set_mask_value
std::vector<paddle::Tensor> SetMaskValueOp(const paddle::Tensor& input_data,
                                           const paddle::Tensor& seq_lens,
                                           const paddle::Tensor& stop_flags) {
    auto seq_lens_out = seq_lens.copy_to(seq_lens.place(), false);
    
    return {seq_lens_out};

//   auto dev_ctx = static_cast<const phi::CustomContext*>(
//       paddle::experimental::DeviceContextPool::Instance().Get(input_data.place()));
//   auto stream = static_cast<aclrtStream>(dev_ctx->stream());

//   auto input_data_tensor = static_cast<const phi::DenseTensor*>(input_data.impl().get());
//   auto seq_lens_tensor = static_cast<const phi::DenseTensor*>(seq_lens.impl().get());
//   auto stop_flags_tensor = static_cast<const phi::DenseTensor*>(stop_flags.impl().get());

//   std::shared_ptr<phi::DenseTensor> out_tensor =
//       std::make_shared<phi::DenseTensor>();
//   out_tensor->Resize(seq_lens_tensor->dims());
//   dev_ctx->Alloc(out_tensor.get(), seq_lens_tensor->dtype());

//   const auto& runner =
//       NpuOpRunner("SetMaskValue", {*input_data_tensor, *stop_flags_tensor, *seq_lens_tensor}, {*out_tensor}, {});
//   runner.Run(stream);

//   return {paddle::Tensor(out_tensor)};
}

std::vector<std::vector<int64_t>> SetMaskValueOpInferShape(
    const std::vector<int64_t>& input_data_shape,
    const std::vector<int64_t>& seq_lens_shape,
    const std::vector<int64_t>& stop_flags_shape) {
  return {seq_lens_shape};
}

// set_value_by_flags_and_idx
std::vector<paddle::Tensor> SetValueByFlagsAndIdxOp(const paddle::Tensor& pre_ids_all,
                                                    const paddle::Tensor& pre_ids_now,
                                                    const paddle::Tensor& step_idx,
											                          		const paddle::Tensor& stop_flags) {
    auto stop_flags_out = stop_flags.copy_to(stop_flags.place(), false);
    
    return {stop_flags_out};                                                                                

//   auto dev_ctx = static_cast<const phi::CustomContext*>(
//       paddle::experimental::DeviceContextPool::Instance().Get(stop_flags.place()));
//   auto stream = static_cast<aclrtStream>(dev_ctx->stream());

//   auto pre_ids_all_tensor = static_cast<const phi::DenseTensor*>(pre_ids_all.impl().get());
//   auto pre_ids_now_tensor = static_cast<const phi::DenseTensor*>(pre_ids_now.impl().get());
//   auto step_idx_tensor = static_cast<const phi::DenseTensor*>(step_idx.impl().get());
//   auto stop_flags_tensor = static_cast<const phi::DenseTensor*>(stop_flags.impl().get());

//   std::shared_ptr<phi::DenseTensor> out_tensor =
//       std::make_shared<phi::DenseTensor>();
//   out_tensor->Resize(stop_flags_tensor->dims());
//   dev_ctx->Alloc(out_tensor.get(), stop_flags_tensor->dtype());

//   const auto& runner =
//       NpuOpRunner("SetValueByFlagsAndIdx", {*pre_ids_all_tensor, *pre_ids_now_tensor, *step_idx_tensor, *stop_flags_tensor}, {*out_tensor}, {});
//   runner.Run(stream);

//   return {paddle::Tensor(out_tensor)};
}

std::vector<std::vector<int64_t>> SetValueByFlagsAndIdxOpInferShape(
    const std::vector<int64_t>& pre_ids_all_shape,
    const std::vector<int64_t>& pre_ids_now_shape,
    const std::vector<int64_t>& step_idx_shape,
	  const std::vector<int64_t>& stop_flags_shape) {
  return {stop_flags_shape};
}

// set_stop_value_multi_ends
std::vector<paddle::Tensor> SetStopValueMultiEndsOp(const paddle::Tensor& end_ids,
                                                    const paddle::Tensor& stop_flags,
                                                    const paddle::Tensor& topk_ids,
												                          	int64_t mode) {

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(end_ids.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto end_ids_tensor = static_cast<const phi::DenseTensor*>(end_ids.impl().get());
  auto stop_flags_tensor = static_cast<const phi::DenseTensor*>(stop_flags.impl().get());
  auto topk_ids_tensor = static_cast<const phi::DenseTensor*>(topk_ids.impl().get());

  std::shared_ptr<phi::DenseTensor> stop_flags_out =
      std::make_shared<phi::DenseTensor>();
  stop_flags_out->Resize(stop_flags_tensor->dims());
  dev_ctx->Alloc(stop_flags_out.get(), stop_flags_tensor->dtype());
  
  
  
  std::shared_ptr<phi::DenseTensor> topk_ids_out =
      std::make_shared<phi::DenseTensor>();
  topk_ids_out->Resize(topk_ids_tensor->dims());
  dev_ctx->Alloc(topk_ids_out.get(), topk_ids_tensor->dtype());
  
  int32_t attr_mode = mode;
  NPUAttributeMap attr_input = {{"mode", attr_mode}};
  
  const auto& runner =
      NpuOpRunner("SetStopValueMultiEnds", {*topk_ids_tensor, *stop_flags_tensor, *end_ids_tensor}, {*topk_ids_out, *stop_flags_out}, attr_input);
  runner.Run(stream);
  
  return {paddle::Tensor(stop_flags_out), paddle::Tensor(topk_ids_out)};
}

std::vector<std::vector<int64_t>> SetStopValueMultiEndsOpInferShape(
    const std::vector<int64_t>& end_ids_shape,
    const std::vector<int64_t>& stop_flags_shape,
    const std::vector<int64_t>& topk_ids_shape) {
  return {stop_flags_shape, topk_ids_shape};
}

// PD_BUILD_OP
PD_BUILD_OP(set_stop_value_multi_ends)
    .Inputs({"end_ids", "stop_flags", "topk_ids"})
    .Outputs({"stop_flags_out", "topk_ids_out"})
	  .Attrs({"mode: int64_t"})
    .SetKernelFn(PD_KERNEL(SetStopValueMultiEndsOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        SetStopValueMultiEndsOpInferShape));  // neccessary if the op has muti_inputs

PD_BUILD_OP(set_value_by_flags_and_idx)
    .Inputs({"pre_ids_all", "pre_ids_now", "step_idx", "stop_flags"})
    .Outputs({"stop_flags_out"})
    .SetKernelFn(PD_KERNEL(SetValueByFlagsAndIdxOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        SetValueByFlagsAndIdxOpInferShape));  // neccessary if the op has muti_inputs

PD_BUILD_OP(set_mask_value)
    .Inputs({"input_data", "seq_lens", "stop_flags"})
    .Outputs({"sequence_lengths"})
    .SetKernelFn(PD_KERNEL(SetMaskValueOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        SetMaskValueOpInferShape));  // neccessary if the op has muti_inputs

PD_BUILD_OP(llama_layer)
    .Inputs({"x", "residual", "input_ids", "padding_offset", "seq_len_encoder", "cache_kv", "mask", "rotary_emb", "ln_scale", "qkv_weight", "out_proj_weight", "ffn_in_scale", "ffn1_weight", "ffn2_weight"})
    .Outputs({"write_cache_kv", "q", "k", "v", "hidden", "residual_out", "rotary_kv_out", "rotary_q_out"})
    .SetKernelFn(PD_KERNEL(LlamaLayerOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        LlamaLayerOpInferShape));  // neccessary if the op has muti_inputs
		
PD_BUILD_OP(encode_rotary_qk)
    .Inputs({"kv", "q", "rotary_emb", "seq_lens"})
    .Outputs({"rotary_kv_out", "rotary_q_out"})
    .SetKernelFn(PD_KERNEL(EncodeRotaryQkOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        EncodeRotaryQkOpInferShape));  // neccessary if the op has muti_inputs

PD_BUILD_OP(variable_length_memory_efficient_attention)
    .Inputs({"key", "kv_seq_lens", "mask", "query", "seq_lens", "value"})
    .Outputs({"out"})
	  .Attrs({"causal: bool"})
    .SetKernelFn(PD_KERNEL(VariablAttOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        VariablAttOpInferShape));  // neccessary if the op has muti_inputs

PD_BUILD_OP(qkv_transpose_split)
    .Inputs({"input_ids", "padding_offset", "qkv", "seq_lens"})
    .Outputs({"k_out", "q_out", "v_out"})
  	.Attrs({"head_size: int",
		    "num_head: int"})
    .SetKernelFn(PD_KERNEL(QkvTransposeSplitOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        QkvTransposeSplitOpInferShape));  // neccessary if the op has muti_inputs

PD_BUILD_OP(write_cache_kv)
    .Inputs({"cache_kv", "input_k", "input_v", "sequence_lengths"})
    .Outputs({"cache_kv_out"})
    .SetKernelFn(PD_KERNEL(WriteCacheKvOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        WriteCacheKvOpInferShape));  // neccessary if the op has muti_inputs

PD_BUILD_OP(transpose_remove_padding)
    .Inputs({"input", "padding_offset", "seq_lens"})
    .Outputs({"fmha_out"})
    .SetKernelFn(PD_KERNEL(TransposeRemovePaddingOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        TransposeRemovePaddingOpInferShape));  // neccessary if the op has muti_inputs

// llama_layer
int64_t out_cached_num = 3;
std::vector<std::vector<int64_t>> LlamaCachedLayerOpInferShape(
    const std::vector<int64_t>& shape_1,
    const std::vector<int64_t>& shape_2,
    const std::vector<int64_t>& shape_3,
    const std::vector<int64_t>& shape_4,
    const std::vector<int64_t>& shape_5,
    const std::vector<int64_t>& shape_6,
    const std::vector<int64_t>& shape_7,
    const std::vector<int64_t>& shape_8,
    const std::vector<int64_t>& shape_9,
    const std::vector<int64_t>& shape_10,
    const std::vector<int64_t>& shape_11,
    const std::vector<int64_t>& shape_12) {
  std::cout<<">>>>>LlamaLayerOpInferShape"<<std::endl;
  std::vector<int64_t> out_shape = shape_1;
  std::vector<std::vector<int64_t>> out_shapes(out_cached_num, out_shape);
  return out_shapes;
}

std::vector<paddle::Tensor> LlamaCachedLayerOp(
    const paddle::Tensor& tensor_1,
    const paddle::Tensor& tensor_2,
    const paddle::Tensor& tensor_3,
    const paddle::Tensor& tensor_4,
    const paddle::Tensor& tensor_5,
    const paddle::Tensor& tensor_6,
    const paddle::Tensor& tensor_7,
    const paddle::Tensor& tensor_8,
    const paddle::Tensor& tensor_9,
    const paddle::Tensor& tensor_10,
    const paddle::Tensor& tensor_11,
    const paddle::Tensor& tensor_12) {
  std::cout<<">>>>>LlamaLayerOp"<<std::endl;
  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  std::vector<paddle::Tensor> out_tensors(out_cached_num, paddle::Tensor(out_tensor));
  return out_tensors;
}

PD_BUILD_OP(llama_cached_layer)
    .Inputs({"in_scale", "rms_norm_residual", "matmul_", "qkv_weight", "cache_kvs", "rotary_t", "sequence_l", "mask", "proj_weight", "ffn_in_scale", "ffn1_weight", "ffn2_weight"})
    .Outputs({"norm_out", "norm_residual", "cached_kv"})
    .SetKernelFn(PD_KERNEL(LlamaCachedLayerOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        LlamaCachedLayerOpInferShape));  // neccessary if the op has muti_inputs

bool is_in_end(const int64_t id, const int64_t *end_ids, int length) {
    bool flag = false;
    for (int i = 0; i < length; i++) {
        if (id == end_ids[i]) {
            return true;
        }
    }
    return flag;
}

std::vector<paddle::Tensor> RebuildPadding(const paddle::Tensor& tmp_out, 
                                           const paddle::Tensor& padding_offset, 
                                           const paddle::Tensor& seq_lens,
                                           const paddle::Tensor& input_ids) {
    return {tmp_out};
}

std::vector<std::vector<int64_t>> RebuildPaddingInferShape(const std::vector<int64_t>& tmp_out_shape,
                                                           const std::vector<int64_t>& padding_offset_shape,
                                                           const std::vector<int64_t>& seq_lens_shape,
                                                           const std::vector<int64_t>& input_ids_shape) {
    int64_t bsz = seq_lens_shape[0];
    int64_t dim_embed = tmp_out_shape[1];
    return {{bsz, dim_embed}};
}

std::vector<paddle::DataType> RebuildPaddingInferDtype(const paddle::DataType& tmp_out_dtype,
                                                       const paddle::DataType& padding_offset_dtype,
                                                       const paddle::DataType& seq_lens_dtype,
                                                       const paddle::DataType& input_ids_dtype) {
    return {tmp_out_dtype};
}

PD_BUILD_OP(rebuild_padding)
    .Inputs({"tmp_out", "padding_offset", "seq_lens", "input_ids"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(RebuildPadding))
    .SetInferShapeFn(PD_INFER_SHAPE(RebuildPaddingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RebuildPaddingInferDtype));

std::vector<paddle::Tensor> GetPaddingOffset(const paddle::Tensor& input_ids,
                                             const paddle::Tensor& cum_offsets,
                                             const paddle::Tensor& token_num,
                                             const paddle::Tensor& seq_len) {

    return {cum_offsets, seq_len, input_ids};
}

std::vector<std::vector<int64_t>> GetPaddingOffsetInferShape(const std::vector<int64_t>& input_ids_shape,
                                                             const std::vector<int64_t>& cum_offsets_shape,
                                                             const std::vector<int64_t>& token_num_shape,
                                                             const std::vector<int64_t>& seq_len_shape) {
    int64_t bsz = input_ids_shape[0];
    int64_t seq_len = input_ids_shape[1];
    return {{-1}, {bsz}, {-1}};
}

std::vector<paddle::DataType> GetPaddingOffsetInferDtype(const paddle::DataType& input_ids_dtype,
                                                         const paddle::DataType& cum_offsets_dtype,
                                                         const paddle::DataType& token_num_dtype,
                                                         const paddle::DataType& seq_len_dtype) {
    return {input_ids_dtype, seq_len_dtype, seq_len_dtype};
}

PD_BUILD_OP(get_padding_offset)
    .Inputs({"input_ids", "cum_offsets", "token_num", "seq_len"})
    .Outputs({"x_remove_padding", "cum_offsets_out", "padding_offset"})
    .SetKernelFn(PD_KERNEL(GetPaddingOffset))
    .SetInferShapeFn(PD_INFER_SHAPE(GetPaddingOffsetInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetPaddingOffsetInferDtype));            

// get_token_penalty_multi_scores
std::vector<paddle::Tensor> TokenPenaltyMultiScores(const paddle::Tensor& cur_len,
                                                    const paddle::Tensor& eos_token_id,
                                                    const paddle::Tensor& frequency_scores,
                                                    const paddle::Tensor& logits,
                                                    const paddle::Tensor& min_len,
                                                    const paddle::Tensor& penalty_scores,
                                                    const paddle::Tensor& pre_ids,
                                                    const paddle::Tensor& presence_scores) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(pre_ids.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto cur_len_tensor = static_cast<const phi::DenseTensor*>(cur_len.impl().get());
  auto eos_token_id_tensor = static_cast<const phi::DenseTensor*>(eos_token_id.impl().get());
  auto frequency_scores_tensor = static_cast<const phi::DenseTensor*>(frequency_scores.impl().get());
  auto logits_tensor = static_cast<const phi::DenseTensor*>(logits.impl().get());
  auto min_len_tensor = static_cast<const phi::DenseTensor*>(min_len.impl().get());
  auto penalty_scores_tensor = static_cast<const phi::DenseTensor*>(penalty_scores.impl().get());
  auto pre_ids_tensor = static_cast<const phi::DenseTensor*>(pre_ids.impl().get());
  auto presence_scores_tensor = static_cast<const phi::DenseTensor*>(presence_scores.impl().get());
  
  std::shared_ptr<phi::DenseTensor> logits_out =
      std::make_shared<phi::DenseTensor>();
  logits_out->Resize(logits_tensor->dims());
  dev_ctx->Alloc(logits_out.get(), logits_tensor->dtype());
  
  std::vector<phi::DenseTensor> inputs = {*cur_len_tensor,
                                          *eos_token_id_tensor,
										  *frequency_scores_tensor,
										  *logits_tensor,
										  *min_len_tensor,
										  *penalty_scores_tensor,
										  *pre_ids_tensor,
										  *presence_scores_tensor};
  
  std::vector<phi::DenseTensor> outputs = {*logits_out};
  
  const auto& runner =
      NpuOpRunner("TokenPenaltyMultiScores", inputs, outputs);
  runner.Run(stream);
  
  return {paddle::Tensor(logits_out)};
}

std::vector<std::vector<int64_t>> TokenPenaltyMultiScoresInferShape(const std::vector<int64_t>& cur_len_shape,
                                                                    const std::vector<int64_t>& eos_token_id_shape,
                                                                    const std::vector<int64_t>& frequency_scores_shape,
                                                                    const std::vector<int64_t>& logits_shape,
                                                                    const std::vector<int64_t>& min_len_shape,
                                                                    const std::vector<int64_t>& penalty_scores_shape,
                                                                    const std::vector<int64_t>& pre_ids_shape,
                                                                    const std::vector<int64_t>& presence_scores_shape) {
    return {logits_shape};
}

std::vector<paddle::DataType> TokenPenaltyMultiScoresInferDtype(const paddle::DataType& cur_len_dtype,
                                                                const paddle::DataType& eos_token_id_dtype,
                                                                const paddle::DataType& frequency_scores_dtype,
                                                                const paddle::DataType& logits_dtype,
                                                                const paddle::DataType& min_len_dtype,
                                                                const paddle::DataType& penalty_scores_dtype,
                                                                const paddle::DataType& pre_ids_dtype,
                                                                const paddle::DataType& presence_scores_dtype) {
    return {logits_dtype};
}


PD_BUILD_OP(get_token_penalty_multi_scores)
    .Inputs({"cur_len", "eos_token_id", "frequency_scores", "logits", "min_len", "penalty_scores", "pre_ids", "presence_scores"})
    .Outputs({"logits_out"})
    .SetKernelFn(PD_KERNEL(TokenPenaltyMultiScores))
    .SetInferShapeFn(PD_INFER_SHAPE(TokenPenaltyMultiScoresInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TokenPenaltyMultiScoresInferDtype));  

// top_p_sampling
std::vector<std::vector<int64_t>> TopPSamplingOpInferShape(
    const std::vector<int64_t>& top_ps_shape,
    const std::vector<int64_t>& x_shape) {
  std::cout<<">>>>>TopPSamplingOp"<<std::endl;
  std::vector<int64_t> out_shape = top_ps_shape;
  return {out_shape, out_shape};
}

std::vector<paddle::Tensor> TopPSamplingOp(
    const paddle::Tensor& top_ps,
    const paddle::Tensor& x) {
  std::vector<int64_t> shape = x.shape();
  int bs = shape[0];
  auto topp_ids = paddle::full({bs, 1}, 1, paddle::DataType::INT64, x.place());

  auto dev_ctx = static_cast<const phi::CustomContext*>(
        paddle::experimental::DeviceContextPool::Instance().Get(top_p.place()));            
  std::cout<<">>>>>TopPSamplingOp"<<std::endl;
  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(phi::make_ddim({bs, 1}));
  dev_ctx->Alloc(out_tensor.get(), paddle::DataType::FLOAT32);
         
  return {topp_ids, paddle::Tensor(out_tensor)};

//   auto dev_ctx = static_cast<const phi::CustomContext*>(
//       paddle::experimental::DeviceContextPool::Instance().Get(top_ps.place()));
//   auto stream = static_cast<aclrtStream>(dev_ctx->stream());

//   auto top_ps_tensor = static_cast<const phi::DenseTensor*>(top_ps.impl().get());
//   auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  
  
//   std::shared_ptr<phi::DenseTensor> topp_ids =
//       std::make_shared<phi::DenseTensor>();
//   topp_ids->Resize(top_ps_tensor->dims());
//   dev_ctx->Alloc(topp_ids.get(), top_ps_tensor->dtype());
  
//   std::shared_ptr<phi::DenseTensor> topp_probs =
//       std::make_shared<phi::DenseTensor>();
//   topp_probs->Resize(top_ps_tensor->dims());
//   dev_ctx->Alloc(topp_probs.get(), top_ps_tensor->dtype());

//   const auto& runner =
//       NpuOpRunner("TopPSampling", {*top_ps_tensor, *x_tensor}, {*topp_ids, *topp_probs});
//   runner.Run(stream);
  
//   return {paddle::Tensor(topp_ids), paddle::Tensor(topp_probs)};
}

PD_BUILD_OP(top_p_sampling)
    .Inputs({"top_ps", "x"})
    .Outputs({"topp_ids", "topp_probs"})
    .SetKernelFn(PD_KERNEL(TopPSamplingOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        TopPSamplingOpInferShape));  // neccessary if the op has muti_inputs

template<typename T, int N>
struct GetPackType {
  using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template<typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template<typename T, int N>
union Pack {
  static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
  Pack() {
    // do nothing
  }
  PackType<T, N> storage;
  T elem[N];
};

void fused_get_rotary_embedding(const int64_t* position_ids, 
                                const int32_t bsz, 
                                const int32_t max_seq_length, 
                                const int32_t max_position_seq_length,
                                const int32_t head_dim, 
                                const int32_t prompt_num,
                                const float inv_head_dim, 
                                const int32_t elem_cnt, 
                                float* rope_embedding) {
    /*
    In Naive implementation, it will stacks [freqs, freqs]
    And actually, each threads can process 1 values, and store continuous 2 same values. 
    So here We construct a Pack to store 2 values. 
    */
    constexpr int PackSize = 2; 
    Pack<float, PackSize> SinStorePack{}; 
    Pack<float, PackSize> CosStorePack{}; 

    const int half_head_dim = head_dim / PackSize; 
    const int32_t global_thread_idx = 0; 
    for(int idx = global_thread_idx; idx < elem_cnt; idx += 1){
        const int32_t bsz_seq_idx = idx / half_head_dim;
        const int32_t bsz_idx =  bsz_seq_idx / max_seq_length;
        const int32_t seq_idx = bsz_seq_idx % max_seq_length;
        const int64_t position_offset = bsz_idx * max_position_seq_length + seq_idx + prompt_num;
        const int32_t half_head_idx = (idx % half_head_dim) * PackSize; 
        const float exponent_factor = -static_cast<float>(half_head_idx) * inv_head_dim; // * inv_head_dim equals to / head_dim. 
        const float inv_freq_val = powf(10000.0f, exponent_factor); 
        const float freqs_val = static_cast<float>(position_ids[position_offset]) * inv_freq_val; 
        const float cos_embedding_val = cos(freqs_val); 
        const float sin_embedding_val = sin(freqs_val); 

        /*
        Since After stack, the continuous 2 elements value is same. 
        So here each threads store 2 computed embedding value. 
        */
        #pragma unroll 
        for(int unroll_idx = 0; unroll_idx < PackSize; unroll_idx++){
            CosStorePack.elem[unroll_idx] = cos_embedding_val; 
            SinStorePack.elem[unroll_idx] = sin_embedding_val; 
        }

        const int32_t cos_offset = bsz_seq_idx * head_dim + half_head_idx; 
        const int32_t sin_offset = bsz * max_seq_length * head_dim + cos_offset; 
        *(reinterpret_cast<PackType<float, PackSize>*>(rope_embedding + cos_offset)) = CosStorePack.storage;
        *(reinterpret_cast<PackType<float, PackSize>*>(rope_embedding + sin_offset)) = SinStorePack.storage;
    }
}

std::vector<std::vector<int64_t>> GetRoPEInferShape(const std::vector<int64_t>& head_dim_shape_tensor_shape, 
                                                    const std::vector<int64_t>& input_ids_shape, 
                                                    const std::vector<int64_t>& position_ids_shape) {
    const int64_t batch_size = position_ids_shape[0]; 
    const int64_t max_seq_length = input_ids_shape[1]; 
    const int64_t head_dim = head_dim_shape_tensor_shape[0]; 
    std::vector<int64_t> out_shape = {2, batch_size, 1, max_seq_length, head_dim}; 
    return {out_shape};
}

std::vector<paddle::DataType> GetRoPEInferDtype(const paddle::DataType& head_dim_shape_tensor_dtype, 
                                                const paddle::DataType& input_ids_dtype, 
                                                const paddle::DataType& position_ids_dtype) {
    // RoPE output dtype is Float. 
    return {paddle::DataType::FLOAT32};
}

std::vector<paddle::Tensor> GetRoPE(const paddle::Tensor& head_dim_shape_tensor, 
                                    const paddle::Tensor& input_ids, 
                                    const paddle::Tensor& position_ids,
                                    int prompt_num,
                                    bool use_neox) {
	auto dev_ctx = static_cast<const phi::CustomContext*>(
        paddle::experimental::DeviceContextPool::Instance().Get(head_dim_shape_tensor.place()));
	
	const int64_t batch_size = position_ids.shape()[0]; 
    const int64_t max_seq_length = input_ids.shape()[1]; 
    const int64_t head_dim = head_dim_shape_tensor.shape()[0]; 
    std::vector<int64_t> out_shape = {2, batch_size, 1, max_seq_length, head_dim};
	
	std::shared_ptr<phi::DenseTensor> rotary_embedding =
      std::make_shared<phi::DenseTensor>();
	rotary_embedding->Resize(phi::make_ddim(out_shape));
    dev_ctx->Alloc(rotary_embedding.get(), paddle::DataType::FLOAT32);
    
	return {paddle::Tensor(rotary_embedding)};
}

PD_BUILD_OP(fused_get_rotary_embedding)
    .Inputs({"head_dim_shape_tensor", "input_ids", "position_ids"})
    .Outputs({"rotary_embedding"})
    .Attrs({"prompt_num: int",
            "use_neox: bool"})
    .SetKernelFn(PD_KERNEL(GetRoPE))
    .SetInferShapeFn(PD_INFER_SHAPE(GetRoPEInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetRoPEInferDtype));

    // save_with_output
std::vector<paddle::Tensor> SaveWithOutputOp(const paddle::Tensor& batch_idx,
                                             const paddle::Tensor& step_idx,
                                             const paddle::Tensor& x) {
  std::cout<<"SaveWithOutputOp"<<std::endl;

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  
  return {paddle::Tensor(out_tensor)};
}

std::vector<std::vector<int64_t>> SaveWithOutputOpInferShape(
    const std::vector<int64_t>& batch_idx_shape,
    const std::vector<int64_t>& step_idx_shape,
    const std::vector<int64_t>& x_shape) {
  return {x_shape};
}

// PD_BUILD_OP
PD_BUILD_OP(save_with_output)
    .Inputs({"batch_idx", "step_idx", "x"})
    .Outputs({"out"})
	.Attrs({"file_path: std::string"})
    .SetKernelFn(PD_KERNEL(SaveWithOutputOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        SaveWithOutputOpInferShape));  // neccessary if the op has muti_inputs
