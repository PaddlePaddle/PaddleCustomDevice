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

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(input_data.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto input_data_tensor = static_cast<const phi::DenseTensor*>(input_data.impl().get());
  auto seq_lens_tensor = static_cast<const phi::DenseTensor*>(seq_lens.impl().get());
  auto stop_flags_tensor = static_cast<const phi::DenseTensor*>(stop_flags.impl().get());

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(seq_lens_tensor->dims());
  dev_ctx->Alloc(out_tensor.get(), seq_lens_tensor->dtype());

  const auto& runner =
      NpuOpRunner("SetMaskValue", {*input_data_tensor, *stop_flags_tensor, *seq_lens_tensor}, {*out_tensor}, {});
  runner.Run(stream);

  return {paddle::Tensor(out_tensor)};
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

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(stop_flags.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto pre_ids_all_tensor = static_cast<const phi::DenseTensor*>(pre_ids_all.impl().get());
  auto pre_ids_now_tensor = static_cast<const phi::DenseTensor*>(pre_ids_now.impl().get());
  auto step_idx_tensor = static_cast<const phi::DenseTensor*>(step_idx.impl().get());
  auto stop_flags_tensor = static_cast<const phi::DenseTensor*>(stop_flags.impl().get());

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(stop_flags_tensor->dims());
  dev_ctx->Alloc(out_tensor.get(), stop_flags_tensor->dtype());

  const auto& runner =
      NpuOpRunner("SetValueByFlagsAndIdx", {*pre_ids_all_tensor, *pre_ids_now_tensor, *step_idx_tensor, *stop_flags_tensor}, {*out_tensor}, {});
  runner.Run(stream);

  return {paddle::Tensor(out_tensor)};
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
												                          	int mode) {

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
	  .Attrs({"mode: int"})
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
