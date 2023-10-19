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
std::vector<paddle::Tensor> QKVTransposeSplit(const paddle::Tensor& qkv, 
                                              const paddle::Tensor& padding_offset, 
                                              const paddle::Tensor& seq_lens,
                                              const paddle::Tensor& input_ids,
                                              int num_head,
                                              int head_size) {
  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();

  return {paddle::Tensor(out_tensor), paddle::Tensor(out_tensor), paddle::Tensor(out_tensor)};
}

std::vector<std::vector<int64_t>> QKVTransposeSplitInferShape(const std::vector<int64_t>& qkv_shape,
                                                              const std::vector<int64_t>& padding_offset_shape,
                                                              const std::vector<int64_t>& seq_lens_shape,
                                                              const std::vector<int64_t>& input_ids_shape,
                                                              int num_head,
                                                              int head_size) {
    int64_t bsz = seq_lens_shape[0];
    return {{bsz, num_head, -1, head_size}, {bsz, num_head, -1, head_size}, {bsz, num_head, -1, head_size}};
}

std::vector<paddle::DataType> QKVTransposeSplitInferDtype(const paddle::DataType& qkv_dtype,
                                                          const paddle::DataType& padding_offset_dtype,
                                                          const paddle::DataType& seq_lens_dtype,
                                                          const paddle::DataType& input_ids_dtype) {
    return {qkv_dtype, qkv_dtype, qkv_dtype};
}

PD_BUILD_OP(qkv_transpose_split)
    .Inputs({"qkv", "padding_offset", "seq_lens", "input_ids"})
    .Outputs({"q_out", "k_out", "v_out"})
    .Attrs({"num_head: int",
            "head_size: int"})
    .SetKernelFn(PD_KERNEL(QKVTransposeSplit))
    .SetInferShapeFn(PD_INFER_SHAPE(QKVTransposeSplitInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(QKVTransposeSplitInferDtype));

// write_cache_kv
std::vector<std::vector<int64_t>> WriteCacheKvOpInferShape(
    const std::vector<int64_t>& cache_kv_shape,
    const std::vector<int64_t>& input_k_shape,
    const std::vector<int64_t>& input_v_shape,
    const std::vector<int64_t>& sequence_lengths_shape) {

  std::vector<int64_t> out_shape = cache_kv_shape;
  return {out_shape};
}

std::vector<paddle::DataType> WriteCacheKvOpInferDtype(const paddle::DataType& input_ids_dtype,
                                                       const paddle::DataType& padding_offset_dtype,
                                                       const paddle::DataType& qkv_dtype,
                                                       const paddle::DataType& seq_len_dtype) {
    return {input_ids_dtype};
}

void WriteCacheKV(const paddle::Tensor& input_k,
                  const paddle::Tensor& input_v,
                  const paddle::Tensor& cache_kv,
                  const paddle::Tensor& sequence_lengths_shape) {
  std::cout<<">>>>>WriteCacheKV"<<std::endl;
}

PD_BUILD_OP(write_cache_kv)
    .Inputs({"input_k", "input_v", "cache_kv", "sequence_lengths"})
    .Outputs({"cache_kv_out"})
    .SetInplaceMap({{"cache_kv", "cache_kv_out"}})
    .SetKernelFn(PD_KERNEL(WriteCacheKV));

// transpose_remove_padding
std::vector<paddle::Tensor> ApplyTransposeRemovingPadding(const paddle::Tensor& input, 
                                                          const paddle::Tensor& seq_lens, 
                                                          const paddle::Tensor& padding_offset) {
  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  return {paddle::Tensor(out_tensor)};
}

std::vector<std::vector<int64_t>> ApplyTransposeRemovingPaddingInferShape(
        const std::vector<int64_t>& input_shape, 
        const std::vector<int64_t>& seq_lens_shape,
        const std::vector<int64_t>& padding_offset_shape) {
  return {{padding_offset_shape[0], input_shape[1] * input_shape[3]}};
}

std::vector<paddle::DataType> ApplyTransposeRemovingPaddingInferDtype(
        const paddle::DataType& input_dtype, 
        const paddle::DataType& seq_lens_dtype,
        const paddle::DataType& padding_offset_dtype) {
  return {input_dtype};
}

PD_BUILD_OP(transpose_remove_padding)
    .Inputs({"input", "seq_lens", "padding_offset"})
    .Outputs({"fmha_out"})
    .SetKernelFn(PD_KERNEL(ApplyTransposeRemovingPadding))
    .SetInferShapeFn(PD_INFER_SHAPE(ApplyTransposeRemovingPaddingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ApplyTransposeRemovingPaddingInferDtype));

// encode_rotary_qk
void RotaryQK(const paddle::Tensor& q, 
              const paddle::Tensor& kv, 
              const paddle::Tensor& rotary_emb, 
              const paddle::Tensor& seq_lens,
              const int32_t rotary_emb_dims, 
              bool use_neox) {
  std::cout<<">>>>>RotaryQK"<<std::endl;
}

PD_BUILD_OP(encode_rotary_qk)
    .Inputs({"q", "kv", "rotary_emb", "seq_lens"})
    .Outputs({"rotary_q_out", "rotary_kv_out"})
    .Attrs({"rotary_emb_dims: int", "use_neox: bool"})
	.SetInplaceMap({{"q", "rotary_q_out"}, {"kv", "rotary_kv_out"}})
    .SetKernelFn(PD_KERNEL(RotaryQK));

// set_mask_value
std::vector<paddle::Tensor> SetMaskValue(const paddle::Tensor& input_data,
                                         const paddle::Tensor& stop_flags,
                                         const paddle::Tensor& seq_lens) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(seq_lens.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  paddle::Tensor input_data_npu = input_data.copy_to(seq_lens.place(), false);
  auto input_data_tensor = static_cast<const phi::DenseTensor*>(input_data_npu.impl().get());
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

std::vector<std::vector<int64_t>> SetMaskValueInferShape(const std::vector<int64_t>& input_data_shape,
	                                                     const std::vector<int64_t>& stop_flags_shape,
	                                                     const std::vector<int64_t>& seq_lens_shape) {
  return {seq_lens_shape};
}

std::vector<paddle::DataType> SetMaskValueInferDtype(
    const paddle::DataType& input_data_dtype,
    const paddle::DataType& stop_flags_dtype,
    const paddle::DataType& seq_lens_dtype) {
  return {seq_lens_dtype};
}

PD_BUILD_OP(set_mask_value)
    .Inputs({"input_data", "stop_flags", "seq_lens"})
    .Outputs({"sequence_lengths"})
    .SetKernelFn(PD_KERNEL(SetMaskValue))
    .SetInferShapeFn(PD_INFER_SHAPE(SetMaskValueInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SetMaskValueInferDtype));

// set_value_by_flags_and_idx
std::vector<paddle::Tensor> SetValueByFlagsAndIdx(const paddle::Tensor& pre_ids_all,
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

std::vector<std::vector<int64_t>> SetValueByFlagsAndIdxInferShape(const std::vector<int64_t>& pre_ids_all_shape, 
                                                                  const std::vector<int64_t>& pre_ids_now_shape,
                                                                  const std::vector<int64_t>& step_idx_shape, 
                                                                  const std::vector<int64_t>& stop_flags_shape) {
  return {stop_flags_shape};
}

std::vector<paddle::DataType> SetValueByFlagsAndIdxInferDtype(const paddle::DataType& pre_ids_all_dtype,
                                                              const paddle::DataType& pre_ids_now_dtype,
                                                              const paddle::DataType& step_idx_dtype,
                                                              const paddle::DataType& stop_flags_dtype) {
  return {stop_flags_dtype};
}

PD_BUILD_OP(set_value_by_flags_and_idx)
    .Inputs({"pre_ids_all", "pre_ids_now", "step_idx", "stop_flags"})
    .Outputs({"stop_flags_out"})
    .SetKernelFn(PD_KERNEL(SetValueByFlagsAndIdx))
    .SetInferShapeFn(PD_INFER_SHAPE(SetValueByFlagsAndIdxInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SetValueByFlagsAndIdxInferDtype));

// set_stop_value_multi_ends
std::vector<paddle::Tensor> GetStopFlagsMulti(const paddle::Tensor& topk_ids, 
                                              const paddle::Tensor& stop_flags, 
                                              const paddle::Tensor& end_ids, 
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

  auto topk_ids_out = topk_ids.copy_to(topk_ids.place(), false);
  auto topk_ids_out_tensor = static_cast<const phi::DenseTensor*>(topk_ids_out.impl().get());

  int32_t attr_mode = mode;
  NPUAttributeMap attr_input = {{"mode", attr_mode}};
  
  const auto& runner =
      NpuOpRunner("SetStopValueMultiEnds",
                  {*topk_ids_out_tensor, *stop_flags_tensor, *end_ids_tensor},
                  {*topk_ids_out_tensor, *stop_flags_out},
                  attr_input);
  runner.Run(stream);
  
  return {paddle::Tensor(topk_ids_out), paddle::Tensor(stop_flags_out)};
}

std::vector<std::vector<int64_t>> GetStopFlagsMultiInferShape(const std::vector<int64_t>& topk_ids_shape, 
                                                              const std::vector<int64_t>& stop_flags_shape, 
															  const std::vector<int64_t>& end_ids_shape) {
  return {topk_ids_shape, stop_flags_shape};
}

std::vector<paddle::DataType> GetStopFlagsMultiInferDtype(const paddle::DataType& topk_ids_dtype, 
                                                          const paddle::DataType& stop_flags_dtype, 
                                                          const paddle::DataType& end_ids_dtype) {
  return {topk_ids_dtype, stop_flags_dtype};
}

PD_BUILD_OP(set_stop_value_multi_ends)
    .Inputs({"topk_ids", "stop_flags", "end_ids"})
    .Outputs({"topk_ids_out", "stop_flags_out"})
    .Attrs({"mode: int64_t"})
    .SetKernelFn(PD_KERNEL(GetStopFlagsMulti))
    .SetInferShapeFn(PD_INFER_SHAPE(GetStopFlagsMultiInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetStopFlagsMultiInferDtype));

bool is_in_end(const int64_t id, const int64_t *end_ids, int length) {
    bool flag = false;
    for (int i = 0; i < length; i++) {
        if (id == end_ids[i]) {
            return true;
        }
    }
    return flag;
}

// rebuild_padding
std::vector<paddle::Tensor> RebuildPadding(const paddle::Tensor& tmp_out, 
                                           const paddle::Tensor& padding_offset, 
                                           const paddle::Tensor& seq_lens,
                                           const paddle::Tensor& input_ids) {
  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  return {paddle::Tensor(out_tensor)};
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

// get_padding_offset
std::vector<paddle::Tensor> GetPaddingOffset(const paddle::Tensor& input_ids,
                                             const paddle::Tensor& cum_offsets,
                                             const paddle::Tensor& token_num,
                                             const paddle::Tensor& seq_len) {

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  return {paddle::Tensor(out_tensor), paddle::Tensor(out_tensor), paddle::Tensor(out_tensor)};
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
std::vector<paddle::Tensor> TokenPenaltyMultiScores(const paddle::Tensor& pre_ids,
                                                    const paddle::Tensor& logits,
                                                    const paddle::Tensor& penalty_scores,
                                                    const paddle::Tensor& frequency_scores,
                                                    const paddle::Tensor& presence_scores,
                                                    const paddle::Tensor& cur_len,
                                                    const paddle::Tensor& min_len,
                                                    const paddle::Tensor& eos_token_id) {
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
  auto repeat_times = paddle::full(logits.shape(), 0, paddle::DataType::INT32, pre_ids.place()); 
  auto repeat_times_tensor = static_cast<const phi::DenseTensor*>(repeat_times.impl().get());
  
  std::shared_ptr<phi::DenseTensor> logits_out =
      std::make_shared<phi::DenseTensor>();
  logits_out->Resize(logits_tensor->dims());
  dev_ctx->Alloc(logits_out.get(), logits_tensor->dtype());
  
  std::vector<phi::DenseTensor> inputs = {*pre_ids_tensor,
                                          *logits_tensor,
                                          *repeat_times_tensor,
                                          *penalty_scores_tensor,
                                          *frequency_scores_tensor,
                                          *presence_scores_tensor,
                                          *cur_len_tensor,
                                          *min_len_tensor,
                                          *eos_token_id_tensor};
  
  std::vector<phi::DenseTensor> outputs = {*logits_out};
  
  const auto& runner =
      NpuOpRunner("TokenPenaltyMultiScores", inputs, outputs);
  runner.Run(stream);
  
  return {paddle::Tensor(logits_out)};
}

std::vector<std::vector<int64_t>> TokenPenaltyMultiScoresInferShape(const std::vector<int64_t>& pre_ids_shape,
                                                                    const std::vector<int64_t>& logits_shape,
                                                                    const std::vector<int64_t>& penalty_scores_shape,
                                                                    const std::vector<int64_t>& frequency_scores_shape,
                                                                    const std::vector<int64_t>& presence_scores_shape,
                                                                    const std::vector<int64_t>& cur_len_shape,
                                                                    const std::vector<int64_t>& min_len_shape,
                                                                    const std::vector<int64_t>& eos_token_id_shape) {
  return {logits_shape};
}

std::vector<paddle::DataType> TokenPenaltyMultiScoresInferDtype(const paddle::DataType& pre_ids_dtype,
                                                                const paddle::DataType& logits_dtype,
                                                                const paddle::DataType& penalty_scores_dtype,
                                                                const paddle::DataType& frequency_scores_dtype,
                                                                const paddle::DataType& presence_scores_dtype,
                                                                const paddle::DataType& cur_len_dtype,
                                                                const paddle::DataType& min_len_dtype,
                                                                const paddle::DataType& eos_token_id_dtype) {
  return {logits_dtype};
}

PD_BUILD_OP(get_token_penalty_multi_scores)
    .Inputs({"pre_ids", "logits", "penalty_scores", "frequency_scores", "presence_scores", "cur_len", "min_len", "eos_token_id"})
    .Outputs({"logits_out"})
    .SetKernelFn(PD_KERNEL(TokenPenaltyMultiScores))
    .SetInferShapeFn(PD_INFER_SHAPE(TokenPenaltyMultiScoresInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TokenPenaltyMultiScoresInferDtype));

// top_p_sampling
std::vector<paddle::Tensor> TopPSampling(const paddle::Tensor& x, 
                                         const paddle::Tensor& top_ps, 
                                         int random_seed) {
    std::vector<int64_t> shape = x.shape();
  int bs = shape[0];
  auto topp_ids = paddle::full({bs, 1}, 1, paddle::DataType::INT64, x.place());

  auto dev_ctx = static_cast<const phi::CustomContext*>(
        paddle::experimental::DeviceContextPool::Instance().Get(top_ps.place()));            
  std::cout<<">>>>>TopPSamplingOp"<<std::endl;
  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(phi::make_ddim({bs, 1}));
  dev_ctx->Alloc(out_tensor.get(), paddle::DataType::FLOAT32);

  return {paddle::Tensor(out_tensor), topp_ids};
}

std::vector<std::vector<int64_t>> TopPSamplingInferShape(const std::vector<int64_t>& x_shape,
                                                         const std::vector<int64_t>& top_ps_shape) {
  std::vector<int64_t> out_probs_shape = {x_shape[0], 1};                                                          
  std::vector<int64_t> out_ids_shape = {x_shape[0], 1};
  return {out_probs_shape, out_ids_shape};
}

std::vector<paddle::DataType> TopPSamplingInferDtype(const paddle::DataType& x_dtype,
                                                     const paddle::DataType& top_ps_dtype) {
  return {x_dtype, paddle::DataType::INT64};
}

// PD_BUILD_OP(top_p_sampling)
//     .Inputs({"x", "top_ps"})
//     .Outputs({"topp_probs", "topp_ids"})
//     .Attrs({"random_seed: int"})
//     .SetKernelFn(PD_KERNEL(TopPSampling))
//     .SetInferShapeFn(PD_INFER_SHAPE(TopPSamplingInferShape))
//     .SetInferDtypeFn(PD_INFER_DTYPE(TopPSamplingInferDtype));

constexpr char kSEP = '/';

std::string DirName(const std::string &filepath) {
  auto pos = filepath.rfind(kSEP);
  if (pos == std::string::npos) {
    return "";
  }
  return filepath.substr(0, pos);
}

bool FileExists(const std::string &filepath) {
  struct stat buffer;
  return (stat(filepath.c_str(), &buffer) == 0);
}

void MkDir(const char *path) {
  std::string path_error(path);
  path_error += " mkdir failed!";
  if (mkdir(path, 0755)) {
    if (errno != EEXIST) {
      throw std::runtime_error(path_error);
    }
  }
}

void MkDirRecursively(const char *fullpath) {
  if (*fullpath == '\0') return;  // empty string
  if (FileExists(fullpath)) return;
  MkDirRecursively(DirName(fullpath).c_str());
  MkDir(fullpath);
}


template<typename data_t>
void saveToFile(std::ostream & os, const void* x_data, std::vector<int64_t> shape, int64_t x_numel, const char type_id) {
  // 1.type
  os.write(reinterpret_cast<const char *>(&type_id),sizeof(type_id));
  // 2.data
  uint64_t size = x_numel * sizeof(data_t);
  os.write(static_cast<const char*>(x_data),static_cast<std::streamsize>(size));

}

template<typename data_t>
void save_with_output_kernel(const paddle::Tensor& x,
                             const paddle::Tensor& batch_idx,
                             const paddle::Tensor& step_idx,
                             std::string file_path,
                             int64_t rank_id,
                             char type_id) {
  std::vector<int64_t> x_shape = x.shape();

  if(rank_id >= 0) {
      file_path += "_rank_" + std::to_string(rank_id);
  }

  int batch_idx_data = -1, step_idx_data = -1;

  if(batch_idx.is_custom_device()) {
    paddle::Tensor batch_idx_cpu = batch_idx.copy_to(paddle::CPUPlace(), true);
    batch_idx_data = batch_idx_cpu.data<int32_t>()[0];
  } else {
    batch_idx_data = batch_idx.data<int32_t>()[0];
  }
  if(step_idx.is_custom_device()) {
    paddle::Tensor step_idx_cpu = step_idx.copy_to(paddle::CPUPlace(), true);
    step_idx_data = step_idx_cpu.data<int64_t>()[0];
  } else {
    step_idx_data = step_idx.data<int64_t>()[0];
  }
  auto x_data = x.data<data_t>();

  if(batch_idx_data >= 0) {
    file_path += "_batch_" + std::to_string(batch_idx_data);
  }
  if(step_idx_data >= 0) {
    file_path += "_step_" + std::to_string(step_idx_data);
  }
  MkDirRecursively(DirName(file_path).c_str());
  std::ofstream fout(file_path, std::ios::binary);
  fout.write("0",1);
  saveToFile<data_t>(fout, x_data, x_shape, x.numel(),type_id);
  fout.seekp(std::ios::beg);
  fout.write("1",1);
  fout.close();

}

void print_shape(const paddle::Tensor& tmp, char *tmp_str){
    std::vector<int64_t> shape = tmp.shape();
    printf("%s's shape: \n", tmp_str);
    for(int i=0; i < shape.size(); i++) {
        printf("%d ", (int)shape[i]);
    }
    printf("\n");
}

std::vector<paddle::Tensor> SaveWithOutputForward(const paddle::Tensor& x,
                                                  const paddle::Tensor& batch_idx,
                                                  const paddle::Tensor& step_idx,
                                                  std::string file_path,
                                                  int64_t rank_id) {
    auto out = x.copy_to(paddle::CPUPlace(), true);
    switch(x.type()) {
      case paddle::DataType::FLOAT32:
         save_with_output_kernel<float>(out, batch_idx, step_idx, file_path, rank_id, '0');
         break;
      case paddle::DataType::INT64:
        save_with_output_kernel<int64_t>(out, batch_idx, step_idx, file_path, rank_id,'1');
         break;
      case paddle::DataType::INT32:
        save_with_output_kernel<int32_t>(out, batch_idx, step_idx, file_path, rank_id, '2');
         break;
      default:
        PD_THROW("function SaveWithOutputForward is not implemented for data type");
    }
   return {out};
}

std::vector<std::vector<int64_t>> SaveWithOutputInferShape(const std::vector<int64_t>& x_shape,
                                                           const std::vector<int64_t>& batch_idx_shape,
                                                           const std::vector<int64_t>& step_idx_shape) {
    return {x_shape};
}

std::vector<paddle::DataType> SaveWithOutputInferDtype(const paddle::DataType& x_dtype,
                                                       const paddle::DataType& batch_idx_dtype,
                                                       const paddle::DataType& step_idx_dtype) {
    return {x_dtype};
}

PD_BUILD_OP(save_with_output)
    .Inputs({"x", "batch_idx", "step_idx"})
    .Attrs({"file_path: std::string",
            "rank_id: int64_t"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(SaveWithOutputForward))
    .SetInferShapeFn(PD_INFER_SHAPE(SaveWithOutputInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SaveWithOutputInferDtype));
