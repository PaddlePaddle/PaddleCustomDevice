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
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>

#include "kernels/funcs/npu_op_runner.h"
#include "paddle/extension.h"
#include "kernels/funcs/format_utils.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

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
              const paddle::Tensor& cos_table, 
              const paddle::Tensor& sin_table, 
              const paddle::Tensor& seq_lens,
              const int32_t rotary_emb_dims, 
              bool use_neox) {
  std::cout<<">>>>>RotaryQK"<<std::endl;
}

PD_BUILD_OP(encode_rotary_qk)
    .Inputs({"q", "kv", "cos_table", "sin_table", "seq_lens"})
    .Outputs({"rotary_q_out", "rotary_kv_out"})
    .Attrs({"rotary_emb_dims: int", "use_neox: bool"})
	.SetInplaceMap({{"q", "rotary_q_out"}, {"kv", "rotary_kv_out"}})
    .SetKernelFn(PD_KERNEL(RotaryQK));

// masked_multihead_attention_npu
std::vector<paddle::Tensor> MaskedMultiheadAttentionNpu(const paddle::Tensor& x, 
              const paddle::Tensor& cache_kv, 
              const paddle::Tensor& src_mask,
              const paddle::Tensor& sequence_lengths,
              const paddle::Tensor& cos_table,
              const paddle::Tensor& sin_table,
              const int32_t rotary_emb_dims, 
              bool use_neox_rotary_style) {
  std::cout<<">>>>>MaskedMultiheadAttentionNpu"<<std::endl;
  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  return {paddle::Tensor(out_tensor)};
}

std::vector<std::vector<int64_t>> MaskedMultiheadAttentionNpuInferShape(const std::vector<int64_t>& x_shape, 
              const std::vector<int64_t>& cache_kv_shape, 
              const std::vector<int64_t>& src_mask_shape,
              const std::vector<int64_t>& sequence_lengths_shape,
              const std::vector<int64_t>& cos_table_shape,
              const std::vector<int64_t>& sin_table_shape) {
  return {{x_shape[0], 1024}};
}

std::vector<paddle::DataType> MaskedMultiheadAttentionNpuInferDtype(const paddle::DataType& x_dtype, 
              const paddle::DataType& cache_kv_dtype, 
              const paddle::DataType& src_mask_dtype,
              const paddle::DataType& sequence_lengths_dtype,
              const paddle::DataType& cos_table_dtype,
              const paddle::DataType& sin_table_dtype) {
  return {src_mask_dtype};
}

PD_BUILD_OP(masked_multihead_attention_npu)
    .Inputs({"x", "cache_kv", "src_mask", "sequence_lengths", "cos_table", "sin_table"})
    .Outputs({"out"})
    .Attrs({"rotary_emb_dims: int", "use_neox_rotary_style: bool"})
    .SetKernelFn(PD_KERNEL(MaskedMultiheadAttentionNpu))
    .SetInferShapeFn(PD_INFER_SHAPE(MaskedMultiheadAttentionNpuInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MaskedMultiheadAttentionNpuInferDtype));

// set_mask_value
std::vector<paddle::Tensor> SetMaskValue(const paddle::Tensor& input_data,
                                         const paddle::Tensor& stop_flags,
                                         const paddle::Tensor& seq_lens) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(seq_lens.place()));
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

  // if(batch_idx_data >= 0) {
  //   file_path += "_batch_" + std::to_string(batch_idx_data);
  // }
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

// save_with_output_delay
std::vector<paddle::Tensor> SaveWithOutputDelay(const paddle::Tensor& x,
                                                const paddle::Tensor& batch_idx,
                                                const paddle::Tensor& step_idx,
                                                const paddle::Tensor& no_use,
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
   return {x, no_use};
}

std::vector<std::vector<int64_t>> SaveWithOutputDelayInferShape(const std::vector<int64_t>& x_shape,
                                                                const std::vector<int64_t>& batch_idx_shape,
                                                                const std::vector<int64_t>& step_idx_shape,
                                                                const std::vector<int64_t>& no_use_shape) {
    return {x_shape, no_use_shape};
}

std::vector<paddle::DataType> SaveWithOutputDelayInferDtype(const paddle::DataType& x_dtype,
                                                            const paddle::DataType& batch_idx_dtype,
                                                            const paddle::DataType& step_idx_dtype,
                                                            const paddle::DataType& no_use_dtype) {
    return {x_dtype, no_use_dtype};
}

PD_BUILD_OP(save_with_output_delay)
    .Inputs({"x", "batch_idx", "step_idx", "no_use"})
    .Attrs({"file_path: std::string",
            "rank_id: int64_t"})
    .Outputs({"out", "no_use_out"})
    .SetKernelFn(PD_KERNEL(SaveWithOutputDelay))
    .SetInferShapeFn(PD_INFER_SHAPE(SaveWithOutputDelayInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SaveWithOutputDelayInferDtype));

std::vector<paddle::Tensor> GetPaddingOffsetV2(const paddle::Tensor& input_ids,
                                               const paddle::Tensor& cum_offsets,
                                               const paddle::Tensor& token_num,
                                               const paddle::Tensor& seq_len) {
    auto dev_ctx = static_cast<const phi::CustomContext*>(
        paddle::experimental::DeviceContextPool::Instance().Get(input_ids.place()));
    auto stream = static_cast<aclrtStream>(dev_ctx->stream());
    std::vector<int64_t> input_ids_shape = input_ids.shape();
    const int bsz = seq_len.shape()[0];
    const int seq_length = input_ids_shape[1];
    auto cum_offsets_out = cum_offsets.copy_to(cum_offsets.place(), true);
    auto cpu_token_num = token_num.copy_to(paddle::CPUPlace(), true);

    const int token_num_data = cpu_token_num.data<int64_t>()[0];
    std::cout << "get_padding_offset_v2  token_num_data:" << token_num_data << std::endl;
    auto x_remove_padding = paddle::full({bsz * seq_length}, 0, paddle::DataType::INT64, input_ids.place());
    auto padding_offset = paddle::full({bsz * seq_length}, 0, paddle::DataType::INT32, input_ids.place());
    auto cu_seqlens_q = paddle::full({bsz + 1}, 0, paddle::DataType::INT32, input_ids.place());
    auto cu_seqlens_k = paddle::full({bsz + 1}, 0, paddle::DataType::INT32, input_ids.place());


    auto input_ids_tensor = static_cast<const phi::DenseTensor*>(input_ids.impl().get());
    auto cum_offsets_tensor = static_cast<const phi::DenseTensor*>(cum_offsets.impl().get());
    auto token_num_tensor = static_cast<phi::DenseTensor*>(token_num.impl().get());
    auto seq_len_tensor = static_cast<const phi::DenseTensor*>(seq_len.impl().get());
    token_num_tensor->Resize(phi::make_ddim({1, 1}));

    auto x_remove_padding_out_tensor = static_cast<phi::DenseTensor*>(x_remove_padding.impl().get());
    auto cum_offsets_out_tensor = static_cast<const phi::DenseTensor*>(cum_offsets_out.impl().get());
    auto padding_offset_tensor = static_cast<phi::DenseTensor*>(padding_offset.impl().get());
    auto cu_seqlens_q_tensor = static_cast<const phi::DenseTensor*>(cu_seqlens_q.impl().get());
    auto cu_seqlens_k_tensor = static_cast<const phi::DenseTensor*>(cu_seqlens_k.impl().get());

    std::vector<phi::DenseTensor> inputs = {*input_ids_tensor,
                                            *cum_offsets_tensor,
                                            *token_num_tensor,
                                            *seq_len_tensor,};

    std::vector<phi::DenseTensor> outputs = {*x_remove_padding_out_tensor,
                                             *cum_offsets_out_tensor,
                                             *padding_offset_tensor,
                                             *cu_seqlens_q_tensor,
                                             *cu_seqlens_k_tensor};

    const auto& runner =
        NpuOpRunner("GetPaddingOffset", inputs, outputs);
    runner.Run(stream);

    x_remove_padding_out_tensor->Resize(phi::make_ddim({1, token_num_data})); // 补一个1，作为bs维

    return {x_remove_padding, cum_offsets_out, padding_offset, cu_seqlens_q, cu_seqlens_k}; // , enc_token_num, dec_token_num};
}

std::vector<std::vector<int64_t>> GetPaddingOffsetV2InferShape(const std::vector<int64_t>& input_ids_shape,
                                                             const std::vector<int64_t>& cum_offsets_shape,
                                                             const std::vector<int64_t>& token_num_shape,
                                                             const std::vector<int64_t>& seq_len_shape) {
    int64_t bsz = seq_len_shape[0];
    int64_t seq_len = input_ids_shape[1];
    return {{-1}, {bsz}, {-1}, {bsz + 1}, {bsz + 1}};
}

std::vector<paddle::DataType> GetPaddingOffsetV2InferDtype(const paddle::DataType& input_ids_dtype,
                                                         const paddle::DataType& cum_offsets_dtype,
                                                         const paddle::DataType& token_num_dtype,
                                                         const paddle::DataType& seq_len_dtype) {
    return {input_ids_dtype, seq_len_dtype, seq_len_dtype, seq_len_dtype, seq_len_dtype};
}

PD_BUILD_OP(get_padding_offset_v2)
    .Inputs({"input_ids", "cum_offsets", "token_num", "seq_len"})
    .Outputs({"x_remove_padding", "cum_offsets_out", "padding_offset", "cu_seqlens_q", "cu_seqlens_k"})
    .SetKernelFn(PD_KERNEL(GetPaddingOffsetV2))
    .SetInferShapeFn(PD_INFER_SHAPE(GetPaddingOffsetV2InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetPaddingOffsetV2InferDtype));

void SetValueByFlagsAndIdxV2(const paddle::Tensor& pre_ids_all, 
                             const paddle::Tensor& input_ids,
                             const paddle::Tensor& seq_lens_this_time,
                             const paddle::Tensor& seq_lens_encoder,
                             const paddle::Tensor& seq_lens_decoder,
                             const paddle::Tensor& step_idx, 
                             const paddle::Tensor& stop_flags) {
    std::cout << "set_value_by_flags_and_idx_v2" << std::endl;
    auto dev_ctx = static_cast<const phi::CustomContext*>(
        paddle::experimental::DeviceContextPool::Instance().Get(pre_ids_all.place()));
    auto stream = static_cast<aclrtStream>(dev_ctx->stream());

    auto pre_ids_all_tensor = static_cast<const phi::DenseTensor*>(pre_ids_all.impl().get());
    auto input_ids_tensor = static_cast<const phi::DenseTensor*>(input_ids.impl().get());
    auto seq_lens_this_time_tensor = static_cast<const phi::DenseTensor*>(seq_lens_this_time.impl().get()); 
    auto seq_lens_encoder_tensor = static_cast<const phi::DenseTensor*>(seq_lens_encoder.impl().get());
    auto seq_lens_decoder_tensor = static_cast<const phi::DenseTensor*>(seq_lens_decoder.impl().get()); 
    auto step_idx_tensor = static_cast<const phi::DenseTensor*>(step_idx.impl().get());
    auto stop_flags_tensor = static_cast<const phi::DenseTensor*>(stop_flags.impl().get()); 

    const auto& runner =
        NpuOpRunner("SetValueByFlagsAndIdxV2",
                    {*pre_ids_all_tensor, *input_ids_tensor, *seq_lens_this_time_tensor, *seq_lens_encoder_tensor, *seq_lens_decoder_tensor, *step_idx_tensor, *stop_flags_tensor},
                    {*pre_ids_all_tensor},
                    {});
    runner.Run(stream);
}

PD_BUILD_OP(set_value_by_flags_and_idx_v2)
    .Inputs({"pre_ids_all", "input_ids", "seq_lens_this_time", "seq_lens_encoder", "seq_lens_decoder", "step_idx", "stop_flags"})
    .Outputs({"pre_ids_all_out"})
    .SetInplaceMap({{"pre_ids_all", "pre_ids_all_out"}})
    .SetKernelFn(PD_KERNEL(SetValueByFlagsAndIdxV2));

void TokenPenaltyMultiScoresV2(const paddle::Tensor& pre_ids,
                             const paddle::Tensor& logits,
                             const paddle::Tensor& penalty_scores,
                             const paddle::Tensor& frequency_scores,
                             const paddle::Tensor& presence_scores,
                             const paddle::Tensor& temperatures,
                             const paddle::Tensor& bad_tokens,
                             const paddle::Tensor& cur_len,
                             const paddle::Tensor& min_len,
                             const paddle::Tensor& eos_token_id) {
    std::cout << "get_token_penalty_multi_scores_v2" << std::endl;

    auto dev_ctx = static_cast<const phi::CustomContext*>(
        paddle::experimental::DeviceContextPool::Instance().Get(pre_ids.place()));
    auto stream = static_cast<aclrtStream>(dev_ctx->stream());

    auto pre_ids_tensor = static_cast<const phi::DenseTensor*>(pre_ids.impl().get());
    auto logits_tensor = static_cast<const phi::DenseTensor*>(logits.impl().get());

    auto repeat_times = paddle::full(logits.shape(), 0, paddle::DataType::INT32, pre_ids.place());
    auto repeat_times_tensor = static_cast<const phi::DenseTensor*>(repeat_times.impl().get());

    auto penalty_scores_tensor = static_cast<const phi::DenseTensor*>(penalty_scores.impl().get()); 
    auto frequency_scores_tensor = static_cast<const phi::DenseTensor*>(frequency_scores.impl().get());
    auto presence_scores_tensor = static_cast<const phi::DenseTensor*>(presence_scores.impl().get()); 
    auto temperatures_tensor = static_cast<const phi::DenseTensor*>(temperatures.impl().get());
    auto bad_tokens_tensor = static_cast<const phi::DenseTensor*>(bad_tokens.impl().get()); 
    auto cur_len_tensor = static_cast<const phi::DenseTensor*>(cur_len.impl().get()); 
    auto min_len_tensor = static_cast<const phi::DenseTensor*>(min_len.impl().get());
    auto eos_token_id_tensor = static_cast<const phi::DenseTensor*>(eos_token_id.impl().get()); 

    std::shared_ptr<phi::DenseTensor> logits_out = std::make_shared<phi::DenseTensor>();
    logits_out->Resize(logits_tensor->dims());
    dev_ctx->Alloc(logits_out.get(), logits_tensor->dtype());

    const auto& runner =
        NpuOpRunner("TokenPenaltyMultiScoresV2",
                    {*pre_ids_tensor, *logits_tensor, *repeat_times_tensor, *penalty_scores_tensor, *frequency_scores_tensor, 
                        *presence_scores_tensor, *temperatures_tensor, *bad_tokens_tensor,
                        *cur_len_tensor, *min_len_tensor, *eos_token_id_tensor},
                    {*logits_out},
                    {});
    runner.Run(stream);
}

PD_BUILD_OP(get_token_penalty_multi_scores_v2)
    .Inputs({"pre_ids", 
             "logits", 
             "penalty_scores", 
             "frequency_scores", 
             "presence_scores", 
             "temperatures", 
             "bad_tokens", 
             "cur_len", 
             "min_len", 
             "eos_token_id"})
    .Outputs({"logits_out"})
    .SetInplaceMap({{"logits", "logits_out"}})
    .SetKernelFn(PD_KERNEL(TokenPenaltyMultiScoresV2));

void GetStopFlagsMultiV2(const paddle::Tensor& topk_ids, 
                         const paddle::Tensor& stop_flags, 
                         const paddle::Tensor& seq_lens, 
                         const paddle::Tensor& end_ids,
                         const paddle::Tensor& next_tokens) {
    std::cout << "set_stop_value_multi_ends_v2" << std::endl;
    auto dev_ctx = static_cast<const phi::CustomContext*>(
        paddle::experimental::DeviceContextPool::Instance().Get(topk_ids.place()));
    auto stream = static_cast<aclrtStream>(dev_ctx->stream());

    auto topk_ids_tensor = static_cast<const phi::DenseTensor*>(topk_ids.impl().get());
    auto stop_flags_tensor = static_cast<const phi::DenseTensor*>(stop_flags.impl().get());
    auto seq_lens_tensor = static_cast<const phi::DenseTensor*>(seq_lens.impl().get()); 
    auto end_ids_tensor = static_cast<const phi::DenseTensor*>(end_ids.impl().get());
    auto next_tokens_tensor = static_cast<const phi::DenseTensor*>(next_tokens.impl().get()); 


    std::shared_ptr<phi::DenseTensor> topk_ids_out = std::make_shared<phi::DenseTensor>();
    topk_ids_out->Resize(topk_ids_tensor->dims());
    dev_ctx->Alloc(topk_ids_out.get(), topk_ids_tensor->dtype());

    std::shared_ptr<phi::DenseTensor> stop_flags_out = std::make_shared<phi::DenseTensor>();
    stop_flags_out->Resize(stop_flags_tensor->dims());
    dev_ctx->Alloc(stop_flags_out.get(), stop_flags_tensor->dtype());

    std::shared_ptr<phi::DenseTensor> next_tokens_out = std::make_shared<phi::DenseTensor>();
    next_tokens_out->Resize(next_tokens_tensor->dims());
    dev_ctx->Alloc(next_tokens_out.get(), next_tokens_tensor->dtype());

    const auto& runner =
        NpuOpRunner("SetStopValueMultiEndsV2",
                    {*topk_ids_tensor, *stop_flags_tensor, *seq_lens_tensor, *end_ids_tensor, *next_tokens_tensor},
                    {*topk_ids_out, *stop_flags_out, *next_tokens_out},
                    {});
    runner.Run(stream);
}

PD_BUILD_OP(set_stop_value_multi_ends_v2)
    .Inputs({"topk_ids", "stop_flags", "seq_lens", "end_ids", "next_tokens"})
    .Outputs({"topk_ids_out", "stop_flags_out", "next_tokens_out"})
    .SetInplaceMap({{"topk_ids", "topk_ids_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"next_tokens", "next_tokens_out"}})
    .SetKernelFn(PD_KERNEL(GetStopFlagsMultiV2));

#define MAX_BSZ 512

struct msgdata {
    long mtype;
    int mtext[MAX_BSZ + 2];   // stop_flag, bsz, tokens
};

void SaveOutMmsg(const paddle::Tensor& x,
                 const paddle::Tensor& not_need_stop,
                 int64_t rank_id) {
    if (rank_id > 0) return;
    auto x_cpu = x.copy_to(paddle::CPUPlace(), true);
    int64_t *x_data = x_cpu.data<int64_t>();
    static struct msgdata msg_sed;
    static key_t key = ftok("./", 1);
    static int msgid = msgget(key, IPC_CREAT | 0666);

    msg_sed.mtype = 1;
    bool not_need_stop_data = not_need_stop.data<bool>()[0];
    msg_sed.mtext[0] = not_need_stop_data ? 1 : -1;
    int bsz = x.shape()[0];
    msg_sed.mtext[1] = bsz;
    for (int i = 2; i < bsz + 2; i++) {
        msg_sed.mtext[i] = (int)x_data[i - 2];
    }
    if ((msgsnd(msgid, &msg_sed, (MAX_BSZ + 2) * 4, 0)) == -1) {
    //   printf("full msg buffer\n");
    }
    return;
}


PD_BUILD_OP(save_output)
    .Inputs({"x", "not_need_stop"})
    .Attrs({"rank_id: int64_t"})
    .Outputs({"x_out"})
    .SetInplaceMap({{"x", "x_out"}})
    .SetKernelFn(PD_KERNEL(SaveOutMmsg));

void GetOutput(const paddle::Tensor& x,
               int64_t rank_id,
               bool wait_flag) {
  if (rank_id > 0) return;

  static struct msgdata msg_rcv;

  static key_t key = ftok("./", 1);

  static int msgid = msgget(key, IPC_CREAT | 0666);

  int64_t *out_data = const_cast<int64_t*>(x.data<int64_t>());
  int ret = -1;
  if (!wait_flag) {
    ret = msgrcv(msgid, &msg_rcv, (MAX_BSZ + 2) * 4, 0, IPC_NOWAIT);
  } else {
    ret = msgrcv(msgid, &msg_rcv, (MAX_BSZ + 2) * 4, 0, 0);
  }
  if(ret == -1)
	{
    // read none
    out_data[0] = -2;
    out_data[1] = 0;
		return;
	}

  int bsz = msg_rcv.mtext[1];

  for (int64_t i = 0; i < bsz + 2; i++) {
    out_data[i] = (int64_t)msg_rcv.mtext[i];
  }
  return;
}

PD_BUILD_OP(get_output)
    .Inputs({"x"})
    .Attrs({"rank_id: int64_t",
            "wait_flag: bool"})
    .Outputs({"x_out"})
    .SetInplaceMap({{"x", "x_out"}})
    .SetKernelFn(PD_KERNEL(GetOutput));

void UpdateInputes(const paddle::Tensor& stop_flags,
                   const paddle::Tensor& not_need_stop, // cpu
                   const paddle::Tensor& seq_lens_this_time,
                   const paddle::Tensor& seq_lens_encoder,
                   const paddle::Tensor& seq_lens_decoder,
                   const paddle::Tensor& input_ids,
                   const paddle::Tensor& stop_nums,
                   const paddle::Tensor& next_tokens,
                   const paddle::Tensor& is_block_step) {
    std::cout << "update_inputs" << std::endl;
    auto dev_ctx = static_cast<const phi::CustomContext*>(
        paddle::experimental::DeviceContextPool::Instance().Get(stop_flags.place()));
    auto stream = static_cast<aclrtStream>(dev_ctx->stream());

    auto not_need_stop_npu = not_need_stop.copy_to(stop_flags.place(), false);

    auto stop_flags_tensor = static_cast<const phi::DenseTensor*>(stop_flags.impl().get());
    auto not_need_stop_tensor = static_cast<const phi::DenseTensor*>(not_need_stop_npu.impl().get());
    auto seq_lens_this_time_tensor = static_cast<const phi::DenseTensor*>(seq_lens_this_time.impl().get()); 
    auto seq_lens_encoder_tensor = static_cast<const phi::DenseTensor*>(seq_lens_encoder.impl().get());
    auto seq_lens_decoder_tensor = static_cast<const phi::DenseTensor*>(seq_lens_decoder.impl().get()); 
    auto input_ids_tensor = static_cast<const phi::DenseTensor*>(input_ids.impl().get());
    auto stop_nums_tensor = static_cast<const phi::DenseTensor*>(stop_nums.impl().get()); 
    auto next_tokens_tensor = static_cast<const phi::DenseTensor*>(next_tokens.impl().get());
    auto is_block_step_tensor = static_cast<const phi::DenseTensor*>(is_block_step.impl().get()); 

    const auto& runner =
        NpuOpRunner("UpdateInputs",
                    {*stop_flags_tensor, *not_need_stop_tensor, *seq_lens_this_time_tensor, *seq_lens_encoder_tensor, *seq_lens_decoder_tensor,
                        *input_ids_tensor, *stop_nums_tensor, *next_tokens_tensor, *is_block_step_tensor},
                    {*not_need_stop_tensor, *seq_lens_this_time_tensor, *seq_lens_encoder_tensor, *seq_lens_decoder_tensor, *input_ids_tensor},
                    {});
    runner.Run(stream);

    auto not_need_stop_cpu = not_need_stop_npu.copy_to(not_need_stop.place(), true);
    bool *not_need_stop_data = const_cast<bool*>(not_need_stop.data<bool>());
    not_need_stop_data[0] = not_need_stop_cpu.data<bool>()[0];
}

PD_BUILD_OP(update_inputs)
    .Inputs({"stop_flags", 
             "not_need_stop", 
             "seq_lens_this_time", 
             "seq_lens_encoder", 
             "seq_lens_decoder",
             "input_ids",
             "stop_nums",
             "next_tokens",
             "is_block_step"})
    .Outputs({"not_need_stop_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "input_ids_out"})
    .SetInplaceMap({{"not_need_stop", "not_need_stop_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"input_ids", "input_ids_out"}})
    .SetKernelFn(PD_KERNEL(UpdateInputes));

void GetMaxLen(const paddle::Tensor& seq_lens_encoder, const paddle::Tensor& seq_lens_decoder) {
    std::cout << "get_max_len" << std::endl;
    auto dev_ctx = static_cast<const phi::CustomContext*>(
        paddle::experimental::DeviceContextPool::Instance().Get(seq_lens_encoder.place()));
    auto stream = static_cast<aclrtStream>(dev_ctx->stream());

    auto seq_lens_encoder_tensor = static_cast<const phi::DenseTensor*>(seq_lens_encoder.impl().get());
    auto seq_lens_decoder_tensor = static_cast<const phi::DenseTensor*>(seq_lens_decoder.impl().get());

    std::shared_ptr<phi::DenseTensor> max_len_encoder = std::make_shared<phi::DenseTensor>();
    max_len_encoder->Resize({1});
    dev_ctx->Alloc(max_len_encoder.get(), paddle::DataType::INT32);

    std::shared_ptr<phi::DenseTensor> max_len_decoder = std::make_shared<phi::DenseTensor>();
    max_len_decoder->Resize({1});
    dev_ctx->Alloc(max_len_decoder.get(), paddle::DataType::INT32);

    const auto& runner =
        NpuOpRunner("GetMaxLen",
                    {*seq_lens_encoder_tensor, *seq_lens_decoder_tensor},
                    {*max_len_encoder, *max_len_decoder},
                    {});
    runner.Run(stream);

    int max_len_encoder_data = paddle::Tensor(max_len_encoder).copy_to(paddle::CPUPlace(), true).data<int>()[0];
    int max_len_decoder_data = paddle::Tensor(max_len_decoder).copy_to(paddle::CPUPlace(), true).data<int>()[0];

    std::ofstream outfile;
    outfile.open("max_len.txt", std::ios::out);
    outfile << max_len_encoder_data << "\n" << max_len_decoder_data;
    outfile.close();
}

PD_BUILD_OP(get_max_len)
    .Inputs({"seq_lens_encoder", "seq_lens_decoder"})
    .Outputs({"seq_lens_encoder_out", "seq_lens_decoder_out"})
    .SetInplaceMap({{"seq_lens_encoder", "seq_lens_encoder_out"}, {"seq_lens_decoder", "seq_lens_decoder_out"}})
    .SetKernelFn(PD_KERNEL(GetMaxLen));

std::vector<paddle::Tensor> DequantInt8(const paddle::Tensor& input,
                                        const paddle::Tensor& out_scale,
                                        std::string dtype
                                        ) {
    std::cout << "dequant_int8" << std::endl;
    return {input};
}

std::vector<std::vector<int64_t>> DequantInt8Shape(const std::vector<int64_t>& input_shape) {
    return {input_shape};
}

std::vector<paddle::DataType> DequantInt8Dtype(const paddle::DataType& input_dtype, const paddle::DataType& out_scale_dtype, std::string dtype) {
    paddle::DataType data_type;
    if (dtype == "float32")
        data_type = paddle::DataType::FLOAT32;
    else if (dtype == "bfloat16")
        data_type = paddle::DataType::BFLOAT16;
    else if (dtype ==  "float16")
        data_type = paddle::DataType::FLOAT16;
    else 
        PD_THROW(
                "NOT supported data type. "
                "Only bfloat16, float16 and float32 are supported. ");

    return {data_type};
}

PD_BUILD_OP(dequant_int8)
    .Inputs({"intput","out_scale"})
    .Outputs({"output"})
    .Attrs({"dtype: std::string"})
    .SetKernelFn(PD_KERNEL(DequantInt8))
    .SetInferShapeFn(PD_INFER_SHAPE(DequantInt8Shape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DequantInt8Dtype));

std::vector<paddle::Tensor> QuantInt8(const paddle::Tensor& input,
                                      const paddle::optional<paddle::Tensor>& shift,
                                      const paddle::optional<paddle::Tensor>& smooth,
                                      float scale,
                                      int32_t round_type,
                                      float max_bound,
                                      float min_bound) {
    std::cout << "quant_int8" << std::endl;
    return {input};
}

std::vector<std::vector<int64_t>> QuantInt8Shape(const std::vector<int64_t>& input_shape,
                                                const paddle::optional<std::vector<int64_t>>& shift_shape,
                                                const paddle::optional<std::vector<int64_t>>& smooth_shape
                                                ) {
    return {input_shape};
}

std::vector<paddle::DataType> QuantInt8Dtype(const paddle::DataType& input_dtype,
                                            const paddle::optional<paddle::DataType>& shift_dtype,
                                            const paddle::optional<paddle::DataType>& smooth_dtype
                                            ) {
    return {paddle::DataType::INT8};
}

PD_BUILD_OP(quant_int8)
    .Inputs({"intput", paddle::Optional("shift"),paddle::Optional("smooth") })
    .Outputs({"output"})
    .Attrs({"scale: float","round_type: int","max_bound: float", "min_bound: float"})
    .SetKernelFn(PD_KERNEL(QuantInt8))
    .SetInferShapeFn(PD_INFER_SHAPE(QuantInt8Shape))
    .SetInferDtypeFn(PD_INFER_DTYPE(QuantInt8Dtype));

std::vector<paddle::Tensor> RebuildPaddingV2(const paddle::Tensor& tmp_out, // [token_num, dim_embed]
                                             const paddle::Tensor& cum_offsets, // [bsz, 1]
                                             const paddle::Tensor& seq_lens_decoder,
                                             const paddle::Tensor& seq_lens_encoder,
                                             int max_input_length) {
    std::cout << "rebuild_padding_v2" << std::endl;
    auto dev_ctx = static_cast<const phi::CustomContext*>(
        paddle::experimental::DeviceContextPool::Instance().Get(tmp_out.place()));
    auto stream = static_cast<aclrtStream>(dev_ctx->stream());
    auto tmp_out_tensor = static_cast<phi::DenseTensor*>(tmp_out.impl().get());
    auto cum_offsets_tensor = static_cast<const phi::DenseTensor*>(cum_offsets.impl().get());
    auto seq_lens_decoder_tensor = static_cast<const phi::DenseTensor*>(seq_lens_decoder.impl().get()); 
    auto seq_lens_encoder_tensor = static_cast<const phi::DenseTensor*>(seq_lens_encoder.impl().get());

    const int dim_embed = tmp_out.shape().back();
    const int bsz = cum_offsets.shape()[0];

    if (tmp_out.shape().size() == 3) { // 需要[token_num, dim_embed]的shape
        tmp_out_tensor->Resize({tmp_out.shape()[0] * tmp_out.shape()[1], tmp_out.shape()[2]});
    }

    std::shared_ptr<phi::DenseTensor> out = std::make_shared<phi::DenseTensor>();
    out->Resize({bsz, dim_embed});
    dev_ctx->Alloc(out.get(), tmp_out_tensor->dtype());

    const auto& runner =
        NpuOpRunner("RebuildPadding",
                    {*tmp_out_tensor, *cum_offsets_tensor, *seq_lens_decoder_tensor, *seq_lens_encoder_tensor},
                    {*out},
                    {{"max_input_length", max_input_length}});
    runner.Run(stream);

    return {paddle::Tensor(out)};
}

std::vector<std::vector<int64_t>> RebuildPaddingV2InferShape(const std::vector<int64_t>& tmp_out_shape,
                                                             const std::vector<int64_t>& cum_offsets_shape,
                                                             const std::vector<int64_t>& seq_lens_decoder_shape,
                                                             const std::vector<int64_t>& seq_lens_encoder_shape) {
    int64_t bsz = cum_offsets_shape[0];
    int64_t dim_embed = tmp_out_shape[1];
    return {{bsz, dim_embed}};
}

std::vector<paddle::DataType> RebuildPaddingV2InferDtype(const paddle::DataType& tmp_out_dtype,
                                                         const paddle::DataType& cum_offsets_dtype,
                                                         const paddle::DataType& seq_lens_decoder_dtype,
                                                         const paddle::DataType& seq_lens_encoder_dtype) {
    return {tmp_out_dtype};
}

PD_BUILD_OP(rebuild_padding_v2)
    .Inputs({"tmp_out", "cum_offsets", "seq_lens_decoder", "seq_lens_encoder"})
    .Outputs({"out"})
    .Attrs({"max_input_length: int"})
    .SetKernelFn(PD_KERNEL(RebuildPaddingV2))
    .SetInferShapeFn(PD_INFER_SHAPE(RebuildPaddingV2InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RebuildPaddingV2InferDtype));
