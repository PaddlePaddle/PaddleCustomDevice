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
#include "habanalabs/perf_lib_layer_params.h"
#include "kernels/funcs.h"
#include "kernels/hpu_funcs.h"
#include "kernels/hpu_operator.h"
#include "paddle/extension.h"
#include "utils/utils.h"

namespace custom_kernel {

struct BlockPPLogitsGeneratorParams {
  ns_ConstantKernel::Params const_params;
  ns_ScatterKernel::Params scatter_params;
};

enum TENSOR_IDS_IN {
  PRE_IDS = 0,
  INPUT_IDS,
  LEN_ENC,
  LEN_DEC,
  STEP_IDX,
  STOP_FLAGS,
  LOGITS,
  PENALTY,
  FREQUENCY,
  PRESENCE,
  TEMPERATURE,
  BAD_TOKEN,
  CUR_LEN,
  MIN_LEN,
  EOS_TOKEN
};

enum TENSOR_IDS_OUT { PRE_IDS_OUT = 0, LOGITS_OUT };

class BlockPPLogitsGeneratorOp : public HpuFusedOperator {
 public:
  BlockPPLogitsGeneratorOp()
      : HpuFusedOperator("BlockPPLogitsGenerator_", false) {}

  void AddNodeSetValueBFAIBlkOp(std::vector<synTensor> inputs,
                                std::vector<synTensor> outputs) {
    std::string node = "custom_set_value_bfai_blk";
    AddNode_IO(inputs, outputs, node, guid_ + node);
  }

  void AddNodeMinLengthLogits(std::vector<synTensor> inputs,
                              std::vector<synTensor> outputs) {
    std::string node = "custom_blk_process_eos";
    AddNode_IO(inputs, outputs, node, guid_ + node);
  }

  void AddNodeWrapBlk(std::vector<synTensor> inputs,
                      std::vector<synTensor> outputs) {
    std::string node = "custom_wrap_update_blk";
    // std::string node = "custom_wrap_update";
    AddNode_IO(inputs, outputs, node, guid_ + node);
  }

  void AddNodeLogitsByRepeats(std::vector<synTensor> inputs,
                              std::vector<synTensor> outputs) {
    std::string node = "custom_logits_score";
    AddNode_IO(inputs, outputs, node, guid_ + node);
  }

  void AddNodeLogitsMarkBadToken(std::vector<synTensor> inputs,
                                 std::vector<synTensor> outputs) {
    std::string node = "custom_logits_wo_badtoken";
    AddNode_IO(inputs, outputs, node, guid_ + node);
  }

  template <typename T>
  void AddNode(ConvertTensors* ct, BlockPPLogitsGeneratorParams params) {
    std::vector<DIMS> inputs_dims = ct->GetDims();

    // cast pre_ids to i32
    synSectionHandle section_pre_ids = createSection();
    synTensor pre_ids = createTensorFromCT(ct, PRE_IDS, true, section_pre_ids);
    synTensor pre_ids_i32 = createTensorNoPresist(
        "pre_ids_i32", syn_type_int32, inputs_dims[PRE_IDS]);
    std::vector<synTensor> cast_pre_id_in = {pre_ids};
    std::vector<synTensor> cast_pre_id_out = {pre_ids_i32};
    AddNodeCast(cast_pre_id_in,
                cast_pre_id_out,
                "cast_i64_to_i32",
                guid_ + "cast_pre_id");

    // cast input_ids to i32
    synTensor input_ids = createTensorFromCT(ct, INPUT_IDS);
    synTensor input_ids_i32 = createTensorNoPresist(
        "input_ids_i32", syn_type_int32, inputs_dims[INPUT_IDS]);
    std::vector<synTensor> cast_input_id_in = {input_ids};
    std::vector<synTensor> cast_input_id_out = {input_ids_i32};
    AddNodeCast(cast_input_id_in,
                cast_input_id_out,
                "cast_i64_to_i32",
                guid_ + "cast_input_id");

    // cast step_idx to i32
    synTensor step_idx = createTensorFromCT(ct, STEP_IDX);
    synTensor step_idx_i32 = createTensorNoPresist(
        "step_idx_i32", syn_type_int32, inputs_dims[STEP_IDX]);
    std::vector<synTensor> cast_step_idx_in = {step_idx};
    std::vector<synTensor> cast_step_idx_out = {step_idx_i32};
    AddNodeCast(cast_step_idx_in,
                cast_step_idx_out,
                "cast_i64_to_i32",
                guid_ + "cast_step_idx");

    // set_flags_bfai_blk
    synTensor len_enc = createTensorFromCT(ct, LEN_ENC);
    synTensor len_dec = createTensorFromCT(ct, LEN_DEC);
    synTensor stop_flags = createTensorFromCT(ct, STOP_FLAGS);
    synTensor pre_ids_out_i32 = createTensorNoPresist(
        "pre_ids_out_i32", syn_type_int32, inputs_dims[PRE_IDS]);
    std::vector<synTensor> set_value_inputs = {
        pre_ids_i32, input_ids_i32, len_enc, len_dec, step_idx_i32, stop_flags};
    std::vector<synTensor> set_value_outputs = {pre_ids_out_i32};
    AddNodeSetValueBFAIBlkOp(set_value_inputs, set_value_outputs);

    // convert pre_ids output back
    synTensor pre_ids_out =
        createTensorFromCT(ct, PRE_IDS_OUT, false, section_pre_ids);
    std::vector<synTensor> cast_pre_ids_back_in = {pre_ids_out_i32};
    std::vector<synTensor> cast_pre_ids_back_out = {pre_ids_out};
    AddNodeCast(cast_pre_ids_back_in,
                cast_pre_ids_back_out,
                "cast_i32_to_i64",
                guid_ + "cast_pre_ids_back");

    // cast cur_len to i32
    synTensor cur_len = createTensorFromCT(ct, CUR_LEN);
    synTensor cur_len_i32 = createTensorNoPresist(
        "cur_len_i32", syn_type_int32, inputs_dims[CUR_LEN]);
    std::vector<synTensor> cast_cur_len_in = {cur_len};
    std::vector<synTensor> cast_cur_len_out = {cur_len_i32};
    AddNodeCast(cast_cur_len_in,
                cast_cur_len_out,
                "cast_i64_to_i32",
                guid_ + "cast_cur_len");

    // cast min_len to i32
    synTensor min_len = createTensorFromCT(ct, MIN_LEN);
    synTensor min_len_i32 = createTensorNoPresist(
        "min_len_i32", syn_type_int32, inputs_dims[MIN_LEN]);
    std::vector<synTensor> cast_min_len_in = {min_len};
    std::vector<synTensor> cast_min_len_out = {min_len_i32};
    AddNodeCast(cast_min_len_in,
                cast_min_len_out,
                "cast_i64_to_i32",
                guid_ + "cast_min_len");

    // cast eos_token to i32
    synTensor eos_token = createTensorFromCT(ct, EOS_TOKEN);
    synTensor eos_token_i32 = createTensorNoPresist(
        "eos_token_i32", syn_type_int32, inputs_dims[EOS_TOKEN]);
    std::vector<synTensor> cast_eos_token_in = {eos_token};
    std::vector<synTensor> cast_eos_token_out = {eos_token_i32};
    AddNodeCast(cast_eos_token_in,
                cast_eos_token_out,
                "cast_i64_to_i32",
                guid_ + "cast_eos_token");

    // set logit to eos when needed
    synSectionHandle section_logits = createSection();
    synTensor logits = createTensorFromCT(ct, LOGITS, true, section_logits);
    synTensor logits_mark_eos = createTensorNoPresist(
        "logits_mark_eos", syn_type_float, inputs_dims[LOGITS]);
    std::vector<synTensor> min_len_logits_in = {
        logits, cur_len_i32, min_len_i32, eos_token_i32};
    std::vector<synTensor> min_len_logits_out = {logits_mark_eos};
    AddNodeMinLengthLogits(min_len_logits_in, min_len_logits_out);

    // update repeat times
    // zero tensor as dest, pre_ids as index, ones as update
    synTensor zeros =
        createTensorNoPresist("zeros", syn_type_float, inputs_dims[LOGITS]);
    params.const_params.constant.f = 0.0f;
    std::vector<synTensor> full_out = {zeros};
    AddNodeFull<T>(full_out, params.const_params, guid_ + "full_zeros");

    synTensor update =
        createTensorNoPresist("update", syn_type_float, inputs_dims[PRE_IDS]);
    std::vector<synTensor> wrap_blk_inputs = {pre_ids_i32};
    std::vector<synTensor> wrap_blk_outputs = {update};
    AddNodeWrapBlk(wrap_blk_inputs, wrap_blk_outputs);

    synTensor pre_ids_times = createTensorNoPresist(
        "pre_ids_times", syn_type_float, inputs_dims[LOGITS]);
    std::vector<synTensor> scatter_in = {zeros, pre_ids_out_i32, update};
    std::vector<synTensor> scatter_out = {pre_ids_times};
    params.scatter_params.axis = 0;
    AddNodeScatterAdd<T>(
        scatter_in, scatter_out, params.scatter_params, guid_ + "scatter");

    // calculate score
    synTensor penalty = createTensorFromCT(ct, PENALTY);
    synTensor frequency = createTensorFromCT(ct, FREQUENCY);
    synTensor presence = createTensorFromCT(ct, PRESENCE);
    synTensor temperature = createTensorFromCT(ct, TEMPERATURE);
    synTensor logits_score = createTensorNoPresist(
        "logits_score", syn_type_float, inputs_dims[LOGITS]);
    std::vector<synTensor> logits_by_repeats_in = {pre_ids_times,
                                                   logits_mark_eos,
                                                   penalty,
                                                   frequency,
                                                   presence,
                                                   temperature};
    std::vector<synTensor> logits_by_repeats_out = {logits_score};
    AddNodeLogitsByRepeats(logits_by_repeats_in, logits_by_repeats_out);

    // convert bad_word_token to i32
    synTensor bad_token = createTensorFromCT(ct, BAD_TOKEN);
    synTensor bad_token_i32 = createTensorNoPresist(
        "bad_token_i32", syn_type_int32, inputs_dims[BAD_TOKEN]);
    std::vector<synTensor> cast_bad_token_in = {bad_token};
    std::vector<synTensor> cast_bad_token_out = {bad_token_i32};
    AddNodeCast(cast_bad_token_in,
                cast_bad_token_out,
                "cast_i64_to_i32",
                guid_ + "cast_bad_token");

    // exclude bad_token from logits
    synTensor logits_mark_bad_token =
        createTensorFromCT(ct, LOGITS_OUT, false, section_logits);
    std::vector<synTensor> logits_mark_bad_token_in = {logits_score,
                                                       bad_token_i32};
    std::vector<synTensor> logits_mark_bad_token_out = {logits_mark_bad_token};
    AddNodeLogitsMarkBadToken(logits_mark_bad_token_in,
                              logits_mark_bad_token_out);
  }
};

#define INSERT_TENSOR_TO_CT(tnsr, ct, rank, id, is_input)                     \
  {                                                                           \
    auto dims = tnsr.dims();                                                  \
    int dim_size = dims.size();                                               \
                                                                              \
    PD_CHECK(dim_size == rank,                                                \
             #tnsr "'s dimensions should be " #rank " but it is      ",       \
             dim_size);                                                       \
                                                                              \
    PD_CHECK(ct.GetDims(is_input).size() == id,                               \
             #tnsr "'s offset is not correct");                               \
                                                                              \
    auto tnsr##_dt = static_cast<const phi::DenseTensor*>(tnsr.impl().get()); \
    ct.Add(*tnsr##_dt, is_input);                                             \
  }

template <typename T>
void set_preids_token_penalty_multi_scores(
    const paddle::Tensor& pre_ids,
    const paddle::Tensor& input_ids,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& step_idx,
    const paddle::Tensor& stop_flags,
    const paddle::Tensor& logits,
    const paddle::Tensor& penalty_scores,
    const paddle::Tensor& frequency_scores,
    const paddle::Tensor& presence_scores,
    const paddle::Tensor& temperatures,
    const paddle::Tensor& bad_tokens,
    const paddle::Tensor& cur_len,
    const paddle::Tensor& min_len,
    const paddle::Tensor& eos_token_id) {
  custom_kernel::ConvertTensors ct;

  INSERT_TENSOR_TO_CT(pre_ids, ct, 2, PRE_IDS, true);
  INSERT_TENSOR_TO_CT(input_ids, ct, 2, INPUT_IDS, true);
  INSERT_TENSOR_TO_CT(seq_lens_encoder, ct, 2, LEN_ENC, true);
  INSERT_TENSOR_TO_CT(seq_lens_decoder, ct, 2, LEN_DEC, true);
  INSERT_TENSOR_TO_CT(step_idx, ct, 2, STEP_IDX, true);
  INSERT_TENSOR_TO_CT(stop_flags, ct, 2, STOP_FLAGS, true);
  INSERT_TENSOR_TO_CT(logits, ct, 2, LOGITS, true);
  INSERT_TENSOR_TO_CT(penalty_scores, ct, 2, PENALTY, true);
  INSERT_TENSOR_TO_CT(frequency_scores, ct, 2, FREQUENCY, true);
  INSERT_TENSOR_TO_CT(presence_scores, ct, 2, PRESENCE, true);
  INSERT_TENSOR_TO_CT(temperatures, ct, 2, TEMPERATURE, true);
  INSERT_TENSOR_TO_CT(bad_tokens, ct, 1, BAD_TOKEN, true);
  INSERT_TENSOR_TO_CT(cur_len, ct, 2, CUR_LEN, true);
  INSERT_TENSOR_TO_CT(min_len, ct, 2, MIN_LEN, true);
  INSERT_TENSOR_TO_CT(eos_token_id, ct, 2, EOS_TOKEN, true);
  INSERT_TENSOR_TO_CT(pre_ids, ct, 2, PRE_IDS_OUT, false);
  INSERT_TENSOR_TO_CT(logits, ct, 2, LOGITS_OUT, false);

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(pre_ids.place()));

  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, nullptr_t>(
      "SetPreidsTokenPenaltyMultiScores", inputs_dims, nullptr);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    BlockPPLogitsGeneratorOp op;
    BlockPPLogitsGeneratorParams params;

    op.AddNode<T>(&ct, params);

    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx->stream()), tensors);
}

}  // namespace custom_kernel

void SetPreidsTokenPenaltyMultiScores(const paddle::Tensor& pre_ids,
                                      const paddle::Tensor& input_ids,
                                      const paddle::Tensor& seq_lens_encoder,
                                      const paddle::Tensor& seq_lens_decoder,
                                      const paddle::Tensor& step_idx,
                                      const paddle::Tensor& stop_flags,
                                      const paddle::Tensor& logits,
                                      const paddle::Tensor& penalty_scores,
                                      const paddle::Tensor& frequency_scores,
                                      const paddle::Tensor& presence_scores,
                                      const paddle::Tensor& temperatures,
                                      const paddle::Tensor& bad_tokens,
                                      const paddle::Tensor& cur_len,
                                      const paddle::Tensor& min_len,
                                      const paddle::Tensor& eos_token_id) {
  switch (logits.type()) {
    case paddle::DataType::FLOAT32: {
      return custom_kernel::set_preids_token_penalty_multi_scores<float>(
          pre_ids,
          input_ids,
          seq_lens_encoder,
          seq_lens_decoder,
          step_idx,
          stop_flags,
          logits,
          penalty_scores,
          frequency_scores,
          presence_scores,
          temperatures,
          bad_tokens,
          cur_len,
          min_len,
          eos_token_id);
    }
    default: {
      PD_THROW(
          "NOT supported data type. "
          "Only float32 are supported. ");
      break;
    }
  }
}

std::vector<std::vector<int64_t>> SetPreidsTokenPenaltyMultiScoresInferShape(
    const std::vector<int64_t>& pre_ids_shape,
    const std::vector<int64_t>& input_ids_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& step_idx_shape,
    const std::vector<int64_t>& stop_flags_shape,
    const std::vector<int64_t>& logits_shape,
    const std::vector<int64_t>& penalty_scores_shape,
    const std::vector<int64_t>& frequency_scores_shape,
    const std::vector<int64_t>& presence_scores_shape,
    const std::vector<int64_t>& temperatures_shape,
    const std::vector<int64_t>& bad_tokens_shape,
    const std::vector<int64_t>& cur_len_shape,
    const std::vector<int64_t>& min_len_shape,
    const std::vector<int64_t>& eos_token_id_shape) {
  return {pre_ids_shape};
}

std::vector<paddle::DataType> SetPreidsTokenPenaltyMultiScoresInferDtype(
    const paddle::DataType& pre_ids_dtype,
    const paddle::DataType& input_ids_dtype,
    const paddle::DataType& seq_lens_encoder_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& step_idx_dtype,
    const paddle::DataType& stop_flags_dtype,
    const paddle::DataType& logits_dtype,
    const paddle::DataType& penalty_scores_dtype,
    const paddle::DataType& frequency_scores_dtype,
    const paddle::DataType& presence_scores_dtype,
    const paddle::DataType& temperatures_dtype,
    const paddle::DataType& bad_tokens_dtype,
    const paddle::DataType& cur_len_dtype,
    const paddle::DataType& min_len_dtype,
    const paddle::DataType& eos_token_id_dtype) {
  return {pre_ids_dtype};
}

PD_BUILD_OP(set_preids_token_penalty_multi_scores)
    .Inputs({"pre_ids",
             "input_ids",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "step_idx",
             "stop_flags",
             "logits",
             "penalty_scores",
             "frequency_scores",
             "presence_scores",
             "temperatures",
             "bad_tokens",
             "cur_len",
             "min_len",
             "eos_token_id"})
    .Outputs({"logits_out", "pre_ids_out"})
    .SetInplaceMap({{"logits", "logits_out"}, {"pre_ids", "pre_ids_out"}})
    .SetKernelFn(PD_KERNEL(SetPreidsTokenPenaltyMultiScores))
    .SetInferShapeFn(PD_INFER_SHAPE(SetPreidsTokenPenaltyMultiScoresInferShape))
    .SetInferDtypeFn(
        PD_INFER_DTYPE(SetPreidsTokenPenaltyMultiScoresInferDtype));
