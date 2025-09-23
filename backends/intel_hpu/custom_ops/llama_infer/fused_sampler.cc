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

#include "habanalabs/perf_lib_layer_params.h"
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "kernels/hpu_funcs.h"
#include "kernels/hpu_operator.h"
#include "paddle/extension.h"
#include "utils/utils.h"

enum TENSOR_IDS_IN {
  PRE_IDS = 0,
  INPUT_IDS,
  LENS_ENC,
  LENS_DEC,
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
  EOS_TOKEN,
  TOP_P,
  TENSOR_IDS_IN_COUNT
};
enum TENSOR_IDS_OUT { PRE_IDS_OUT = 0, TOP_SCORE, TOP_ID };

struct FusedSamplerParam {
  int rank_num;
  int rank_id;
  int bs;
  int slice_start;
  size_t slice_num;
};

namespace custom_kernel {

class FusedSamplerKernel : public HpuFusedOperator {
 public:
  FusedSamplerKernel() : HpuFusedOperator("FusedSampler") {}

  synTensor createSlicedTensorNoPresist(ConvertTensors* ct,
                                        int id,
                                        int slice_axis,
                                        int slice_width,
                                        std::string name,
                                        synDataType type,
                                        bool is_input = true) {
    if (is_input) {
      std::vector<DIMS> inputs_dims = ct->GetDims();
      auto dims = inputs_dims[id];
      dims[slice_axis] = slice_width;
      auto t = createTensorNoPresist(name, type, dims);
      return t;
    }

    std::vector<DIMS> outputs_dims = ct->GetDims(false);
    auto dims = outputs_dims[id];
    dims[slice_axis] = slice_width;
    auto t = createTensorNoPresist(name, type, dims);
    return t;
  }

  synTensor AddNodeSliceX(ConvertTensors* ct,
                          int input_id,
                          int slice_axis,
                          int slice_start,
                          int slice_end,
                          synSectionHandle section_in = nullptr,
                          synTensor* source = nullptr) {
    auto inputs = ct->GetTensors();
    std::vector<DIMS> inputs_dims = ct->GetDims();
    synSliceParamsV2 params;
    int dim_size = inputs_dims[input_id].size();

    for (size_t i = 0; i < inputs.size(); i++) {
      params.axes[i] = i;
      params.steps[i] = 1;
      params.starts[i] = 0;
      params.ends[i] = inputs_dims[input_id][dim_size - 1 - i];
    }
    int axis = dim_size - 1 - slice_axis;
    params.starts[axis] = slice_start;
    params.ends[axis] = slice_end;

    auto x = createTensorFromCT(ct, input_id, true, section_in);
    std::string y_name = "tmp_out_" + std::to_string(input_id);
    auto y_dims = inputs_dims[input_id];
    y_dims[slice_axis] = slice_end - slice_start;
    auto y = createTensorNoPresist(y_name, inputs[input_id].type, y_dims);

    std::vector<synTensor> ins = {x};
    std::vector<synTensor> outs = {y};
    std::string node_name = guid_ + "_slice_" + std::to_string(input_id);
    AddNodeSlice(ins, outs, params, node_name);

    if (source != nullptr) {
      *source = x;
    }

    return y;
  }

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

  void AddNode(ConvertTensors* ct, const FusedSamplerParam& sampler_param) {
    std::vector<synTensor> sampling_meta(TENSOR_IDS_IN_COUNT);
    std::vector<DIMS> inputs_dims = ct->GetDims();
    bool is_slice_needed = false;

    if (sampler_param.rank_num > 1) {
      is_slice_needed = true;
    }

    synSectionHandle section_pre_ids = createSection();
    synTensor pre_ids;

#define SLICE_INPUT(ID)                        \
  sampling_meta[ID] =                          \
      AddNodeSliceX(ct,                        \
                    ID,                        \
                    0,                         \
                    sampler_param.slice_start, \
                    sampler_param.slice_start + sampler_param.slice_num);

    if (is_slice_needed) {
      sampling_meta[PRE_IDS] =
          AddNodeSliceX(ct,
                        PRE_IDS,
                        0,
                        sampler_param.slice_start,
                        sampler_param.slice_start + sampler_param.slice_num,
                        section_pre_ids,
                        &pre_ids);
      SLICE_INPUT(INPUT_IDS);
      SLICE_INPUT(LENS_ENC);
      SLICE_INPUT(LENS_DEC);
      SLICE_INPUT(STEP_IDX);
      SLICE_INPUT(STOP_FLAGS);
      SLICE_INPUT(LOGITS);
      SLICE_INPUT(PENALTY);
      SLICE_INPUT(FREQUENCY);
      SLICE_INPUT(PRESENCE);
      SLICE_INPUT(TEMPERATURE);
      sampling_meta[BAD_TOKEN] = createTensorFromCT(ct, BAD_TOKEN);
      SLICE_INPUT(CUR_LEN);
      SLICE_INPUT(MIN_LEN);
      sampling_meta[EOS_TOKEN] = createTensorFromCT(ct, EOS_TOKEN);
      SLICE_INPUT(TOP_P);
    } else {
      sampling_meta[PRE_IDS] =
          createTensorFromCT(ct, PRE_IDS, true, section_pre_ids);
      sampling_meta[INPUT_IDS] = createTensorFromCT(ct, INPUT_IDS);
      sampling_meta[LENS_ENC] = createTensorFromCT(ct, LENS_ENC);
      sampling_meta[LENS_DEC] = createTensorFromCT(ct, LENS_DEC);
      sampling_meta[STEP_IDX] = createTensorFromCT(ct, STEP_IDX);
      sampling_meta[STOP_FLAGS] = createTensorFromCT(ct, STOP_FLAGS);
      sampling_meta[LOGITS] = createTensorFromCT(ct, LOGITS);
      sampling_meta[PENALTY] = createTensorFromCT(ct, PENALTY);
      sampling_meta[FREQUENCY] = createTensorFromCT(ct, FREQUENCY);
      sampling_meta[PRESENCE] = createTensorFromCT(ct, PRESENCE);
      sampling_meta[TEMPERATURE] = createTensorFromCT(ct, TEMPERATURE);
      sampling_meta[BAD_TOKEN] = createTensorFromCT(ct, BAD_TOKEN);
      sampling_meta[CUR_LEN] = createTensorFromCT(ct, CUR_LEN);
      sampling_meta[MIN_LEN] = createTensorFromCT(ct, MIN_LEN);
      sampling_meta[EOS_TOKEN] = createTensorFromCT(ct, EOS_TOKEN);
      sampling_meta[TOP_P] = createTensorFromCT(ct, TOP_P);
    }

    int slice_num = sampler_param.slice_num;

    // cast pre_ids to i32
    synTensor pre_ids_i32 = createSlicedTensorNoPresist(
        ct, PRE_IDS, 0, slice_num, "pre_ids_i32", syn_type_int32);
    std::vector<synTensor> cast_pre_id_in = {sampling_meta[PRE_IDS]};
    std::vector<synTensor> cast_pre_id_out = {pre_ids_i32};
    AddNodeCast(cast_pre_id_in,
                cast_pre_id_out,
                "cast_i64_to_i32",
                guid_ + "cast_pre_id");

    // cast input_ids to i32
    synTensor input_ids_i32 = createSlicedTensorNoPresist(
        ct, INPUT_IDS, 0, slice_num, "input_ids_i32", syn_type_int32);
    std::vector<synTensor> cast_input_id_in = {sampling_meta[INPUT_IDS]};
    std::vector<synTensor> cast_input_id_out = {input_ids_i32};
    AddNodeCast(cast_input_id_in,
                cast_input_id_out,
                "cast_i64_to_i32",
                guid_ + "cast_input_id");

    // cast step_idx to i32
    synTensor step_idx_i32 = createSlicedTensorNoPresist(
        ct, STEP_IDX, 0, slice_num, "step_idx_i32", syn_type_int32);
    std::vector<synTensor> cast_step_idx_in = {sampling_meta[STEP_IDX]};
    std::vector<synTensor> cast_step_idx_out = {step_idx_i32};
    AddNodeCast(cast_step_idx_in,
                cast_step_idx_out,
                "cast_i64_to_i32",
                guid_ + "cast_step_idx");

    // set_flags_bfai_blk
    synTensor pre_ids_out_i32 = createSlicedTensorNoPresist(
        ct, PRE_IDS, 0, slice_num, "pre_ids_out_i32", syn_type_int32);
    std::vector<synTensor> set_value_inputs = {pre_ids_i32,
                                               input_ids_i32,
                                               sampling_meta[LENS_ENC],
                                               sampling_meta[LENS_DEC],
                                               step_idx_i32,
                                               sampling_meta[STOP_FLAGS]};
    std::vector<synTensor> set_value_outputs = {pre_ids_out_i32};
    AddNodeSetValueBFAIBlkOp(set_value_inputs, set_value_outputs);

    // indices to scatter slices back
    ns_RangeKernel::Params range_params;
    range_params.start.i = sampler_param.slice_start;
    range_params.delta.i = 1;
    range_params.limit.i = sampler_param.slice_start + sampler_param.slice_num;
    std::vector<int64_t> scatter_dims = {slice_num};
    auto scatter_indices_1d = createTensorNoPresist(
        "scatter_indices_1d", syn_type_int32, scatter_dims);
    std::vector<synTensor> outputs = {scatter_indices_1d};
    AddNodeRange<int>(outputs, range_params, guid_ + "_range_as_indices");
    scatter_dims.push_back(1);
    auto scatter_indices =
        createTensorNoPresist("scatter_indices", syn_type_int32, scatter_dims);
    std::vector<synTensor> reshape_ins = {scatter_indices_1d};
    std::vector<synTensor> reshape_outs = {scatter_indices};
    AddNodeReshape(reshape_ins, reshape_outs, guid_ + "reshape_indices");

    ns_ScatterKernel::Params scatter_slice_params;
    scatter_slice_params.axis = 1;

    std::vector<DIMS> outputs_dims = ct->GetDims(false);
    synTensor pre_ids_full_i32;
    if (is_slice_needed) {
      // scatter source
      auto pre_ids_source = createTensorNoPresist(
          "pre_ids_source_i32", syn_type_int32, outputs_dims[PRE_IDS_OUT]);
      std::vector<synTensor> cast_pre_ids_back_in = {pre_ids};
      std::vector<synTensor> cast_pre_ids_back_out = {pre_ids_source};
      AddNodeCast(cast_pre_ids_back_in,
                  cast_pre_ids_back_out,
                  "cast_i64_to_i32",
                  guid_ + "cast_pre_ids_source");

      // scatter to pre_ids_i32
      pre_ids_full_i32 = createTensorNoPresist(
          "pre_ids_full_i32", syn_type_int32, outputs_dims[PRE_IDS_OUT]);
      std::vector<synTensor> scatter_ins = {
          pre_ids_source, scatter_indices, pre_ids_out_i32};
      std::vector<synTensor> scatter_outs = {pre_ids_full_i32};

      AddNodeScatterFwd<int>(scatter_ins,
                             scatter_outs,
                             scatter_slice_params,
                             guid_ + "_pre_ids_scatter");

    } else {
      pre_ids_full_i32 = pre_ids_i32;
    }

    // overwrite original pre_ids
    auto pre_ids_out =
        createTensorFromCT(ct, PRE_IDS_OUT, false, section_pre_ids);
    std::vector<synTensor> cast_pre_ids_back_in = {pre_ids_full_i32};
    std::vector<synTensor> cast_pre_ids_back_out = {pre_ids_out};
    AddNodeCast(cast_pre_ids_back_in,
                cast_pre_ids_back_out,
                "cast_i32_to_i64",
                guid_ + "cast_pre_ids_back");

    // cast cur_len to i32
    synTensor cur_len_i32 = createSlicedTensorNoPresist(
        ct, CUR_LEN, 0, slice_num, "cur_len_i32", syn_type_int32);
    std::vector<synTensor> cast_cur_len_in = {sampling_meta[CUR_LEN]};
    std::vector<synTensor> cast_cur_len_out = {cur_len_i32};
    AddNodeCast(cast_cur_len_in,
                cast_cur_len_out,
                "cast_i64_to_i32",
                guid_ + "cast_cur_len");

    // cast min_len to i32
    synTensor min_len_i32 = createSlicedTensorNoPresist(
        ct, MIN_LEN, 0, slice_num, "min_len_i32", syn_type_int32);
    std::vector<synTensor> cast_min_len_in = {sampling_meta[MIN_LEN]};
    std::vector<synTensor> cast_min_len_out = {min_len_i32};
    AddNodeCast(cast_min_len_in,
                cast_min_len_out,
                "cast_i64_to_i32",
                guid_ + "cast_min_len");

    // cast eos_token to i32
    synTensor eos_token_i32 = createTensorNoPresist(
        "eos_token_i32", syn_type_int32, inputs_dims[EOS_TOKEN]);
    std::vector<synTensor> cast_eos_token_in = {sampling_meta[EOS_TOKEN]};
    std::vector<synTensor> cast_eos_token_out = {eos_token_i32};

    AddNodeCast(cast_eos_token_in,
                cast_eos_token_out,
                "cast_i64_to_i32",
                guid_ + "cast_eos_token");

    // conditionally set logit to eos
    synTensor logits_mark_eos = createSlicedTensorNoPresist(
        ct, LOGITS, 0, slice_num, "logits_mark_eos", syn_type_float);
    std::vector<synTensor> min_len_logits_in = {
        sampling_meta[LOGITS], cur_len_i32, min_len_i32, eos_token_i32};
    std::vector<synTensor> min_len_logits_out = {logits_mark_eos};
    AddNodeMinLengthLogits(min_len_logits_in, min_len_logits_out);

    // update repeat times
    // zero tensor as dest, pre_ids as index, ones as update
    synTensor zeros = createSlicedTensorNoPresist(
        ct, LOGITS, 0, slice_num, "zeros", syn_type_float);
    ns_ConstantKernel::Params const_params;
    const_params.constant.f = 0.0f;
    std::vector<synTensor> full_out = {zeros};
    AddNodeFull<float>(full_out, const_params, guid_ + "full_zeros");

    synTensor update = createSlicedTensorNoPresist(
        ct, PRE_IDS, 0, slice_num, "update", syn_type_float);
    std::vector<synTensor> wrap_blk_inputs = {pre_ids_i32};
    std::vector<synTensor> wrap_blk_outputs = {update};
    AddNodeWrapBlk(wrap_blk_inputs, wrap_blk_outputs);

    synTensor pre_ids_times = createSlicedTensorNoPresist(
        ct, LOGITS, 0, slice_num, "pre_ids_times", syn_type_float);
    std::vector<synTensor> scatter_in = {zeros, pre_ids_out_i32, update};
    std::vector<synTensor> scatter_out = {pre_ids_times};
    ns_ScatterKernel::Params scatter_params;
    scatter_params.axis = 0;
    AddNodeScatterAdd<float>(
        scatter_in, scatter_out, scatter_params, guid_ + "scatter");

    // calculate score
    synTensor logits_score = createSlicedTensorNoPresist(
        ct, LOGITS, 0, slice_num, "logits_score", syn_type_float);
    std::vector<synTensor> logits_by_repeats_in = {pre_ids_times,
                                                   logits_mark_eos,
                                                   sampling_meta[PENALTY],
                                                   sampling_meta[FREQUENCY],
                                                   sampling_meta[PRESENCE],
                                                   sampling_meta[TEMPERATURE]};
    std::vector<synTensor> logits_by_repeats_out = {logits_score};
    AddNodeLogitsByRepeats(logits_by_repeats_in, logits_by_repeats_out);

    // convert bad_word_token to i32
    synTensor bad_token_i32 = createTensorNoPresist(
        "bad_token_i32", syn_type_int32, inputs_dims[BAD_TOKEN]);
    std::vector<synTensor> cast_bad_token_in = {sampling_meta[BAD_TOKEN]};
    std::vector<synTensor> cast_bad_token_out = {bad_token_i32};
    AddNodeCast(cast_bad_token_in,
                cast_bad_token_out,
                "cast_i64_to_i32",
                guid_ + "cast_bad_token");

    // exclude bad_token from logits
    synTensor logits_mark_bad_token = createSlicedTensorNoPresist(
        ct, LOGITS, 0, slice_num, "logits_mark_bad_token", syn_type_float);
    std::vector<synTensor> logits_mark_bad_token_in = {logits_score,
                                                       bad_token_i32};
    std::vector<synTensor> logits_mark_bad_token_out = {logits_mark_bad_token};
    AddNodeLogitsMarkBadToken(logits_mark_bad_token_in,
                              logits_mark_bad_token_out);

    // softmax
    ns_Softmax::Params softmax_params;
    softmax_params.dim = 0;
    synTensor probs = createSlicedTensorNoPresist(
        ct, LOGITS, 0, slice_num, "probs", syn_type_float);
    std::vector<synTensor> softmax_in = {logits_mark_bad_token};
    std::vector<synTensor> softmax_out = {probs};
    AddNodeSoftmax<float>(
        softmax_in, softmax_out, softmax_params, guid_ + "softmax");

    // top_p_sampling
    int candidates = 4096;
    if (inputs_dims[LOGITS][1] < candidates) {
      candidates = inputs_dims[LOGITS][1];
    }
    std::vector<int64_t> k_dims = {slice_num, candidates};

    // TopK Node to get sorted probs and indices
    synBeamParams topk_params{};
    topk_params.bsw = candidates;
    topk_params.axis = 0;
    topk_params.bottomK = false;
    auto sorted_probs =
        createTensorNoPresist("sorted_probs", syn_type_float, k_dims);
    auto sorted_indices =
        createTensorNoPresist("sorted_indices", syn_type_int32, k_dims);
    std::vector<synTensor> topk_ins = {probs};
    std::vector<synTensor> topk_outs = {sorted_probs, sorted_indices};
    AddNodeTopK(topk_ins, topk_outs, topk_params, guid_ + "topk");

    // Cumsum Node to get cumsum probs
    auto cumsum_probs =
        createTensorNoPresist("cumsum_probs", syn_type_float, k_dims);
    std::vector<synTensor> cumsum_ins = {sorted_probs};
    std::vector<synTensor> cumsum_outs = {cumsum_probs};
    ns_CumSumKernel::Params cumsum_params{0, 0, 0};
    AddNodeCumsum<float>(
        cumsum_ins, cumsum_outs, cumsum_params, guid_ + "cumsum");

    // Sub to get the cumulative sum before the current element.
    auto sub_probs = createTensorNoPresist("sub_probs", syn_type_float, k_dims);
    std::vector<synTensor> sub_ins = {cumsum_probs, sorted_probs};
    std::vector<synTensor> sub_outs = {sub_probs};
    AddNodeSub<float>(sub_ins, sub_outs, guid_ + "sub");

    // Less Equal to get the selected_index
    auto mask = createTensorNoPresist("mask", syn_type_int8, k_dims);
    std::vector<synTensor> less_equal_ins = {sub_probs, sampling_meta[TOP_P]};
    std::vector<synTensor> less_equal_outs = {mask};
    AddNodeLessEqual<float>(
        less_equal_ins, less_equal_outs, guid_ + "less_equal");

    // Scalar Node Zero
    std::vector<int64_t> scalar_dims = {1};
    auto zero = createTensorNoPresist("zero", syn_type_float, scalar_dims);
    const_params.constant.f = 0.0f;
    full_out[0] = zero;
    AddNodeFull<float>(full_out, const_params, guid_ + "full_zero");

    // Populate unwanted probs with -inf
    auto filtered_probs =
        createTensorNoPresist("filtered_probs", syn_type_float, k_dims);
    std::vector<synTensor> where_ins = {mask, sorted_probs, zero};
    std::vector<synTensor> where_outs = {filtered_probs};
    AddNodeWhere<float>(where_ins, where_outs, guid_ + "where");

    // Multinomial to select indices of sorted indices
    auto selected_indices = createSlicedTensorNoPresist(
        ct, TOP_ID, 0, slice_num, "selected_indices", syn_type_int32, false);
    std::random_device rd;
    auto time_seed =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    int32_t rd_seed = static_cast<int32_t>(rd() ^ time_seed);
    ns_RandomMultinomial::ParamsV2 multinomial_params;
    multinomial_params.num_samples = 1;
    multinomial_params.replacement = false;
    multinomial_params.seed = rd_seed;
    std::vector<synTensor> multinomial_ins = {filtered_probs};
    std::vector<synTensor> multinomial_outs = {selected_indices};
    AddNodeMultinomial<float>(multinomial_ins,
                              multinomial_outs,
                              multinomial_params,
                              guid_ + "multinomial");

    // Gather Elements to get selected token indices
    auto token_indices = createSlicedTensorNoPresist(
        ct, TOP_ID, 0, slice_num, "token_indices", syn_type_int32, false);
    ns_GatherKernel::Params gather_params;
    gather_params.axis = 0;
    std::vector<synTensor> index_sample_ins = {sorted_indices,
                                               selected_indices};
    std::vector<synTensor> index_sample_outs = {token_indices};
    AddNodeIndexSample<int32_t>(index_sample_ins,
                                index_sample_outs,
                                gather_params,
                                guid_ + "index_sample");

    synTensor token_out_i32;
    if (is_slice_needed) {
      auto token_indices_zero = createTensorNoPresist(
          "token_indices_zero", syn_type_int32, outputs_dims[TOP_ID]);
      const_params.constant.i = 0;
      full_out[0] = token_indices_zero;
      AddNodeFull<int>(full_out, const_params, guid_ + "token_indices_zero");

      token_out_i32 = createTensorNoPresist(
          "token_out_i32", syn_type_int32, outputs_dims[TOP_ID]);

      std::vector<synTensor> scatter_ins = {
          token_indices_zero, scatter_indices, token_indices};
      std::vector<synTensor> scatter_outs = {token_out_i32};
      AddNodeScatterFwd<int>(scatter_ins,
                             scatter_outs,
                             scatter_slice_params,
                             guid_ + "_top_ids_scatter");
    } else {
      token_out_i32 = token_indices;
    }

    // Cast to output format int64
    auto token_indices_out = createTensorFromCT(ct, TOP_ID, false);
    std::vector<synTensor> cast_ins = {token_out_i32};
    std::vector<synTensor> cast_outs = {token_indices_out};

    AddNodeCast(
        cast_ins, cast_outs, "cast_i32_to_i64", guid_ + "cast_token_ids");
  }
};

}  // namespace custom_kernel

#define INSERT_TENSOR_TO_CT(tnsr, ct, exp_size, id, is_input)                 \
  {                                                                           \
    auto dims = tnsr.dims();                                                  \
    int dim_size = dims.size();                                               \
                                                                              \
    PD_CHECK(dim_size == exp_size,                                            \
             #tnsr "'s dimensions should be " #exp_size " but it is      ",   \
             dim_size);                                                       \
                                                                              \
    PD_CHECK(ct.GetDims(is_input).size() == id,                               \
             #tnsr "'s offset is not correct");                               \
                                                                              \
    auto tnsr##_dt = static_cast<const phi::DenseTensor*>(tnsr.impl().get()); \
    ct.Add(*tnsr##_dt, is_input);                                             \
  }

static std::vector<int> get_sampler_slices(int rank_num, int rank_id, int bs) {
  std::vector<int> task_ids;

  if (rank_id < 0 || rank_id >= rank_num || bs <= 0) {
    return task_ids;
  }

  int base_chunks = (bs / 8) / rank_num;
  int extra_chunk_count = (bs / 8) % rank_num;
  int final_remainder = bs % 8;

  int workload_for_id;
  if (rank_id < extra_chunk_count) {
    workload_for_id = (base_chunks + 1) * 8;
  } else {
    workload_for_id = base_chunks * 8;
  }

  if (rank_id == extra_chunk_count) {
    workload_for_id += final_remainder;
  }

  int start_id = 0;
  for (int i = 0; i < rank_id; ++i) {
    int workload_before;
    if (i < extra_chunk_count) {
      workload_before = (base_chunks + 1) * 8;
    } else {
      workload_before = base_chunks * 8;
    }
    if (i == extra_chunk_count) {
      workload_before += final_remainder;
    }
    start_id += workload_before;
  }

  for (int i = 0; i < workload_for_id; ++i) {
    task_ids.push_back(start_id + i);
  }

  return task_ids;
}

std::vector<paddle::Tensor> FusedSampler(const paddle::Tensor& pre_ids,
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
                                         const paddle::Tensor& eos_token_id,
                                         const paddle::Tensor& top_p,
                                         const int rank_num,
                                         const int rank_id) {
  PD_CHECK(logits.type() == paddle::DataType::FLOAT32,
           "Unsupported logits data type. Logits should be in float32.");

  auto dims = min_len.dims();
  int bs = dims[0];
  auto out =
      paddle::full({bs, 1}, 0, paddle::DataType::FLOAT32, pre_ids.place());
  auto ids = paddle::full({bs, 1}, 0, paddle::DataType::INT64, pre_ids.place());

  std::vector<int> slices = get_sampler_slices(rank_num, rank_id, bs);

  if (rank_num == 1) {
    PD_CHECK(slices.size() == static_cast<size_t>(bs),
             "Slicing should not happen for rank_num 1 case.");
  }

  if (slices.size() == 0) {
    // early out when no workload for current rank
    return {out, ids};
  }

  FusedSamplerParam sampler_param = {
      rank_num, rank_id, bs, slices[0], slices.size()};

  custom_kernel::ConvertTensors ct;
  INSERT_TENSOR_TO_CT(pre_ids, ct, 2, PRE_IDS, true);
  INSERT_TENSOR_TO_CT(input_ids, ct, 2, INPUT_IDS, true);
  INSERT_TENSOR_TO_CT(seq_lens_encoder, ct, 2, LENS_ENC, true);
  INSERT_TENSOR_TO_CT(seq_lens_decoder, ct, 2, LENS_DEC, true);
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
  INSERT_TENSOR_TO_CT(top_p, ct, 2, TOP_P, true);
  INSERT_TENSOR_TO_CT(pre_ids, ct, 2, PRE_IDS_OUT, false);
  INSERT_TENSOR_TO_CT(out, ct, 2, TOP_SCORE, false);
  INSERT_TENSOR_TO_CT(ids, ct, 2, TOP_ID, false);

  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<float, FusedSamplerParam>(
      "FusedSamplerKernel", inputs_dims, &sampler_param);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    custom_kernel::FusedSamplerKernel op;
    op.AddNode(&ct, sampler_param);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(pre_ids.place()));
  auto tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx->stream()), tensors);
  return {out, ids};
}

std::vector<std::vector<int64_t>> FusedSamplerInferShape(
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
    const std::vector<int64_t>& eos_token_id_shape,
    const std::vector<int64_t>& top_p_shape) {
  return {pre_ids_shape};
}

std::vector<paddle::DataType> FusedSamplerInferDtype(
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
    const paddle::DataType& eos_token_id_dtype,
    const paddle::DataType& top_p_dtype) {
  return {pre_ids_dtype};
}

PD_BUILD_OP(fused_sampler)
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
             "eos_token_id",
             "top_p"})
    .Attrs({"rank_num: int", "rank_id: int"})
    .Outputs({"top_score", "top_id"})
    .SetKernelFn(PD_KERNEL(FusedSampler))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedSamplerInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedSamplerInferDtype));
