// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
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

struct FusedBlockAttentionParams {
  ns_ConstantKernel::Params const_params;
  ns_GatherKernel::Params index_select_params;
  ns_Reduction::Params reduce_params;
  ns_IndexReduce::Params index_reduce_params;
  ns_LayerNormKernel::Params rmsnorm_params;

  int head_dim;
  int num_head;
  int num_kv_head;

  bool use_neox_style = true;
  bool with_qkv_biases = false;
  bool transpose = true;
  bool use_fp8 = false;
  bool use_qk_rmsnorm = false;
};

class FusedBlockAttentionBase : public HpuFusedOperator {
 public:
  explicit FusedBlockAttentionBase(const std::string& guid, bool is_eager)
      : HpuFusedOperator(guid, is_eager) {}

  template <typename T>
  inline void AddNodeMixedPrecisionGemm(bool use_fp8,
                                        ConvertTensors& ct,
                                        int scale_x_index,
                                        int scale_y_index,
                                        int reciprocal_scale_x,
                                        int reciprocal_scale_y,
                                        std::vector<synTensor> inputs,
                                        std::vector<synTensor> outputs,
                                        synGEMMParams gemm_params,
                                        const std::string& suffix) {
    if (use_fp8) {
      synTensor scale_x = createTensorFromCT(&ct, scale_x_index);
      synTensor scale_y = createTensorFromCT(&ct, scale_y_index);

      if (reciprocal_scale_x) {
        std::vector<synTensor> reciprocal_in = {scale_x};
        auto reciprocal_scale_x = cloneTensor(
            "reciprocal_scale_x_" + suffix, scale_x, syn_type_float);

        std::vector<synTensor> reciprocal_outs = {reciprocal_scale_x};
        AddNode_IO(reciprocal_in,
                   reciprocal_outs,
                   "reciprocal_fwd_f32",
                   "reciprocal_scale_x_" + suffix);
        inputs.push_back(reciprocal_scale_x);
      } else {
        inputs.push_back(scale_x);
      }

      if (reciprocal_scale_y) {
        std::vector<synTensor> reciprocal_in = {
            createTensorFromCT(&ct, scale_y_index)};
        auto reciprocal_scale_y = cloneTensor(
            "reciprocal_scale_y_" + suffix, scale_y, syn_type_float);

        std::vector<synTensor> reciprocal_outs = {reciprocal_scale_y};
        AddNode_IO(reciprocal_in,
                   reciprocal_outs,
                   "reciprocal_fwd_f32",
                   "reciprocal_scale_y_" + suffix);
        inputs.push_back(reciprocal_scale_y);
      } else {
        inputs.push_back(scale_y);
      }

      AddNodeFusedFP8Gemm<T>(
          inputs, outputs, gemm_params, guid_ + "fused_fp8_gemm_" + suffix);
    } else {
      AddNodeBatchGemm(
          inputs, outputs, gemm_params, guid_ + "batchgemm_" + suffix);
    }
  }
};

class FusedMHABlockAttention : public FusedBlockAttentionBase {
 public:
  explicit FusedMHABlockAttention(std::string guid_prefix, synDataType dtype)
      : FusedBlockAttentionBase(guid_prefix, false), dtype_(dtype) {}
  template <typename T>
  void AddNode(ConvertTensors& ct, FusedBlockAttentionParams& params) {
    auto ins = ct.GetTensors();
    auto outs = ct.GetTensors(false);
    auto kv_dtype = params.use_fp8 ? synDataType::syn_type_fp8_143 : dtype_;

    int index_base = 0;
    int src_index = (index_base++);             // 0
    int rotary_embs_index = (index_base++);     // 1
    int key_cache_index = (index_base++);       // 2
    int value_cache_index = (index_base++);     // 3
    int block_groups_index = (index_base++);    // 4
    int block_list_index = (index_base++);      // 5
    int block_mapping_index = (index_base++);   // 6
    int block_bias_index = (index_base++);      // 7
    int block_indices_index = (index_base++);   // 8
    int block_offsets_index = (index_base++);   // 9
    int qkv_weights_index = (index_base++);     // 10
    int linear_weights_index = (index_base++);  // 11

    int src_scale_index = -1, qkv_weights_scale_index = -1, q_scale_index = -1,
        k_scale_index = -1, a_scale_index = -1, v_scale_index = -1,
        o_linear_scale_x_index = -1, o_linear_scale_y_index = -1,
        qkv_biases_index = -1, q_gamma_index = -1, k_gamma_index = -1;
    if (params.use_fp8) {
      src_scale_index = (index_base++);
      qkv_weights_scale_index = (index_base++);
      q_scale_index = (index_base++);
      k_scale_index = (index_base++);
      a_scale_index = (index_base++);
      v_scale_index = (index_base++);
      o_linear_scale_x_index = (index_base++);
      o_linear_scale_y_index = (index_base++);
    }

    if (params.with_qkv_biases) {
      qkv_biases_index = (index_base++);
    }

    if (params.use_qk_rmsnorm) {
      q_gamma_index = (index_base++);
      k_gamma_index = (index_base++);
    }

    std::vector<int64_t> src_dims = std::vector<int64_t>(ins[src_index].dims);

    int64_t batch_size = src_dims[0];
    int64_t hidden_size = ins[linear_weights_index].dims[0];
    int64_t block_size = ins[key_cache_index].dims[1];
    int64_t num_of_block = ins[block_list_index].dims[0];

    int64_t num_head = params.num_head;
    int64_t head_dim = params.head_dim;
    int64_t num_kv_head = params.num_kv_head;

    synGEMMParams gemm_params_f_f;
    gemm_params_f_f.transpose_a = false;
    gemm_params_f_f.transpose_b = false;

    synGEMMParams gemm_params_t_f;
    gemm_params_t_f.transpose_a = true;
    gemm_params_t_f.transpose_b = false;

    synGEMMParams gemm_params_f_t;
    gemm_params_f_t.transpose_a = false;
    gemm_params_f_t.transpose_b = true;

    auto src = createTensorFromCT(&ct, src_index);
    auto qkv_weights = createTensorFromCT(&ct, qkv_weights_index);

    std::vector<synTensor> linear_inputs;
    linear_inputs.push_back(src);
    linear_inputs.push_back(qkv_weights);
    synTensor qkv_biases;
    if (params.with_qkv_biases) {
      qkv_biases = createTensorFromCT(&ct, qkv_biases_index);
    }

    auto tmp_dims = src_dims;
    auto wt_dims = ins[qkv_weights_index].dims;
    tmp_dims[1] = params.transpose ? wt_dims[0] : wt_dims[1];
    auto qkv_out = createTensorNoPresist("qkv_out", dtype_, tmp_dims);
    std::vector<synTensor> linear_outputs;
    linear_outputs.push_back(qkv_out);

    std::vector<synTensor> reshape_inputs;

    if (params.transpose) {
      if (params.with_qkv_biases) {
        linear_inputs.push_back(qkv_biases);
      }
      AddNodeLinear<T>(linear_inputs, linear_outputs, guid_ + "linear");
      reshape_inputs.push_back(qkv_out);
    } else {
      synGEMMParams gemm_params;
      gemm_params.transpose_a = false;
      gemm_params.transpose_b = params.transpose;

      AddNodeMixedPrecisionGemm<T>(params.use_fp8,
                                   ct,
                                   src_scale_index,
                                   qkv_weights_scale_index,
                                   0,
                                   0,
                                   linear_inputs,
                                   linear_outputs,
                                   gemm_params,
                                   "batchgemm");

      if (params.with_qkv_biases) {
        auto qkv_out_with_bias =
            createTensorNoPresist("qkv_out_with_bias", dtype_, tmp_dims);
        std::vector<synTensor> qkv_add_inputs;
        qkv_add_inputs.push_back(qkv_out);
        qkv_add_inputs.push_back(qkv_biases);
        std::vector<synTensor> qkv_add_outputs = {qkv_out_with_bias};
        AddNodeAdd<T>(qkv_add_inputs, qkv_add_outputs, guid_ + "add_bias");
        reshape_inputs.push_back(qkv_out_with_bias);
      } else {
        reshape_inputs.push_back(qkv_out);
      }
    }

    auto reshape_dims = src_dims;
    reshape_dims[1] = num_head + 2 * num_kv_head;
    reshape_dims.push_back(head_dim);

    std::vector<synTensor> reshape_outputs;
    auto reshape_out =
        createTensorNoPresist("reshape_out", dtype_, reshape_dims);
    reshape_outputs.push_back(reshape_out);
    AddNodeReshape(reshape_inputs, reshape_outputs, guid_ + "reshape_qkv");

    std::vector<int64_t> q_dims;
    q_dims.push_back(batch_size);
    q_dims.push_back(num_head);
    q_dims.push_back(head_dim);
    std::vector<int64_t> kv_dims;
    kv_dims.push_back(batch_size);
    kv_dims.push_back(num_kv_head);
    kv_dims.push_back(head_dim);

    auto q_split = createTensorNoPresist("q_split", dtype_, q_dims);
    auto k_split = createTensorNoPresist("k_split", dtype_, kv_dims);
    auto v_split = createTensorNoPresist("v_split", dtype_, kv_dims);
    std::vector<synTensor> split_outpus;
    split_outpus.push_back(q_split);
    split_outpus.push_back(k_split);
    split_outpus.push_back(v_split);

    synSplitParams splitParams;
    splitParams.axis = 1;
    AddNodeSplit(reshape_outputs, split_outpus, splitParams, guid_ + "split");

    std::vector<synTensor> rotary_embs_inputs;
    auto rotary_embs_c = createTensorFromCT(&ct, rotary_embs_index);
    rotary_embs_inputs.push_back(rotary_embs_c);

    auto rotary_embs_dims = ins[rotary_embs_index].dims;
    rotary_embs_dims[0] = 1;

    std::vector<synTensor> cos_inputs;
    auto cos_in = createTensorNoPresist("cos_in", dtype_, rotary_embs_dims);
    cos_inputs.push_back(cos_in);

    synSliceParamsV2 sliceParams;
    for (uint64_t i = 0; i < rotary_embs_dims.size(); i++) {
      sliceParams.axes[i] = i;
      sliceParams.steps[i] = 1;
      sliceParams.starts[i] = 0;
      sliceParams.ends[i] = rotary_embs_dims[rotary_embs_dims.size() - 1 - i];
    }
    AddNodeSlice(
        rotary_embs_inputs, cos_inputs, sliceParams, guid_ + "slice_cos");

    std::vector<synTensor> sin_inputs;
    auto sin_in = createTensorNoPresist("sin_in", dtype_, rotary_embs_dims);
    sin_inputs.push_back(sin_in);
    sliceParams.starts[rotary_embs_dims.size() - 1] = 1;
    sliceParams.ends[rotary_embs_dims.size() - 1] = 2;
    AddNodeSlice(
        rotary_embs_inputs, sin_inputs, sliceParams, guid_ + "slice_sin");

    rotary_embs_dims.erase(rotary_embs_dims.begin());
    auto sin_sq =
        createTensorNoPresist("sin_squeezed", dtype_, rotary_embs_dims);
    std::vector<synTensor> sin_squeezed;
    sin_squeezed.push_back(sin_sq);

    synSqueezeParams squeezeParams;
    squeezeParams.axis = 3;
    AddNodeSqueeze(
        sin_inputs, sin_squeezed, squeezeParams, guid_ + "squeeze_sin");

    auto cos_sq =
        createTensorNoPresist("cos_squeezed", dtype_, rotary_embs_dims);
    std::vector<synTensor> cos_squeezed;
    cos_squeezed.push_back(cos_sq);
    AddNodeSqueeze(
        cos_inputs, cos_squeezed, squeezeParams, guid_ + "squeeze_cos");

    std::vector<synTensor> inputs_q;
    std::vector<synTensor> outputs_q;

    std::vector<synTensor> inputs_k;
    std::vector<synTensor> outputs_k;

    if (params.use_qk_rmsnorm) {
      auto q_rmsnorm = createTensorNoPresist("q_rmsnorm", dtype_, q_dims);
      auto k_rmsnorm = createTensorNoPresist("k_rmsnorm", dtype_, kv_dims);

      synTensor q_gamma = createTensorFromCT(&ct, q_gamma_index);
      synTensor k_gamma = createTensorFromCT(&ct, k_gamma_index);

      auto tmp_q_dims = q_dims;
      tmp_q_dims[2] = 1;
      auto q_rmsnorm_var =
          createTensorNoPresist("q_rmsnorm_var", dtype_, tmp_q_dims);

      auto tmp_k_dims = kv_dims;
      tmp_k_dims[2] = 1;
      auto k_rmsnorm_var =
          createTensorNoPresist("k_rmsnorm_var", dtype_, tmp_k_dims);

      std::vector<synTensor> rmsnorm_inputs_q;
      rmsnorm_inputs_q.push_back(q_split);
      rmsnorm_inputs_q.push_back(q_gamma);

      std::vector<synTensor> rmsnorm_outputs_q;
      rmsnorm_outputs_q.push_back(q_rmsnorm);
      rmsnorm_outputs_q.push_back(q_rmsnorm_var);

      AddNodeRmsNorm<T>(rmsnorm_inputs_q,
                        rmsnorm_outputs_q,
                        params.rmsnorm_params,
                        guid_ + "q_rmsnorm");

      std::vector<synTensor> rmsnorm_inputs_k;
      rmsnorm_inputs_k.push_back(k_split);
      rmsnorm_inputs_k.push_back(k_gamma);
      std::vector<synTensor> rmsnorm_outputs_k;
      rmsnorm_outputs_k.push_back(k_rmsnorm);
      rmsnorm_outputs_k.push_back(k_rmsnorm_var);

      AddNodeRmsNorm<T>(rmsnorm_inputs_k,
                        rmsnorm_outputs_k,
                        params.rmsnorm_params,
                        guid_ + "k_rmsnorm");

      inputs_q.push_back(q_rmsnorm);
      inputs_k.push_back(k_rmsnorm);
    } else {
      inputs_q.push_back(q_split);
      inputs_k.push_back(k_split);
    }
    inputs_q.push_back(sin_sq);
    inputs_q.push_back(cos_sq);

    auto q_states = createTensorNoPresist("q_states", dtype_, q_dims);
    outputs_q.push_back(q_states);

    ns_RoPESt2::ParamsV2 ropeParams;
    ropeParams.offset = 0;
    ropeParams.mode = params.use_neox_style
                          ? ROTARY_POS_EMBEDDING_MODE_BLOCKWISE
                          : ROTARY_POS_EMBEDDING_MODE_PAIRWISE;
    AddNodeRope<T>(inputs_q, outputs_q, ropeParams, guid_ + "rope_q");

    inputs_k.push_back(sin_sq);
    inputs_k.push_back(cos_sq);

    auto k_rope = createTensorNoPresist("k_rope", dtype_, kv_dims);
    outputs_k.push_back(k_rope);
    AddNodeRope<T>(inputs_k, outputs_k, ropeParams, guid_ + "rope_k");

    //////////////////////////////////////////////////////////////////
    std::vector<int64_t> indices_concat_dims =
        std::vector<int64_t>(ins[block_indices_index].dims);
    indices_concat_dims.emplace_back(1);

    std::vector<synTensor> inputs_concat;
    inputs_concat.push_back(createTensor(indices_concat_dims.size(),
                                         ins[block_indices_index].type,
                                         indices_concat_dims,
                                         true,
                                         ins[block_indices_index].name));
    inputs_concat.push_back(createTensor(indices_concat_dims.size(),
                                         ins[block_offsets_index].type,
                                         indices_concat_dims,
                                         true,
                                         ins[block_offsets_index].name));

    std::vector<synTensor> outputs_concat;
    indices_concat_dims.back() = 2;
    auto indices_concat = createTensor(indices_concat_dims.size(),
                                       ins[block_indices_index].type,
                                       indices_concat_dims,
                                       false,
                                       "indices_concat");
    outputs_concat.push_back(indices_concat);

    synConcatenateParams concatParams;
    concatParams.axis = 0;
    AddNodeConcat(
        inputs_concat, outputs_concat, concatParams, guid_ + "concat");

    synSectionHandle kCache_section = createSection();
    auto key_cache =
        createTensorFromCT(&ct, key_cache_index, true, kCache_section);
    auto kCache_out = createTensorFromCT(&ct, 1, false, kCache_section);
    std::vector<synTensor> inputs_scatter_k;
    inputs_scatter_k.push_back(key_cache);
    inputs_scatter_k.push_back(indices_concat);
    std::vector<synTensor> outputs_scatter_k;
    outputs_scatter_k.push_back(kCache_out);

    ns_CastKernel::Params convert_fp8_params;
    convert_fp8_params.round_mode = CAST_ROUND_HALF_NE;

    if (kv_dtype == synDataType::syn_type_fp8_143) {
      auto k_rope_fp8 = createTensorNoPresist(
          "k_rope_fp8", synDataType::syn_type_fp8_143, kv_dims);
      auto k_scale = createTensorFromCT(&ct, k_scale_index);
      std::vector<synTensor> k_convert_inputs;
      k_convert_inputs.push_back(k_rope);
      k_convert_inputs.push_back(k_scale);
      std::vector<synTensor> k_convert_outputs;
      k_convert_outputs.push_back(k_rope_fp8);
      AddNodeConvertToFP8<T>(k_convert_inputs,
                             k_convert_outputs,
                             convert_fp8_params,
                             guid_ + "cast_k_fp8");
      inputs_scatter_k.push_back(k_rope_fp8);
      AddNodeScatter<phi::dtype::float8_e4m3fn>(
          inputs_scatter_k, outputs_scatter_k, guid_ + "index_put_k_fp8");
    } else {
      inputs_scatter_k.push_back(k_rope);
      AddNodeScatter<T>(
          inputs_scatter_k, outputs_scatter_k, guid_ + "index_put_k");
    }

    synSectionHandle vCache_section = createSection();
    auto value_cache =
        createTensorFromCT(&ct, value_cache_index, true, vCache_section);
    auto vCache_out = createTensorFromCT(&ct, 2, false, vCache_section);
    std::vector<synTensor> inputs_scatter_v;
    inputs_scatter_v.push_back(value_cache);
    inputs_scatter_v.push_back(indices_concat);
    std::vector<synTensor> outputs_scatter_v;
    outputs_scatter_v.push_back(vCache_out);

    if (kv_dtype == synDataType::syn_type_fp8_143) {
      auto v_split_fp8 = createTensorNoPresist(
          "v_split_fp8", synDataType::syn_type_fp8_143, kv_dims);
      auto v_scale = createTensorFromCT(&ct, v_scale_index);
      std::vector<synTensor> v_convert_inputs;
      v_convert_inputs.push_back(v_split);
      v_convert_inputs.push_back(v_scale);
      std::vector<synTensor> v_convert_outputs;
      v_convert_outputs.push_back(v_split_fp8);
      AddNodeConvertToFP8<T>(v_convert_inputs,
                             v_convert_outputs,
                             convert_fp8_params,
                             guid_ + "cast_v_fp8");
      inputs_scatter_v.push_back(v_split_fp8);
      AddNodeScatter<phi::dtype::float8_e4m3fn>(
          inputs_scatter_v, outputs_scatter_v, guid_ + "index_put_v_fp8");
    } else {
      inputs_scatter_v.push_back(v_split);
      AddNodeScatter<T>(
          inputs_scatter_v, outputs_scatter_v, guid_ + "index_put_v");
    }

    //////////////////////////////////////////////////////////////////

    std::vector<int64_t> scaler_dims = {1};
    auto scaler_tensor =
        createTensorNoPresist("scaler_tensor", syn_type_bf16, scaler_dims);
    std::vector<synTensor> scaler;
    scaler.push_back(scaler_tensor);
    AddNodeFull<T>(scaler, params.const_params, guid_ + "full_scale");

    std::vector<synTensor> scaled_q_in;
    scaled_q_in.push_back(q_states);
    scaled_q_in.push_back(scaler_tensor);

    auto scaled_q = createTensorNoPresist("scaled_q", dtype_, q_dims);
    std::vector<synTensor> scaled_q_out;
    scaled_q_out.push_back(scaled_q);

    AddNodeMultiply<T>(scaled_q_in, scaled_q_out, guid_ + "mul_scale_q");

    std::vector<int64_t> reshape_q_dims;
    reshape_q_dims.push_back(batch_size);
    reshape_q_dims.push_back(hidden_size);

    auto reshaped_q =
        createTensorNoPresist("reshaped_q", dtype_, reshape_q_dims);
    std::vector<synTensor> reshape_q_out;
    reshape_q_out.push_back(reshaped_q);

    AddNodeReshape(scaled_q_out, reshape_q_out, guid_ + "reshape_scale_q");

    /*******************************/

    std::vector<synTensor> map_q_in;
    auto block_mapping = createTensorFromCT(&ct, block_mapping_index);
    map_q_in.push_back(block_mapping);
    map_q_in.push_back(reshaped_q);

    std::vector<int64_t> map_q_dims;
    map_q_dims.push_back(num_of_block);
    map_q_dims.push_back(hidden_size);
    auto mapped_q = createTensorNoPresist("mapped_q", dtype_, map_q_dims);
    std::vector<synTensor> map_q_out;
    map_q_out.push_back(mapped_q);

    AddNodeGemm(map_q_in, map_q_out, gemm_params_f_f, guid_ + "gemm_map_q");

    std::vector<int64_t> reshape_map_q_dims;
    reshape_map_q_dims.push_back(num_of_block);
    reshape_map_q_dims.push_back(num_head);
    reshape_map_q_dims.push_back(1);
    reshape_map_q_dims.push_back(head_dim);

    auto reshaped_map_q =
        createTensorNoPresist("reshaped_map_q", dtype_, reshape_map_q_dims);
    std::vector<synTensor> reshape_map_q_out;
    reshape_map_q_out.push_back(reshaped_map_q);

    AddNodeReshape(map_q_out, reshape_map_q_out, guid_ + "reshape_map_q");

    /*******************************/

    std::vector<synTensor> index_select_k_in;
    std::vector<synTensor> index_select_v_in;

    auto block_list = createTensorFromCT(&ct, block_list_index);

    index_select_k_in.push_back(kCache_out);
    index_select_v_in.push_back(vCache_out);
    index_select_k_in.push_back(block_list);
    index_select_v_in.push_back(block_list);

    std::vector<int64_t> index_selected_dims;
    index_selected_dims.push_back(num_of_block);
    index_selected_dims.push_back(block_size);
    index_selected_dims.push_back(num_kv_head);
    index_selected_dims.push_back(head_dim);

    auto index_select_k_i = createTensorNoPresist(
        "index_select_k_i", kv_dtype, index_selected_dims);
    auto index_select_v_i = createTensorNoPresist(
        "index_select_v_i", kv_dtype, index_selected_dims);
    std::vector<synTensor> index_select_k_out;
    index_select_k_out.push_back(index_select_k_i);
    std::vector<synTensor> index_select_v_out;
    index_select_v_out.push_back(index_select_v_i);

    if (kv_dtype == synDataType::syn_type_fp8_143) {
      AddNodeIndexSelect<phi::dtype::float8_e4m3fn>(
          index_select_k_in,
          index_select_k_out,
          params.index_select_params,
          guid_ + "index_select_k_i_fp8");
      AddNodeIndexSelect<phi::dtype::float8_e4m3fn>(
          index_select_v_in,
          index_select_v_out,
          params.index_select_params,
          guid_ + "index_select_v_i_fp8");
    } else {
      AddNodeIndexSelect<T>(index_select_k_in,
                            index_select_k_out,
                            params.index_select_params,
                            guid_ + "index_select_k_i");
      AddNodeIndexSelect<T>(index_select_v_in,
                            index_select_v_out,
                            params.index_select_params,
                            guid_ + "index_select_v_i");
    }

    std::vector<int> axis = {0, 2, 1, 3};
    synTransposeParams trans_params;
    for (size_t i = 0; i < axis.size(); i++) {
      trans_params.permutation[i] =
          static_cast<TransposePermutationDim>(axis[i]);
    }
    trans_params.tensorDim = 4;

    std::vector<int64_t> transpose_dims;
    transpose_dims.push_back(num_of_block);
    transpose_dims.push_back(num_kv_head);
    transpose_dims.push_back(block_size);
    transpose_dims.push_back(head_dim);

    auto transpose_k =
        createTensorNoPresist("transpose_k", kv_dtype, transpose_dims);
    std::vector<synTensor> trans_index_select_k;
    trans_index_select_k.push_back(transpose_k);

    AddNodeTranspose(index_select_k_out,
                     trans_index_select_k,
                     trans_params,
                     guid_ + "transpose_k");

    auto transpose_v =
        createTensorNoPresist("transpose_v", kv_dtype, transpose_dims);
    std::vector<synTensor> trans_index_select_v;
    trans_index_select_v.push_back(transpose_v);

    AddNodeTranspose(index_select_v_out,
                     trans_index_select_v,
                     trans_params,
                     guid_ + "transpose_v");

    std::vector<synTensor> q_k_in;
    q_k_in.push_back(reshaped_map_q);
    q_k_in.push_back(transpose_k);

    std::vector<int64_t> q_k_dims;
    q_k_dims.push_back(num_of_block);
    q_k_dims.push_back(num_head);
    q_k_dims.push_back(1);
    q_k_dims.push_back(block_size);
    auto q_k = createTensorNoPresist("q_k", dtype_, q_k_dims);
    std::vector<synTensor> q_k_out;
    q_k_out.push_back(q_k);

    // Q*k^T
    AddNodeMixedPrecisionGemm<T>(params.use_fp8,
                                 ct,
                                 q_scale_index,
                                 k_scale_index,
                                 0,
                                 1,
                                 q_k_in,
                                 q_k_out,
                                 gemm_params_f_t,
                                 "q_k");

    /*******************************/
    auto block_bias = createTensorFromCT(&ct, block_bias_index);
    std::vector<synTensor> block_bias_in;
    block_bias_in.push_back(block_bias);

    std::vector<int64_t> reshaped_bias_dims;
    reshaped_bias_dims.push_back(num_of_block);
    reshaped_bias_dims.push_back(1);
    reshaped_bias_dims.push_back(1);
    reshaped_bias_dims.push_back(block_size);

    auto reshaped_bias =
        createTensorNoPresist("reshaped_bias", dtype_, reshaped_bias_dims);
    std::vector<synTensor> block_bias_out;
    block_bias_out.push_back(reshaped_bias);

    AddNodeReshape(block_bias_in, block_bias_out, guid_ + "reshaped_bias");

    std::vector<synTensor> add_bias_in;
    add_bias_in.push_back(q_k);
    add_bias_in.push_back(reshaped_bias);

    auto add_bias = createTensorNoPresist("add_bias", dtype_, q_k_dims);
    std::vector<synTensor> add_bias_out;
    add_bias_out.push_back(add_bias);

    AddNodeAdd<T>(add_bias_in, add_bias_out, guid_ + "add_bias");
    /*******************************/

    std::vector<int64_t> block_max_dims;
    block_max_dims.push_back(num_of_block);
    block_max_dims.push_back(num_head);
    block_max_dims.push_back(1);
    block_max_dims.push_back(1);

    auto block_max = createTensorNoPresist("block_max", dtype_, block_max_dims);
    std::vector<synTensor> block_max_out;
    block_max_out.push_back(block_max);

    AddNodeReduceMax<T>(
        add_bias_out, block_max_out, params.reduce_params, guid_ + "reduceMax");

    /**************************************************************/

    std::vector<int64_t> sum_adjusted_dims;
    sum_adjusted_dims.push_back(num_of_block);
    sum_adjusted_dims.push_back(num_head);

    auto block_max_2D =
        createTensorNoPresist("block_max_2D", dtype_, sum_adjusted_dims);
    std::vector<synTensor> block_max_2D_out;
    block_max_2D_out.push_back(block_max_2D);

    AddNodeReshape(
        block_max_out, block_max_2D_out, guid_ + "squeeze_block_max");

    std::vector<int64_t> group_max_dims;
    group_max_dims.push_back(batch_size + 1);
    group_max_dims.push_back(num_head);

    auto group_max = createTensorNoPresist("group_max", dtype_, group_max_dims);
    std::vector<synTensor> group_max_tensor;
    group_max_tensor.push_back(group_max);

    params.const_params.constant.f = -std::numeric_limits<float>::infinity();

    AddNodeFull<T>(group_max_tensor, params.const_params, guid_ + "full_inf");

    auto block_groups = createTensorFromCT(&ct, block_groups_index);
    std::vector<synTensor> index_reduce_in;
    index_reduce_in.push_back(group_max);
    index_reduce_in.push_back(block_groups);
    index_reduce_in.push_back(block_max_2D);

    auto reduced_group_max =
        createTensorNoPresist("reduced_group_max", dtype_, group_max_dims);
    std::vector<synTensor> index_reduce_out;
    index_reduce_out.push_back(reduced_group_max);

    AddNodeIndexReduce<T>(index_reduce_in,
                          index_reduce_out,
                          params.index_reduce_params,
                          guid_ + "index_reduce_amax");

    std::vector<synTensor> index_select_groupmax_in;
    index_select_groupmax_in.push_back(reduced_group_max);
    index_select_groupmax_in.push_back(block_groups);

    auto selected_group_max =
        createTensorNoPresist("selected_group_max", dtype_, sum_adjusted_dims);
    std::vector<synTensor> index_select_groupmax_out;
    index_select_groupmax_out.push_back(selected_group_max);
    params.index_select_params.axis = 1;
    AddNodeIndexSelect<T>(index_select_groupmax_in,
                          index_select_groupmax_out,
                          params.index_select_params,
                          guid_ + "index_select_groupmax");

    std::vector<synTensor> sub_group_max_in;
    sub_group_max_in.push_back(block_max_2D);
    sub_group_max_in.push_back(selected_group_max);

    auto sub_group_max =
        createTensorNoPresist("sub_group_max", dtype_, sum_adjusted_dims);
    std::vector<synTensor> sub_group_max_out;
    sub_group_max_out.push_back(sub_group_max);

    AddNodeSub<T>(sub_group_max_in, sub_group_max_out, guid_ + "sub_group_max");

    auto block_adjustment =
        createTensorNoPresist("block_adjustment", dtype_, sum_adjusted_dims);
    std::vector<synTensor> block_adjustment_out;
    block_adjustment_out.push_back(block_adjustment);
    AddNodeExp<T>(sub_group_max_out,
                  block_adjustment_out,
                  guid_ + "exp_block_adjustment");

    /**************************************************************/

    std::vector<synTensor> sub_block_max_in;
    sub_block_max_in.push_back(add_bias);
    sub_block_max_in.push_back(block_max);

    auto sub_block_max =
        createTensorNoPresist("sub_block_max", dtype_, q_k_dims);
    std::vector<synTensor> sub_block_max_out;
    sub_block_max_out.push_back(sub_block_max);
    AddNodeSub<T>(sub_block_max_in, sub_block_max_out, guid_ + "sub_block_max");

    auto score = createTensorNoPresist("score", dtype_, q_k_dims);
    std::vector<synTensor> score_out;
    score_out.push_back(score);
    AddNodeExp<T>(sub_block_max_out, score_out, guid_ + "exp_score");

    /*******************************/

    std::vector<synTensor> score_v_in;
    score_v_in.push_back(score);
    score_v_in.push_back(transpose_v);

    auto score_v = createTensorNoPresist("score_v", dtype_, reshape_map_q_dims);
    std::vector<synTensor> score_v_out;
    score_v_out.push_back(score_v);

    // Score*V
    AddNodeMixedPrecisionGemm<T>(params.use_fp8,
                                 ct,
                                 a_scale_index,
                                 v_scale_index,
                                 0,
                                 1,
                                 score_v_in,
                                 score_v_out,
                                 gemm_params_f_f,
                                 "score_v");

    auto reduceSum = createTensorNoPresist("reduceSum", dtype_, block_max_dims);
    std::vector<synTensor> reduceSum_out;
    reduceSum_out.push_back(reduceSum);

    AddNodeReduceSum<T>(
        score_out, reduceSum_out, params.reduce_params, guid_ + "reduceSum");

    auto block_sums_2D =
        createTensorNoPresist("block_sums_2D", dtype_, sum_adjusted_dims);
    std::vector<synTensor> block_sums_2D_out;
    block_sums_2D_out.push_back(block_sums_2D);

    AddNodeReshape(
        reduceSum_out, block_sums_2D_out, guid_ + "squeeze_block_sums");

    std::vector<synTensor> sum_adjusted_in;
    sum_adjusted_in.push_back(block_sums_2D);
    sum_adjusted_in.push_back(block_adjustment);

    auto sum_adjusted =
        createTensorNoPresist("sum_adjusted", dtype_, sum_adjusted_dims);
    std::vector<synTensor> sum_adjusted_out;
    sum_adjusted_out.push_back(sum_adjusted);

    AddNodeMultiply<T>(
        sum_adjusted_in, sum_adjusted_out, guid_ + "mul_sum_adjusted");
    /***************************************************************/

    std::vector<synTensor> map_sum_adjusted_in;
    map_sum_adjusted_in.push_back(block_mapping);
    map_sum_adjusted_in.push_back(sum_adjusted);

    std::vector<int64_t> map_sum_adjusted_dims;
    map_sum_adjusted_dims.push_back(batch_size);
    map_sum_adjusted_dims.push_back(num_head);
    auto mapped_sum_adjusted = createTensorNoPresist(
        "mapped_sum_adjusted", dtype_, map_sum_adjusted_dims);
    std::vector<synTensor> map_sum_adjusted_out;
    map_sum_adjusted_out.push_back(mapped_sum_adjusted);

    AddNodeGemm(map_sum_adjusted_in,
                map_sum_adjusted_out,
                gemm_params_t_f,
                guid_ + "gemm_map_sum_adjusted");

    std::vector<synTensor> group_sum_adjusted_in;
    group_sum_adjusted_in.push_back(block_mapping);
    group_sum_adjusted_in.push_back(mapped_sum_adjusted);

    auto group_sum_adjusted =
        createTensorNoPresist("group_sum_adjusted", dtype_, sum_adjusted_dims);
    std::vector<synTensor> group_sum_adjusted_out;
    group_sum_adjusted_out.push_back(group_sum_adjusted);

    AddNodeGemm(group_sum_adjusted_in,
                group_sum_adjusted_out,
                gemm_params_f_f,
                guid_ + "gemm_group_sum_adjusted");

    /*******************************/

    auto reshaped_group_sum_adjusted = createTensorNoPresist(
        "reshaped_group_sum_adjusted", dtype_, block_max_dims);
    std::vector<synTensor> group_sum_adjusted_4D;
    group_sum_adjusted_4D.push_back(reshaped_group_sum_adjusted);

    AddNodeReshape(group_sum_adjusted_out,
                   group_sum_adjusted_4D,
                   guid_ + "reshaped_group_sum_adjusted");

    auto reshaped_sum_adjusted =
        createTensorNoPresist("reshaped_sum_adjusted", dtype_, block_max_dims);
    std::vector<synTensor> sum_adjusted_4D;
    sum_adjusted_4D.push_back(reshaped_sum_adjusted);

    AddNodeReshape(
        sum_adjusted_out, sum_adjusted_4D, guid_ + "reshaped_sum_adjusted");

    auto reshaped_block_adjustment = createTensorNoPresist(
        "reshaped_block_adjustment", dtype_, block_max_dims);
    std::vector<synTensor> block_adjustment_4D;
    block_adjustment_4D.push_back(reshaped_block_adjustment);

    AddNodeReshape(block_adjustment_out,
                   block_adjustment_4D,
                   guid_ + "reshaped_block_adjustment");

    std::vector<synTensor> max_sum_adjust_in;
    max_sum_adjust_in.push_back(reshaped_group_sum_adjusted);
    max_sum_adjust_in.push_back(reshaped_sum_adjusted);
    auto max_sum_adjust =
        createTensorNoPresist("max_sum_adjust", dtype_, block_max_dims);
    std::vector<synTensor> max_sum_adjust_out;
    max_sum_adjust_out.push_back(max_sum_adjust);

    AddNodeMaximum<T>(
        max_sum_adjust_in, max_sum_adjust_out, guid_ + "max_sum_adjust");

    std::vector<synTensor> rescale_in;
    rescale_in.push_back(reshaped_block_adjustment);
    rescale_in.push_back(max_sum_adjust);
    auto rescale = createTensorNoPresist("rescale", dtype_, block_max_dims);
    std::vector<synTensor> rescale_out;
    rescale_out.push_back(rescale);
    AddNodeDivide<T>(rescale_in, rescale_out, guid_ + "div_rescale");

    std::vector<synTensor> rescale_v_in;
    rescale_v_in.push_back(rescale);
    rescale_v_in.push_back(score_v);

    auto rescale_v =
        createTensorNoPresist("rescale_v", dtype_, reshape_map_q_dims);
    std::vector<synTensor> rescale_v_out;
    rescale_v_out.push_back(rescale_v);

    AddNodeMultiply<T>(rescale_v_in, rescale_v_out, guid_ + "mul_rescale_v");

    auto reshape_attn =
        createTensorNoPresist("reshape_attn", dtype_, map_q_dims);
    std::vector<synTensor> reshape_attn_out;
    reshape_attn_out.push_back(reshape_attn);

    AddNodeReshape(rescale_v_out, reshape_attn_out, guid_ + "reshape_attn");

    std::vector<synTensor> map_attn_in;
    map_attn_in.push_back(block_mapping);
    map_attn_in.push_back(reshape_attn);

    auto mapped_attn =
        createTensorNoPresist("mapped_attn", dtype_, reshape_q_dims);
    std::vector<synTensor> map_attn_out;
    map_attn_out.push_back(mapped_attn);

    AddNodeGemm(
        map_attn_in, map_attn_out, gemm_params_t_f, guid_ + "gemm_map_attn");

    std::vector<synTensor> proj_in;
    auto linear_weights = createTensorFromCT(&ct, linear_weights_index);
    proj_in.push_back(mapped_attn);
    proj_in.push_back(linear_weights);

    auto linear_out = createTensorFromCT(&ct, 0, false);
    std::vector<synTensor> proj_out;
    proj_out.push_back(linear_out);

    // Final linear
    AddNodeMixedPrecisionGemm<T>(params.use_fp8,
                                 ct,
                                 o_linear_scale_x_index,
                                 o_linear_scale_y_index,
                                 0,
                                 0,
                                 proj_in,
                                 proj_out,
                                 gemm_params_f_f,
                                 "proj");
  }

 protected:
  synDataType dtype_;
};

class FusedGQABlockAttention : public FusedBlockAttentionBase {
 public:
  explicit FusedGQABlockAttention(std::string guid_prefix, synDataType dtype)
      : FusedBlockAttentionBase(guid_prefix, false), dtype_(dtype) {}
  template <typename T>
  void AddNode(ConvertTensors& ct, FusedBlockAttentionParams& params) {
    auto ins = ct.GetTensors();
    auto outs = ct.GetTensors(false);
    auto kv_dtype = params.use_fp8 ? synDataType::syn_type_fp8_143 : dtype_;

    int index_base = 0;
    int src_index = (index_base++);             // 0
    int rotary_embs_index = (index_base++);     // 1
    int key_cache_index = (index_base++);       // 2
    int value_cache_index = (index_base++);     // 3
    int block_groups_index = (index_base++);    // 4
    int block_list_index = (index_base++);      // 5
    int block_mapping_index = (index_base++);   // 6
    int block_bias_index = (index_base++);      // 7
    int block_indices_index = (index_base++);   // 8
    int block_offsets_index = (index_base++);   // 9
    int qkv_weights_index = (index_base++);     // 10
    int linear_weights_index = (index_base++);  // 11

    int src_scale_index = -1, qkv_weights_scale_index = -1, q_scale_index = -1,
        k_scale_index = -1, a_scale_index = -1, v_scale_index = -1,
        o_linear_scale_x_index = -1, o_linear_scale_y_index = -1,
        qkv_biases_index = -1, q_gamma_index = -1, k_gamma_index = -1;
    if (params.use_fp8) {
      src_scale_index = (index_base++);
      qkv_weights_scale_index = (index_base++);
      q_scale_index = (index_base++);
      k_scale_index = (index_base++);
      a_scale_index = (index_base++);
      v_scale_index = (index_base++);
      o_linear_scale_x_index = (index_base++);
      o_linear_scale_y_index = (index_base++);
    }

    if (params.with_qkv_biases) {
      qkv_biases_index = (index_base++);
    }

    if (params.use_qk_rmsnorm) {
      q_gamma_index = (index_base++);
      k_gamma_index = (index_base++);
    }

    std::vector<int64_t> src_dims = std::vector<int64_t>(ins[src_index].dims);

    int64_t batch_size = src_dims[0];
    int64_t hidden_size = ins[linear_weights_index].dims[0];
    int64_t block_size = ins[key_cache_index].dims[1];
    int64_t num_of_block = ins[block_list_index].dims[0];

    int64_t num_head = params.num_head;
    int64_t head_dim = params.head_dim;
    int64_t num_kv_head = params.num_kv_head;
    int64_t ngroups = num_head / num_kv_head;

    synGEMMParams gemm_params_f_f;
    gemm_params_f_f.transpose_a = false;
    gemm_params_f_f.transpose_b = false;

    synGEMMParams gemm_params_t_f;
    gemm_params_t_f.transpose_a = true;
    gemm_params_t_f.transpose_b = false;

    synGEMMParams gemm_params_f_t;
    gemm_params_f_t.transpose_a = false;
    gemm_params_f_t.transpose_b = true;

    auto src = createTensorFromCT(&ct, src_index);
    auto qkv_weights = createTensorFromCT(&ct, qkv_weights_index);

    std::vector<synTensor> linear_inputs;
    linear_inputs.push_back(src);
    linear_inputs.push_back(qkv_weights);
    synTensor qkv_biases;
    if (params.with_qkv_biases) {
      qkv_biases = createTensorFromCT(&ct, qkv_biases_index);
    }

    auto tmp_dims = src_dims;
    auto wt_dims = ins[qkv_weights_index].dims;
    tmp_dims[1] = params.transpose ? wt_dims[0] : wt_dims[1];
    auto qkv_out = createTensorNoPresist("qkv_out", dtype_, tmp_dims);
    std::vector<synTensor> linear_outputs;
    linear_outputs.push_back(qkv_out);

    std::vector<synTensor> reshape_inputs;

    if (params.transpose) {
      if (params.with_qkv_biases) {
        linear_inputs.push_back(qkv_biases);
      }
      AddNodeLinear<T>(linear_inputs, linear_outputs, guid_ + "linear");
      reshape_inputs.push_back(qkv_out);
    } else {
      synGEMMParams gemm_params;
      gemm_params.transpose_a = false;
      gemm_params.transpose_b = params.transpose;

      AddNodeMixedPrecisionGemm<T>(params.use_fp8,
                                   ct,
                                   src_scale_index,
                                   qkv_weights_scale_index,
                                   0,
                                   0,
                                   linear_inputs,
                                   linear_outputs,
                                   gemm_params,
                                   "batchgemm");

      if (params.with_qkv_biases) {
        auto qkv_out_with_bias =
            createTensorNoPresist("qkv_out_with_bias", dtype_, tmp_dims);
        std::vector<synTensor> qkv_add_inputs;
        qkv_add_inputs.push_back(qkv_out);
        qkv_add_inputs.push_back(qkv_biases);
        std::vector<synTensor> qkv_add_outputs = {qkv_out_with_bias};
        AddNodeAdd<T>(qkv_add_inputs, qkv_add_outputs, guid_ + "add_bias");
        reshape_inputs.push_back(qkv_out_with_bias);
      } else {
        reshape_inputs.push_back(qkv_out);
      }
    }

    auto reshape_dims = src_dims;
    reshape_dims[1] = num_head + 2 * num_kv_head;
    reshape_dims.push_back(head_dim);

    std::vector<synTensor> reshape_outputs;
    auto reshape_out =
        createTensorNoPresist("reshape_out", dtype_, reshape_dims);
    reshape_outputs.push_back(reshape_out);
    AddNodeReshape(reshape_inputs, reshape_outputs, guid_ + "reshape_qkv");

    std::vector<int64_t> q_dims;
    q_dims.push_back(batch_size);
    q_dims.push_back(num_head);
    q_dims.push_back(head_dim);
    std::vector<int64_t> kv_dims;
    kv_dims.push_back(batch_size);
    kv_dims.push_back(num_kv_head);
    kv_dims.push_back(head_dim);

    auto q_split = createTensorNoPresist("q_split", dtype_, q_dims);
    auto k_split = createTensorNoPresist("k_split", dtype_, kv_dims);
    auto v_split = createTensorNoPresist("v_split", dtype_, kv_dims);
    std::vector<synTensor> split_outpus;
    split_outpus.push_back(q_split);
    split_outpus.push_back(k_split);
    split_outpus.push_back(v_split);

    synSplitParams splitParams;
    splitParams.axis = 1;
    AddNodeSplit(reshape_outputs, split_outpus, splitParams, guid_ + "split");

    std::vector<synTensor> rotary_embs_inputs;
    auto rotary_embs_c = createTensorFromCT(&ct, rotary_embs_index);
    rotary_embs_inputs.push_back(rotary_embs_c);

    auto rotary_embs_dims = ins[rotary_embs_index].dims;
    rotary_embs_dims[0] = 1;

    std::vector<synTensor> cos_inputs;
    auto cos_in = createTensorNoPresist("cos_in", dtype_, rotary_embs_dims);
    cos_inputs.push_back(cos_in);

    synSliceParamsV2 sliceParams;
    for (uint64_t i = 0; i < rotary_embs_dims.size(); i++) {
      sliceParams.axes[i] = i;
      sliceParams.steps[i] = 1;
      sliceParams.starts[i] = 0;
      sliceParams.ends[i] = rotary_embs_dims[rotary_embs_dims.size() - 1 - i];
    }
    AddNodeSlice(
        rotary_embs_inputs, cos_inputs, sliceParams, guid_ + "slice_cos");

    std::vector<synTensor> sin_inputs;
    auto sin_in = createTensorNoPresist("sin_in", dtype_, rotary_embs_dims);
    sin_inputs.push_back(sin_in);
    sliceParams.starts[rotary_embs_dims.size() - 1] = 1;
    sliceParams.ends[rotary_embs_dims.size() - 1] = 2;
    AddNodeSlice(
        rotary_embs_inputs, sin_inputs, sliceParams, guid_ + "slice_sin");

    rotary_embs_dims.erase(rotary_embs_dims.begin());
    auto sin_sq =
        createTensorNoPresist("sin_squeezed", dtype_, rotary_embs_dims);
    std::vector<synTensor> sin_squeezed;
    sin_squeezed.push_back(sin_sq);

    synSqueezeParams squeezeParams;
    squeezeParams.axis = 3;
    AddNodeSqueeze(
        sin_inputs, sin_squeezed, squeezeParams, guid_ + "squeeze_sin");

    auto cos_sq =
        createTensorNoPresist("cos_squeezed", dtype_, rotary_embs_dims);
    std::vector<synTensor> cos_squeezed;
    cos_squeezed.push_back(cos_sq);
    AddNodeSqueeze(
        cos_inputs, cos_squeezed, squeezeParams, guid_ + "squeeze_cos");

    std::vector<synTensor> inputs_q;
    std::vector<synTensor> outputs_q;

    std::vector<synTensor> inputs_k;
    std::vector<synTensor> outputs_k;

    if (params.use_qk_rmsnorm) {
      auto q_rmsnorm = createTensorNoPresist("q_rmsnorm", dtype_, q_dims);
      auto k_rmsnorm = createTensorNoPresist("k_rmsnorm", dtype_, kv_dims);
      synTensor q_gamma = createTensorFromCT(&ct, q_gamma_index);
      synTensor k_gamma = createTensorFromCT(&ct, k_gamma_index);

      auto tmp_q_dims = q_dims;
      tmp_q_dims[2] = 1;
      auto q_rmsnorm_var =
          createTensorNoPresist("q_rmsnorm_var", dtype_, tmp_q_dims);

      auto tmp_k_dims = kv_dims;
      tmp_k_dims[2] = 1;
      auto k_rmsnorm_var =
          createTensorNoPresist("k_rmsnorm_var", dtype_, tmp_k_dims);

      std::vector<synTensor> rmsnorm_inputs_q;
      rmsnorm_inputs_q.push_back(q_split);
      rmsnorm_inputs_q.push_back(q_gamma);
      std::vector<synTensor> rmsnorm_outputs_q;
      rmsnorm_outputs_q.push_back(q_rmsnorm);
      rmsnorm_outputs_q.push_back(q_rmsnorm_var);

      AddNodeRmsNorm<T>(rmsnorm_inputs_q,
                        rmsnorm_outputs_q,
                        params.rmsnorm_params,
                        guid_ + "q_rmsnorm");

      std::vector<synTensor> rmsnorm_inputs_k;
      rmsnorm_inputs_k.push_back(k_split);
      rmsnorm_inputs_k.push_back(k_gamma);
      std::vector<synTensor> rmsnorm_outputs_k;
      rmsnorm_outputs_k.push_back(k_rmsnorm);
      rmsnorm_outputs_k.push_back(k_rmsnorm_var);

      AddNodeRmsNorm<T>(rmsnorm_inputs_k,
                        rmsnorm_outputs_k,
                        params.rmsnorm_params,
                        guid_ + "k_rmsnorm");

      inputs_q.push_back(q_rmsnorm);
      inputs_k.push_back(k_rmsnorm);
    } else {
      inputs_q.push_back(q_split);
      inputs_k.push_back(k_split);
    }

    inputs_q.push_back(sin_sq);
    inputs_q.push_back(cos_sq);

    auto q_states = createTensorNoPresist("q_states", dtype_, q_dims);
    outputs_q.push_back(q_states);

    ns_RoPESt2::ParamsV2 ropeParams;
    ropeParams.offset = 0;
    ropeParams.mode = params.use_neox_style
                          ? ROTARY_POS_EMBEDDING_MODE_BLOCKWISE
                          : ROTARY_POS_EMBEDDING_MODE_PAIRWISE;
    AddNodeRope<T>(inputs_q, outputs_q, ropeParams, guid_ + "rope_q");

    inputs_k.push_back(sin_sq);
    inputs_k.push_back(cos_sq);

    auto k_rope = createTensorNoPresist("k_rope", dtype_, kv_dims);
    outputs_k.push_back(k_rope);
    AddNodeRope<T>(inputs_k, outputs_k, ropeParams, guid_ + "rope_k");

    //////////////////////////////////////////////////////////////////
    std::vector<int64_t> indices_concat_dims =
        std::vector<int64_t>(ins[block_indices_index].dims);
    indices_concat_dims.emplace_back(1);

    std::vector<synTensor> inputs_concat;
    inputs_concat.push_back(createTensor(indices_concat_dims.size(),
                                         ins[block_indices_index].type,
                                         indices_concat_dims,
                                         true,
                                         ins[block_indices_index].name));
    inputs_concat.push_back(createTensor(indices_concat_dims.size(),
                                         ins[block_offsets_index].type,
                                         indices_concat_dims,
                                         true,
                                         ins[block_offsets_index].name));

    std::vector<synTensor> outputs_concat;
    indices_concat_dims.back() = 2;
    auto indices_concat = createTensor(indices_concat_dims.size(),
                                       ins[block_indices_index].type,
                                       indices_concat_dims,
                                       false,
                                       "indices_concat");
    outputs_concat.push_back(indices_concat);

    synConcatenateParams concatParams;
    concatParams.axis = 0;
    AddNodeConcat(
        inputs_concat, outputs_concat, concatParams, guid_ + "concat");

    synSectionHandle kCache_section = createSection();
    auto key_cache =
        createTensorFromCT(&ct, key_cache_index, true, kCache_section);
    auto kCache_out = createTensorFromCT(&ct, 1, false, kCache_section);
    std::vector<synTensor> inputs_scatter_k;
    inputs_scatter_k.push_back(key_cache);
    inputs_scatter_k.push_back(indices_concat);
    std::vector<synTensor> outputs_scatter_k;
    outputs_scatter_k.push_back(kCache_out);

    ns_CastKernel::Params convert_fp8_params;
    convert_fp8_params.round_mode = CAST_ROUND_HALF_NE;

    if (kv_dtype == synDataType::syn_type_fp8_143) {
      auto k_rope_fp8 = createTensorNoPresist(
          "k_rope_fp8", synDataType::syn_type_fp8_143, kv_dims);
      auto k_scale = createTensorFromCT(&ct, k_scale_index);
      std::vector<synTensor> k_convert_inputs;
      k_convert_inputs.push_back(k_rope);
      k_convert_inputs.push_back(k_scale);
      std::vector<synTensor> k_convert_outputs;
      k_convert_outputs.push_back(k_rope_fp8);
      AddNodeConvertToFP8<T>(k_convert_inputs,
                             k_convert_outputs,
                             convert_fp8_params,
                             guid_ + "cast_k_fp8");
      inputs_scatter_k.push_back(k_rope_fp8);
      AddNodeScatter<phi::dtype::float8_e4m3fn>(
          inputs_scatter_k, outputs_scatter_k, guid_ + "index_put_k_fp8");
    } else {
      inputs_scatter_k.push_back(k_rope);
      AddNodeScatter<T>(
          inputs_scatter_k, outputs_scatter_k, guid_ + "index_put_k");
    }

    synSectionHandle vCache_section = createSection();
    auto value_cache =
        createTensorFromCT(&ct, value_cache_index, true, vCache_section);
    auto vCache_out = createTensorFromCT(&ct, 2, false, vCache_section);
    std::vector<synTensor> inputs_scatter_v;
    inputs_scatter_v.push_back(value_cache);
    inputs_scatter_v.push_back(indices_concat);
    std::vector<synTensor> outputs_scatter_v;
    outputs_scatter_v.push_back(vCache_out);

    if (kv_dtype == synDataType::syn_type_fp8_143) {
      auto v_split_fp8 = createTensorNoPresist(
          "v_split_fp8", synDataType::syn_type_fp8_143, kv_dims);
      auto v_scale = createTensorFromCT(&ct, v_scale_index);
      std::vector<synTensor> v_convert_inputs;
      v_convert_inputs.push_back(v_split);
      v_convert_inputs.push_back(v_scale);
      std::vector<synTensor> v_convert_outputs;
      v_convert_outputs.push_back(v_split_fp8);
      AddNodeConvertToFP8<T>(v_convert_inputs,
                             v_convert_outputs,
                             convert_fp8_params,
                             guid_ + "cast_v_fp8");
      inputs_scatter_v.push_back(v_split_fp8);
      AddNodeScatter<phi::dtype::float8_e4m3fn>(
          inputs_scatter_v, outputs_scatter_v, guid_ + "index_put_v_fp8");
    } else {
      inputs_scatter_v.push_back(v_split);
      AddNodeScatter<T>(
          inputs_scatter_v, outputs_scatter_v, guid_ + "index_put_v");
    }

    //////////////////////////////////////////////////////////////////

    std::vector<int64_t> scaler_dims = {1};
    auto scaler_tensor =
        createTensorNoPresist("scaler_tensor", syn_type_bf16, scaler_dims);
    std::vector<synTensor> scaler;
    scaler.push_back(scaler_tensor);
    AddNodeFull<T>(scaler, params.const_params, guid_ + "full_scale");

    std::vector<synTensor> scaled_q_in;
    scaled_q_in.push_back(q_states);
    scaled_q_in.push_back(scaler_tensor);

    auto scaled_q = createTensorNoPresist("scaled_q", dtype_, q_dims);
    std::vector<synTensor> scaled_q_out;
    scaled_q_out.push_back(scaled_q);

    AddNodeMultiply<T>(scaled_q_in, scaled_q_out, guid_ + "mul_scale_q");

    std::vector<int64_t> reshape_q_dims;
    reshape_q_dims.push_back(batch_size);
    reshape_q_dims.push_back(hidden_size);

    auto reshaped_q =
        createTensorNoPresist("reshaped_q", dtype_, reshape_q_dims);
    std::vector<synTensor> reshape_q_out;
    reshape_q_out.push_back(reshaped_q);

    AddNodeReshape(scaled_q_out, reshape_q_out, guid_ + "reshape_scale_q");

    /*******************************/

    std::vector<synTensor> map_q_in;
    auto block_mapping = createTensorFromCT(&ct, block_mapping_index);
    map_q_in.push_back(block_mapping);
    map_q_in.push_back(reshaped_q);

    std::vector<int64_t> map_q_dims;
    map_q_dims.push_back(num_of_block);
    map_q_dims.push_back(hidden_size);
    auto mapped_q = createTensorNoPresist("mapped_q", dtype_, map_q_dims);
    std::vector<synTensor> map_q_out;
    map_q_out.push_back(mapped_q);

    AddNodeGemm(map_q_in, map_q_out, gemm_params_f_f, guid_ + "gemm_map_q");

    std::vector<int64_t> reshape_map_q_dims;
    reshape_map_q_dims.push_back(num_of_block);
    reshape_map_q_dims.push_back(num_kv_head);
    reshape_map_q_dims.push_back(ngroups);
    reshape_map_q_dims.push_back(1);
    reshape_map_q_dims.push_back(head_dim);

    auto reshaped_map_q =
        createTensorNoPresist("reshaped_map_q", dtype_, reshape_map_q_dims);
    std::vector<synTensor> reshape_map_q_out;
    reshape_map_q_out.push_back(reshaped_map_q);

    AddNodeReshape(map_q_out, reshape_map_q_out, guid_ + "reshape_map_q");

    /*******************************/

    std::vector<synTensor> index_select_k_in;
    std::vector<synTensor> index_select_v_in;

    auto block_list = createTensorFromCT(&ct, block_list_index);

    index_select_k_in.push_back(kCache_out);
    index_select_v_in.push_back(vCache_out);
    index_select_k_in.push_back(block_list);
    index_select_v_in.push_back(block_list);

    std::vector<int64_t> index_selected_dims;
    index_selected_dims.push_back(num_of_block);
    index_selected_dims.push_back(block_size);
    index_selected_dims.push_back(num_kv_head);
    index_selected_dims.push_back(head_dim);

    auto index_select_k_i = createTensorNoPresist(
        "index_select_k_i", kv_dtype, index_selected_dims);
    auto index_select_v_i = createTensorNoPresist(
        "index_select_v_i", kv_dtype, index_selected_dims);
    std::vector<synTensor> index_select_k_out;
    index_select_k_out.push_back(index_select_k_i);
    std::vector<synTensor> index_select_v_out;
    index_select_v_out.push_back(index_select_v_i);

    if (kv_dtype == synDataType::syn_type_fp8_143) {
      AddNodeIndexSelect<phi::dtype::float8_e4m3fn>(
          index_select_k_in,
          index_select_k_out,
          params.index_select_params,
          guid_ + "index_select_k_i_fp8");
      AddNodeIndexSelect<phi::dtype::float8_e4m3fn>(
          index_select_v_in,
          index_select_v_out,
          params.index_select_params,
          guid_ + "index_select_v_i_fp8");
    } else {
      AddNodeIndexSelect<T>(index_select_k_in,
                            index_select_k_out,
                            params.index_select_params,
                            guid_ + "index_select_k_i");
      AddNodeIndexSelect<T>(index_select_v_in,
                            index_select_v_out,
                            params.index_select_params,
                            guid_ + "index_select_v_i");
    }
    std::vector<int> axis = {0, 2, 1, 3};
    synTransposeParams trans_params;
    for (size_t i = 0; i < axis.size(); i++) {
      trans_params.permutation[i] =
          static_cast<TransposePermutationDim>(axis[i]);
    }
    trans_params.tensorDim = 4;

    std::vector<int64_t> transpose_dims;
    transpose_dims.push_back(num_of_block);
    transpose_dims.push_back(num_kv_head);
    transpose_dims.push_back(block_size);
    transpose_dims.push_back(head_dim);

    auto transpose_k =
        createTensorNoPresist("transpose_k", kv_dtype, transpose_dims);
    std::vector<synTensor> trans_index_select_k;
    trans_index_select_k.push_back(transpose_k);

    AddNodeTranspose(index_select_k_out,
                     trans_index_select_k,
                     trans_params,
                     guid_ + "transpose_k");

    auto transpose_v =
        createTensorNoPresist("transpose_v", kv_dtype, transpose_dims);
    std::vector<synTensor> trans_index_select_v;
    trans_index_select_v.push_back(transpose_v);

    AddNodeTranspose(index_select_v_out,
                     trans_index_select_v,
                     trans_params,
                     guid_ + "transpose_v");

    std::vector<int64_t> reshape_kv_dims;
    reshape_kv_dims.push_back(num_of_block);
    reshape_kv_dims.push_back(num_kv_head);
    reshape_kv_dims.push_back(1);
    reshape_kv_dims.push_back(block_size);
    reshape_kv_dims.push_back(head_dim);

    auto index_select_k =
        createTensorNoPresist("index_select_k", kv_dtype, reshape_kv_dims);
    std::vector<synTensor> reshape_index_select_k;
    reshape_index_select_k.push_back(index_select_k);

    AddNodeReshape(
        trans_index_select_k, reshape_index_select_k, guid_ + "reshape_k");

    auto index_select_v =
        createTensorNoPresist("index_select_v", kv_dtype, reshape_kv_dims);
    std::vector<synTensor> reshape_index_select_v;
    reshape_index_select_v.push_back(index_select_v);

    AddNodeReshape(
        trans_index_select_v, reshape_index_select_v, guid_ + "reshape_v");

    std::vector<synTensor> q_k_in;
    q_k_in.push_back(reshaped_map_q);
    q_k_in.push_back(index_select_k);

    std::vector<int64_t> q_k_dims;
    q_k_dims.push_back(num_of_block);
    q_k_dims.push_back(num_kv_head);
    q_k_dims.push_back(ngroups);
    q_k_dims.push_back(1);
    q_k_dims.push_back(block_size);
    auto q_k = createTensorNoPresist("q_k", dtype_, q_k_dims);
    std::vector<synTensor> q_k_out;
    q_k_out.push_back(q_k);

    // Q*K^T
    AddNodeMixedPrecisionGemm<T>(params.use_fp8,
                                 ct,
                                 q_scale_index,
                                 k_scale_index,
                                 0,
                                 1,
                                 q_k_in,
                                 q_k_out,
                                 gemm_params_f_t,
                                 "q_k");

    /*******************************/

    auto block_bias = createTensorFromCT(&ct, block_bias_index);
    std::vector<synTensor> block_bias_in;
    block_bias_in.push_back(block_bias);

    std::vector<int64_t> reshaped_bias_dims;
    reshaped_bias_dims.push_back(num_of_block);
    reshaped_bias_dims.push_back(1);
    reshaped_bias_dims.push_back(1);
    reshaped_bias_dims.push_back(1);
    reshaped_bias_dims.push_back(block_size);

    auto reshaped_bias =
        createTensorNoPresist("reshaped_bias", dtype_, reshaped_bias_dims);
    std::vector<synTensor> block_bias_out;
    block_bias_out.push_back(reshaped_bias);

    AddNodeReshape(block_bias_in, block_bias_out, guid_ + "reshaped_bias");

    std::vector<synTensor> add_bias_in;
    add_bias_in.push_back(q_k);
    add_bias_in.push_back(reshaped_bias);

    auto add_bias = createTensorNoPresist("add_bias", dtype_, q_k_dims);
    std::vector<synTensor> add_bias_out;
    add_bias_out.push_back(add_bias);

    AddNodeAdd<T>(add_bias_in, add_bias_out, guid_ + "add_bias");
    /*******************************/

    std::vector<int64_t> block_max_dims;
    block_max_dims.push_back(num_of_block);
    block_max_dims.push_back(num_kv_head);
    block_max_dims.push_back(ngroups);
    block_max_dims.push_back(1);
    block_max_dims.push_back(1);

    auto block_max = createTensorNoPresist("block_max", dtype_, block_max_dims);
    std::vector<synTensor> block_max_out;
    block_max_out.push_back(block_max);

    AddNodeReduceMax<T>(
        add_bias_out, block_max_out, params.reduce_params, guid_ + "reduceMax");

    /**************************************************************/

    std::vector<int64_t> sum_adjusted_dims;
    sum_adjusted_dims.push_back(num_of_block);
    sum_adjusted_dims.push_back(num_kv_head);
    sum_adjusted_dims.push_back(ngroups);

    auto block_max_2D =
        createTensorNoPresist("block_max_2D", dtype_, sum_adjusted_dims);
    std::vector<synTensor> block_max_2D_out;
    block_max_2D_out.push_back(block_max_2D);

    AddNodeReshape(
        block_max_out, block_max_2D_out, guid_ + "squeeze_block_max");

    std::vector<int64_t> group_max_dims;
    group_max_dims.push_back(batch_size + 1);
    group_max_dims.push_back(num_kv_head);
    group_max_dims.push_back(ngroups);

    auto group_max = createTensorNoPresist("group_max", dtype_, group_max_dims);
    std::vector<synTensor> group_max_tensor;
    group_max_tensor.push_back(group_max);

    params.const_params.constant.f = -std::numeric_limits<float>::infinity();

    AddNodeFull<T>(group_max_tensor, params.const_params, guid_ + "full_inf");

    auto block_groups = createTensorFromCT(&ct, block_groups_index);
    std::vector<synTensor> index_reduce_in;
    index_reduce_in.push_back(group_max);
    index_reduce_in.push_back(block_groups);
    index_reduce_in.push_back(block_max_2D);

    auto reduced_group_max =
        createTensorNoPresist("reduced_group_max", dtype_, group_max_dims);
    std::vector<synTensor> index_reduce_out;
    index_reduce_out.push_back(reduced_group_max);

    AddNodeIndexReduce<T>(index_reduce_in,
                          index_reduce_out,
                          params.index_reduce_params,
                          guid_ + "index_reduce_amax");

    std::vector<synTensor> index_select_groupmax_in;
    index_select_groupmax_in.push_back(reduced_group_max);
    index_select_groupmax_in.push_back(block_groups);

    auto selected_group_max =
        createTensorNoPresist("selected_group_max", dtype_, sum_adjusted_dims);
    std::vector<synTensor> index_select_groupmax_out;
    index_select_groupmax_out.push_back(selected_group_max);
    params.index_select_params.axis = 2;
    AddNodeIndexSelect<T>(index_select_groupmax_in,
                          index_select_groupmax_out,
                          params.index_select_params,
                          guid_ + "index_select_groupmax");

    std::vector<synTensor> sub_group_max_in;
    sub_group_max_in.push_back(block_max_2D);
    sub_group_max_in.push_back(selected_group_max);

    auto sub_group_max =
        createTensorNoPresist("sub_group_max", dtype_, sum_adjusted_dims);
    std::vector<synTensor> sub_group_max_out;
    sub_group_max_out.push_back(sub_group_max);

    AddNodeSub<T>(sub_group_max_in, sub_group_max_out, guid_ + "sub_group_max");

    auto block_adjustment =
        createTensorNoPresist("block_adjustment", dtype_, sum_adjusted_dims);
    std::vector<synTensor> block_adjustment_out;
    block_adjustment_out.push_back(block_adjustment);
    AddNodeExp<T>(sub_group_max_out,
                  block_adjustment_out,
                  guid_ + "exp_block_adjustment");

    /**************************************************************/

    std::vector<synTensor> sub_block_max_in;
    sub_block_max_in.push_back(add_bias);
    sub_block_max_in.push_back(block_max);

    auto sub_block_max =
        createTensorNoPresist("sub_block_max", dtype_, q_k_dims);
    std::vector<synTensor> sub_block_max_out;
    sub_block_max_out.push_back(sub_block_max);
    AddNodeSub<T>(sub_block_max_in, sub_block_max_out, guid_ + "sub_block_max");

    auto score = createTensorNoPresist("score", dtype_, q_k_dims);
    std::vector<synTensor> score_out;
    score_out.push_back(score);
    AddNodeExp<T>(sub_block_max_out, score_out, guid_ + "exp_score");

    /*******************************/

    std::vector<synTensor> score_v_in;
    score_v_in.push_back(score);
    score_v_in.push_back(index_select_v);

    auto score_v = createTensorNoPresist("score_v", dtype_, reshape_map_q_dims);
    std::vector<synTensor> score_v_out;
    score_v_out.push_back(score_v);

    // Score*V
    AddNodeMixedPrecisionGemm<T>(params.use_fp8,
                                 ct,
                                 a_scale_index,
                                 v_scale_index,
                                 0,
                                 1,
                                 score_v_in,
                                 score_v_out,
                                 gemm_params_f_f,
                                 "a_v");

    auto reduceSum = createTensorNoPresist("reduceSum", dtype_, block_max_dims);
    std::vector<synTensor> reduceSum_out;
    reduceSum_out.push_back(reduceSum);

    AddNodeReduceSum<T>(
        score_out, reduceSum_out, params.reduce_params, guid_ + "reduceSum");

    auto block_sums_2D =
        createTensorNoPresist("block_sums_2D", dtype_, sum_adjusted_dims);
    std::vector<synTensor> block_sums_2D_out;
    block_sums_2D_out.push_back(block_sums_2D);

    AddNodeReshape(
        reduceSum_out, block_sums_2D_out, guid_ + "squeeze_block_sums");

    std::vector<synTensor> sum_adjusted_in;
    sum_adjusted_in.push_back(block_sums_2D);
    sum_adjusted_in.push_back(block_adjustment);

    auto sum_adjusted =
        createTensorNoPresist("sum_adjusted", dtype_, sum_adjusted_dims);
    std::vector<synTensor> sum_adjusted_out;
    sum_adjusted_out.push_back(sum_adjusted);

    AddNodeMultiply<T>(
        sum_adjusted_in, sum_adjusted_out, guid_ + "mul_sum_adjusted");
    /***************************************************************/

    std::vector<int64_t> reshaped_sum_adjusted_dims;
    reshaped_sum_adjusted_dims.push_back(num_of_block);
    reshaped_sum_adjusted_dims.push_back(num_head);

    auto flatten_sum_adjusted = createTensorNoPresist(
        "flatten_sum_adjusted", dtype_, reshaped_sum_adjusted_dims);
    std::vector<synTensor> reshaped_sum_adjusted_out;
    reshaped_sum_adjusted_out.push_back(flatten_sum_adjusted);

    AddNodeReshape(sum_adjusted_out,
                   reshaped_sum_adjusted_out,
                   guid_ + "flatten_sum_adjusted");

    std::vector<synTensor> map_sum_adjusted_in;
    map_sum_adjusted_in.push_back(block_mapping);
    map_sum_adjusted_in.push_back(flatten_sum_adjusted);

    std::vector<int64_t> map_sum_adjusted_dims;
    map_sum_adjusted_dims.push_back(batch_size);
    map_sum_adjusted_dims.push_back(num_head);
    auto mapped_sum_adjusted = createTensorNoPresist(
        "mapped_sum_adjusted", dtype_, map_sum_adjusted_dims);
    std::vector<synTensor> map_sum_adjusted_out;
    map_sum_adjusted_out.push_back(mapped_sum_adjusted);

    AddNodeGemm(map_sum_adjusted_in,
                map_sum_adjusted_out,
                gemm_params_t_f,
                guid_ + "gemm_map_sum_adjusted");

    std::vector<synTensor> group_sum_adjusted_in;
    group_sum_adjusted_in.push_back(block_mapping);
    group_sum_adjusted_in.push_back(mapped_sum_adjusted);

    std::vector<int64_t> matmul_sum_adjusted_dims;
    matmul_sum_adjusted_dims.push_back(num_of_block);
    matmul_sum_adjusted_dims.push_back(num_head);
    auto group_sum_adjusted = createTensorNoPresist(
        "group_sum_adjusted", dtype_, matmul_sum_adjusted_dims);
    std::vector<synTensor> group_sum_adjusted_out;
    group_sum_adjusted_out.push_back(group_sum_adjusted);

    AddNodeGemm(group_sum_adjusted_in,
                group_sum_adjusted_out,
                gemm_params_f_f,
                guid_ + "gemm_group_sum_adjusted");

    /*******************************/

    auto reshaped_group_sum_adjusted = createTensorNoPresist(
        "reshaped_group_sum_adjusted", dtype_, block_max_dims);
    std::vector<synTensor> group_sum_adjusted_4D;
    group_sum_adjusted_4D.push_back(reshaped_group_sum_adjusted);

    AddNodeReshape(group_sum_adjusted_out,
                   group_sum_adjusted_4D,
                   guid_ + "reshaped_group_sum_adjusted");

    auto reshaped_sum_adjusted =
        createTensorNoPresist("reshaped_sum_adjusted", dtype_, block_max_dims);
    std::vector<synTensor> sum_adjusted_4D;
    sum_adjusted_4D.push_back(reshaped_sum_adjusted);

    AddNodeReshape(
        sum_adjusted_out, sum_adjusted_4D, guid_ + "reshaped_sum_adjusted");

    auto reshaped_block_adjustment = createTensorNoPresist(
        "reshaped_block_adjustment", dtype_, block_max_dims);
    std::vector<synTensor> block_adjustment_4D;
    block_adjustment_4D.push_back(reshaped_block_adjustment);

    AddNodeReshape(block_adjustment_out,
                   block_adjustment_4D,
                   guid_ + "reshaped_block_adjustment");

    std::vector<synTensor> max_sum_adjust_in;
    max_sum_adjust_in.push_back(reshaped_group_sum_adjusted);
    max_sum_adjust_in.push_back(reshaped_sum_adjusted);
    auto max_sum_adjust =
        createTensorNoPresist("max_sum_adjust", dtype_, block_max_dims);
    std::vector<synTensor> max_sum_adjust_out;
    max_sum_adjust_out.push_back(max_sum_adjust);

    AddNodeMaximum<T>(
        max_sum_adjust_in, max_sum_adjust_out, guid_ + "max_sum_adjust");

    std::vector<synTensor> rescale_in;
    rescale_in.push_back(reshaped_block_adjustment);
    rescale_in.push_back(max_sum_adjust);
    auto rescale = createTensorNoPresist("rescale", dtype_, block_max_dims);
    std::vector<synTensor> rescale_out;
    rescale_out.push_back(rescale);
    AddNodeDivide<T>(rescale_in, rescale_out, guid_ + "div_rescale");

    std::vector<synTensor> rescale_v_in;
    rescale_v_in.push_back(rescale);
    rescale_v_in.push_back(score_v);

    auto rescale_v =
        createTensorNoPresist("rescale_v", dtype_, reshape_map_q_dims);
    std::vector<synTensor> rescale_v_out;
    rescale_v_out.push_back(rescale_v);

    AddNodeMultiply<T>(rescale_v_in, rescale_v_out, guid_ + "mul_rescale_v");

    auto reshape_attn =
        createTensorNoPresist("reshape_attn", dtype_, map_q_dims);
    std::vector<synTensor> reshape_attn_out;
    reshape_attn_out.push_back(reshape_attn);

    AddNodeReshape(rescale_v_out, reshape_attn_out, guid_ + "reshape_attn");

    std::vector<synTensor> map_attn_in;
    map_attn_in.push_back(block_mapping);
    map_attn_in.push_back(reshape_attn);

    auto mapped_attn =
        createTensorNoPresist("mapped_attn", dtype_, reshape_q_dims);
    std::vector<synTensor> map_attn_out;
    map_attn_out.push_back(mapped_attn);

    AddNodeGemm(
        map_attn_in, map_attn_out, gemm_params_t_f, guid_ + "gemm_map_attn");

    std::vector<synTensor> proj_in;
    auto linear_weights = createTensorFromCT(&ct, linear_weights_index);
    proj_in.push_back(mapped_attn);
    proj_in.push_back(linear_weights);

    auto linear_out = createTensorFromCT(&ct, 0, false);
    std::vector<synTensor> proj_out;
    proj_out.push_back(linear_out);

    // Final Linear
    AddNodeMixedPrecisionGemm<T>(params.use_fp8,
                                 ct,
                                 o_linear_scale_x_index,
                                 o_linear_scale_y_index,
                                 0,
                                 0,
                                 proj_in,
                                 proj_out,
                                 gemm_params_f_f,
                                 "proj");
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void FusedBlockAttentionKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& src,
    const phi::DenseTensor& rotary_embs,
    const phi::DenseTensor& key_cache,
    const phi::DenseTensor& value_cache,
    const phi::DenseTensor& block_groups,
    const phi::DenseTensor& block_list,
    const phi::DenseTensor& block_mapping,
    const phi::DenseTensor& block_bias,
    const phi::DenseTensor& block_indices,
    const phi::DenseTensor& block_offsets,
    const phi::DenseTensor& qkv_weights,
    const paddle::optional<phi::DenseTensor>& qkv_biases,
    const phi::DenseTensor& linear_weights,
    const paddle::optional<phi::DenseTensor>& q_norm_weights,
    const paddle::optional<phi::DenseTensor>& k_norm_weights,
    const paddle::optional<phi::DenseTensor>& src_scale,
    const paddle::optional<phi::DenseTensor>& qkv_weights_scale,
    const paddle::optional<phi::DenseTensor>& qk_scale_x,
    const paddle::optional<phi::DenseTensor>& qk_scale_y,
    const paddle::optional<phi::DenseTensor>& av_scale_x,
    const paddle::optional<phi::DenseTensor>& av_scale_y,
    const paddle::optional<phi::DenseTensor>& o_linear_scale_x,
    const paddle::optional<phi::DenseTensor>& o_linear_scale_y,
    phi::DenseTensor* out_linear,
    const phi::Scalar& head_dim,
    const phi::Scalar& num_head,
    const phi::Scalar& scaling_factor,
    const phi::Scalar& transpose,
    const phi::Scalar& use_neox_style,
    const phi::Scalar& epsilon) {
  std::vector<int64_t> src_dims = phi::vectorize<int64_t>(src.dims());
  std::vector<int64_t> qkv_weights_dims =
      phi::vectorize<int64_t>(qkv_weights.dims());

  int head_dim_ = head_dim.to<int>();
  int num_head_ = num_head.to<int>();
  bool transpose_ = transpose.to<bool>();
  bool use_neox_style_ = use_neox_style.to<bool>();
  const int64_t fused_hidden_size =
      transpose_ ? qkv_weights_dims[0] : qkv_weights_dims[1];
  const int num_kv_head =
      (fused_hidden_size - num_head_ * head_dim_) / head_dim_ / 2;

  ConvertTensors ct;
  ct.Add(src);
  ct.Add(rotary_embs);
  ct.Add(key_cache);
  ct.Add(value_cache);
  ct.Add(block_groups);
  std::vector<DIMS> inputs_dims = ct.GetDims();
  ct.Add(block_list);
  ct.Add(block_mapping);
  ct.Add(block_bias);
  ct.Add(block_indices);
  ct.Add(block_offsets);
  ct.Add(qkv_weights);
  ct.Add(linear_weights);
  ct.Add(out_linear, false);
  ct.Add(key_cache, false);
  ct.Add(value_cache, false);

  std::string guid_prefix = "fused_block_attention_";

  bool use_fp8 = false;
  if (qk_scale_x || qk_scale_y || av_scale_x || av_scale_y ||
      o_linear_scale_x || o_linear_scale_y) {
    if (!qk_scale_x || !qk_scale_y || !av_scale_x || !av_scale_y ||
        !o_linear_scale_x || !o_linear_scale_y) {
      throw std::runtime_error(
          "Please specify all scale values for FusedBlockAttentionKernel");
    }

    use_fp8 = true;
    guid_prefix = "fused_fp8_block_attention_";
    ct.Add(src_scale.get());
    ct.Add(qkv_weights_scale.get());
    ct.Add(qk_scale_x.get());
    ct.Add(qk_scale_y.get());
    ct.Add(av_scale_x.get());
    ct.Add(av_scale_y.get());
    ct.Add(o_linear_scale_x.get());
    ct.Add(o_linear_scale_y.get());
  }

  if (qkv_biases) {
    ct.Add(qkv_biases.get());
    guid_prefix += "bias_";
  }

  bool qk_rmsnorm = false;
  if (q_norm_weights && k_norm_weights) {
    qk_rmsnorm = true;
    ct.Add(q_norm_weights.get());
    ct.Add(k_norm_weights.get());
    guid_prefix += "qk_rmsnorm_";
  }
  if (num_head_ == num_kv_head) {
    guid_prefix += "MHA_";
  } else {
    guid_prefix += "GQA_";
  }
  guid_prefix += "fwd_";

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, nullptr_t>(guid_prefix, inputs_dims, nullptr);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FusedBlockAttentionParams params;
    memset(reinterpret_cast<void*>(&params),
           0x00,
           sizeof(FusedBlockAttentionParams));
    params.const_params.constant.f = scaling_factor.to<float>();
    params.index_select_params.axis = 3;
    params.reduce_params.reductionDimension = 0;
    params.index_reduce_params.mode = INDEX_REDUCE_AMAX;
    params.index_reduce_params.include_self = true;
    params.index_reduce_params.axis = 0;
    params.rmsnorm_params.epsValid = true;
    params.rmsnorm_params.eps = epsilon.to<float>();
    params.use_neox_style = use_neox_style_;
    params.transpose = transpose_;
    params.head_dim = head_dim_;
    params.num_head = num_head_;
    params.num_kv_head = num_kv_head;
    params.use_fp8 = use_fp8;
    if (qkv_biases) {
      params.with_qkv_biases = true;
    }
    if (qk_rmsnorm) {
      params.use_qk_rmsnorm = true;
    }

    if (num_head_ == num_kv_head) {
      FusedMHABlockAttention op(guid_prefix, op_info.datatype_);
      op.AddNode<T>(ct, params);
      op.Compile();
      op_info.setOp(op);
    } else {
      FusedGQABlockAttention op(guid_prefix, op_info.datatype_);
      op.AddNode<T>(ct, params);
      op.Compile();
      op_info.setOp(op);
    }

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

template <typename Context>
void CallFusedBlockAttentionKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& src,
    const phi::DenseTensor& rotary_embs,
    const phi::DenseTensor& key_cache,
    const phi::DenseTensor& value_cache,
    const phi::DenseTensor& block_groups,
    const phi::DenseTensor& block_list,
    const phi::DenseTensor& block_mapping,
    const phi::DenseTensor& block_bias,
    const phi::DenseTensor& block_indices,
    const phi::DenseTensor& block_offsets,
    const phi::DenseTensor& qkv_weights,
    const paddle::optional<phi::DenseTensor>& qkv_biases,
    const phi::DenseTensor& linear_weights,
    const paddle::optional<phi::DenseTensor>& q_norm_weights,
    const paddle::optional<phi::DenseTensor>& k_norm_weights,
    const paddle::optional<phi::DenseTensor>& src_scale,
    const paddle::optional<phi::DenseTensor>& qkv_weights_scale,
    const paddle::optional<phi::DenseTensor>& q_scale,
    const paddle::optional<phi::DenseTensor>& k_scale,
    const paddle::optional<phi::DenseTensor>& a_scale,
    const paddle::optional<phi::DenseTensor>& v_scale,
    const paddle::optional<phi::DenseTensor>& o_linear_scale_x,
    const paddle::optional<phi::DenseTensor>& o_linear_scale_y,
    phi::DenseTensor* out_linear,
    const phi::Scalar& head_dim,
    const phi::Scalar& num_head,
    const phi::Scalar& scaling_factor,
    const phi::Scalar& transpose,
    const phi::Scalar& use_neox_style,
    const phi::Scalar& epsilon) {
  if (src.dtype() == phi::DataType::FLOAT16) {
    custom_kernel::FusedBlockAttentionKernel<phi::dtype::float16>(
        dev_ctx,
        src,
        rotary_embs,
        key_cache,
        value_cache,
        block_groups,
        block_list,
        block_mapping,
        block_bias,
        block_indices,
        block_offsets,
        qkv_weights,
        qkv_biases,
        linear_weights,
        q_norm_weights,
        k_norm_weights,
        src_scale,
        qkv_weights_scale,
        q_scale,
        k_scale,
        a_scale,
        v_scale,
        o_linear_scale_x,
        o_linear_scale_y,
        out_linear,
        head_dim,
        num_head,
        scaling_factor,
        transpose,
        use_neox_style,
        epsilon);
  } else if (src.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::FusedBlockAttentionKernel<phi::dtype::bfloat16>(
        dev_ctx,
        src,
        rotary_embs,
        key_cache,
        value_cache,
        block_groups,
        block_list,
        block_mapping,
        block_bias,
        block_indices,
        block_offsets,
        qkv_weights,
        qkv_biases,
        linear_weights,
        q_norm_weights,
        k_norm_weights,
        src_scale,
        qkv_weights_scale,
        q_scale,
        k_scale,
        a_scale,
        v_scale,
        o_linear_scale_x,
        o_linear_scale_y,
        out_linear,
        head_dim,
        num_head,
        scaling_factor,
        transpose,
        use_neox_style,
        epsilon);
  } else {
    throw std::runtime_error(
        "Unsupported data type for FusedBlockAttentionKernel");
  }
}

std::vector<paddle::Tensor> FusedBlockAttentionForward(
    const paddle::Tensor& src,
    const paddle::Tensor& rotary_embs,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& block_groups,
    const paddle::Tensor& block_list,
    const paddle::Tensor& block_mapping,
    const paddle::Tensor& block_bias,
    const paddle::Tensor& block_indices,
    const paddle::Tensor& block_offsets,
    const paddle::Tensor& qkv_weights,
    const paddle::optional<paddle::Tensor>& qkv_biases,
    const paddle::Tensor& linear_weights,
    const paddle::optional<paddle::Tensor>& q_norm_weights,
    const paddle::optional<paddle::Tensor>& k_norm_weights,
    int head_dim,
    int num_head,
    float scaling_factor,
    bool transpose,
    bool use_neox_style,
    float epsilon) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(src.place()));
  auto src_tensor = static_cast<const phi::DenseTensor*>(src.impl().get());
  auto rotary_embs_tensor =
      static_cast<const phi::DenseTensor*>(rotary_embs.impl().get());
  auto key_cache_tensor =
      static_cast<const phi::DenseTensor*>(key_cache.impl().get());
  auto value_cache_tensor =
      static_cast<const phi::DenseTensor*>(value_cache.impl().get());
  auto block_groups_tensor =
      static_cast<const phi::DenseTensor*>(block_groups.impl().get());
  auto block_list_tensor =
      static_cast<const phi::DenseTensor*>(block_list.impl().get());
  auto block_mapping_tensor =
      static_cast<const phi::DenseTensor*>(block_mapping.impl().get());
  auto block_bias_tensor =
      static_cast<const phi::DenseTensor*>(block_bias.impl().get());
  auto block_indices_tensor =
      static_cast<const phi::DenseTensor*>(block_indices.impl().get());
  auto block_offsets_tensor =
      static_cast<const phi::DenseTensor*>(block_offsets.impl().get());
  auto qkv_weights_tensor =
      static_cast<const phi::DenseTensor*>(qkv_weights.impl().get());
  auto linear_weights_tensor =
      static_cast<const phi::DenseTensor*>(linear_weights.impl().get());

  auto qkv_biases_tensor = paddle::optional<phi::DenseTensor>();
  if (qkv_biases) {
    auto qkv_biases_dt =
        static_cast<phi::DenseTensor*>(qkv_biases->impl().get());
    qkv_biases_tensor = paddle::optional<phi::DenseTensor>(*qkv_biases_dt);
  }

  auto q_norm_weights_tensor = paddle::optional<phi::DenseTensor>();
  if (q_norm_weights) {
    auto q_norm_weights_dt =
        static_cast<phi::DenseTensor*>(q_norm_weights->impl().get());
    q_norm_weights_tensor =
        paddle::optional<phi::DenseTensor>(*q_norm_weights_dt);
  }

  auto k_norm_weights_tensor = paddle::optional<phi::DenseTensor>();
  if (k_norm_weights) {
    auto k_norm_weights_dt =
        static_cast<phi::DenseTensor*>(k_norm_weights->impl().get());
    k_norm_weights_tensor =
        paddle::optional<phi::DenseTensor>(*k_norm_weights_dt);
  }

  // allocate memory on device.
  int64_t batch_size = src.dims()[0];
  int64_t out_features = linear_weights.dims()[1];

  std::shared_ptr<phi::DenseTensor> out_linear =
      std::make_shared<phi::DenseTensor>();
  out_linear->Resize(phi::make_ddim({batch_size, out_features}));
  dev_ctx->Alloc(out_linear.get(), src_tensor->dtype());

  CallFusedBlockAttentionKernel(*dev_ctx,
                                *src_tensor,
                                *rotary_embs_tensor,
                                *key_cache_tensor,
                                *value_cache_tensor,
                                *block_groups_tensor,
                                *block_list_tensor,
                                *block_mapping_tensor,
                                *block_bias_tensor,
                                *block_indices_tensor,
                                *block_offsets_tensor,
                                *qkv_weights_tensor,
                                qkv_biases_tensor,
                                *linear_weights_tensor,
                                q_norm_weights_tensor,
                                k_norm_weights_tensor,
                                paddle::optional<phi::DenseTensor>(),
                                paddle::optional<phi::DenseTensor>(),
                                paddle::optional<phi::DenseTensor>(),
                                paddle::optional<phi::DenseTensor>(),
                                paddle::optional<phi::DenseTensor>(),
                                paddle::optional<phi::DenseTensor>(),
                                paddle::optional<phi::DenseTensor>(),
                                paddle::optional<phi::DenseTensor>(),
                                out_linear.get(),
                                phi::Scalar(head_dim),
                                phi::Scalar(num_head),
                                phi::Scalar(scaling_factor),
                                phi::Scalar(transpose),
                                phi::Scalar(use_neox_style),
                                phi::Scalar(epsilon));
  return {paddle::Tensor(out_linear)};
}

std::vector<std::vector<int64_t>> FusedBlockAttentionShape(
    const std::vector<int64_t>& src_shape,
    const std::vector<int64_t>& rotary_embs_shape,
    const std::vector<int64_t>& key_cache_shape,
    const std::vector<int64_t>& value_cache_shape,
    const std::vector<int64_t>& block_groups_shape,
    const std::vector<int64_t>& block_list_shape,
    const std::vector<int64_t>& block_mapping_shape,
    const std::vector<int64_t>& block_bias_shape,
    const std::vector<int64_t>& block_indices_shape,
    const std::vector<int64_t>& block_offsets_shape,
    const std::vector<int64_t>& qkv_weights_shape,
    const paddle::optional<std::vector<int64_t>>& qkv_biases_shape,
    const std::vector<int64_t>& linear_weights_shape,
    int head_dim,
    int num_head,
    float scaling_factor,
    bool transpose,
    bool use_neox_style) {
  int64_t batch_size = src_shape[0];
  int64_t out_features = linear_weights_shape[1];
  return {{batch_size, 1, out_features}};
}

std::vector<paddle::DataType> FusedBlockAttentionDtype(
    const paddle::DataType& src_dtype,
    const paddle::DataType& rotary_embs_dtype,
    const paddle::DataType& key_cache_dtype,
    const paddle::DataType& value_cache_dtype,
    const paddle::DataType& block_groups_dtype,
    const paddle::DataType& block_list_dtype,
    const paddle::DataType& block_mapping_dtype,
    const paddle::DataType& block_bias_dtype,
    const paddle::DataType& block_indices_dtype,
    const paddle::DataType& block_offsets_dtype,
    const paddle::DataType& qkv_weights_dtype,
    const paddle::optional<paddle::DataType>& qkv_biases_dtype,
    const paddle::DataType& linear_weights_dtype) {
  return {src_dtype};
}

PD_BUILD_OP(fused_block_attention)
    .Inputs({"src",
             "rotary_embs",
             "key_cache",
             "value_cache",
             "block_groups",
             "block_list",
             "block_mapping",
             "block_bias",
             "block_indices",
             "block_offsets",
             "qkv_weights",
             paddle::Optional("qkv_biases"),
             "linear_weights",
             paddle::Optional("q_norm_weights"),
             paddle::Optional("k_norm_weights")})
    .Outputs({"out_linear"})
    .Attrs({"head_dim: int",
            "num_head: int",
            "scaling_factor: float",
            "transpose: bool",
            "use_neox_style: bool",
            "epsilon: float"})
    .SetKernelFn(PD_KERNEL(FusedBlockAttentionForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedBlockAttentionShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedBlockAttentionDtype));

std::vector<paddle::Tensor> FusedFp8BlockAttentionForward(
    const paddle::Tensor& src,
    const paddle::Tensor& rotary_embs,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& block_groups,
    const paddle::Tensor& block_list,
    const paddle::Tensor& block_mapping,
    const paddle::Tensor& block_bias,
    const paddle::Tensor& block_indices,
    const paddle::Tensor& block_offsets,
    const paddle::Tensor& qkv_weights,
    const paddle::optional<paddle::Tensor>& qkv_biases,
    const paddle::Tensor& linear_weights,
    const paddle::optional<paddle::Tensor>& q_norm_weights,
    const paddle::optional<paddle::Tensor>& k_norm_weights,
    const paddle::Tensor& src_scale,
    const paddle::Tensor& qkv_weights_scale,
    const paddle::Tensor& q_scale,
    const paddle::Tensor& k_scale,
    const paddle::Tensor& a_scale,
    const paddle::Tensor& v_scale,
    const paddle::Tensor& o_linear_scale_x,
    const paddle::Tensor& o_linear_scale_y,
    int head_dim,
    int num_head,
    float scaling_factor,
    bool transpose,
    bool use_neox_style,
    float epsilon) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(src.place()));
  auto src_tensor = static_cast<const phi::DenseTensor*>(src.impl().get());
  auto rotary_embs_tensor =
      static_cast<const phi::DenseTensor*>(rotary_embs.impl().get());
  auto key_cache_tensor =
      static_cast<const phi::DenseTensor*>(key_cache.impl().get());
  auto value_cache_tensor =
      static_cast<const phi::DenseTensor*>(value_cache.impl().get());
  auto block_groups_tensor =
      static_cast<const phi::DenseTensor*>(block_groups.impl().get());
  auto block_list_tensor =
      static_cast<const phi::DenseTensor*>(block_list.impl().get());
  auto block_mapping_tensor =
      static_cast<const phi::DenseTensor*>(block_mapping.impl().get());
  auto block_bias_tensor =
      static_cast<const phi::DenseTensor*>(block_bias.impl().get());
  auto block_indices_tensor =
      static_cast<const phi::DenseTensor*>(block_indices.impl().get());
  auto block_offsets_tensor =
      static_cast<const phi::DenseTensor*>(block_offsets.impl().get());
  auto qkv_weights_tensor =
      static_cast<const phi::DenseTensor*>(qkv_weights.impl().get());
  auto linear_weights_tensor =
      static_cast<const phi::DenseTensor*>(linear_weights.impl().get());

  auto qkv_biases_tensor = paddle::optional<phi::DenseTensor>();
  if (qkv_biases) {
    auto qkv_biases_dt =
        static_cast<phi::DenseTensor*>(qkv_biases->impl().get());
    qkv_biases_tensor = paddle::optional<phi::DenseTensor>(*qkv_biases_dt);
  }

  auto q_norm_weights_tensor = paddle::optional<phi::DenseTensor>();
  if (q_norm_weights) {
    auto q_norm_weights_dt =
        static_cast<phi::DenseTensor*>(q_norm_weights->impl().get());
    q_norm_weights_tensor =
        paddle::optional<phi::DenseTensor>(*q_norm_weights_dt);
  }

  auto k_norm_weights_tensor = paddle::optional<phi::DenseTensor>();
  if (k_norm_weights) {
    auto k_norm_weights_dt =
        static_cast<phi::DenseTensor*>(k_norm_weights->impl().get());
    k_norm_weights_tensor =
        paddle::optional<phi::DenseTensor>(*k_norm_weights_dt);
  }

  auto src_scale_tensor =
      static_cast<const phi::DenseTensor*>(src_scale.impl().get());
  auto qkv_weights_scale_tensor =
      static_cast<const phi::DenseTensor*>(qkv_weights_scale.impl().get());
  auto k_scale_tensor =
      static_cast<const phi::DenseTensor*>(q_scale.impl().get());
  auto q_scale_tensor =
      static_cast<const phi::DenseTensor*>(k_scale.impl().get());
  auto a_scale_tensor =
      static_cast<const phi::DenseTensor*>(a_scale.impl().get());
  auto v_scale_tensor =
      static_cast<const phi::DenseTensor*>(v_scale.impl().get());
  auto o_linear_scale_x_tensor =
      static_cast<const phi::DenseTensor*>(o_linear_scale_x.impl().get());
  auto o_linear_scale_y_tensor =
      static_cast<const phi::DenseTensor*>(o_linear_scale_y.impl().get());

  // allocate memory on device.
  int64_t batch_size = src.dims()[0];
  int64_t out_features = linear_weights.dims()[1];

  std::shared_ptr<phi::DenseTensor> out_linear =
      std::make_shared<phi::DenseTensor>();
  out_linear->Resize(phi::make_ddim({batch_size, out_features}));
  dev_ctx->Alloc(out_linear.get(), src_tensor->dtype());

  CallFusedBlockAttentionKernel(*dev_ctx,
                                *src_tensor,
                                *rotary_embs_tensor,
                                *key_cache_tensor,
                                *value_cache_tensor,
                                *block_groups_tensor,
                                *block_list_tensor,
                                *block_mapping_tensor,
                                *block_bias_tensor,
                                *block_indices_tensor,
                                *block_offsets_tensor,
                                *qkv_weights_tensor,
                                qkv_biases_tensor,
                                *linear_weights_tensor,
                                q_norm_weights_tensor,
                                k_norm_weights_tensor,
                                *src_scale_tensor,
                                *qkv_weights_scale_tensor,
                                *q_scale_tensor,
                                *k_scale_tensor,
                                *a_scale_tensor,
                                *v_scale_tensor,
                                *o_linear_scale_x_tensor,
                                *o_linear_scale_y_tensor,
                                out_linear.get(),
                                phi::Scalar(head_dim),
                                phi::Scalar(num_head),
                                phi::Scalar(scaling_factor),
                                phi::Scalar(transpose),
                                phi::Scalar(use_neox_style),
                                phi::Scalar(epsilon));
  return {paddle::Tensor(out_linear)};
}

PD_BUILD_OP(fused_fp8_block_attention)
    .Inputs({"src",
             "rotary_embs",
             "key_cache",
             "value_cache",
             "block_groups",
             "block_list",
             "block_mapping",
             "block_bias",
             "block_indices",
             "block_offsets",
             "qkv_weights",
             paddle::Optional("qkv_biases"),
             "linear_weights",
             paddle::Optional("q_norm_weights"),
             paddle::Optional("k_norm_weights"),
             "src_scale",
             "qkv_weights_scale",
             "q_scale",
             "k_scale",
             "a_scale",
             "v_scale",
             "o_linear_scale_x",
             "o_linear_scale_y"})
    .Outputs({"out_linear"})
    .Attrs({"head_dim: int",
            "num_head: int",
            "scaling_factor: float",
            "transpose: bool",
            "use_neox_style: bool",
            "epsilon: float"})
    .SetKernelFn(PD_KERNEL(FusedFp8BlockAttentionForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedBlockAttentionShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedBlockAttentionDtype));
