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

struct FusedFlatPaParams {
  ns_ConstantKernel::Params const_params;
  ns_GatherKernel::Params index_select_params;
  ns_Reduction::Params reduce_params;
  ns_IndexReduce::Params index_reduce_params;
  bool use_fp8;
};

class FusedFlatPaBase : public HpuFusedOperator {
 public:
  explicit FusedFlatPaBase(const std::string& guid, bool is_eager)
      : HpuFusedOperator(guid, is_eager) {}

  template <typename T>
  inline void AddNodeMixedPrecisionGemm(bool use_fp8,
                                        ConvertTensors& ct,
                                        int scale_x_index,
                                        std::vector<synTensor> inputs,
                                        std::vector<synTensor> outputs,
                                        synGEMMParams gemm_params,
                                        const std::string& suffix) {
    if (use_fp8) {
      synTensor scale_x = createTensorFromCT(&ct, scale_x_index);
      synTensor scale_y = createTensorFromCT(&ct, scale_x_index + 1);
      inputs.push_back(scale_x);
      inputs.push_back(scale_y);
      AddNodeFusedFP8Gemm<T>(
          inputs, outputs, gemm_params, guid_ + "fused_fp8_gemm_" + suffix);
    } else {
      AddNodeBatchGemm(
          inputs, outputs, gemm_params, guid_ + "batchgemm_" + suffix);
    }
  }
};

class FusedFlatPaMHAProj : public FusedFlatPaBase {
 public:
  explicit FusedFlatPaMHAProj(std::string name, synDataType dtype)
      : FusedFlatPaBase(name, false), dtype_(dtype) {}

  template <typename T>
  void AddNode(ConvertTensors& ct, FusedFlatPaParams params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    std::vector<int64_t> q_dims = std::vector<int64_t>(inputs[0].dims);
    std::vector<int64_t> kv_dims = std::vector<int64_t>(inputs[1].dims);
    std::vector<int64_t> block_list_dims = std::vector<int64_t>(inputs[4].dims);
    std::vector<int64_t> linear_weights_dims =
        std::vector<int64_t>(inputs[7].dims);
    int64_t batch_size = q_dims[0];
    int64_t num_head = q_dims[2];
    int64_t head_dim = q_dims[3];
    int64_t block_size = kv_dims[1];
    int64_t num_kv_head = kv_dims[2];
    int64_t num_of_block = block_list_dims[0];
    int64_t hidden_size =
        params.use_fp8 ? linear_weights_dims[1] : linear_weights_dims[0];

    synGEMMParams gemm_params_f_f;
    gemm_params_f_f.transpose_a = false;
    gemm_params_f_f.transpose_b = false;

    synGEMMParams gemm_params_t_f;
    gemm_params_t_f.transpose_a = true;
    gemm_params_t_f.transpose_b = false;

    synGEMMParams gemm_params_f_t;
    gemm_params_f_t.transpose_a = false;
    gemm_params_f_t.transpose_b = true;

    std::vector<int64_t> scaler_dims = {1};
    auto scaler_tensor =
        createTensorNoPresist("scaler_tensor", syn_type_bf16, scaler_dims);
    std::vector<synTensor> scaler;
    scaler.push_back(scaler_tensor);
    AddNodeFull<T>(scaler, params.const_params, guid_ + "full_scale");

    auto q_tensor = createTensorFromCT(&ct, 0);
    std::vector<synTensor> scaled_q_in;
    scaled_q_in.push_back(q_tensor);
    scaled_q_in.push_back(scaler_tensor);

    auto scaled_q =
        createTensorNoPresist("scaled_q", inputs[0].type, inputs[0].dims);
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
    auto block_mapping = createTensorFromCT(&ct, 5);
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
    auto key_cache = createTensorFromCT(&ct, 1);
    auto value_cache = createTensorFromCT(&ct, 2);
    auto block_list = createTensorFromCT(&ct, 4);

    index_select_k_in.push_back(key_cache);
    index_select_v_in.push_back(value_cache);
    index_select_k_in.push_back(block_list);
    index_select_v_in.push_back(block_list);

    std::vector<int64_t> index_selected_dims;
    index_selected_dims.push_back(num_of_block);
    index_selected_dims.push_back(block_size);
    index_selected_dims.push_back(num_kv_head);
    index_selected_dims.push_back(head_dim);

    auto index_select_k_i =
        createTensorNoPresist("index_select_k_i", dtype_, index_selected_dims);
    auto index_select_v_i =
        createTensorNoPresist("index_select_v_i", dtype_, index_selected_dims);
    std::vector<synTensor> index_select_k_out;
    index_select_k_out.push_back(index_select_k_i);
    std::vector<synTensor> index_select_v_out;
    index_select_v_out.push_back(index_select_v_i);

    AddNodeIndexSelect<T>(index_select_k_in,
                          index_select_k_out,
                          params.index_select_params,
                          guid_ + "index_select_k_i");
    AddNodeIndexSelect<T>(index_select_v_in,
                          index_select_v_out,
                          params.index_select_params,
                          guid_ + "index_select_v_i");

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
        createTensorNoPresist("transpose_k", dtype_, transpose_dims);
    std::vector<synTensor> trans_index_select_k;
    trans_index_select_k.push_back(transpose_k);

    AddNodeTranspose(index_select_k_out,
                     trans_index_select_k,
                     trans_params,
                     guid_ + "transpose_k");

    auto transpose_v =
        createTensorNoPresist("transpose_v", dtype_, transpose_dims);
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
    // qk_scale here
    auto q_k = createTensorNoPresist("q_k", dtype_, q_k_dims);
    std::vector<synTensor> q_k_out;
    q_k_out.push_back(q_k);

    AddNodeMixedPrecisionGemm<T>(
        params.use_fp8, ct, 8, q_k_in, q_k_out, gemm_params_f_t, "q_k");

    /*******************************/

    auto block_bias = createTensorFromCT(&ct, 6);
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

    auto block_groups = createTensorFromCT(&ct, 3);
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

    AddNodeMixedPrecisionGemm<T>(params.use_fp8,
                                 ct,
                                 10,
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

    std::vector<int64_t> reshape_attn_dims;
    reshape_attn_dims.push_back(batch_size);
    reshape_attn_dims.push_back(1);
    reshape_attn_dims.push_back(hidden_size);
    auto attn = createTensorNoPresist("attn", dtype_, reshape_attn_dims);
    std::vector<synTensor> attn_out;
    attn_out.push_back(attn);

    AddNodeReshape(map_attn_out, attn_out, guid_ + "attn");

    std::vector<synTensor> proj_in;
    auto linear_weights = createTensorFromCT(&ct, 7);
    proj_in.push_back(attn);
    proj_in.push_back(linear_weights);

    auto linear_out = createTensorFromCT(&ct, 0, false);
    std::vector<synTensor> proj_out;
    proj_out.push_back(linear_out);

    synGEMMParams gemm_params =
        params.use_fp8 ? gemm_params_f_t : gemm_params_f_f;
    AddNodeMixedPrecisionGemm<T>(
        params.use_fp8, ct, 12, proj_in, proj_out, gemm_params, "proj");
  }

 protected:
  synDataType dtype_;
};

class FusedFlatPaGQAProj : public FusedFlatPaBase {
 public:
  explicit FusedFlatPaGQAProj(std::string name, synDataType dtype)
      : FusedFlatPaBase(name, false), dtype_(dtype) {}
  template <typename T>
  void AddNode(ConvertTensors& ct, FusedFlatPaParams params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    std::vector<int64_t> q_dims = std::vector<int64_t>(inputs[0].dims);
    std::vector<int64_t> kv_dims = std::vector<int64_t>(inputs[1].dims);
    std::vector<int64_t> block_list_dims = std::vector<int64_t>(inputs[4].dims);
    std::vector<int64_t> linear_weights_dims =
        std::vector<int64_t>(inputs[7].dims);
    int64_t batch_size = q_dims[0];
    int64_t num_head = q_dims[2];
    int64_t head_dim = q_dims[3];
    int64_t block_size = kv_dims[1];
    int64_t num_kv_head = kv_dims[2];
    int64_t num_of_block = block_list_dims[0];
    int64_t hidden_size =
        params.use_fp8 ? linear_weights_dims[1] : linear_weights_dims[0];
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

    std::vector<int64_t> scaler_dims = {1};
    auto scaler_tensor =
        createTensorNoPresist("scaler_tensor", syn_type_bf16, scaler_dims);
    std::vector<synTensor> scaler;
    scaler.push_back(scaler_tensor);
    AddNodeFull<T>(scaler, params.const_params, guid_ + "full_scale");

    auto q_tensor = createTensorFromCT(&ct, 0);
    std::vector<synTensor> scaled_q_in;
    scaled_q_in.push_back(q_tensor);
    scaled_q_in.push_back(scaler_tensor);

    auto scaled_q =
        createTensorNoPresist("scaled_q", inputs[0].type, inputs[0].dims);
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
    auto block_mapping = createTensorFromCT(&ct, 5);
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
    auto key_cache = createTensorFromCT(&ct, 1);
    auto value_cache = createTensorFromCT(&ct, 2);
    auto block_list = createTensorFromCT(&ct, 4);

    index_select_k_in.push_back(key_cache);
    index_select_v_in.push_back(value_cache);
    index_select_k_in.push_back(block_list);
    index_select_v_in.push_back(block_list);

    std::vector<int64_t> index_selected_dims;
    index_selected_dims.push_back(num_of_block);
    index_selected_dims.push_back(block_size);
    index_selected_dims.push_back(num_kv_head);
    index_selected_dims.push_back(head_dim);

    auto index_select_k_i =
        createTensorNoPresist("index_select_k_i", dtype_, index_selected_dims);
    auto index_select_v_i =
        createTensorNoPresist("index_select_v_i", dtype_, index_selected_dims);
    std::vector<synTensor> index_select_k_out;
    index_select_k_out.push_back(index_select_k_i);
    std::vector<synTensor> index_select_v_out;
    index_select_v_out.push_back(index_select_v_i);

    AddNodeIndexSelect<T>(index_select_k_in,
                          index_select_k_out,
                          params.index_select_params,
                          guid_ + "index_select_k_i");
    AddNodeIndexSelect<T>(index_select_v_in,
                          index_select_v_out,
                          params.index_select_params,
                          guid_ + "index_select_v_i");

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
        createTensorNoPresist("transpose_k", dtype_, transpose_dims);
    std::vector<synTensor> trans_index_select_k;
    trans_index_select_k.push_back(transpose_k);

    AddNodeTranspose(index_select_k_out,
                     trans_index_select_k,
                     trans_params,
                     guid_ + "transpose_k");

    auto transpose_v =
        createTensorNoPresist("transpose_v", dtype_, transpose_dims);
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
        createTensorNoPresist("index_select_k", dtype_, reshape_kv_dims);
    std::vector<synTensor> reshape_index_select_k;
    reshape_index_select_k.push_back(index_select_k);

    AddNodeReshape(
        trans_index_select_k, reshape_index_select_k, guid_ + "reshape_k");

    auto index_select_v =
        createTensorNoPresist("index_select_v", dtype_, reshape_kv_dims);
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

    AddNodeMixedPrecisionGemm<T>(
        params.use_fp8, ct, 8, q_k_in, q_k_out, gemm_params_f_t, "q_k");

    /*******************************/

    auto block_bias = createTensorFromCT(&ct, 6);
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

    auto block_groups = createTensorFromCT(&ct, 3);
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

    AddNodeMixedPrecisionGemm<T>(params.use_fp8,
                                 ct,
                                 10,
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

    std::vector<int64_t> reshape_attn_dims;
    reshape_attn_dims.push_back(batch_size);
    reshape_attn_dims.push_back(1);
    reshape_attn_dims.push_back(hidden_size);
    auto attn = createTensorNoPresist("attn", dtype_, reshape_attn_dims);
    std::vector<synTensor> attn_out;
    attn_out.push_back(attn);

    AddNodeReshape(map_attn_out, attn_out, guid_ + "attn");

    std::vector<synTensor> proj_in;
    auto linear_weights = createTensorFromCT(&ct, 7);
    proj_in.push_back(attn);
    proj_in.push_back(linear_weights);

    auto linear_out = createTensorFromCT(&ct, 0, false);
    std::vector<synTensor> proj_out;
    proj_out.push_back(linear_out);

    synGEMMParams gemm_params =
        params.use_fp8 ? gemm_params_f_t : gemm_params_f_f;
    AddNodeMixedPrecisionGemm<T>(
        params.use_fp8, ct, 12, proj_in, proj_out, gemm_params, "proj");
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void FusedFlatPaProjKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& query,
    const phi::DenseTensor& key_cache,
    const phi::DenseTensor& value_cache,
    const phi::DenseTensor& block_groups,
    const phi::DenseTensor& block_list,
    const phi::DenseTensor& block_mapping,
    const phi::DenseTensor& block_bias,
    const phi::DenseTensor& linear_weights,
    const paddle::optional<phi::DenseTensor>& qk_scale_x,
    const paddle::optional<phi::DenseTensor>& qk_scale_y,
    const paddle::optional<phi::DenseTensor>& av_scale_x,
    const paddle::optional<phi::DenseTensor>& av_scale_y,
    const paddle::optional<phi::DenseTensor>& o_linear_scale_x,
    const paddle::optional<phi::DenseTensor>& o_linear_scale_y,
    phi::DenseTensor* out_linear,
    const phi::Scalar& scaling_factor) {
  ConvertTensors ct;
  ct.Add(query);
  ct.Add(key_cache);
  ct.Add(value_cache);
  ct.Add(block_groups);
  ct.Add(block_list);
  ct.Add(block_mapping);
  ct.Add(block_bias);
  ct.Add(linear_weights);
  ct.Add(out_linear, false);

  std::string guid_prefix = "fused_flatpa_proj_fwd_";
  bool use_fp8 = false;
  if (qk_scale_x || qk_scale_y || av_scale_x || av_scale_y ||
      o_linear_scale_x || o_linear_scale_y) {
    if (!qk_scale_x || !qk_scale_y || !av_scale_x || !av_scale_y ||
        !o_linear_scale_x || !o_linear_scale_y) {
      throw std::runtime_error(
          "Please specify all scale values for FusedFp8FlatPaProjKernel");
    }

    use_fp8 = true;
    guid_prefix = "fused_fp8_flatpa_proj_fwd_";
    ct.Add(qk_scale_x.get());
    ct.Add(qk_scale_y.get());
    ct.Add(av_scale_x.get());
    ct.Add(av_scale_y.get());
    ct.Add(o_linear_scale_x.get());
    ct.Add(o_linear_scale_y.get());
  }
  std::vector<DIMS> inputs_dims = ct.GetDims();

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, nullptr_t>(
      guid_prefix.c_str(), inputs_dims, nullptr);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FusedFlatPaParams params;
    memset(reinterpret_cast<void*>(&params), 0x00, sizeof(FusedFlatPaParams));

    params.const_params.constant.f = scaling_factor.to<float>();
    params.index_select_params.axis = 3;
    params.reduce_params.reductionDimension = 0;
    params.index_reduce_params.mode = INDEX_REDUCE_AMAX;
    params.index_reduce_params.include_self = true;
    params.index_reduce_params.axis = 0;
    params.use_fp8 = use_fp8;

    std::vector<int64_t> query_dims = phi::vectorize<int64_t>(query.dims());
    std::vector<int64_t> key_cache_dims =
        phi::vectorize<int64_t>(key_cache.dims());
    int64_t q_head = query_dims[1];
    int64_t kv_head = key_cache_dims[1];

    if (q_head == kv_head) {
      FusedFlatPaMHAProj op(guid_prefix + "MHA_", op_info.datatype_);
      op.AddNode<T>(ct, params);
      op.Compile();
      op_info.setOp(op);
    } else {
      FusedFlatPaGQAProj op(guid_prefix + "GAQ_", op_info.datatype_);
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
void CallFusedFlatPaProjKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& query,
    const phi::DenseTensor& key_cache,
    const phi::DenseTensor& value_cache,
    const phi::DenseTensor& block_groups,
    const phi::DenseTensor& block_list,
    const phi::DenseTensor& block_mapping,
    const phi::DenseTensor& block_bias,
    const phi::DenseTensor& linear_weights,
    const paddle::optional<phi::DenseTensor>& qk_scale_x,
    const paddle::optional<phi::DenseTensor>& qk_scale_y,
    const paddle::optional<phi::DenseTensor>& av_scale_x,
    const paddle::optional<phi::DenseTensor>& av_scale_y,
    const paddle::optional<phi::DenseTensor>& o_linear_scale_x,
    const paddle::optional<phi::DenseTensor>& o_linear_scale_y,
    phi::DenseTensor* out_linear,
    const phi::Scalar& scaling_factor) {
  if (query.dtype() == phi::DataType::FLOAT16) {
    custom_kernel::FusedFlatPaProjKernel<phi::dtype::float16>(dev_ctx,
                                                              query,
                                                              key_cache,
                                                              value_cache,
                                                              block_groups,
                                                              block_list,
                                                              block_mapping,
                                                              block_bias,
                                                              linear_weights,
                                                              qk_scale_x,
                                                              qk_scale_y,
                                                              av_scale_x,
                                                              av_scale_y,
                                                              o_linear_scale_x,
                                                              o_linear_scale_y,
                                                              out_linear,
                                                              scaling_factor);
  } else if (query.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::FusedFlatPaProjKernel<phi::dtype::bfloat16>(dev_ctx,
                                                               query,
                                                               key_cache,
                                                               value_cache,
                                                               block_groups,
                                                               block_list,
                                                               block_mapping,
                                                               block_bias,
                                                               linear_weights,
                                                               qk_scale_x,
                                                               qk_scale_y,
                                                               av_scale_x,
                                                               av_scale_y,
                                                               o_linear_scale_x,
                                                               o_linear_scale_y,
                                                               out_linear,
                                                               scaling_factor);
  } else {
    throw std::runtime_error("Unsupported data type for FusedFlatPaProjKernel");
  }
}

std::vector<paddle::Tensor> FusedFlatPaProj(
    const paddle::Tensor& query,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& block_groups,
    const paddle::Tensor& block_list,
    const paddle::Tensor& block_mapping,
    const paddle::Tensor& block_bias,
    const paddle::Tensor& linear_weights,
    float scaling_factor) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(query.place()));
  auto query_tensor = static_cast<const phi::DenseTensor*>(query.impl().get());
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
  auto linear_weights_tensor =
      static_cast<const phi::DenseTensor*>(linear_weights.impl().get());

  // allocate memory on device.
  int64_t batch_size = query.dims()[0];
  int out_features = linear_weights.dims()[1];
  // int64_t block_size = key_cache.dims()[2];
  // int64_t num_of_block = block_mapping.dims()[0];

  std::shared_ptr<phi::DenseTensor> out_linear =
      std::make_shared<phi::DenseTensor>();
  out_linear->Resize(phi::make_ddim({batch_size, 1, out_features}));

  dev_ctx->Alloc(out_linear.get(), query_tensor->dtype());

  CallFusedFlatPaProjKernel(*dev_ctx,
                            *query_tensor,
                            *key_cache_tensor,
                            *value_cache_tensor,
                            *block_groups_tensor,
                            *block_list_tensor,
                            *block_mapping_tensor,
                            *block_bias_tensor,
                            *linear_weights_tensor,
                            paddle::optional<phi::DenseTensor>(),
                            paddle::optional<phi::DenseTensor>(),
                            paddle::optional<phi::DenseTensor>(),
                            paddle::optional<phi::DenseTensor>(),
                            paddle::optional<phi::DenseTensor>(),
                            paddle::optional<phi::DenseTensor>(),
                            out_linear.get(),
                            phi::Scalar(scaling_factor));
  return {paddle::Tensor(out_linear)};
}

std::vector<std::vector<int64_t>> FusedFlatPaProjShape(
    const std::vector<int64_t>& query_shape,
    const std::vector<int64_t>& key_cache_shape,
    const std::vector<int64_t>& value_cache_shape,
    const std::vector<int64_t>& block_groups_shape,
    const std::vector<int64_t>& block_list_shape,
    const std::vector<int64_t>& block_mapping_shape,
    const std::vector<int64_t>& block_bias_shape,
    const std::vector<int64_t>& linear_weights_shape) {
  int64_t batch_size = query_shape[0];
  int out_features = linear_weights_shape[1];
  return {{batch_size, 1, out_features}};
}

std::vector<paddle::DataType> FusedFlatPaProjDtype(
    const paddle::DataType& query_dtype,
    const paddle::DataType& key_cache_dtype,
    const paddle::DataType& value_cache_dtype,
    const paddle::DataType& block_groups_dtype,
    const paddle::DataType& block_list_dtype,
    const paddle::DataType& block_mapping_dtype,
    const paddle::DataType& block_bias_dtype,
    const paddle::DataType& linear_weights_dtype) {
  return {query_dtype};
}

PD_BUILD_OP(fused_flatpa_proj)
    .Inputs({"query",
             "key_cache",
             "value_cache",
             "block_groups",
             "block_list",
             "block_mapping",
             "block_bias",
             "linear_weights"})
    .Outputs({"out_linear"})
    .Attrs({"scaling_factor: float"})
    .SetKernelFn(PD_KERNEL(FusedFlatPaProj))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedFlatPaProjShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedFlatPaProjDtype));

std::vector<paddle::Tensor> FusedFp8FlatPaProj(
    const paddle::Tensor& query,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& block_groups,
    const paddle::Tensor& block_list,
    const paddle::Tensor& block_mapping,
    const paddle::Tensor& block_bias,
    const paddle::Tensor& linear_weights,
    const paddle::Tensor& qk_scale_x,
    const paddle::Tensor& qk_scale_y,
    const paddle::Tensor& av_scale_x,
    const paddle::Tensor& av_scale_y,
    const paddle::Tensor& o_linear_scale_x,
    const paddle::Tensor& o_linear_scale_y,
    float scaling_factor) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(query.place()));
  auto query_tensor = static_cast<const phi::DenseTensor*>(query.impl().get());
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
  auto linear_weights_tensor =
      static_cast<const phi::DenseTensor*>(linear_weights.impl().get());
  auto qk_scale_x_tensor =
      static_cast<const phi::DenseTensor*>(qk_scale_x.impl().get());
  auto qk_scale_y_tensor =
      static_cast<const phi::DenseTensor*>(qk_scale_y.impl().get());
  auto av_scale_x_tensor =
      static_cast<const phi::DenseTensor*>(av_scale_x.impl().get());
  auto av_scale_y_tensor =
      static_cast<const phi::DenseTensor*>(av_scale_y.impl().get());
  auto o_linear_scale_x_tensor =
      static_cast<const phi::DenseTensor*>(o_linear_scale_x.impl().get());
  auto o_linear_scale_y_tensor =
      static_cast<const phi::DenseTensor*>(o_linear_scale_y.impl().get());

  // allocate memory on device.
  int64_t batch_size = query.dims()[0];
  int out_features = linear_weights.dims()[0];

  std::shared_ptr<phi::DenseTensor> out_linear =
      std::make_shared<phi::DenseTensor>();
  out_linear->Resize(phi::make_ddim({batch_size, 1, out_features}));

  dev_ctx->Alloc(out_linear.get(), query_tensor->dtype());

  CallFusedFlatPaProjKernel(*dev_ctx,
                            *query_tensor,
                            *key_cache_tensor,
                            *value_cache_tensor,
                            *block_groups_tensor,
                            *block_list_tensor,
                            *block_mapping_tensor,
                            *block_bias_tensor,
                            *linear_weights_tensor,
                            *qk_scale_x_tensor,
                            *qk_scale_y_tensor,
                            *av_scale_x_tensor,
                            *av_scale_y_tensor,
                            *o_linear_scale_x_tensor,
                            *o_linear_scale_y_tensor,
                            out_linear.get(),
                            phi::Scalar(scaling_factor));

  return {paddle::Tensor(out_linear)};
}

PD_BUILD_OP(fused_fp8_flatpa_proj)
    .Inputs({"query",
             "key_cache",
             "value_cache",
             "block_groups",
             "block_list",
             "block_mapping",
             "block_bias",
             "linear_weights",
             "qk_scale_x",
             "qk_scale_y",
             "av_scale_x",
             "av_scale_y",
             "o_linear_scale_x",
             "o_linear_scale_y"})
    .Outputs({"out_linear"})
    .Attrs({"scaling_factor: float"})
    .SetKernelFn(PD_KERNEL(FusedFp8FlatPaProj))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedFlatPaProjShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedFlatPaProjDtype));
