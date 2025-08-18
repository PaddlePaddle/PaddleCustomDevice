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
#include <chrono>
#include <limits>
#include <random>

#include "habanalabs/perf_lib_layer_params.h"
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "kernels/hpu_funcs.h"
#include "kernels/hpu_operator.h"
#include "utils/utils.h"

const float NEG_INF = std::numeric_limits<float>::lowest();

namespace custom_kernel {

class TopP : public HpuFusedOperator {
 public:
  TopP() : HpuFusedOperator("top_p_sampling") {}

  template <typename T>
  void AddNode(ConvertTensors& ct) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    // TopK Node to get sorted probs and indices
    ns_TopkNodeV2::ParamsV4 topk_params{};
    topk_params.bsw = inputs[0].dims[1];
    topk_params.axis = 0;
    topk_params.bottomK = false;
    topk_params.isVcData = false;
    topk_params.isStable = false;

    auto probs = createTensorFromCT(&ct, 0);
    auto sorted_probs =
        createTensorNoPresist("sorted_probs", inputs[0].type, inputs[0].dims);
    auto sorted_indices =
        createTensorNoPresist("sorted_indices", syn_type_int32, inputs[0].dims);
    std::vector<synTensor> topk_ins = {probs};
    std::vector<synTensor> topk_outs = {sorted_probs, sorted_indices};
    AddNodeTopK(topk_ins, topk_outs, topk_params, guid_ + "topk");

    // Cumsum Node to get cumsum probs
    auto cumsum_probs =
        createTensorNoPresist("cumsum_probs", inputs[0].type, inputs[0].dims);
    std::vector<synTensor> cumsum_ins = {sorted_probs};
    std::vector<synTensor> cumsum_outs = {cumsum_probs};
    ns_CumSumKernel::Params cumsum_params{0, 0, 0};
    AddNodeCumsum<T>(cumsum_ins, cumsum_outs, cumsum_params, guid_ + "cumsum");

    // Sub to get the cumulative sum before the current element.
    auto sub_probs =
        createTensorNoPresist("sub_probs", inputs[0].type, inputs[0].dims);
    std::vector<synTensor> sub_ins = {cumsum_probs, sorted_probs};
    std::vector<synTensor> sub_outs = {sub_probs};
    AddNodeSub<T>(sub_ins, sub_outs, guid_ + "sub");

    // Less Equal to get the selected_index
    auto top_p_value = createTensorFromCT(&ct, 1);
    auto mask = createTensorNoPresist("mask", syn_type_int8, inputs[0].dims);
    std::vector<synTensor> less_equal_ins = {sub_probs, top_p_value};
    std::vector<synTensor> less_equal_outs = {mask};
    AddNodeLessEqual<T>(less_equal_ins, less_equal_outs, guid_ + "less_equal");

    // Scalar Node neg_inf
    std::vector<int64_t> scalar_dims = {1};
    auto neg_inf =
        createTensorNoPresist("neg_inf", syn_type_float, scalar_dims);
    ns_ConstantKernel::Params const_params;
    const_params.constant.f = NEG_INF;
    std::vector<synTensor> full_out = {neg_inf};
    AddNodeFull<T>(full_out, const_params, guid_ + "full_neg_inf");

    // Where to populate unwanted probs with -inf
    auto filtered_probs =
        createTensorNoPresist("filtered_probs", inputs[0].type, inputs[0].dims);
    std::vector<synTensor> where_ins = {mask, sorted_probs, neg_inf};
    std::vector<synTensor> where_outs = {filtered_probs};
    AddNodeWhere<T>(where_ins, where_outs, guid_ + "where");

    // Multinomial to select indices of sorted indices
    auto selected_indices = createTensorNoPresist(
        "selected_indices", syn_type_int32, outputs[1].dims);
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
    AddNodeMultinomial<T>(multinomial_ins,
                          multinomial_outs,
                          multinomial_params,
                          guid_ + "multinomial");

    // Gather Elements to get selected token indices
    auto token_indices =
        createTensorNoPresist("token_indices", syn_type_int32, outputs[1].dims);
    ns_GatherKernel::Params gather_params;
    gather_params.axis = 0;
    std::vector<synTensor> index_sample_ins = {sorted_indices,
                                               selected_indices};
    std::vector<synTensor> index_sample_outs = {token_indices};
    AddNodeIndexSample<int32_t>(index_sample_ins,
                                index_sample_outs,
                                gather_params,
                                guid_ + "index_sample");

    // Cast to output format int64
    auto token_indices_out = createTensorFromCT(&ct, 1, false);
    std::vector<synTensor> cast_ins = {token_indices};
    std::vector<synTensor> cast_outs = {token_indices_out};
    AddNodeCast(
        cast_ins, cast_outs, "cast_i32_to_i64", guid_ + "cast_token_ids");
  }
};
}  // namespace custom_kernel

namespace custom_kernel {
template <typename T, typename Context>
void TopPSamplingKernel_hpu(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            const phi::DenseTensor& ps,
                            const paddle::optional<phi::DenseTensor>& threshold,
                            const paddle::optional<phi::DenseTensor>& topp_seed,
                            int seed,
                            int k,
                            const std::string& mode,
                            phi::DenseTensor* out,
                            phi::DenseTensor* ids,
                            phi::DenseTensor* topk_scores,
                            phi::DenseTensor* topk_ids) {
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<int64_t>(ids);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(ps);
  ct.Add(*out, false);
  ct.Add(*ids, false);

  auto x_dims = phi::vectorize<int64_t>(x.dims());
  int length = x_dims[1];
  ns_TopkNodeV2::ParamsV4 params{};
  params.bsw = length;
  params.axis = 0;
  params.bottomK = false;
  params.isVcData = false;
  params.isStable = false;

  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, ns_TopkNodeV2::ParamsV4>(
      "ToppKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    TopP op;
    op.AddNode<T>(ct);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(top_p_sampling,
                          intel_hpu,
                          ALL_LAYOUT,
                          custom_kernel::TopPSamplingKernel_hpu,
                          float) {}
