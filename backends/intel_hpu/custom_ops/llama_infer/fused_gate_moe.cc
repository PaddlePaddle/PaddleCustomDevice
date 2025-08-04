// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#define MIN_FP8_VALUES -240.0
#define MAX_FP8_VALUES 240.0

struct FusedGateMoeParams {
  char activation_mode[32];

  int topk;
  int32_t num_experts;
  int32_t experts_min;
  int32_t experts_max;
  int32_t block_size;

  bool moe_use_gate_correction_bias;
  bool norm_topk_prob;
  bool permuted_weights;

  bool fused_gemm;
  bool measurement_mode;
  bool dynamic_scale;
};

enum TENSOR_IDS_IN {
  HIDDEN_STATES = 0,
  GATE_OUT = 1,
  BIAS_OR_WEIGHTS,  // 2 + bias_offset
  EOS_TOKEN
};

static const std::map<std::string_view, MoeActivationMode_t> activationModeMap =
    {{"selu", MoeActivationMode_t::MOE_ACTIVATION_MODE_SELU},
     {"gelu", MoeActivationMode_t::MOE_ACTIVATION_MODE_GELU},
     {"relu", MoeActivationMode_t::MOE_ACTIVATION_MODE_RELU},
     {"silu", MoeActivationMode_t::MOE_ACTIVATION_MODE_SILU}};

std::shared_ptr<ns_MoeKernel::ParamsV3> FillMixtureOfExpertsParams(
    const FusedGateMoeParams& config) {
  auto moe_params = std::make_shared<ns_MoeKernel::ParamsV3>();
  memset(reinterpret_cast<void*>(moe_params.get()),
         0x00,
         sizeof(ns_MoeKernel::ParamsV3));

  std::string activation_str(config.activation_mode);
  auto activationIterator = activationModeMap.find(activation_str);
  moe_params->experts.activation = activationIterator->second;

  moe_params->router.experts_min = config.experts_min;
  moe_params->router.experts_max = config.experts_max;

  moe_params->flags =
      config.permuted_weights ? MoeFlags_t::MOE_FLAGS_PERMUTED_WEIGHTS : 0;
  moe_params->flags |=
      (config.fused_gemm ? MoeFlags_t::MOE_FLAGS_FUSED_GEMM : 0);
  moe_params->flags |=
      (config.measurement_mode ? MoeFlags_t::MOE_FLAGS_CALC_AMAX : 0);
  moe_params->flags |=
      (config.dynamic_scale ? MoeFlags_t::MOE_FLAGS_DYNAMIC_SCALE : 0);
  moe_params->flags |=
      (config.block_size > 0
           ? MoeFlags_t::MOE_FLAGS_BLOCKWISE_WEIGHT_QUANTIZATION
           : 0);
  moe_params->block_size = config.block_size;

  return moe_params;
}

class FusedGateMoe : public HpuFusedOperator {
 public:
  explicit FusedGateMoe(synDataType dtype)
      : HpuFusedOperator("fused_gate_moe_fwd_", false), dtype_(dtype) {}
  template <typename T, typename TMoe>
  void AddNode(ConvertTensors& ct, FusedGateMoeParams params) {
    auto ins = ct.GetTensors();
    auto gate_data_type = ins[GATE_OUT].type;

    /* ---------------- MoE Gate ---------------- */

    // weights = paddle.nn.functional.softmax(gate_out, axis=-1)
    auto gate_out = createTensorFromCT(&ct, GATE_OUT);
    std::vector<synTensor> softmax_in;
    softmax_in.push_back(gate_out);

    auto weights =
        createTensorNoPresist("weights", gate_data_type, ins[GATE_OUT].dims);
    std::vector<synTensor> softmax_out;
    softmax_out.push_back(weights);

    ns_Softmax::Params softmax_params;
    softmax_params.dim = 0;
    AddNodeSoftmax<float>(
        softmax_in, softmax_out, softmax_params, guid_ + "softmax");

    ns_TopkNodeV2::ParamsV4 topk_params{};
    topk_params.bsw = params.topk;
    topk_params.axis = 0;
    topk_params.bottomK = false;
    topk_params.isVcData = false;
    topk_params.isStable = false;

    std::vector<int64_t> topk_dims = std::vector<int64_t>(ins[GATE_OUT].dims);
    topk_dims[1] = params.topk;
    auto routing_weights_fp32 = createTensorNoPresist(
        "routing_weights_fp32", gate_data_type, topk_dims);
    auto selected_experts =
        createTensorNoPresist("selected_experts", syn_type_int32, topk_dims);

    int bias_offset = 0;
    // if layer.moe_use_gate_correction_bias:
    if (params.moe_use_gate_correction_bias) {
      // scores = weights + layer.gate_correction_bias
      bias_offset = 1;
      int bias_base = BIAS_OR_WEIGHTS;
      auto gate_correction_bias = createTensorFromCT(&ct, bias_base);
      auto gate_correction_out = createTensorNoPresist(
          "gate_correction_out", gate_data_type, ins[GATE_OUT].dims);
      std::vector<synTensor> gate_correction_in;
      gate_correction_in.push_back(weights);
      gate_correction_in.push_back(gate_correction_bias);
      std::vector<synTensor> gate_correction_outs;
      gate_correction_outs.push_back(gate_correction_out);
      AddNodeAdd<float>(
          gate_correction_in, gate_correction_outs, guid_ + "gate_add");

      // _, selected_experts = paddle.topk(scores, layer.top_k, axis=-1)
      auto drop_data =
          createTensorNoPresist("drop_data", syn_type_int32, topk_dims);

      std::vector<synTensor> topk_outs;
      topk_outs.push_back(drop_data);
      topk_outs.push_back(selected_experts);
      AddNodeTopK(gate_correction_outs, topk_outs, topk_params, guid_ + "topk");

      // routing_weights = paddle.index_sample(weights, selected_experts)
      std::vector<synTensor> index_sample_in;
      index_sample_in.push_back(weights);
      index_sample_in.push_back(selected_experts);

      std::vector<synTensor> index_sample_out;
      index_sample_out.push_back(routing_weights_fp32);

      ns_GatherKernel::Params index_sample_params;
      index_sample_params.axis = 0;
      AddNodeIndexSample<float>(index_sample_in,
                                index_sample_out,
                                index_sample_params,
                                guid_ + "index_sample");
    } else {
      // routing_weights, selected_experts = paddle.topk(weights, layer.top_k,
      // axis=-1)
      std::vector<synTensor> topk_outs;
      topk_outs.push_back(routing_weights_fp32);
      topk_outs.push_back(selected_experts);
      AddNodeTopK(softmax_out, topk_outs, topk_params, guid_ + "topk");
    }

    // routing_weights /= paddle.sum(routing_weights, axis=-1, keepdim=True)
    std::vector<synTensor> cast_in;
    if (params.norm_topk_prob) {
      // Normalize the top-k probabilities
      std::vector<synTensor> reduceSum_in;
      reduceSum_in.push_back(routing_weights_fp32);

      auto reduceSum = createTensorNoPresist(
          "reduceSum", gate_data_type, {ins[GATE_OUT].dims[0], 1});
      std::vector<synTensor> reduceSum_out;
      reduceSum_out.push_back(reduceSum);

      ns_Reduction::Params reduce_sum_params;
      reduce_sum_params.reductionDimension = 0;

      AddNodeReduceSum<float>(
          reduceSum_in, reduceSum_out, reduce_sum_params, guid_ + "reduceSum");

      std::vector<synTensor> norm_in;
      norm_in.push_back(routing_weights_fp32);
      norm_in.push_back(reduceSum);

      auto routing_weights_norm = createTensorNoPresist(
          "routing_weights_norm", gate_data_type, topk_dims);
      std::vector<synTensor> norm_out;
      norm_out.push_back(routing_weights_norm);
      AddNodeDivide<float>(norm_in, norm_out, guid_ + "divide");

      cast_in.push_back(routing_weights_norm);
    } else {
      cast_in.push_back(routing_weights_fp32);
    }

    // routing_weights.cast("bfloat16")
    std::vector<synTensor> cast_out;
    auto routing_weights_bf16 = createTensorNoPresist(
        "routing_weights_bf16", ins[HIDDEN_STATES].type, topk_dims);
    cast_out.push_back(routing_weights_bf16);
    AddNodeCast(cast_in, cast_out, "cast_f32_to_bf16", guid_ + "cast");

    std::vector<synTensor> inputs;
    synTensor hidden_states = createTensorFromCT(&ct, HIDDEN_STATES);
    synTensor fp8_scale = nullptr;

    /* ---------------- quant_fn for fp8 MoE  ---------------- */
    // x, x_scale = self.quant_fn(x)
    if (dtype_ == syn_type_fp8_143) {
      ns_ConstantKernel::Params const_params;
      synTensor q_min =
          createTensorNoPresist("q_min", ins[HIDDEN_STATES].type, {1});
      const_params.constant.f = MIN_FP8_VALUES;
      std::vector<synTensor> min_tensor = {q_min};
      AddNodeFull<T>(min_tensor, const_params, guid_ + "full_min");

      synTensor q_max =
          createTensorNoPresist("q_max", ins[HIDDEN_STATES].type, {1});
      const_params.constant.f = MAX_FP8_VALUES;
      std::vector<synTensor> max_tensor = {q_max};
      AddNodeFull<T>(max_tensor, const_params, guid_ + "full_max");

      synTensor zeropoint =
          createTensorNoPresist("zeropoint", ins[HIDDEN_STATES].type, {1});
      const_params.constant.f = 0;
      std::vector<synTensor> zeropoint_tensor = {zeropoint};
      AddNodeFull<T>(zeropoint_tensor, const_params, guid_ + "full_zero");

      std::vector<synTensor> abs_in;
      abs_in.push_back(hidden_states);
      auto hidden_states_abs = createTensorNoPresist("hidden_states_abs",
                                                     ins[HIDDEN_STATES].type,
                                                     ins[HIDDEN_STATES].dims);
      std::vector<synTensor> abs_out;
      abs_out.push_back(hidden_states_abs);
      AddNodeAbs<T>(abs_in, abs_out, guid_ + "abs");

      auto max_out =
          createTensorNoPresist("max_out", ins[HIDDEN_STATES].type, {1});
      std::vector<synTensor> max_outputs;
      max_outputs.push_back(max_out);

      ns_Reduction::ParamsV2 reduce_max_params{};
      AddNodeMaximumMultidimensional<T>(
          abs_out, max_outputs, reduce_max_params, guid_ + "reduceMax");

      std::vector<synTensor> div_inputs;
      div_inputs.push_back(max_out);
      div_inputs.push_back(q_max);
      std::vector<synTensor> div_outputs;
      // w/a Tensor fp8_scale was already mapped
      unsigned int seed = time(NULL);
      std::string fp8_scale_name = "fp8_scale_" + std::to_string(rand_r(&seed));

      fp8_scale =
          createTensorNoPresist(fp8_scale_name, ins[HIDDEN_STATES].type, {1});
      div_outputs.push_back(fp8_scale);
      AddNodeDivide<T>(div_inputs, div_outputs, guid_ + "div");

      std::vector<synTensor> quant_inputs;
      quant_inputs.push_back(hidden_states);
      quant_inputs.push_back(fp8_scale);
      quant_inputs.push_back(zeropoint);
      quant_inputs.push_back(q_min);
      quant_inputs.push_back(q_max);

      std::vector<synTensor> quant_outputs;
      synTensor scaled_hidden_states = createTensorNoPresist(
          "scaled_hidden_states", dtype_, ins[HIDDEN_STATES].dims);
      quant_outputs.push_back(scaled_hidden_states);
      AddNodeQuantizePerTensor<T>(quant_inputs, quant_outputs, guid_ + "quant");

      // fp8
      inputs.push_back(scaled_hidden_states);
    } else {
      // bf16 / blockwise fp8
      inputs.push_back(hidden_states);
    }

    /* ---------------- mixture_of_experts ---------------- */
    // fused_moe_out, _ = mixture_of_experts(...)
    auto weights_per_expert = params.fused_gemm ? 2 : 3;
    inputs.push_back(selected_experts);
    inputs.push_back(routing_weights_bf16);

    // Add gate_up_weights and down_weights
    int64_t input_count = params.num_experts * weights_per_expert;
    int weight_base = BIAS_OR_WEIGHTS + bias_offset;
    for (int64_t i = weight_base; i < weight_base + input_count; i++) {
      inputs.push_back(createTensorFromCT(&ct, i));
    }
    int scale_base = weight_base + input_count;

    if (dtype_ == syn_type_fp8_143) {
      // fp8
      // hidden_states_scales
      inputs.push_back(fp8_scale);
      auto scales_per_expert = params.fused_gemm ? 2 : 3;
      if (!params.dynamic_scale) scales_per_expert += 1;
      input_count = params.num_experts * scales_per_expert;
    } else if (params.block_size > 0) {
      // blockwise fp8
      auto scales_per_expert = params.fused_gemm ? 2 : 3;
      input_count = params.num_experts * scales_per_expert;
    } else {
      // bf16
      input_count = 0;
    }

    // Add gate_up_weights_scales and down_weights_scales
    for (int64_t i = scale_base; i < scale_base + input_count; i++) {
      inputs.push_back(createTensorFromCT(&ct, i));
    }

    std::vector<synTensor> outputs;
    outputs.push_back(createTensorFromCT(&ct, 0, false));

    auto moe_params = FillMixtureOfExpertsParams(params);
    AddNodeMoeForward<TMoe>(inputs, outputs, moe_params, guid_ + "moe");
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename TMoe, typename Context>
void FusedGateMoeKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& hidden_states,
    const phi::DenseTensor& gate_out,
    const paddle::optional<phi::DenseTensor>& gate_correction_bias,
    const std::vector<phi::DenseTensor>& gate_up_weights,
    const std::vector<phi::DenseTensor>& down_weights,
    const paddle::optional<std::vector<phi::DenseTensor>>& scales,
    phi::DenseTensor* final_hidden_states,
    const int top_k,
    const bool moe_use_gate_correction_bias,
    const bool norm_topk_prob,
    const bool permuted_weights,
    const std::string& activation,
    const int experts_min,
    const int experts_max,
    const bool measurement_mode,
    const bool dynamic_scale,
    const int block_size) {
  FusedGateMoeParams params;
  memset(reinterpret_cast<void*>(&params), 0x00, sizeof(FusedGateMoeParams));
  params.topk = top_k;
  params.moe_use_gate_correction_bias = moe_use_gate_correction_bias;
  params.norm_topk_prob = norm_topk_prob;
  params.permuted_weights = permuted_weights;
  params.fused_gemm = (gate_up_weights.size() == down_weights.size());
  params.num_experts = down_weights.size();
  params.experts_min = experts_min;
  params.experts_max = experts_max;
  params.dynamic_scale = dynamic_scale;
  params.block_size = block_size;
  strncpy(params.activation_mode,
          activation.c_str(),
          sizeof(params.activation_mode) - 1);

  ConvertTensors ct;
  ct.Add(hidden_states);
  ct.Add(gate_out);
  if (moe_use_gate_correction_bias) {
    ct.Add(gate_correction_bias.get());
  }
  for (const auto& t : gate_up_weights) {
    ct.Add(t);
  }
  for (const auto& t : down_weights) {
    ct.Add(t);
  }
  if (scales) {
    for (const auto& t : scales.get()) {
      ct.Add(t);
    }
  }

  ct.Add(*final_hidden_states, false);

  std::vector<DIMS> inputs_dims = ct.GetDims();

  OpCacheOperator op_info;
  std::string recipe_name = "FusedGateMoeKernel";
  op_info.prepareOpInfo<TMoe, FusedGateMoeParams>(
      recipe_name, inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FusedGateMoe op(op_info.datatype_);
    op.AddNode<T, TMoe>(ct, params);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

template <typename Context>
void CallFusedGateMoeKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& hidden_states,
    const phi::DenseTensor& gate_out,
    const paddle::optional<phi::DenseTensor>& gate_correction_bias,
    const std::vector<phi::DenseTensor>& gate_up_weights,
    const std::vector<phi::DenseTensor>& down_weights,
    const paddle::optional<std::vector<phi::DenseTensor>>& scales,
    phi::DenseTensor* final_hidden_states,
    const int top_k,
    const bool moe_use_gate_correction_bias,
    const bool norm_topk_prob,
    const bool permuted_weights,
    const std::string& activation,
    const int experts_min,
    const int experts_max,
    const bool is_bf16_moe_input,
    const bool measurement_mode,
    const bool dynamic_scale,
    const int block_size) {
  if (hidden_states.dtype() == phi::DataType::BFLOAT16) {
    if (is_bf16_moe_input) {
      // bf16 & blockwise fp8
      custom_kernel::FusedGateMoeKernel<phi::dtype::bfloat16,
                                        phi::dtype::bfloat16>(
          dev_ctx,
          hidden_states,
          gate_out,
          gate_correction_bias,
          gate_up_weights,
          down_weights,
          scales,
          final_hidden_states,
          top_k,
          moe_use_gate_correction_bias,
          norm_topk_prob,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          measurement_mode,
          dynamic_scale,
          block_size);
    } else {
      custom_kernel::FusedGateMoeKernel<phi::dtype::bfloat16,
                                        phi::dtype::float8_e4m3fn>(
          dev_ctx,
          hidden_states,
          gate_out,
          gate_correction_bias,
          gate_up_weights,
          down_weights,
          scales,
          final_hidden_states,
          top_k,
          moe_use_gate_correction_bias,
          norm_topk_prob,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          measurement_mode,
          dynamic_scale,
          block_size);
    }
  } else {
    throw std::runtime_error("Unsupported data type for FusedGateMoeKernel");
  }
}

std::vector<paddle::Tensor> FusedGateMoeForward(
    const paddle::Tensor& hidden_states,
    const paddle::Tensor& gate_out,
    const paddle::optional<paddle::Tensor>& gate_correction_bias,
    const std::vector<paddle::Tensor>& gate_up_weights,
    const std::vector<paddle::Tensor>& down_weights,
    const int top_k,
    const bool moe_use_gate_correction_bias,
    const bool norm_topk_prob,
    const bool permuted_weights,
    const std::string& activation,
    const int experts_min,
    const int experts_max) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          hidden_states.place()));

  auto hidden_states_tensor =
      static_cast<const phi::DenseTensor*>(hidden_states.impl().get());
  auto gate_out_tensor =
      static_cast<const phi::DenseTensor*>(gate_out.impl().get());

  auto gate_correction_tensor = paddle::optional<phi::DenseTensor>();
  if (gate_correction_bias) {
    auto gate_correction_bias_dt =
        static_cast<phi::DenseTensor*>(gate_correction_bias->impl().get());
    gate_correction_tensor =
        paddle::optional<phi::DenseTensor>(*gate_correction_bias_dt);
  }

  std::vector<phi::DenseTensor> gate_up_weights_vec;
  for (const auto& t : gate_up_weights) {
    gate_up_weights_vec.push_back(
        *static_cast<const phi::DenseTensor*>(t.impl().get()));
  }
  std::vector<phi::DenseTensor> down_weights_vec;
  for (const auto& t : down_weights) {
    down_weights_vec.push_back(
        *static_cast<const phi::DenseTensor*>(t.impl().get()));
  }

  std::shared_ptr<phi::DenseTensor> final_hidden_states =
      std::make_shared<phi::DenseTensor>();
  final_hidden_states->Resize(hidden_states.dims());
  dev_ctx->Alloc(final_hidden_states.get(), hidden_states.dtype());

  CallFusedGateMoeKernel(
      *dev_ctx,
      *hidden_states_tensor,
      *gate_out_tensor,
      gate_correction_tensor,
      gate_up_weights_vec,
      down_weights_vec,
      paddle::optional<std::vector<phi::DenseTensor>>(), /* scales */
      final_hidden_states.get(),
      top_k,
      moe_use_gate_correction_bias,
      norm_topk_prob,
      permuted_weights,
      activation,
      experts_min,
      experts_max,
      true,  /* moe input = bf16 */
      false, /* measurement_mode, so far not need */
      false, /* dynamic_scale */
      -1 /* block_size */);

  return {paddle::Tensor(final_hidden_states)};
}

std::vector<paddle::Tensor> FusedGateMoeFP8Forward(
    const paddle::Tensor& hidden_states,
    const paddle::Tensor& gate_out,
    const paddle::optional<paddle::Tensor>& gate_correction_bias,
    const std::vector<paddle::Tensor>& gate_up_weights,
    const std::vector<paddle::Tensor>& down_weights,
    const paddle::optional<std::vector<paddle::Tensor>>&
        intermediate_hidden_states_scales,
    const std::vector<paddle::Tensor>& gate_up_weights_scales,
    const std::vector<paddle::Tensor>& down_weights_scales,
    const int top_k,
    const bool moe_use_gate_correction_bias,
    const bool norm_topk_prob,
    const bool permuted_weights,
    const std::string& activation,
    const int experts_min,
    const int experts_max) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          hidden_states.place()));

  auto hidden_states_tensor =
      static_cast<const phi::DenseTensor*>(hidden_states.impl().get());
  auto gate_out_tensor =
      static_cast<const phi::DenseTensor*>(gate_out.impl().get());

  auto gate_correction_tensor = paddle::optional<phi::DenseTensor>();
  if (gate_correction_bias) {
    auto gate_correction_bias_dt =
        static_cast<phi::DenseTensor*>(gate_correction_bias->impl().get());
    gate_correction_tensor =
        paddle::optional<phi::DenseTensor>(*gate_correction_bias_dt);
  }

  std::vector<phi::DenseTensor> gate_up_weights_vec;
  for (const auto& t : gate_up_weights) {
    gate_up_weights_vec.push_back(
        *static_cast<const phi::DenseTensor*>(t.impl().get()));
  }
  std::vector<phi::DenseTensor> down_weights_vec;
  for (const auto& t : down_weights) {
    down_weights_vec.push_back(
        *static_cast<const phi::DenseTensor*>(t.impl().get()));
  }

  bool dynamic_scale = true;
  std::vector<phi::DenseTensor> scales_vec;
  if (intermediate_hidden_states_scales) {
    dynamic_scale = false;
    for (const auto& t : intermediate_hidden_states_scales.get()) {
      scales_vec.push_back(
          *static_cast<const phi::DenseTensor*>(t.impl().get()));
    }
  }
  for (const auto& t : gate_up_weights_scales) {
    scales_vec.push_back(*static_cast<const phi::DenseTensor*>(t.impl().get()));
  }
  for (const auto& t : down_weights_scales) {
    scales_vec.push_back(*static_cast<const phi::DenseTensor*>(t.impl().get()));
  }

  std::shared_ptr<phi::DenseTensor> final_hidden_states =
      std::make_shared<phi::DenseTensor>();
  final_hidden_states->Resize(hidden_states.dims());
  dev_ctx->Alloc(final_hidden_states.get(), hidden_states.dtype());

  CallFusedGateMoeKernel(
      *dev_ctx,
      *hidden_states_tensor,
      *gate_out_tensor,
      gate_correction_tensor,
      gate_up_weights_vec,
      down_weights_vec,
      scales_vec,
      final_hidden_states.get(),
      top_k,
      moe_use_gate_correction_bias,
      norm_topk_prob,
      permuted_weights,
      activation,
      experts_min,
      experts_max,
      false, /* moe input = fp8*/
      false, /* measurement_mode, so far not supported on FP8 */
      dynamic_scale,
      -1 /* block_size */);
  return {paddle::Tensor(final_hidden_states)};
}

std::vector<paddle::Tensor> FusedGateMoeBlockWiseFP8Forward(
    const paddle::Tensor& hidden_states,
    const paddle::Tensor& gate_out,
    const paddle::optional<paddle::Tensor>& gate_correction_bias,
    const std::vector<paddle::Tensor>& gate_up_weights,
    const std::vector<paddle::Tensor>& down_weights,
    const std::vector<paddle::Tensor>& gate_up_weights_scales,
    const std::vector<paddle::Tensor>& down_weights_scales,
    const int top_k,
    const bool moe_use_gate_correction_bias,
    const bool norm_topk_prob,
    const bool permuted_weights,
    const std::string& activation,
    const int experts_min,
    const int experts_max,
    const int block_size) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          hidden_states.place()));

  auto hidden_states_tensor =
      static_cast<const phi::DenseTensor*>(hidden_states.impl().get());
  auto gate_out_tensor =
      static_cast<const phi::DenseTensor*>(gate_out.impl().get());

  auto gate_correction_tensor = paddle::optional<phi::DenseTensor>();
  if (gate_correction_bias) {
    auto gate_correction_bias_dt =
        static_cast<phi::DenseTensor*>(gate_correction_bias->impl().get());
    gate_correction_tensor =
        paddle::optional<phi::DenseTensor>(*gate_correction_bias_dt);
  }

  std::vector<phi::DenseTensor> gate_up_weights_vec;
  for (const auto& t : gate_up_weights) {
    gate_up_weights_vec.push_back(
        *static_cast<const phi::DenseTensor*>(t.impl().get()));
  }
  std::vector<phi::DenseTensor> down_weights_vec;
  for (const auto& t : down_weights) {
    down_weights_vec.push_back(
        *static_cast<const phi::DenseTensor*>(t.impl().get()));
  }

  std::vector<phi::DenseTensor> scales_vec;
  for (const auto& t : gate_up_weights_scales) {
    scales_vec.push_back(*static_cast<const phi::DenseTensor*>(t.impl().get()));
  }
  for (const auto& t : down_weights_scales) {
    scales_vec.push_back(*static_cast<const phi::DenseTensor*>(t.impl().get()));
  }

  std::shared_ptr<phi::DenseTensor> final_hidden_states =
      std::make_shared<phi::DenseTensor>();
  final_hidden_states->Resize(hidden_states.dims());
  dev_ctx->Alloc(final_hidden_states.get(), hidden_states.dtype());

  CallFusedGateMoeKernel(
      *dev_ctx,
      *hidden_states_tensor,
      *gate_out_tensor,
      gate_correction_tensor,
      gate_up_weights_vec,
      down_weights_vec,
      scales_vec,
      final_hidden_states.get(),
      top_k,
      moe_use_gate_correction_bias,
      norm_topk_prob,
      permuted_weights,
      activation,
      experts_min,
      experts_max,
      true,  /* moe input = bf16 */
      false, /* measurement_mode, so far not supported on FP8 */
      false, /*dynamic_scale*/
      block_size);
  return {paddle::Tensor(final_hidden_states)};
}

std::vector<std::vector<int64_t>> FusedGateMoeInferShape(
    const std::vector<int64_t>& hidden_states_shape,
    const std::vector<int64_t>& gate_out_shape,
    const paddle::optional<std::vector<int64_t>>& gate_correction_bias_shape,
    const std::vector<int64_t>& gate_up_weights_shape,
    const std::vector<int64_t>& down_weights_shape) {
  return {hidden_states_shape};
}

std::vector<paddle::DataType> FusedGateMoeInferDtype(
    const paddle::DataType& hidden_states_dtype,
    const paddle::DataType& gate_out_dtype,
    const paddle::optional<paddle::DataType>& gate_correction_bias_dtype,
    const paddle::DataType& gate_up_weights_dtype,
    const paddle::DataType& down_weights_dtype) {
  return {hidden_states_dtype};
}

// hidden_states        : bf16
// gate_out             : fp32
// gate_correction_bias : fp32 [BT, 1] <optional>
// final_hidden_states  : bf16
// moe_use_gate_correction_bias -> gate_correction_bias (False->None)
PD_BUILD_OP(fused_gate_moe)
    .Inputs({"hidden_states",
             "gate_out",
             paddle::Optional("gate_correction_bias"),
             paddle::Vec("gate_up_weights"),
             paddle::Vec("down_weights")})
    .Outputs({"final_hidden_states"})
    .Attrs({"top_k: int",
            "moe_use_gate_correction_bias: bool",
            "norm_topk_prob: bool",
            "permuted_weights: bool",
            "activation: std::string",
            "experts_min: int",
            "experts_max: int"})
    .SetKernelFn(PD_KERNEL(FusedGateMoeForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedGateMoeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedGateMoeInferDtype));

// hidden_states        : bf16 --> quant --> fp8 --> moe
// gate_out             : fp32
// gate_correction_bias : fp32 [BT, 1] <optional>
// gate_up/down_weights : fp8
// final_hidden_states  : internel fp8 --> bf16
// moe_use_gate_correction_bias -> gate_correction_bias (False->None)
// dynamic_scale <-> intermediate_hidden_states_scales (Ture->None)
PD_BUILD_OP(fused_gate_moe_fp8)
    .Inputs({"hidden_states",
             "gate_out",
             paddle::Optional("gate_correction_bias"),
             paddle::Vec("gate_up_weights"),
             paddle::Vec("down_weights"),
             paddle::Optional(paddle::Vec("intermediate_hidden_states_scales")),
             paddle::Vec("gate_up_weights_scales"),
             paddle::Vec("down_weights_scales")})
    .Outputs({"final_hidden_states"})
    .Attrs({"top_k: int",
            "moe_use_gate_correction_bias: bool",
            "norm_topk_prob: bool",
            "permuted_weights: bool",
            "activation: std::string",
            "experts_min: int",
            "experts_max: int"})
    .SetKernelFn(PD_KERNEL(FusedGateMoeFP8Forward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedGateMoeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedGateMoeInferDtype));

// hidden_states        : bf16 --> moe(internel fp8)
// gate_out             : fp32
// gate_correction_bias : fp32 [BT, 1] <optional>
// gate_up/down_weights : fp8
// final_hidden_states  : internel fp8 --> bf16
// moe_use_gate_correction_bias -> gate_correction_bias (False->None)
PD_BUILD_OP(fused_gate_moe_blockwise_fp8)
    .Inputs({"hidden_states",
             "gate_out",
             paddle::Optional("gate_correction_bias"),
             paddle::Vec("gate_up_weights"),
             paddle::Vec("down_weights"),
             paddle::Vec("gate_up_weights_scales"),
             paddle::Vec("down_weights_scales")})
    .Outputs({"final_hidden_states"})
    .Attrs({"top_k: int",
            "moe_use_gate_correction_bias: bool",
            "norm_topk_prob: bool",
            "permuted_weights: bool",
            "activation: std::string",
            "experts_min: int",
            "experts_max: int",
            "block_size: int"})
    .SetKernelFn(PD_KERNEL(FusedGateMoeBlockWiseFP8Forward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedGateMoeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedGateMoeInferDtype));
