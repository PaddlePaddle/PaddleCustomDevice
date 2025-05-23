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

static const std::map<std::string_view, MoeActivationMode_t> activationModeMap =
    {{"gelu", MoeActivationMode_t::MOE_ACTIVATION_MODE_GELU},
     {"relu", MoeActivationMode_t::MOE_ACTIVATION_MODE_RELU},
     {"silu", MoeActivationMode_t::MOE_ACTIVATION_MODE_SILU}};

struct FusedMoEConfig {
  bool permuted_weights;
  bool fused_gemm;
  bool measurement_mode;
  std::string_view activation_mode;
  int32_t num_experts;
  int32_t experts_min;
  int32_t experts_max;
  bool dynamic_scale;
  bool blockwise_quantization;
  int32_t block_size;
};

std::shared_ptr<ns_MoeKernel::ParamsV2> FillMixtureOfExpertsParams(
    const FusedMoEConfig& config) {
  auto moe_params = std::make_shared<ns_MoeKernel::ParamsV2>();
  memset(reinterpret_cast<void*>(moe_params.get()),
         0x00,
         sizeof(ns_MoeKernel::ParamsV2));

  auto activationIterator = activationModeMap.find(config.activation_mode);
  moe_params->experts.activation = activationIterator->second;

  moe_params->router.experts_min = config.experts_min;
  moe_params->router.experts_max = config.experts_max;

  moe_params->flags =
      config.permuted_weights ? MoeFlags_t::MOE_FLAGS_PERMUTED_WEIGHTS : 0;
  moe_params->flags |=
      (config.fused_gemm ? MoeFlags_t::MOE_FLAGS_FUSED_GEMM : 0);
  moe_params->flags |=
      (config.measurement_mode ? MoeFlags_t::MOE_FLAGS_CALC_AMAX : 0);

  return moe_params;
}

class FusedMixtureOfExperts : public HpuFusedOperator {
 public:
  explicit FusedMixtureOfExperts(synDataType dtype)
      : HpuFusedOperator("moe_", false), dtype_(dtype) {}

  template <typename T>
  void AddNodeMoeForward(std::vector<synTensor> inputs,
                         std::vector<synTensor> outputs,
                         std::shared_ptr<ns_MoeKernel::ParamsV2> params) {
    std::string node_name = "moe_fwd";

    std::string guid = guid_ + guid_dtype<T>();

    AddNode_IOP<ns_MoeKernel::ParamsV2>(
        inputs, outputs, *params, guid, node_name);
  }

  template <typename T>
  void AddNode(ConvertTensors* ct, FusedMoEConfig config) {
    auto weights_per_expert = config.fused_gemm ? 2 : 3;
    std::vector<synTensor> inputs;

    int64_t input_count = 3 + config.num_experts * weights_per_expert;
    for (int64_t i = 0; i < input_count; i++) {
      inputs.push_back(createTensorFromCT(ct, i));
    }

    const bool measurement_mode = config.measurement_mode;
    std::vector<synTensor> outputs;
    if (measurement_mode) {
      for (size_t i = 0; i < 2; i++) {
        outputs.push_back(createTensorFromCT(ct, i, false));
      }
    } else {
      outputs.push_back(createTensorFromCT(ct, 0, false));
    }

    auto params = FillMixtureOfExpertsParams(config);
    AddNodeMoeForward<T>(inputs, outputs, params);
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void FusedMoEKernel(const Context& dev_ctx,
                    const phi::DenseTensor& hidden_states,
                    const phi::DenseTensor& expert_routing_table,
                    const phi::DenseTensor& router_weights,
                    const std::vector<phi::DenseTensor>& gate_up_weights,
                    const std::vector<phi::DenseTensor>& down_weights,
                    const bool permuted_weights,
                    const std::string& activation,
                    const int experts_min,
                    const int experts_max,
                    const bool measurement_mode,
                    phi::DenseTensor* final_hidden_states,
                    phi::DenseTensor* amax_per_expert) {
  ConvertTensors ct;
  ct.Add(hidden_states);
  ct.Add(expert_routing_table);
  ct.Add(router_weights);
  for (const auto& t : gate_up_weights) {
    ct.Add(t);
  }
  for (const auto& t : down_weights) {
    ct.Add(t);
  }
  std::vector<DIMS> inputs_dims = ct.GetDims();

  ct.Add(final_hidden_states, false);
  ct.Add(amax_per_expert, false);

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, nullptr_t>("fused_moe_", inputs_dims, nullptr);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FusedMoEConfig config;
    memset(reinterpret_cast<void*>(&config), 0x00, sizeof(FusedMoEConfig));

    config.permuted_weights = permuted_weights;
    config.fused_gemm = (gate_up_weights.size() == down_weights.size());
    config.measurement_mode = measurement_mode;
    config.activation_mode = activation;
    config.experts_min = experts_min;
    config.experts_max = experts_max;
    config.num_experts = router_weights.dims()[1];

    FusedMixtureOfExperts op(op_info.datatype_);
    op.AddNode<T>(&ct, config);
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
void CallFusedMoEKernel(const Context& dev_ctx,
                        const phi::DenseTensor& hidden_states,
                        const phi::DenseTensor& expert_routing_table,
                        const phi::DenseTensor& router_weights,
                        const std::vector<phi::DenseTensor>& gate_up_weights,
                        const std::vector<phi::DenseTensor>& down_weights,
                        const bool permuted_weights,
                        const std::string& activation,
                        const int experts_min,
                        const int experts_max,
                        const bool measurement_mode,
                        phi::DenseTensor* final_hidden_states,
                        phi::DenseTensor* amax_per_expert) {
  if (hidden_states.dtype() == phi::DataType::FLOAT16) {
    custom_kernel::FusedMoEKernel<phi::dtype::float16>(dev_ctx,
                                                       hidden_states,
                                                       expert_routing_table,
                                                       router_weights,
                                                       gate_up_weights,
                                                       down_weights,
                                                       permuted_weights,
                                                       activation,
                                                       experts_min,
                                                       experts_max,
                                                       measurement_mode,
                                                       final_hidden_states,
                                                       amax_per_expert);
  } else if (hidden_states.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::FusedMoEKernel<phi::dtype::bfloat16>(dev_ctx,
                                                        hidden_states,
                                                        expert_routing_table,
                                                        router_weights,
                                                        gate_up_weights,
                                                        down_weights,
                                                        permuted_weights,
                                                        activation,
                                                        experts_min,
                                                        experts_max,
                                                        measurement_mode,
                                                        final_hidden_states,
                                                        amax_per_expert);
  } else {
    throw std::runtime_error("Unsupported data type for FusedMoEKernel");
  }
}

std::vector<paddle::Tensor> MixtureOfExpertsForward(
    const paddle::Tensor& hidden_states,
    const paddle::Tensor& expert_routing_table,
    const paddle::Tensor& router_weights,
    const std::vector<paddle::Tensor>& gate_up_weights,
    const std::vector<paddle::Tensor>& down_weights,
    const bool permuted_weights,
    const std::string& activation,
    const int experts_min,
    const int experts_max,
    const bool measurement_mode) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          hidden_states.place()));
  auto hidden_states_tensor =
      static_cast<const phi::DenseTensor*>(hidden_states.impl().get());
  auto expert_routing_table_tensor =
      static_cast<const phi::DenseTensor*>(expert_routing_table.impl().get());
  auto router_weights_tensor =
      static_cast<const phi::DenseTensor*>(router_weights.impl().get());

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

  // allocate memory on device.
  int64_t num_tokens = hidden_states.dims()[0];
  int64_t hidden_dims = hidden_states.dims()[1];
  int64_t num_experts = router_weights.dims()[1];

  std::shared_ptr<phi::DenseTensor> final_hidden_states =
      std::make_shared<phi::DenseTensor>();
  final_hidden_states->Resize(phi::make_ddim({num_tokens, hidden_dims}));
  dev_ctx->Alloc(final_hidden_states.get(), hidden_states.dtype());

  std::shared_ptr<phi::DenseTensor> amax_per_expert =
      std::make_shared<phi::DenseTensor>();
  amax_per_expert->Resize(phi::make_ddim({num_experts}));
  dev_ctx->Alloc(amax_per_expert.get(), paddle::DataType::FLOAT32);

  CallFusedMoEKernel(*dev_ctx,
                     *hidden_states_tensor,
                     *expert_routing_table_tensor,
                     *router_weights_tensor,
                     gate_up_weights_vec,
                     down_weights_vec,
                     permuted_weights,
                     activation,
                     experts_min,
                     experts_max,
                     measurement_mode,
                     final_hidden_states.get(),
                     amax_per_expert.get());

  return {paddle::Tensor(final_hidden_states), paddle::Tensor(amax_per_expert)};
}

std::vector<std::vector<int64_t>> MixtureOfExpertsInferShape(
    const std::vector<int64_t>& hidden_states_shape,
    const std::vector<int64_t>& expert_routing_table_shape,
    const std::vector<int64_t>& router_weights_shape,
    const std::vector<int64_t>& gate_up_weights_shape,
    const std::vector<int64_t>& down_weights_shape) {
  int64_t num_tokens = hidden_states_shape[0];
  int64_t hidden_dims = hidden_states_shape[1];
  int64_t num_experts = router_weights_shape[1];
  return {{num_tokens, hidden_dims}, {num_experts}};
}

std::vector<paddle::DataType> MixtureOfExpertsInferDtype(
    const paddle::DataType& hidden_states_dtype,
    const paddle::DataType& expert_routing_table_dtype,
    const paddle::DataType& router_weights_dtype,
    const paddle::DataType& gate_up_weights_dtype,
    const paddle::DataType& down_weights_dtype) {
  return {hidden_states_dtype, paddle::DataType::FLOAT32};
}

PD_BUILD_OP(mixture_of_experts)
    .Inputs({"hidden_states",
             "expert_routing_table",
             "router_weights",
             paddle::Vec("gate_up_weights"),
             paddle::Vec("down_weights")})
    .Outputs({"final_hidden_states", paddle::Optional("amax_per_expert")})
    .Attrs({"permuted_weights: bool",
            "activation: std::string",
            "experts_min: int",
            "experts_max: int",
            "measurement_mode: bool"})
    .SetKernelFn(PD_KERNEL(MixtureOfExpertsForward))
    .SetInferShapeFn(PD_INFER_SHAPE(MixtureOfExpertsInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MixtureOfExpertsInferDtype));
