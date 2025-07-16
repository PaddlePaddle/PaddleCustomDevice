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
    {{"selu", MoeActivationMode_t::MOE_ACTIVATION_MODE_SELU},
     {"gelu", MoeActivationMode_t::MOE_ACTIVATION_MODE_GELU},
     {"relu", MoeActivationMode_t::MOE_ACTIVATION_MODE_RELU},
     {"silu", MoeActivationMode_t::MOE_ACTIVATION_MODE_SILU}};

struct FusedMoEConfig {
  bool permuted_weights;
  bool fused_gemm;
  bool measurement_mode;
  char activation_mode[32];
  int32_t num_experts;
  int32_t experts_min;
  int32_t experts_max;
  bool dynamic_scale;
  int32_t block_size;
};

std::shared_ptr<ns_MoeKernel::ParamsV3> FillMixtureOfExpertsParams(
    const FusedMoEConfig& config) {
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

class FusedMixtureOfExperts : public HpuFusedOperator {
 public:
  explicit FusedMixtureOfExperts(synDataType dtype)
      : HpuFusedOperator("moe_", false), dtype_(dtype) {}

  template <typename T>
  void AddNodeMoeForward(std::vector<synTensor> inputs,
                         std::vector<synTensor> outputs,
                         std::shared_ptr<ns_MoeKernel::ParamsV3> params) {
    std::string node_name = "moe_fwd";

    std::string guid = guid_ + guid_dtype<T>();

    AddNode_IOP<ns_MoeKernel::ParamsV3>(
        inputs, outputs, *params, guid, node_name);
  }

  template <typename T>
  void AddNode(ConvertTensors* ct, FusedMoEConfig config) {
    auto weights_per_expert = config.fused_gemm ? 2 : 3;
    std::vector<synTensor> inputs;

    int64_t input_count = 3 + config.num_experts * weights_per_expert;
    if (dtype_ == syn_type_fp8_143) {
      auto scales_per_expert = config.fused_gemm ? 2 : 3;
      if (!config.dynamic_scale) scales_per_expert += 1;
      input_count += config.num_experts * scales_per_expert + 1;
    } else if (config.block_size > 0) {
      // blockwise fp8
      auto scales_per_expert = config.fused_gemm ? 2 : 3;
      input_count += config.num_experts * scales_per_expert;
    }

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
void FusedMoEKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& hidden_states,
    const phi::DenseTensor& routing_table,
    const phi::DenseTensor& router_weights,
    const std::vector<phi::DenseTensor>& gate_up_weights,
    const std::vector<phi::DenseTensor>& down_weights,
    const paddle::optional<std::vector<phi::DenseTensor>>& scales,
    const bool permuted_weights,
    const std::string& activation,
    const int experts_min,
    const int experts_max,
    const bool measurement_mode,
    const bool dynamic_scale,
    const int block_size,
    phi::DenseTensor* final_hidden_states,
    phi::DenseTensor* amax_per_expert) {
  ConvertTensors ct;
  ct.Add(hidden_states);
  ct.Add(routing_table);
  ct.Add(router_weights);
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

  std::vector<DIMS> inputs_dims = ct.GetDims();

  ct.Add(final_hidden_states, false);
  if (amax_per_expert) ct.Add(amax_per_expert, false);

  FusedMoEConfig config;
  memset(reinterpret_cast<void*>(&config), 0x00, sizeof(FusedMoEConfig));
  config.permuted_weights = permuted_weights;
  config.fused_gemm = (gate_up_weights.size() == down_weights.size());
  config.measurement_mode = measurement_mode;
  strncpy(config.activation_mode,
          activation.c_str(),
          sizeof(config.activation_mode) - 1);
  config.experts_min = experts_min;
  config.experts_max = experts_max;
  config.num_experts = down_weights.size();
  config.dynamic_scale = dynamic_scale;
  config.block_size = block_size;

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, custom_kernel::FusedMoEConfig>(
      "fused_moe_", inputs_dims, &config);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
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
void CallFusedMoEKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& hidden_states,
    const phi::DenseTensor& routing_table,
    const phi::DenseTensor& router_weights,
    const std::vector<phi::DenseTensor>& gate_up_weights,
    const std::vector<phi::DenseTensor>& down_weights,
    const paddle::optional<std::vector<phi::DenseTensor>>& scales,
    const bool permuted_weights,
    const std::string& activation,
    const int experts_min,
    const int experts_max,
    const bool measurement_mode,
    const bool dynamic_scale,
    const int block_size,
    phi::DenseTensor* final_hidden_states,
    phi::DenseTensor* amax_per_expert) {
  // handle both bfloat16 and blockwise fp8
  if (hidden_states.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::FusedMoEKernel<phi::dtype::bfloat16>(dev_ctx,
                                                        hidden_states,
                                                        routing_table,
                                                        router_weights,
                                                        gate_up_weights,
                                                        down_weights,
                                                        scales,
                                                        permuted_weights,
                                                        activation,
                                                        experts_min,
                                                        experts_max,
                                                        measurement_mode,
                                                        dynamic_scale,
                                                        block_size,
                                                        final_hidden_states,
                                                        amax_per_expert);
  } else if (hidden_states.dtype() == phi::DataType::FLOAT8_E4M3FN) {
    custom_kernel::FusedMoEKernel<phi::dtype::float8_e4m3fn>(
        dev_ctx,
        hidden_states,
        routing_table,
        router_weights,
        gate_up_weights,
        down_weights,
        scales,
        permuted_weights,
        activation,
        experts_min,
        experts_max,
        measurement_mode,
        dynamic_scale,
        block_size,
        final_hidden_states,
        amax_per_expert);
  } else {
    throw std::runtime_error("Unsupported data type for FusedMoEKernel");
  }
}

std::vector<paddle::Tensor> MixtureOfExpertsForward(
    const paddle::Tensor& hidden_states,
    const paddle::Tensor& routing_table,
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
  auto routing_table_tensor =
      static_cast<const phi::DenseTensor*>(routing_table.impl().get());
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
  int64_t num_experts = down_weights.size();

  std::shared_ptr<phi::DenseTensor> final_hidden_states =
      std::make_shared<phi::DenseTensor>();
  final_hidden_states->Resize(hidden_states.dims());
  dev_ctx->Alloc(final_hidden_states.get(), hidden_states.dtype());

  std::shared_ptr<phi::DenseTensor> amax_per_expert =
      std::make_shared<phi::DenseTensor>();
  amax_per_expert->Resize(phi::make_ddim({num_experts}));
  dev_ctx->Alloc(amax_per_expert.get(), paddle::DataType::FLOAT32);

  CallFusedMoEKernel(*dev_ctx,
                     *hidden_states_tensor,
                     *routing_table_tensor,
                     *router_weights_tensor,
                     gate_up_weights_vec,
                     down_weights_vec,
                     paddle::optional<std::vector<phi::DenseTensor>>(),
                     permuted_weights,
                     activation,
                     experts_min,
                     experts_max,
                     measurement_mode,
                     false,
                     -1,
                     final_hidden_states.get(),
                     amax_per_expert.get());

  return {paddle::Tensor(final_hidden_states), paddle::Tensor(amax_per_expert)};
}

std::vector<paddle::Tensor> MixtureOfExpertsFP8Forward(
    const paddle::Tensor& hidden_states,
    const paddle::Tensor& routing_table,
    const paddle::Tensor& router_weights,
    const std::vector<paddle::Tensor>& gate_up_weights,
    const std::vector<paddle::Tensor>& down_weights,
    const paddle::Tensor& hidden_states_scales,
    const paddle::optional<std::vector<paddle::Tensor>>&
        intermediate_hidden_states_scales,
    const std::vector<paddle::Tensor>& gate_up_weights_scales,
    const std::vector<paddle::Tensor>& down_weights_scales,
    const bool permuted_weights,
    const std::string& activation,
    const int experts_min,
    const int experts_max,
    const bool dynamic_scale) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          hidden_states.place()));
  auto hidden_states_tensor =
      static_cast<const phi::DenseTensor*>(hidden_states.impl().get());
  auto routing_table_tensor =
      static_cast<const phi::DenseTensor*>(routing_table.impl().get());
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

  std::vector<phi::DenseTensor> scales_vec;
  scales_vec.push_back(
      *static_cast<const phi::DenseTensor*>(hidden_states_scales.impl().get()));
  if (intermediate_hidden_states_scales) {
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

  // allocate memory on device.
  std::shared_ptr<phi::DenseTensor> final_hidden_states =
      std::make_shared<phi::DenseTensor>();
  final_hidden_states->Resize(hidden_states.dims());
  dev_ctx->Alloc(final_hidden_states.get(), router_weights.dtype());

  CallFusedMoEKernel(*dev_ctx,
                     *hidden_states_tensor,
                     *routing_table_tensor,
                     *router_weights_tensor,
                     gate_up_weights_vec,
                     down_weights_vec,
                     scales_vec,
                     permuted_weights,
                     activation,
                     experts_min,
                     experts_max,
                     false,  // so far not supported on FP8
                     dynamic_scale,
                     -1,
                     final_hidden_states.get(),
                     nullptr);

  return {paddle::Tensor(final_hidden_states)};
}

std::vector<paddle::Tensor> MixtureOfExpertsBlockWiseFP8Forward(
    const paddle::Tensor& hidden_states,
    const paddle::Tensor& routing_table,
    const paddle::Tensor& router_weights,
    const std::vector<paddle::Tensor>& gate_up_weights,
    const std::vector<paddle::Tensor>& down_weights,
    const std::vector<paddle::Tensor>& gate_up_weights_scales,
    const std::vector<paddle::Tensor>& down_weights_scales,
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
  auto routing_table_tensor =
      static_cast<const phi::DenseTensor*>(routing_table.impl().get());
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

  std::vector<phi::DenseTensor> scales_vec;
  for (const auto& t : gate_up_weights_scales) {
    scales_vec.push_back(*static_cast<const phi::DenseTensor*>(t.impl().get()));
  }
  for (const auto& t : down_weights_scales) {
    scales_vec.push_back(*static_cast<const phi::DenseTensor*>(t.impl().get()));
  }

  // allocate memory on device.
  std::shared_ptr<phi::DenseTensor> final_hidden_states =
      std::make_shared<phi::DenseTensor>();
  final_hidden_states->Resize(hidden_states.dims());
  dev_ctx->Alloc(final_hidden_states.get(), router_weights.dtype());

  CallFusedMoEKernel(*dev_ctx,
                     *hidden_states_tensor,
                     *routing_table_tensor,
                     *router_weights_tensor,
                     gate_up_weights_vec,
                     down_weights_vec,
                     scales_vec,
                     permuted_weights,
                     activation,
                     experts_min,
                     experts_max,
                     false,  // so far not supported on FP8
                     false,
                     block_size,
                     final_hidden_states.get(),
                     nullptr);

  return {paddle::Tensor(final_hidden_states)};
}

std::vector<std::vector<int64_t>> MixtureOfExpertsInferShape(
    const std::vector<int64_t>& hidden_states_shape,
    const std::vector<int64_t>& expert_routing_table_shape,
    const std::vector<int64_t>& router_weights_shape,
    const std::vector<int64_t>& gate_up_weights_shape,
    const std::vector<int64_t>& down_weights_shape) {
  int64_t num_experts = router_weights_shape[1];
  return {hidden_states_shape, {num_experts}};
}

std::vector<paddle::DataType> MixtureOfExpertsInferDtype(
    const paddle::DataType& hidden_states_dtype,
    const paddle::DataType& expert_routing_table_dtype,
    const paddle::DataType& router_weights_dtype,
    const paddle::DataType& gate_up_weights_dtype,
    const paddle::DataType& down_weights_dtype) {
  return {router_weights_dtype, paddle::DataType::FLOAT32};
}

std::vector<std::vector<int64_t>> MixtureOfExpertsFP8InferShape(
    const std::vector<int64_t>& hidden_states_shape,
    const std::vector<int64_t>& expert_routing_table_shape,
    const std::vector<int64_t>& router_weights_shape,
    const std::vector<int64_t>& gate_up_weights_shape,
    const std::vector<int64_t>& down_weights_shape) {
  return {hidden_states_shape};
}

std::vector<paddle::DataType> MixtureOfExpertsFP8InferDtype(
    const paddle::DataType& hidden_states_dtype,
    const paddle::DataType& expert_routing_table_dtype,
    const paddle::DataType& router_weights_dtype,
    const paddle::DataType& gate_up_weights_dtype,
    const paddle::DataType& down_weights_dtype) {
  return {router_weights_dtype};
}

PD_BUILD_OP(mixture_of_experts)
    .Inputs({"hidden_states",
             "routing_table",
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

PD_BUILD_OP(mixture_of_experts_fp8)
    .Inputs({"hidden_states",
             "routing_table",
             "router_weights",
             paddle::Vec("gate_up_weights"),
             paddle::Vec("down_weights"),
             "hidden_states_scales",
             paddle::Optional(paddle::Vec("intermediate_hidden_states_scales")),
             paddle::Vec("gate_up_weights_scales"),
             paddle::Vec("down_weights_scales")})
    .Outputs({"final_hidden_states"})
    .Attrs({"permuted_weights: bool",
            "activation: std::string",
            "experts_min: int",
            "experts_max: int",
            "dynamic_scale: bool"})
    .SetKernelFn(PD_KERNEL(MixtureOfExpertsFP8Forward))
    .SetInferShapeFn(PD_INFER_SHAPE(MixtureOfExpertsFP8InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MixtureOfExpertsFP8InferDtype));

PD_BUILD_OP(mixture_of_experts_blockwise_fp8)
    .Inputs({"hidden_states",
             "routing_table",
             "router_weights",
             paddle::Vec("gate_up_weights"),
             paddle::Vec("down_weights"),
             paddle::Vec("gate_up_weights_scales"),
             paddle::Vec("down_weights_scales")})
    .Outputs({"final_hidden_states"})
    .Attrs({"permuted_weights: bool",
            "activation: std::string",
            "experts_min: int",
            "experts_max: int",
            "block_size: int"})
    .SetKernelFn(PD_KERNEL(MixtureOfExpertsBlockWiseFP8Forward))
    .SetInferShapeFn(PD_INFER_SHAPE(MixtureOfExpertsFP8InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MixtureOfExpertsFP8InferDtype));
