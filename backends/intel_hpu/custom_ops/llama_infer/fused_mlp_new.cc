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
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "kernels/hpu_funcs.h"
#include "kernels/hpu_operator.h"
#include "paddle/extension.h"
#include "utils/utils.h"

namespace custom_kernel {

struct FusedMlpParams {
  synSplitParams split_params;
  synGEMMParams gemm_params;

  bool fused_gate_up;
  bool use_fp8;
};

// ZERO_POINT,
// QUANT_MIN,
// QUANT_MAX,
enum TENSOR_IDS_IN {
  HIDDEN_STATES = 0,
  PROJ_WEIGHT,
  DOWN_WEIGHT,
  PROJ_SCALE,
  DOWN_SCALE,
  HID_STE_SCALE,
  INTM_HID_STE_SCALE,
  UP_SCALE = -2,
  UP_WEIGHT = -1
};

#define MIN_FP8_VALUES -240
#define MAX_FP8_VALUES 240

class FusedMlpNew : public HpuFusedOperator {
 public:
  explicit FusedMlpNew(synDataType dtype)
      : HpuFusedOperator("fused_mlp_new_", false), dtype_(dtype) {}
  template <typename T>
  void AddNode(ConvertTensors& ct, FusedMlpParams params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    synTensor hidden_states = createTensorFromCT(&ct, HIDDEN_STATES);
    synTensor proj_weight = createTensorFromCT(&ct, PROJ_WEIGHT);

    std::vector<int64_t> proj_dims = inputs[HIDDEN_STATES].dims;
    if (params.gemm_params.transpose_b == true) {
      proj_dims[inputs[HIDDEN_STATES].dims.size() - 1] =
          inputs[PROJ_WEIGHT].dims[0];
    } else {
      proj_dims[inputs[HIDDEN_STATES].dims.size() - 1] =
          inputs[PROJ_WEIGHT].dims[1];
    }
    synTensor proj_out = createTensorNoPresist("proj_out", dtype_, proj_dims);
    std::vector<synTensor> ffn_ins;
    std::vector<synTensor> ffn_outs = {proj_out};

    synTensor scaled_hidden_states, hidden_states_scale;
    synTensor zero_point, quant_min, quant_max;

    if (params.use_fp8) {
      zero_point = createTensorNoPresist("zero_point", syn_type_float, {1});
      quant_min = createTensorNoPresist("quant_min", syn_type_float, {1});
      quant_max = createTensorNoPresist("quant_max", syn_type_float, {1});
      AddScalarAsTensor<float, T>({zero_point}, 0, guid_ + "zero_point");
      AddScalarAsTensor<float, T>(
          {quant_min}, MIN_FP8_VALUES, guid_ + "quant_min");
      AddScalarAsTensor<float, T>(
          {quant_max}, MAX_FP8_VALUES, guid_ + "quant_max");

      // static quant hidden_states to fp8 with hidden_states_scale
      // move out from FP8Gemm because scaled_hidden_states maybe
      // use twice
      hidden_states_scale = createTensorFromCT(&ct, HID_STE_SCALE);
      std::vector<synTensor> quant_inputs;
      quant_inputs.push_back(hidden_states);
      quant_inputs.push_back(hidden_states_scale);
      // ns_QuantizationPerChannel::ParamsV2 quant_params;
      // quant_params.zero_point = 0;
      // quant_params.quant_min = MIN_FP8_VALUES;
      // quant_params.quant_max = MAX_FP8_VALUES;
      quant_inputs.push_back(zero_point);
      quant_inputs.push_back(quant_min);
      quant_inputs.push_back(quant_max);
      std::vector<synTensor> quant_outputs;
      scaled_hidden_states = createTensorNoPresist(
          "scaled_hidden_states", syn_type_fp8_143, inputs[HIDDEN_STATES].dims);
      quant_outputs.push_back(scaled_hidden_states);

      AddNodeQuantizePerTensor<T>(quant_inputs, quant_outputs, guid_ + "quant");

      auto proj_de_scale = createTensorFromCT(&ct, PROJ_SCALE);
      ffn_ins.push_back(scaled_hidden_states);
      ffn_ins.push_back(proj_weight);
      ffn_ins.push_back(hidden_states_scale);
      ffn_ins.push_back(proj_de_scale);

      AddNodeFusedFP8GemmBF16<T>(
          ffn_ins, ffn_outs, params.gemm_params, guid_ + "proj_gemm");
    } else {
      ffn_ins.push_back(hidden_states);
      ffn_ins.push_back(proj_weight);
      AddNodeGemm(ffn_ins, ffn_outs, params.gemm_params, guid_ + "proj_gemm");
    }

    std::vector<int64_t> swiglu_dims = proj_dims;
    std::vector<synTensor> silu_ins;
    synTensor up_out;

    // Second Gemm or split First Gemm
    if (params.fused_gate_up) {
      // fused weights, split node. bf16 must, fp8 optional
      swiglu_dims[proj_dims.size() - 1] = proj_dims[proj_dims.size() - 1] / 2;
      synTensor gate_out =
          createTensorNoPresist("gate_out", dtype_, swiglu_dims);
      up_out = createTensorNoPresist("up_out", dtype_, swiglu_dims);
      std::vector<synTensor> split_outs = {gate_out, up_out};
      AddNodeSplit(ffn_outs, split_outs, params.split_params, guid_ + "split");
      silu_ins = {gate_out};
    } else if (params.use_fp8) {
      // splitted weights, fp8_gemm node. fp8 branch
      auto up_weight = createTensorFromCT(&ct, inputs.size() + UP_WEIGHT);
      auto up_scale = createTensorFromCT(&ct, inputs.size() + UP_SCALE);
      up_out = createTensorNoPresist("up_out", dtype_, swiglu_dims);
      ffn_ins.clear();
      ffn_ins.push_back(scaled_hidden_states);
      ffn_ins.push_back(up_weight);
      ffn_ins.push_back(hidden_states_scale);
      ffn_ins.push_back(up_scale);
      ffn_outs.clear();
      ffn_outs.push_back(up_out);
      AddNodeFusedFP8GemmBF16<T>(
          ffn_ins, ffn_outs, params.gemm_params, guid_ + "up_gemm");
      silu_ins = {proj_out};
    } else {
      // splitted weights, gemm node. bf16 branch
      auto up_weight = createTensorFromCT(&ct, inputs.size() + UP_WEIGHT);
      up_out = createTensorNoPresist("up_out", dtype_, swiglu_dims);
      ffn_ins.clear();
      ffn_ins.push_back(hidden_states);
      ffn_ins.push_back(up_weight);
      ffn_outs.clear();
      ffn_outs.push_back(up_out);
      AddNodeGemm(ffn_ins, ffn_outs, params.gemm_params, guid_ + "up_gemm");
      silu_ins = {proj_out};
    }

    // silu node
    auto silu_out = createTensorNoPresist("silu_out", dtype_, swiglu_dims);
    std::vector<synTensor> silu_outs = {silu_out};
    AddNodeSilu<T>(silu_ins, silu_outs, guid_ + "silu");

    // multi node
    auto multi_out = createTensorNoPresist("multi_out", dtype_, swiglu_dims);
    std::vector<synTensor> multi_ins = {silu_out, up_out};
    std::vector<synTensor> multi_outs = {multi_out};
    AddNodeMultiply<T>(multi_ins, multi_outs, guid_ + "multi");

    auto down_weight = createTensorFromCT(&ct, DOWN_WEIGHT);
    auto mlp_out = createTensorFromCT(&ct, 0, false);
    std::vector<synTensor> ffn_down_ins = {multi_out, down_weight};
    std::vector<synTensor> ffn_down_outs = {mlp_out};

    // ffn_down gemm node
    if (params.use_fp8) {
      auto intermediate_hidden_states_scale =
          createTensorFromCT(&ct, INTM_HID_STE_SCALE);
      auto down_scale = createTensorFromCT(&ct, DOWN_SCALE);
      ffn_down_ins.push_back(intermediate_hidden_states_scale);
      ffn_down_ins.push_back(down_scale);
      ffn_down_ins.push_back(zero_point);
      ffn_down_ins.push_back(quant_min);
      ffn_down_ins.push_back(quant_max);
      AddNodeFusedFP8GemmBF16<T>(
          ffn_down_ins, ffn_down_outs, params.gemm_params, guid_ + "down_gemm");
    } else {
      AddNodeGemm(
          ffn_down_ins, ffn_down_outs, params.gemm_params, guid_ + "down_gemm");
    }
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void FusedMlpNewKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& hidden_states,
    const phi::DenseTensor& proj_weight,
    const paddle::optional<phi::DenseTensor>& up_weight,
    const phi::DenseTensor& down_weight,
    const paddle::optional<phi::DenseTensor>& hidden_states_scale,
    const paddle::optional<phi::DenseTensor>& proj_scale,
    const paddle::optional<phi::DenseTensor>& up_scale,
    const paddle::optional<phi::DenseTensor>& intermediate_hidden_states_scale,
    const paddle::optional<phi::DenseTensor>& down_scale,
    // const paddle::optional<phi::DenseTensor>& zero_point,
    // const paddle::optional<phi::DenseTensor>& quant_min,
    // const paddle::optional<phi::DenseTensor>& quant_max,
    const bool permuted_weights,
    phi::DenseTensor* out) {
  // allocate memory on device.
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  FusedMlpParams params;
  memset(reinterpret_cast<void*>(&params), 0x00, sizeof(FusedMlpParams));

  params.gemm_params.transpose_a = false;
  params.gemm_params.transpose_b = permuted_weights;

  params.fused_gate_up = true;

  params.use_fp8 = (proj_weight.dtype() == phi::DataType::FLOAT8_E4M3FN);

  ConvertTensors ct;
  ct.Add(hidden_states);
  ct.Add(proj_weight);
  ct.Add(down_weight);

  if (params.use_fp8) {
    ct.Add(proj_scale.get());
    ct.Add(down_scale.get());
    ct.Add(hidden_states_scale.get());
    ct.Add(intermediate_hidden_states_scale.get());
    // ct.Add(zero_point.get());
    // ct.Add(quant_min.get());
    // ct.Add(quant_max.get());
    if (up_scale) {
      ct.Add(up_scale.get());
    }
  }
  if (up_weight) {
    ct.Add(up_weight.get());
    params.fused_gate_up = false;
  }

  ct.Add(*out, false);

  std::vector<DIMS> inputs_dims = ct.GetDims();

  OpCacheOperator op_info;
  std::string recipe_name =
      params.use_fp8 ? "FusedFP8MlpNewKernel" : "FusedMlpNewKernel";
  op_info.prepareOpInfo<T, FusedMlpParams>(recipe_name, inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FusedMlpNew op(op_info.datatype_);
    op.AddNode<T>(ct, params);
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
void CallFusedMlpNewKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& hidden_states,
    const phi::DenseTensor& proj_weight,
    const paddle::optional<phi::DenseTensor>& up_weight,
    const phi::DenseTensor& down_weight,
    const paddle::optional<phi::DenseTensor>& hidden_states_scale,
    const paddle::optional<phi::DenseTensor>& proj_scale,
    const paddle::optional<phi::DenseTensor>& up_scale,
    const paddle::optional<phi::DenseTensor>& intermediate_hidden_states_scale,
    const paddle::optional<phi::DenseTensor>& down_scale,
    // const paddle::optional<phi::DenseTensor>& zero_point,
    // const paddle::optional<phi::DenseTensor>& quant_min,
    // const paddle::optional<phi::DenseTensor>& quant_max,
    const bool permuted_weights,
    phi::DenseTensor* out) {
  if (hidden_states.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::FusedMlpNewKernel<phi::dtype::bfloat16>(
        dev_ctx,
        hidden_states,
        proj_weight,
        up_weight,
        down_weight,
        hidden_states_scale,
        proj_scale,
        up_scale,
        intermediate_hidden_states_scale,
        down_scale,
        // zero_point,
        // quant_min,
        // quant_max,
        permuted_weights,
        out);
  } else {
    throw std::runtime_error("Unsupported data type for FusedRmsMlpKernel");
  }
}

std::vector<paddle::Tensor> FusedMlpNewForward(
    const paddle::Tensor& hidden_states,
    const paddle::Tensor& proj_weight,
    const paddle::optional<paddle::Tensor>& up_weight,
    const paddle::Tensor& down_weight) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          hidden_states.place()));

  auto hidden_states_tensor =
      static_cast<const phi::DenseTensor*>(hidden_states.impl().get());
  auto proj_weight_tensor =
      static_cast<const phi::DenseTensor*>(proj_weight.impl().get());
  auto up_weight_tensor = paddle::optional<phi::DenseTensor>();
  if (up_weight) {
    auto up_weight_dt = static_cast<phi::DenseTensor*>(up_weight->impl().get());
    up_weight_tensor = paddle::optional<phi::DenseTensor>(*up_weight_dt);
  }
  auto down_weight_tensor =
      static_cast<const phi::DenseTensor*>(down_weight.impl().get());
  auto out_tensor = std::make_shared<phi::DenseTensor>();

  out_tensor->Resize(hidden_states_tensor->dims());

  CallFusedMlpNewKernel(*dev_ctx,
                        *hidden_states_tensor,
                        *proj_weight_tensor,
                        up_weight_tensor,
                        *down_weight_tensor,
                        paddle::optional<phi::DenseTensor>(),
                        paddle::optional<phi::DenseTensor>(),
                        paddle::optional<phi::DenseTensor>(),
                        paddle::optional<phi::DenseTensor>(),
                        paddle::optional<phi::DenseTensor>(),
                        // paddle::optional<phi::DenseTensor>(),
                        // paddle::optional<phi::DenseTensor>(),
                        // paddle::optional<phi::DenseTensor>(),
                        false,  // permuted_weights,
                        out_tensor.get());

  paddle::Tensor out(out_tensor);

  return {out};
}

std::vector<paddle::Tensor> FusedFP8MlpNewForward(
    const paddle::Tensor& hidden_states,
    const paddle::Tensor& proj_weight,
    const paddle::optional<paddle::Tensor>& up_weight,
    const paddle::Tensor& down_weight,
    const paddle::Tensor& hidden_states_scale,
    const paddle::Tensor& proj_scale,
    const paddle::optional<paddle::Tensor>& up_scale,
    const paddle::Tensor& intermediate_hidden_states_scale,
    const paddle::Tensor& down_scale,
    const bool permuted_weights) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          hidden_states.place()));

  auto hidden_states_tensor =
      static_cast<const phi::DenseTensor*>(hidden_states.impl().get());
  auto proj_weight_tensor =
      static_cast<const phi::DenseTensor*>(proj_weight.impl().get());
  auto up_weight_tensor = paddle::optional<phi::DenseTensor>();
  if (up_weight) {
    auto up_weight_dt = static_cast<phi::DenseTensor*>(up_weight->impl().get());
    up_weight_tensor = paddle::optional<phi::DenseTensor>(*up_weight_dt);
  }
  auto down_weight_tensor =
      static_cast<const phi::DenseTensor*>(down_weight.impl().get());
  auto hidden_states_scale_tensor =
      static_cast<const phi::DenseTensor*>(hidden_states_scale.impl().get());
  auto proj_scale_tensor =
      static_cast<const phi::DenseTensor*>(proj_scale.impl().get());
  auto up_scale_tensor = paddle::optional<phi::DenseTensor>();
  if (up_scale) {
    auto up_scale_dt = static_cast<phi::DenseTensor*>(up_scale->impl().get());
    up_scale_tensor = paddle::optional<phi::DenseTensor>(*up_scale_dt);
  }
  auto intermediate_hidden_states_scale_tensor =
      static_cast<const phi::DenseTensor*>(
          intermediate_hidden_states_scale.impl().get());
  auto down_scale_tensor =
      static_cast<const phi::DenseTensor*>(down_scale.impl().get());
  auto out_tensor = std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(hidden_states_tensor->dims());

  /*
  auto zero_point_cpu = paddle::full(
      {1}, 0, paddle::DataType::INT32, paddle::CPUPlace());
  auto quant_min_cpu = paddle::full(
      {1}, MIN_FP8_VALUES, paddle::DataType::INT32, paddle::CPUPlace());
  auto quant_max_cpu = paddle::full(
      {1}, MAX_FP8_VALUES, paddle::DataType::INT32, paddle::CPUPlace());

  auto zero_point = std::make_shared<phi::DenseTensor>();
  zero_point->Resize(phi::make_ddim({1}));
  dev_ctx->Alloc(zero_point.get(), phi::DataType::INT32);
  custom_kernel::copy_tensor_wrapper(
      dev_ctx, zero_point_cpu, paddle::Tensor(zero_point));
  auto zero_point_tensor = paddle::optional<phi::DenseTensor>(*zero_point);

  auto quant_min = std::make_shared<phi::DenseTensor>();
  quant_min->Resize(phi::make_ddim({1}));
  dev_ctx->Alloc(quant_min.get(), phi::DataType::INT32);
  custom_kernel::copy_tensor_wrapper(
      dev_ctx, quant_min_cpu, paddle::Tensor(quant_min));
  auto quant_min_tensor = paddle::optional<phi::DenseTensor>(*quant_min);

  auto quant_max = std::make_shared<phi::DenseTensor>();
  quant_max->Resize(phi::make_ddim({1}));
  dev_ctx->Alloc(quant_max.get(), phi::DataType::INT32);
  custom_kernel::copy_tensor_wrapper(
      dev_ctx, quant_max_cpu, paddle::Tensor(quant_max));
  auto quant_max_tensor = paddle::optional<phi::DenseTensor>(*quant_max);
  */

  CallFusedMlpNewKernel(*dev_ctx,
                        *hidden_states_tensor,
                        *proj_weight_tensor,
                        up_weight_tensor,
                        *down_weight_tensor,
                        *hidden_states_scale_tensor,
                        *proj_scale_tensor,
                        up_scale_tensor,
                        *intermediate_hidden_states_scale_tensor,
                        *down_scale_tensor,
                        // zero_point_tensor,
                        // quant_min_tensor,
                        // quant_max_tensor,
                        permuted_weights,
                        out_tensor.get());

  paddle::Tensor out(out_tensor);

  return {out};
}

std::vector<std::vector<int64_t>> FusedMlpNewInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& proj_weight_shape,
    const paddle::optional<std::vector<int64_t>>& up_weight_shape,
    const std::vector<int64_t>& down_weight_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> FusedMlpNewInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::DataType& proj_weight_dtype,
    const paddle::optional<paddle::DataType>& up_weight_dtype,
    const paddle::DataType& down_weight_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(fused_mlp_new)
    .Inputs({"hidden_states",
             "proj_weight",
             paddle::Optional("up_weight"),
             "down_weight"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(FusedMlpNewForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedMlpNewInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedMlpNewInferDtype));

PD_BUILD_OP(fused_fp8_mlp_new)
    .Inputs({"hidden_states",
             "proj_weight",
             paddle::Optional("up_weight"),
             "down_weight",
             "hidden_states_scale",
             "proj_scale",
             paddle::Optional("up_scale"),
             "intermediate_hidden_states_scales",
             "down_scale"})
    .Outputs({"out"})
    .Attrs({"permuted_weights: bool"})
    .SetKernelFn(PD_KERNEL(FusedFP8MlpNewForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedMlpNewInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedMlpNewInferDtype));
