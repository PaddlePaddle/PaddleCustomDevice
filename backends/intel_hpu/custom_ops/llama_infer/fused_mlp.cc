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

class FusedSplitMlp : public HpuOperator {
 public:
  explicit FusedSplitMlp(synDataType dtype)
      : HpuOperator("fused_mlp_fwd", false), dtype_(dtype) {}

  void AddNode(ConvertTensors& ct) {
    auto ins = ct.GetTensors();
    auto outs = ct.GetTensors(false);

    synTensor hidden_states = createTensor(
        ins[0].dims.size(), dtype_, ins[0].dims, true, ins[0].name);
    synTensor gate_weight = createTensor(
        ins[1].dims.size(), dtype_, ins[1].dims, true, ins[1].name);
    synTensor up_weight = createTensor(
        ins[2].dims.size(), dtype_, ins[2].dims, true, ins[2].name);
    synTensor down_weight = createTensor(
        ins[3].dims.size(), dtype_, ins[3].dims, true, ins[3].name);

    std::vector<int64_t> proj_dims = {
        ins[0].dims[0], ins[0].dims[1], ins[1].dims[1]};
    synTensor gate_out =
        createTensor(proj_dims.size(), dtype_, proj_dims, false, "gate_out");
    synTensor up_out =
        createTensor(proj_dims.size(), dtype_, proj_dims, false, "up_out");
    synTensor silu_out =
        createTensor(proj_dims.size(), dtype_, proj_dims, false, "silu_out");
    synTensor mul_out =
        createTensor(proj_dims.size(), dtype_, proj_dims, false, "mul_out");

    synTensor mlp_out = createTensor(
        outs[0].dims.size(), dtype_, outs[0].dims, true, outs[0].name);

    std::vector<synTensor> gate_inputs;
    gate_inputs.push_back(hidden_states);
    gate_inputs.push_back(gate_weight);
    std::vector<synTensor> gate_outputs;
    gate_outputs.push_back(gate_out);

    std::vector<synTensor> up_inputs;
    up_inputs.push_back(hidden_states);
    up_inputs.push_back(up_weight);
    std::vector<synTensor> up_outputs;
    up_outputs.push_back(up_out);

    std::vector<synTensor> silu_inputs;
    silu_inputs.push_back(gate_out);
    std::vector<synTensor> silu_outputs;
    silu_outputs.push_back(silu_out);

    std::vector<synTensor> mul_inputs;
    mul_inputs.push_back(silu_out);
    mul_inputs.push_back(up_out);
    std::vector<synTensor> mul_outputs;
    mul_outputs.push_back(mul_out);

    std::vector<synTensor> down_inputs;
    down_inputs.push_back(mul_out);
    down_inputs.push_back(down_weight);
    std::vector<synTensor> down_outputs;
    down_outputs.push_back(mlp_out);

    std::string matmul = "gemm";
    std::string silu = "silu_fwd_";
    std::string mul = "mult_fwd_";
    if (dtype_ == syn_type_fp16) {
      silu = silu + "f16";
      mul = mul + "f16";
    } else if (dtype_ == syn_type_bf16) {
      silu = silu + "bf16";
      mul = mul + "bf16";
    } else if (dtype_ == syn_type_single) {
      silu = silu + "f32";
      mul = mul + "f32";
    }

    std::string silu_name = guid_ + "_silu";
    std::string mul_name = guid_ + "_mul";
    std::string gate_name = guid_ + "_gate_proj";
    std::string up_name = guid_ + "_up_proj";
    std::string down_name = guid_ + "_down_proj";

    synStatus status = synNodeCreate(graphHandle_,
                                     gate_inputs.data(),
                                     gate_outputs.data(),
                                     gate_inputs.size(),
                                     gate_outputs.size(),
                                     nullptr,
                                     0,
                                     matmul.c_str(),
                                     gate_name.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    status = synNodeCreate(graphHandle_,
                           up_inputs.data(),
                           up_outputs.data(),
                           up_inputs.size(),
                           up_outputs.size(),
                           nullptr,
                           0,
                           matmul.c_str(),
                           up_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    status = synNodeCreate(graphHandle_,
                           silu_inputs.data(),
                           silu_outputs.data(),
                           silu_inputs.size(),
                           silu_outputs.size(),
                           nullptr,
                           0,
                           silu.c_str(),
                           silu_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    status = synNodeCreate(graphHandle_,
                           mul_inputs.data(),
                           mul_outputs.data(),
                           mul_inputs.size(),
                           mul_outputs.size(),
                           nullptr,
                           0,
                           mul.c_str(),
                           mul_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    status = synNodeCreate(graphHandle_,
                           down_inputs.data(),
                           down_outputs.data(),
                           down_inputs.size(),
                           down_outputs.size(),
                           nullptr,
                           0,
                           matmul.c_str(),
                           down_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }

 protected:
  synDataType dtype_;
};

class FusedGateUpMlp : public HpuOperator {
 public:
  explicit FusedGateUpMlp(synDataType dtype)
      : HpuOperator("fused_gate_up_mlp_fwd", false), dtype_(dtype) {}

  void AddNode(ConvertTensors& ct, synSplitParams params) {
    auto ins = ct.GetTensors();
    auto outs = ct.GetTensors(false);

    synTensor hidden_states = createTensor(
        ins[0].dims.size(), dtype_, ins[0].dims, true, ins[0].name);
    synTensor proj_weight = createTensor(
        ins[1].dims.size(), dtype_, ins[1].dims, true, ins[1].name);
    std::vector<int64_t> proj_dims = ins[0].dims;
    proj_dims[ins[0].dims.size() - 1] = ins[1].dims[1];
    synTensor proj_out =
        createTensor(proj_dims.size(), dtype_, proj_dims, false, "proj_out");

    std::vector<int64_t> split_out_dims = proj_dims;
    split_out_dims[proj_dims.size() - 1] = proj_dims[proj_dims.size() - 1] / 2;

    synTensor gate_out = createTensor(
        split_out_dims.size(), dtype_, split_out_dims, false, "gate_out");
    synTensor up_out = createTensor(
        split_out_dims.size(), dtype_, split_out_dims, false, "up_out");

    synTensor down_weight = createTensor(
        ins[2].dims.size(), dtype_, ins[2].dims, true, ins[2].name);

    synTensor silu_out = createTensor(
        split_out_dims.size(), dtype_, split_out_dims, false, "silu_out");
    synTensor mul_out = createTensor(
        split_out_dims.size(), dtype_, split_out_dims, false, "mul_out");

    synTensor mlp_out = createTensor(
        outs[0].dims.size(), dtype_, outs[0].dims, true, outs[0].name);

    std::vector<synTensor> proj_inputs;
    proj_inputs.push_back(hidden_states);
    proj_inputs.push_back(proj_weight);
    std::vector<synTensor> proj_outputs;
    proj_outputs.push_back(proj_out);

    std::vector<synTensor> split_inputs;
    split_inputs.push_back(proj_out);
    std::vector<synTensor> split_outputs;
    split_outputs.push_back(gate_out);
    split_outputs.push_back(up_out);

    std::vector<synTensor> silu_inputs;
    silu_inputs.push_back(gate_out);
    std::vector<synTensor> silu_outputs;
    silu_outputs.push_back(silu_out);

    std::vector<synTensor> mul_inputs;
    mul_inputs.push_back(silu_out);
    mul_inputs.push_back(up_out);
    std::vector<synTensor> mul_outputs;
    mul_outputs.push_back(mul_out);

    std::vector<synTensor> down_inputs;
    down_inputs.push_back(mul_out);
    down_inputs.push_back(down_weight);
    std::vector<synTensor> down_outputs;
    down_outputs.push_back(mlp_out);

    std::string split = "split";
    std::string matmul = "gemm";
    std::string silu = "silu_fwd_";
    std::string mul = "mult_fwd_";
    if (dtype_ == syn_type_fp16) {
      silu = silu + "f16";
      mul = mul + "f16";
    } else if (dtype_ == syn_type_bf16) {
      silu = silu + "bf16";
      mul = mul + "bf16";
    } else if (dtype_ == syn_type_single) {
      silu = silu + "f32";
      mul = mul + "f32";
    }

    std::string proj_name = guid_ + "_proj";
    std::string split_name = guid_ + "_split_proj";
    std::string silu_name = guid_ + "_silu";
    std::string mul_name = guid_ + "_mul";
    std::string down_name = guid_ + "_down_proj";

    synStatus status = synNodeCreate(graphHandle_,
                                     proj_inputs.data(),
                                     proj_outputs.data(),
                                     proj_inputs.size(),
                                     proj_outputs.size(),
                                     nullptr,
                                     0,
                                     matmul.c_str(),
                                     proj_name.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    status = synNodeCreate(graphHandle_,
                           split_inputs.data(),
                           split_outputs.data(),
                           split_inputs.size(),
                           split_outputs.size(),
                           &params,
                           sizeof(params),
                           split.c_str(),
                           split_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    status = synNodeCreate(graphHandle_,
                           silu_inputs.data(),
                           silu_outputs.data(),
                           silu_inputs.size(),
                           silu_outputs.size(),
                           nullptr,
                           0,
                           silu.c_str(),
                           silu_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    status = synNodeCreate(graphHandle_,
                           mul_inputs.data(),
                           mul_outputs.data(),
                           mul_inputs.size(),
                           mul_outputs.size(),
                           nullptr,
                           0,
                           mul.c_str(),
                           mul_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    status = synNodeCreate(graphHandle_,
                           down_inputs.data(),
                           down_outputs.data(),
                           down_inputs.size(),
                           down_outputs.size(),
                           nullptr,
                           0,
                           matmul.c_str(),
                           down_name.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }

 protected:
  synDataType dtype_;
};

struct FusedMlpParams {
  synSplitParams split_params;
  synGEMMParams gemm_params;

  bool fused_gate_up;
  bool use_fp8;
};

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

class FusedMlp : public HpuFusedOperator {
 public:
  explicit FusedMlp(synDataType dtype)
      : HpuFusedOperator("fused_mlp_", false), dtype_(dtype) {}
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

    synTensor scaled_hidden_states;
    synTensor hidden_states_de_scale;
    if (params.use_fp8) {
      // static quant hidden_states to fp8 with hidden_states_scale
      // move out from AddNodeFusedFP8Gemm because scaled_hidden_states maybe
      // use twice
      std::vector<synTensor> quant_inputs;
      synTensor hidden_states_scale = createTensorFromCT(&ct, HID_STE_SCALE);
      quant_inputs.push_back(hidden_states);
      quant_inputs.push_back(hidden_states_scale);
      std::vector<synTensor> quant_outputs;
      scaled_hidden_states = createTensorNoPresist(
          "scaled_hidden_states", syn_type_fp8_143, inputs[HIDDEN_STATES].dims);
      quant_outputs.push_back(scaled_hidden_states);
      ns_CastKernel::Params cast_to_fp8_params;
      cast_to_fp8_params.round_mode = CAST_ROUND_HALF_NE;
      AddNodeConvertToFP8<T>(
          quant_inputs, quant_outputs, cast_to_fp8_params, guid_ + "cast");
      ffn_ins.push_back(scaled_hidden_states);
      ffn_ins.push_back(proj_weight);

      // 1/hidden_states_scale for gemm d_scale
      hidden_states_de_scale = createTensorNoPresist(
          "hidden_states_de_scale", inputs[HIDDEN_STATES].type, {1});
      synTensor one =
          createTensorNoPresist("one", inputs[HIDDEN_STATES].type, {1});
      ns_ConstantKernel::Params const_params;
      const_params.constant.f = 1.0f;
      std::vector<synTensor> one_tensor = {one};
      AddNodeFull<T>(one_tensor, const_params, guid_ + "full_one");
      std::vector<synTensor> div_inputs;
      div_inputs.push_back(one);
      div_inputs.push_back(hidden_states_scale);
      std::vector<synTensor> div_outputs = {hidden_states_de_scale};
      AddNodeDivide<T>(div_inputs, div_outputs, guid_ + "reciprocal");

      ffn_ins.push_back(hidden_states_de_scale);
      auto proj_de_scale = createTensorFromCT(&ct, PROJ_SCALE);
      ffn_ins.push_back(proj_de_scale);

      AddNodeFusedFP8Gemm<T>(
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
      ffn_ins.push_back(hidden_states_de_scale);
      ffn_ins.push_back(up_scale);
      ffn_outs.clear();
      ffn_outs.push_back(up_out);
      AddNodeFusedFP8Gemm<T>(
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
      AddNodeFusedFP8Gemm<T>(
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
void FusedMlpKernel(
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
      params.use_fp8 ? "FusedFP8MlpKernel" : "FusedMlpKernel";
  op_info.prepareOpInfo<T, FusedMlpParams>(recipe_name, inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FusedMlp op(op_info.datatype_);
    op.AddNode<T>(ct, params);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

template <typename T, typename Context>
void FusedSplitMlpKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& gate_weight,
                         const phi::DenseTensor& up_weight,
                         const phi::DenseTensor& down_weight,
                         phi::DenseTensor* out) {
  // allocate memory on device.
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(gate_weight);
  ct.Add(up_weight);
  ct.Add(down_weight);
  ct.Add(*out, false);
  std::vector<DIMS> inputs_dims = ct.GetDims();

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, nullptr_t>("FusedMlpKernel", inputs_dims, nullptr);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FusedSplitMlp op(op_info.datatype_);
    op.AddNode(ct);
    op.Compile();
    op_info.setOp(op);

    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

template <typename T, typename Context>
void FusedGateUpMlpKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& proj_weight,
                          const phi::DenseTensor& down_weight,
                          phi::DenseTensor* out) {
  // allocate memory on device.
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  const phi::Scalar axis_scalar = proj_weight.dims().size() - 1;
  int64_t axis = axis_scalar.to<int64_t>();
  if (axis < 0) {
    axis = proj_weight.dims().size() + axis;
  }
  synSplitParams params = {{0}};
  params.axis = proj_weight.dims().size() - 1 - axis;

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(proj_weight);
  ct.Add(down_weight);
  ct.Add(*out, false);
  std::vector<DIMS> inputs_dims = ct.GetDims();

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, synSplitParams>(
      "FusedGateUpMlpKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FusedGateUpMlp op(op_info.datatype_);
    op.AddNode(ct, params);
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
void CallFusedSplitMlpKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::DenseTensor& gate_weight,
                             const phi::DenseTensor& up_weight,
                             const phi::DenseTensor& down_weight,
                             phi::DenseTensor* out) {
  if (x.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::FusedSplitMlpKernel<phi::dtype::bfloat16>(
        dev_ctx, x, gate_weight, up_weight, down_weight, out);
  } else {
    throw std::runtime_error("Unsupported data type for FusedMlpKernel");
  }
}

template <typename Context>
void CallFusedGateUpMlpKernel(const Context& dev_ctx,
                              const phi::DenseTensor& x,
                              const phi::DenseTensor& proj_weight,
                              const phi::DenseTensor& down_weight,
                              phi::DenseTensor* out) {
  if (x.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::FusedGateUpMlpKernel<phi::dtype::bfloat16>(
        dev_ctx, x, proj_weight, down_weight, out);
  } else {
    throw std::runtime_error("Unsupported data type for FusedGateUpMlpKernel");
  }
}

template <typename Context>
void CallFusedMlpKernel(
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
    const bool permuted_weights,
    phi::DenseTensor* out) {
  if (hidden_states.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::FusedMlpKernel<phi::dtype::bfloat16>(
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
        permuted_weights,
        out);
  } else {
    throw std::runtime_error("Unsupported data type for FusedRmsMlpKernel");
  }
}

std::vector<paddle::Tensor> FusedMlpForward(
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

  CallFusedMlpKernel(*dev_ctx,
                     *hidden_states_tensor,
                     *proj_weight_tensor,
                     up_weight_tensor,
                     *down_weight_tensor,
                     paddle::optional<phi::DenseTensor>(),
                     paddle::optional<phi::DenseTensor>(),
                     paddle::optional<phi::DenseTensor>(),
                     paddle::optional<phi::DenseTensor>(),
                     paddle::optional<phi::DenseTensor>(),
                     false,  // permuted_weights,
                     out_tensor.get());

  paddle::Tensor out(out_tensor);

  return {out};
}

std::vector<paddle::Tensor> FusedFP8MlpForward(
    const paddle::Tensor& hidden_states,
    const paddle::Tensor& proj_weight,
    const paddle::optional<paddle::Tensor>& up_weight,
    const paddle::Tensor& down_weight,
    const paddle::optional<paddle::Tensor>& hidden_states_scale,
    const paddle::optional<paddle::Tensor>& proj_scale,
    const paddle::optional<paddle::Tensor>& up_scale,
    const paddle::optional<paddle::Tensor>& intermediate_hidden_states_scale,
    const paddle::optional<paddle::Tensor>& down_scale,
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

  auto hidden_states_scale_tensor = paddle::optional<phi::DenseTensor>();
  if (hidden_states_scale) {
    auto hidden_states_scale_dt =
        static_cast<phi::DenseTensor*>(hidden_states_scale->impl().get());
    hidden_states_scale_tensor =
        paddle::optional<phi::DenseTensor>(*hidden_states_scale_dt);
  }
  auto proj_scale_tensor = paddle::optional<phi::DenseTensor>();
  if (proj_scale) {
    auto proj_scale_dt =
        static_cast<phi::DenseTensor*>(proj_scale->impl().get());
    proj_scale_tensor = paddle::optional<phi::DenseTensor>(*proj_scale_dt);
  }
  auto up_scale_tensor = paddle::optional<phi::DenseTensor>();
  if (up_scale) {
    auto up_scale_dt = static_cast<phi::DenseTensor*>(up_scale->impl().get());
    up_scale_tensor = paddle::optional<phi::DenseTensor>(*up_scale_dt);
  }
  auto intermediate_hidden_states_scale_tensor =
      paddle::optional<phi::DenseTensor>();
  if (intermediate_hidden_states_scale) {
    auto intermediate_hidden_states_scale_dt = static_cast<phi::DenseTensor*>(
        intermediate_hidden_states_scale->impl().get());
    intermediate_hidden_states_scale_tensor =
        paddle::optional<phi::DenseTensor>(
            *intermediate_hidden_states_scale_dt);
  }
  auto down_scale_tensor = paddle::optional<phi::DenseTensor>();
  if (down_scale) {
    auto down_scale_dt =
        static_cast<phi::DenseTensor*>(down_scale->impl().get());
    down_scale_tensor = paddle::optional<phi::DenseTensor>(*down_scale_dt);
  }
  auto out_tensor = std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(hidden_states_tensor->dims());

  CallFusedMlpKernel(*dev_ctx,
                     *hidden_states_tensor,
                     *proj_weight_tensor,
                     up_weight_tensor,
                     *down_weight_tensor,
                     hidden_states_scale_tensor,
                     proj_scale_tensor,
                     up_scale_tensor,
                     intermediate_hidden_states_scale_tensor,
                     down_scale_tensor,
                     permuted_weights,
                     out_tensor.get());

  paddle::Tensor out(out_tensor);

  return {out};
}

std::vector<paddle::Tensor> FusedSplitMlpForward(
    const paddle::Tensor& x,
    const paddle::Tensor& proj_weight,
    const paddle::optional<paddle::Tensor>& up_weight,
    const paddle::Tensor& down_weight) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());

  auto down_tensor =
      static_cast<const phi::DenseTensor*>(down_weight.impl().get());
  auto out_tensor = std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(x_tensor->dims());

  if (up_weight) {
    auto gate_tensor =
        static_cast<const phi::DenseTensor*>(proj_weight.impl().get());
    auto up_tensor =
        static_cast<const phi::DenseTensor*>(up_weight->impl().get());

    CallFusedSplitMlpKernel(*dev_ctx,
                            *x_tensor,
                            *gate_tensor,
                            *up_tensor,
                            *down_tensor,
                            out_tensor.get());
  } else {
    auto proj_tensor =
        static_cast<const phi::DenseTensor*>(proj_weight.impl().get());

    CallFusedGateUpMlpKernel(
        *dev_ctx, *x_tensor, *proj_tensor, *down_tensor, out_tensor.get());
  }

  paddle::Tensor out(out_tensor);

  return {out};
}

std::vector<std::vector<int64_t>> FusedMlpInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& proj_weight_shape,
    const paddle::optional<std::vector<int64_t>>& up_weight_shape,
    const std::vector<int64_t>& down_weight_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> FusedMlpInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::DataType& proj_weight_dtype,
    const paddle::optional<paddle::DataType>& up_weight_dtype,
    const paddle::DataType& down_weight_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(fused_mlp_bf16)
    .Inputs({"hidden_states",
             "proj_weight",
             paddle::Optional("up_weight"),
             "down_weight"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(FusedMlpForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedMlpInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedMlpInferDtype));

PD_BUILD_OP(fused_mlp)
    .Inputs({"hidden_states",
             "proj_weight",
             paddle::Optional("up_weight"),
             "down_weight",
             paddle::Optional("hidden_states_scale"),
             paddle::Optional("proj_scale"),
             paddle::Optional("up_scale"),
             paddle::Optional("intermediate_hidden_states_scales"),
             paddle::Optional("down_scale")})
    .Outputs({"out"})
    .Attrs({"permuted_weights: bool"})
    .SetKernelFn(PD_KERNEL(FusedFP8MlpForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedMlpInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedMlpInferDtype));
