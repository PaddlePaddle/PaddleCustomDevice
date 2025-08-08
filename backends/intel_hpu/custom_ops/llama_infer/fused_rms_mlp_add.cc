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

struct FusedRmsMlpResParams {
  ns_LayerNormKernel::Params rmsnorm_params;
  synSplitParams split_params;

  bool use_fp8;
  int ffn1_sclale_index;
};

class FusedRmsMlpRes : public HpuFusedOperator {
 public:
  explicit FusedRmsMlpRes(synDataType dtype)
      : HpuFusedOperator("fused_rms_mlp_res_fwd_", false), dtype_(dtype) {}
  template <typename T>
  void AddNode(ConvertTensors& ct, FusedRmsMlpResParams params) {
    auto ins = ct.GetTensors();
    auto outs = ct.GetTensors(false);

    synGEMMParams gemm_params;
    gemm_params.transpose_a = false;
    gemm_params.transpose_b = (params.use_fp8 ? true : false);

    synSectionHandle section = createSection();
    auto hidden_states = createTensorFromCT(&ct, 0);
    auto residual_input = createTensorFromCT(&ct, 4, true, section);
    auto residual_out = createTensorFromCT(&ct, 1, false, section);

    std::vector<synTensor> add_residual_in;
    add_residual_in.push_back(hidden_states);
    add_residual_in.push_back(residual_input);

    std::vector<synTensor> add_residual_out;
    add_residual_out.push_back(residual_out);

    AddNodeAdd<T>(add_residual_in, add_residual_out, guid_ + "add_residual");

    auto ln_scales = createTensorFromCT(&ct, 1);
    std::vector<synTensor> rmsnorm_inputs;
    rmsnorm_inputs.push_back(residual_out);
    rmsnorm_inputs.push_back(ln_scales);

    auto tmp_dims = ins[0].dims;
    tmp_dims[2] = 1;
    auto norm_out = createTensorNoPresist("norm_out", ins[0].type, ins[0].dims);
    auto norm_var = createTensorNoPresist("norm_var", ins[0].type, tmp_dims);
    std::vector<synTensor> rmsnorm_outputs;
    rmsnorm_outputs.push_back(norm_out);
    rmsnorm_outputs.push_back(norm_var);

    AddNodeRmsNorm<T>(rmsnorm_inputs,
                      rmsnorm_outputs,
                      params.rmsnorm_params,
                      guid_ + "rmsnorm");

    std::vector<int64_t> split_out_dims;
    synTensor gate_out;
    synTensor up_out;
    if (params.use_fp8) {
      // ffn1_0 gemm node
      std::vector<int64_t> proj_dims = {
          ins[0].dims[0], ins[0].dims[1], ins[2].dims[0]};
      auto proj_weight0 = createTensorFromCT(&ct, 2);
      auto ffn1_in_scale = createTensorFromCT(&ct, params.ffn1_sclale_index);
      auto ffn1_0_scale = createTensorFromCT(&ct, params.ffn1_sclale_index + 1);
      gate_out = createTensorNoPresist("gate_out", dtype_, proj_dims);
      std::vector<synTensor> ffn1_ins = {
          norm_out, proj_weight0, ffn1_in_scale, ffn1_0_scale};
      std::vector<synTensor> ffn1_outs = {gate_out};
      AddNodeFusedFP8Gemm<T>(
          ffn1_ins, ffn1_outs, gemm_params, guid_ + "_ffn1_0_gemm");

      // ffn1_1 gemm node
      auto proj_weight1 = createTensorFromCT(&ct, 5);
      auto ffn1_1_scale = createTensorFromCT(&ct, params.ffn1_sclale_index + 2);
      up_out = createTensorNoPresist("up_out", dtype_, proj_dims);
      ffn1_ins.clear();
      ffn1_ins.push_back(norm_out);
      ffn1_ins.push_back(proj_weight1);
      ffn1_ins.push_back(ffn1_in_scale);
      ffn1_ins.push_back(ffn1_1_scale);
      ffn1_outs.clear();
      ffn1_outs.push_back(up_out);
      AddNodeFusedFP8Gemm<T>(
          ffn1_ins, ffn1_outs, gemm_params, guid_ + "_ffn1_1_gemm");

      split_out_dims.push_back(proj_dims[0]);
      split_out_dims.push_back(proj_dims[1]);
      split_out_dims.push_back(proj_dims[2]);
    } else {
      // ffn1 gemm node
      std::vector<int64_t> proj_dims = {
          ins[0].dims[0], ins[0].dims[1], ins[2].dims[1]};
      auto proj_weight = createTensorFromCT(&ct, 2);
      auto proj_out = createTensorNoPresist("proj_out", ins[0].type, proj_dims);
      std::vector<synTensor> ffn1_ins = {norm_out, proj_weight};
      std::vector<synTensor> ffn1_outs = {proj_out};
      AddNodeGemm(ffn1_ins, ffn1_outs, gemm_params, guid_ + "_ffn1_gemm");

      // split node
      split_out_dims.push_back(proj_dims[0]);
      split_out_dims.push_back(proj_dims[1]);
      split_out_dims.push_back(proj_dims[2] / 2);
      gate_out = createTensorNoPresist("gate_out", ins[0].type, split_out_dims);
      up_out = createTensorNoPresist("up_out", ins[0].type, split_out_dims);
      std::vector<synTensor> split_ins = {proj_out};
      std::vector<synTensor> split_outs = {gate_out, up_out};
      AddNodeSplit(
          split_ins, split_outs, params.split_params, guid_ + "_split");
    }

    auto silu_out =
        createTensorNoPresist("silu_out", ins[0].type, split_out_dims);
    std::vector<synTensor> silu_inputs;
    silu_inputs.push_back(gate_out);
    std::vector<synTensor> silu_outputs;
    silu_outputs.push_back(silu_out);

    AddNodeSilu<T>(silu_inputs, silu_outputs, guid_ + "silu");

    auto multi_out =
        createTensorNoPresist("multi_out", ins[0].type, split_out_dims);
    std::vector<synTensor> multi_inputs;
    multi_inputs.push_back(silu_out);
    multi_inputs.push_back(up_out);
    std::vector<synTensor> multi_outputs;
    multi_outputs.push_back(multi_out);

    AddNodeMultiply<T>(multi_inputs, multi_outputs, guid_ + "_multi");

    auto down_weight = createTensorFromCT(&ct, 3);
    auto mlp_out = createTensorFromCT(&ct, 0, false);
    std::vector<synTensor> ffn2_ins = {multi_out, down_weight};
    std::vector<synTensor> ffn2_outs = {mlp_out};

    if (params.use_fp8) {
      auto ffn2_in_scale =
          createTensorFromCT(&ct, params.ffn1_sclale_index + 3);
      auto ffn2_scale = createTensorFromCT(&ct, params.ffn1_sclale_index + 4);
      ffn2_ins.push_back(ffn2_in_scale);
      ffn2_ins.push_back(ffn2_scale);
      AddNodeFusedFP8Gemm<T>(
          ffn2_ins, ffn2_outs, gemm_params, guid_ + "_ffn2_gemm");
    } else {
      AddNodeGemm(ffn2_ins, ffn2_outs, gemm_params, guid_ + "_ffn2_gemm");
    }
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void FusedRmsMlpResKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const phi::DenseTensor& residual,
    const phi::DenseTensor& ln_scales,
    const phi::DenseTensor& proj_weight0,
    const paddle::optional<phi::DenseTensor>& proj_weight1,
    const phi::DenseTensor& down_weight,
    const paddle::optional<phi::DenseTensor>& ffn1_in_scale,
    const paddle::optional<phi::DenseTensor>& proj_scale0,
    const paddle::optional<phi::DenseTensor>& proj_scale1,
    const paddle::optional<phi::DenseTensor>& ffn2_in_scale,
    const paddle::optional<phi::DenseTensor>& down_scale,
    const phi::Scalar& epsilon,
    phi::DenseTensor* out) {
  // allocate memory on device.
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  std::vector<int64_t> ln_scales_dims =
      phi::vectorize<int64_t>(ln_scales.dims());

  const phi::Scalar axis_scalar = proj_weight0.dims().size() - 1;
  int64_t axis = axis_scalar.to<int64_t>();
  if (axis < 0) {
    axis = proj_weight0.dims().size() + axis;
  }
  FusedRmsMlpResParams params;
  memset(reinterpret_cast<void*>(&params), 0x00, sizeof(FusedRmsMlpResParams));
  params.rmsnorm_params.epsValid = true;
  params.rmsnorm_params.eps = epsilon.to<float>();

  params.split_params = {{0}};
  params.split_params.axis = proj_weight0.dims().size() - 1 - axis;

  params.use_fp8 = (proj_weight0.dtype() == phi::DataType::FLOAT8_E4M3FN);

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(ln_scales);
  ct.Add(proj_weight0);
  ct.Add(down_weight);
  ct.Add(residual);

  if (params.use_fp8) {
    ct.Add(proj_weight1.get());
    params.ffn1_sclale_index = 6;
    ct.Add(ffn1_in_scale.get());
    ct.Add(proj_scale0.get());
    ct.Add(proj_scale1.get());
    ct.Add(ffn2_in_scale.get());
    ct.Add(down_scale.get());
  }

  ct.Add(*out, false);
  ct.Add(residual, false);
  std::vector<DIMS> inputs_dims = ct.GetDims();

  OpCacheOperator op_info;
  std::string recipe_name =
      params.use_fp8 ? "FusedFP8RmsMlpResKernel" : "FusedRmsMlpResKernel";
  op_info.prepareOpInfo<T, FusedRmsMlpResParams>(
      recipe_name, inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FusedRmsMlpRes op(op_info.datatype_);
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
void CallFusedRmsMlpResKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const phi::DenseTensor& residual,
    const phi::DenseTensor& ln_scales,
    const phi::DenseTensor& proj_weight0,
    const paddle::optional<phi::DenseTensor>& proj_weight1,
    const phi::DenseTensor& down_weight,
    const paddle::optional<phi::DenseTensor>& ffn1_in_scale,
    const paddle::optional<phi::DenseTensor>& proj_scale0,
    const paddle::optional<phi::DenseTensor>& proj_scale1,
    const paddle::optional<phi::DenseTensor>& ffn2_in_scale,
    const paddle::optional<phi::DenseTensor>& down_scale,
    const phi::Scalar& epsilon,
    phi::DenseTensor* out) {
  if (x.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::FusedRmsMlpResKernel<phi::dtype::bfloat16>(dev_ctx,
                                                              x,
                                                              residual,
                                                              ln_scales,
                                                              proj_weight0,
                                                              proj_weight1,
                                                              down_weight,
                                                              ffn1_in_scale,
                                                              proj_scale0,
                                                              proj_scale1,
                                                              ffn2_in_scale,
                                                              down_scale,
                                                              epsilon,
                                                              out);
  } else {
    throw std::runtime_error("Unsupported data type for FusedRmsMlpResKernel");
  }
}

std::vector<paddle::Tensor> FusedRmsMlpResForward(
    const paddle::Tensor& x,
    const paddle::Tensor& ln_scales,
    const paddle::Tensor& proj_weight,
    const paddle::Tensor& down_weight,
    const paddle::Tensor& residual,
    const float epsilon) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto residual_tensor =
      static_cast<const phi::DenseTensor*>(residual.impl().get());

  auto ln_scales_tensor =
      static_cast<const phi::DenseTensor*>(ln_scales.impl().get());
  auto down_tensor =
      static_cast<const phi::DenseTensor*>(down_weight.impl().get());
  auto proj_tensor =
      static_cast<const phi::DenseTensor*>(proj_weight.impl().get());

  auto out_tensor = std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(x_tensor->dims());

  CallFusedRmsMlpResKernel(*dev_ctx,
                           *x_tensor,
                           *residual_tensor,
                           *ln_scales_tensor,
                           *proj_tensor,
                           paddle::optional<phi::DenseTensor>(),
                           *down_tensor,
                           paddle::optional<phi::DenseTensor>(),
                           paddle::optional<phi::DenseTensor>(),
                           paddle::optional<phi::DenseTensor>(),
                           paddle::optional<phi::DenseTensor>(),
                           paddle::optional<phi::DenseTensor>(),
                           phi::Scalar(epsilon),
                           out_tensor.get());

  paddle::Tensor out(out_tensor);

  return {out};
}

std::vector<std::vector<int64_t>> FusedRmsMlpResInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& ln_scales_shape,
    const std::vector<int64_t>& proj_weight_shape,
    const std::vector<int64_t>& down_weight_shape,
    const std::vector<int64_t>& residual_shape) {
  return {x_shape, residual_shape};
}

std::vector<paddle::DataType> FusedRmsMlpResInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::DataType& ln_scales_dtype,
    const paddle::DataType& proj_weight_dtype,
    const paddle::DataType& down_weight_dtype,
    const paddle::DataType& residual_dtype) {
  return {x_dtype, residual_dtype};
}

PD_BUILD_OP(fused_rms_mlp_res)
    .Inputs({"x", "ln_scales", "proj_weight", "down_weight", "residual_in"})
    .Outputs({"out"})
    .Attrs({"epsilon: float"})
    .SetKernelFn(PD_KERNEL(FusedRmsMlpResForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedRmsMlpResInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedRmsMlpResInferDtype));

std::vector<paddle::Tensor> FusedFP8RmsMlpResForward(
    const paddle::Tensor& x,
    const paddle::Tensor& ln_scales,
    const paddle::Tensor& proj_weight0,
    const paddle::Tensor& proj_weight1,
    const paddle::Tensor& down_weight,
    const paddle::Tensor& residual,
    const paddle::Tensor& ffn1_in_scale,
    const paddle::Tensor& proj_scale0,
    const paddle::Tensor& proj_scale1,
    const paddle::Tensor& ffn2_in_scale,
    const paddle::Tensor& down_scale,
    const float epsilon) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto residual_tensor =
      static_cast<const phi::DenseTensor*>(residual.impl().get());

  auto ln_scales_tensor =
      static_cast<const phi::DenseTensor*>(ln_scales.impl().get());
  auto proj_weight0_tensor =
      static_cast<const phi::DenseTensor*>(proj_weight0.impl().get());
  auto proj_weight1_tensor =
      static_cast<const phi::DenseTensor*>(proj_weight1.impl().get());
  auto down_weight_tensor =
      static_cast<const phi::DenseTensor*>(down_weight.impl().get());
  auto ffn1_in_scale_tensor =
      static_cast<const phi::DenseTensor*>(ffn1_in_scale.impl().get());
  auto proj_scale0_tensor =
      static_cast<const phi::DenseTensor*>(proj_scale0.impl().get());
  auto proj_scale1_tensor =
      static_cast<const phi::DenseTensor*>(proj_scale1.impl().get());
  auto ffn2_in_scale_tensor =
      static_cast<const phi::DenseTensor*>(ffn2_in_scale.impl().get());
  auto down_scale_tensor =
      static_cast<const phi::DenseTensor*>(down_scale.impl().get());

  auto out_tensor = std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(x_tensor->dims());

  CallFusedRmsMlpResKernel(*dev_ctx,
                           *x_tensor,
                           *residual_tensor,
                           *ln_scales_tensor,
                           *proj_weight0_tensor,
                           *proj_weight1_tensor,
                           *down_weight_tensor,
                           *ffn1_in_scale_tensor,
                           *proj_scale0_tensor,
                           *proj_scale1_tensor,
                           *ffn2_in_scale_tensor,
                           *down_scale_tensor,
                           phi::Scalar(epsilon),
                           out_tensor.get());

  paddle::Tensor out(out_tensor);

  return {out};
}

std::vector<std::vector<int64_t>> FusedFP8RmsMlpResInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& ln_scales_shape,
    const std::vector<int64_t>& proj_weight0_shape,
    const std::vector<int64_t>& proj_weight1_shape,
    const std::vector<int64_t>& down_weight_shape,
    const std::vector<int64_t>& residual_shape,
    const std::vector<int64_t>& ffn1_in_scale_shape,
    const std::vector<int64_t>& proj_scale0_shape,
    const std::vector<int64_t>& proj_scale1_shape,
    const std::vector<int64_t>& ffn2_in_scale_shape,
    const std::vector<int64_t>& down_scale_shape) {
  return {x_shape, residual_shape};
}

std::vector<paddle::DataType> FusedFP8RmsMlpResInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::DataType& ln_scales_dtype,
    const paddle::DataType& proj_weight0_dtype,
    const paddle::DataType& proj_weight1_dtype,
    const paddle::DataType& down_weight_dtype,
    const paddle::DataType& residual_dtype,
    const paddle::DataType& ffn1_in_scale_dtype,
    const paddle::DataType& proj_scale0_dtype,
    const paddle::DataType& proj_scale1_dtype,
    const paddle::DataType& ffn2_in_scale_dtype,
    const paddle::DataType& down_scale_dtype) {
  return {x_dtype, residual_dtype};
}

PD_BUILD_OP(fused_fp8_rms_mlp_res)
    .Inputs({"x",
             "ln_scales",
             "proj_weight0",
             "proj_weight1",
             "down_weight",
             "residual_in",
             "ff1_in_scale",
             "proj_scale0",
             "proj_scale1",
             "ff2_in_scale",
             "down_scale"})
    .Outputs({"out"})
    .Attrs({"epsilon: float"})
    .SetKernelFn(PD_KERNEL(FusedFP8RmsMlpResForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedFP8RmsMlpResInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedFP8RmsMlpResInferDtype));
