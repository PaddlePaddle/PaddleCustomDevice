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
    gemm_params.transpose_b = false;

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

    auto proj_weight = createTensorFromCT(&ct, 2);
    std::vector<int64_t> proj_dims = {
        ins[0].dims[0], ins[0].dims[1], ins[2].dims[1]};
    auto proj_out = createTensorNoPresist("proj_out", ins[0].type, proj_dims);

    std::vector<synTensor> proj_inputs;
    proj_inputs.push_back(norm_out);
    proj_inputs.push_back(proj_weight);
    std::vector<synTensor> proj_outputs;
    proj_outputs.push_back(proj_out);

    AddNodeGemm(proj_inputs, proj_outputs, gemm_params, guid_ + "gemm_up_proj");

    std::vector<int64_t> split_out_dims = {
        proj_dims[0], proj_dims[1], proj_dims[2] / 2};
    auto gate_out =
        createTensorNoPresist("gate_out", ins[0].type, split_out_dims);
    auto up_out = createTensorNoPresist("up_out", ins[0].type, split_out_dims);
    auto down_weight = createTensorFromCT(&ct, 3);

    std::vector<synTensor> split_inputs;
    split_inputs.push_back(proj_out);
    std::vector<synTensor> split_outputs;
    split_outputs.push_back(gate_out);
    split_outputs.push_back(up_out);

    AddNodeSplit(
        split_inputs, split_outputs, params.split_params, guid_ + "split");

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

    auto mlp_out = createTensorFromCT(&ct, 0, false);
    std::vector<synTensor> down_inputs;
    down_inputs.push_back(multi_out);
    down_inputs.push_back(down_weight);
    std::vector<synTensor> down_outputs;
    down_outputs.push_back(mlp_out);

    AddNodeGemm(
        down_inputs, down_outputs, gemm_params, guid_ + "gemm_down_proj");
  }

 protected:
  synDataType dtype_;
};

template <typename T, typename Context>
void FusedRmsMlpResKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& residual,
                          const phi::DenseTensor& ln_scales,
                          const phi::DenseTensor& proj_weight,
                          const phi::DenseTensor& down_weight,
                          const phi::Scalar& epsilon,
                          phi::DenseTensor* out) {
  // allocate memory on device.
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  std::vector<int64_t> ln_scales_dims =
      phi::vectorize<int64_t>(ln_scales.dims());

  const phi::Scalar axis_scalar = proj_weight.dims().size() - 1;
  int64_t axis = axis_scalar.to<int64_t>();
  if (axis < 0) {
    axis = proj_weight.dims().size() + axis;
  }
  FusedRmsMlpResParams params;
  memset(reinterpret_cast<void*>(&params), 0x00, sizeof(FusedRmsMlpResParams));
  params.rmsnorm_params.epsValid = true;
  params.rmsnorm_params.eps = epsilon.to<float>();

  params.split_params = {{0}};
  params.split_params.axis = proj_weight.dims().size() - 1 - axis;

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(ln_scales);
  ct.Add(proj_weight);
  ct.Add(down_weight);
  ct.Add(residual);
  ct.Add(*out, false);
  ct.Add(residual, false);
  std::vector<DIMS> inputs_dims = ct.GetDims();

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, FusedRmsMlpResParams>(
      "FusedRmsMlpResKernel", inputs_dims, &params);
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
void CallFusedRmsMlpResKernel(const Context& dev_ctx,
                              const phi::DenseTensor& x,
                              const phi::DenseTensor& residual,
                              const phi::DenseTensor& ln_scales,
                              const phi::DenseTensor& proj_weight,
                              const phi::DenseTensor& down_weight,
                              const phi::Scalar& epsilon,
                              phi::DenseTensor* out) {
  if (x.dtype() == phi::DataType::BFLOAT16) {
    custom_kernel::FusedRmsMlpResKernel<phi::dtype::bfloat16>(dev_ctx,
                                                              x,
                                                              residual,
                                                              ln_scales,
                                                              proj_weight,
                                                              down_weight,
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
                           *down_tensor,
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
