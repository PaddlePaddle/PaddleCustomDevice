// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "kernels/hpu_operator.h"
#include "paddle/extension.h"
#include "utils/utils.h"

#define MAX_FP8_VALUES 240

namespace custom_kernel {

class FusedQuant : public HpuOperator {
 public:
  FusedQuant() : HpuOperator("fused_quant_fwd") {}

  void AddNode(ConvertTensors& ct, ns_Reduction::ParamsV2& params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    std::string guid_abs = "abs_fwd_bf16";
    std::string guid_max = "reduce_max_multi_dim_fwd_bf16";
    std::string guid_div = "div_fwd_f32";
    std::string guid_quant = "quantize_per_tensor_bf16";

    std::string name_abs = guid_ + "_abs";
    std::string name_div = guid_ + "_div";
    std::string name_max = guid_ + "_max";
    std::string name_quant = guid_ + "_quant";

    synStatus status = synFail;
    auto in = createTensor(inputs[0].dims.size(),
                           inputs[0].type,
                           inputs[0].dims,
                           true,
                           inputs[0].name);

    std::vector<synTensor> abs_inputs;
    abs_inputs.push_back(in);
    auto abs_out = createTensor(inputs[0].dims.size(),
                                inputs[0].type,
                                inputs[0].dims,
                                false,
                                inputs[0].name);
    std::vector<synTensor> abs_outputs;
    abs_outputs.push_back(abs_out);
    status = synNodeCreate(graphHandle_,
                           abs_inputs.data(),
                           abs_outputs.data(),
                           abs_inputs.size(),
                           abs_outputs.size(),
                           nullptr,
                           0,
                           guid_abs.c_str(),
                           name_abs.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    std::vector<synTensor> max_inputs;
    max_inputs.push_back(abs_out);

    auto max_out = createTensor(outputs[0].dims.size(),
                                outputs[0].type,
                                outputs[0].dims,
                                false,
                                outputs[0].name);

    std::vector<synTensor> max_outputs;
    max_outputs.push_back(max_out);

    status = synNodeCreate(graphHandle_,
                           max_inputs.data(),
                           max_outputs.data(),
                           max_inputs.size(),
                           max_outputs.size(),
                           &params,
                           sizeof(params),
                           guid_max.c_str(),
                           name_max.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    std::vector<synTensor> div_inputs;
    div_inputs.push_back(max_out);
    div_inputs.push_back(createTensor(inputs[3].dims.size(),
                                      inputs[3].type,
                                      inputs[3].dims,
                                      true,
                                      inputs[3].name));

    std::vector<synTensor> div_outputs;
    auto scale = createTensor(outputs[0].dims.size(),
                              outputs[0].type,
                              outputs[0].dims,
                              true,
                              outputs[0].name);
    div_outputs.push_back(scale);
    status = synNodeCreate(graphHandle_,
                           div_inputs.data(),
                           div_outputs.data(),
                           div_inputs.size(),
                           div_outputs.size(),
                           nullptr,
                           0,
                           guid_div.c_str(),
                           name_div.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);

    std::vector<synTensor> quant_inputs;
    quant_inputs.push_back(in);
    quant_inputs.push_back(scale);
    for (size_t i = 1; i < inputs.size(); i++) {
      quant_inputs.push_back(createTensor(inputs[i].dims.size(),
                                          inputs[i].type,
                                          inputs[i].dims,
                                          true,
                                          inputs[i].name));
    }

    std::vector<synTensor> quant_outputs;
    auto out = createTensor(outputs[1].dims.size(),
                            outputs[1].type,
                            outputs[1].dims,
                            true,
                            outputs[1].name);
    quant_outputs.push_back(out);
    status = synNodeCreate(graphHandle_,
                           quant_inputs.data(),
                           quant_outputs.data(),
                           quant_inputs.size(),
                           quant_outputs.size(),
                           nullptr,
                           0,
                           guid_quant.c_str(),
                           name_quant.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

template <typename T, typename Context>
void fused_quant(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 phi::DenseTensor* scale,
                 const phi::DenseTensor& zp,
                 const phi::DenseTensor& min,
                 const phi::DenseTensor& max,
                 phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  dev_ctx.template Alloc<float>(scale);
  if (scale->numel() == 0) {
    return;
  }

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(zp);
  ct.Add(min);
  ct.Add(max);

  ct.Add(*scale, false);
  ct.Add(*out, false);

  ns_Reduction::ParamsV2 params{};

  std::vector<DIMS> inputs_dims = ct.GetDims();

  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, ns_Reduction::ParamsV2>(
      "FusedQuantKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FusedQuant op;
    op.AddNode(ct, params);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  auto tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx.stream()), tensors);
}

}  // namespace custom_kernel

std::vector<paddle::Tensor> FusedQuantForward(const paddle::Tensor& x) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto scale_tensor = std::make_shared<phi::DenseTensor>();
  auto out_tensor = std::make_shared<phi::DenseTensor>();

  paddle::Tensor zp = paddle::full({1}, 0, phi::DataType::BFLOAT16, x.place());
  paddle::Tensor min = paddle::full(
      {1}, -1.0 * MAX_FP8_VALUES, phi::DataType::BFLOAT16, x.place());
  paddle::Tensor max =
      paddle::full({1}, MAX_FP8_VALUES, phi::DataType::BFLOAT16, x.place());

  auto zp_tensor = static_cast<const phi::DenseTensor*>(zp.impl().get());
  auto min_tensor = static_cast<const phi::DenseTensor*>(min.impl().get());
  auto max_tensor = static_cast<const phi::DenseTensor*>(max.impl().get());

  std::vector<int64_t> out_shape = x.shape();

  scale_tensor->Resize(phi::make_ddim({1}));
  out_tensor->Resize(phi::make_ddim(out_shape));

  custom_kernel::fused_quant<phi::dtype::float8_e4m3fn>(*dev_ctx,
                                                        *x_tensor,
                                                        scale_tensor.get(),
                                                        *zp_tensor,
                                                        *min_tensor,
                                                        *max_tensor,
                                                        out_tensor.get());

  paddle::Tensor out(out_tensor);
  paddle::Tensor scale(scale_tensor);

  return {out, scale};
}

std::vector<std::vector<int64_t>> FusedQuantShape(
    const std::vector<int64_t>& x_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> FusedQuantDtype(const paddle::DataType& x_type,
                                              std::string output_dtype) {
  paddle::DataType data_type;
  if (output_dtype == "float8_e4m3fn")
    data_type = paddle::DataType::FLOAT8_E4M3FN;
  else
    PD_THROW("fused_quant only support float8_e4m3fn output");

  return {data_type};
}

PD_BUILD_OP(fused_quant)
    .Inputs({"x"})
    .Outputs({"out", "scale"})
    .SetKernelFn(PD_KERNEL(FusedQuantForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedQuantShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedQuantDtype));
