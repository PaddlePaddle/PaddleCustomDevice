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

namespace custom_kernel {

class FusedFp8Gemm : public HpuOperator {
 public:
  FusedFp8Gemm() : HpuOperator("fp8_gemm_bf16") {}

  void AddNode(ConvertTensors& ct, synGEMMParams& params) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);

    std::vector<synTensor> syn_inputs;
    synStatus status = synFail;

    for (size_t i = 0; i < inputs.size(); i++) {
      syn_inputs.push_back(createTensor(inputs[i].dims.size(),
                                        inputs[i].type,
                                        inputs[i].dims,
                                        true,
                                        inputs[i].name));
    }

    std::vector<synTensor> sync_outputs;
    for (size_t i = 0; i < outputs.size(); i++) {
      sync_outputs.push_back(createTensor(outputs[i].dims.size(),
                                          outputs[i].type,
                                          outputs[i].dims,
                                          true,
                                          outputs[i].name));
    }

    status = synNodeCreate(graphHandle_,
                           syn_inputs.data(),
                           sync_outputs.data(),
                           syn_inputs.size(),
                           sync_outputs.size(),
                           &params,
                           sizeof(params),
                           guid_.c_str(),
                           guid_.c_str(),
                           nullptr,
                           nullptr);
    PD_CHECK(
        status == synSuccess, "[RUNTIME] synNodeCreate () failed = %d", status);
  }
};

template <typename T, typename Context>
void fused_fp8_gemm(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    const phi::DenseTensor& bias,
                    const bool& trans_x,
                    const bool& trans_y,
                    const phi::DenseTensor& sca,
                    const phi::DenseTensor& scb,
                    const std::string& output_dtype,
                    const std::string& activation_type,
                    phi::DenseTensor* out) {
  // allocate memory on device.
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  ConvertTensors ct;
  ct.Add(x);
  ct.Add(y);
  ct.Add(sca);
  ct.Add(scb);
  ct.Add(bias);

  ct.Add(*out, false);
  std::vector<DIMS> inputs_dims = ct.GetDims();

  synGEMMParams params{trans_x, trans_y};
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, synGEMMParams>(
      "FusedFp8GemmKernel", inputs_dims, &params);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    FusedFp8Gemm op;
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

std::vector<paddle::Tensor> FusedFp8GemmForward(
    const paddle::Tensor& x,
    const paddle::Tensor& y,
    const paddle::optional<paddle::Tensor>& bias,
    bool trans_x,
    bool trans_y,
    float scale,  // only support per-tensor quantization
    std::string output_dtype,
    std::string activation_type) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto y_tensor = static_cast<const phi::DenseTensor*>(y.impl().get());
  auto out_tensor = std::make_shared<phi::DenseTensor>();

  paddle::Tensor sca =
      paddle::full(x.shape(), scale, phi::DataType::FLOAT32, x.place());
  auto sca_tensor = static_cast<const phi::DenseTensor*>(sca.impl().get());

  paddle::Tensor scb =
      paddle::full(x.shape(), 1.0, phi::DataType::FLOAT32, x.place());
  auto scb_tensor = static_cast<const phi::DenseTensor*>(scb.impl().get());

  if (activation_type != "identity") {
    PD_THROW("fused_fp8_gemm only support identity activation, but received %s",
             activation_type);
  }

  int rank = x.dims().size();
  int M = 0;
  int N = 0;

  if (!trans_x) {
    M = x.dims()[rank - 2];
  } else {
    M = x.dims()[rank - 1];
  }
  if (!trans_y) {
    N = y.dims()[rank - 1];
  } else {
    N = y.dims()[rank - 2];
  }
  std::vector<int64_t> out_shape = x.shape();
  out_shape[rank - 1] = N;
  out_shape[rank - 2] = M;

  out_tensor->Resize(phi::make_ddim(out_shape));

  phi::DenseTensor* bias_tensor = nullptr;
  paddle::Tensor bias_zero =
      paddle::full(out_shape, 0.0, phi::DataType::BFLOAT16, x.place());

  if (bias) {
    auto bias_ptr = *(bias.get_ptr());
    bias_tensor = static_cast<phi::DenseTensor*>(bias_ptr.impl().get());
  } else {
    bias_tensor = static_cast<phi::DenseTensor*>(bias_zero.impl().get());
  }

  custom_kernel::fused_fp8_gemm<phi::dtype::bfloat16>(*dev_ctx,
                                                      *x_tensor,
                                                      *y_tensor,
                                                      *bias_tensor,
                                                      trans_x,
                                                      trans_y,
                                                      *sca_tensor,
                                                      *scb_tensor,
                                                      output_dtype,
                                                      activation_type,
                                                      out_tensor.get());

  paddle::Tensor out(out_tensor);

  return {out};
}

std::vector<std::vector<int64_t>> FusedFp8GemmShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y_shape,
    const paddle::optional<std::vector<int64_t>>& bias_shape,
    bool trans_x,
    bool trans_y) {
  PADDLE_ENFORCE_EQ(x_shape.size(),
                    y_shape.size(),
                    phi::errors::InvalidArgument(
                        "The rank of input X and Y should be equal, but "
                        "received X's rank is %d, Y's rank is %d.",
                        x_shape.size(),
                        y_shape.size()));

  int rank = x_shape.size();
  int M = 0;
  int N = 0;

  if (!trans_x) {
    M = x_shape[rank - 2];
  } else {
    M = x_shape[rank - 1];
  }
  if (!trans_y) {
    N = y_shape[rank - 1];
  } else {
    N = y_shape[rank - 2];
  }
  std::vector<int64_t> out_shape = x_shape;
  out_shape[rank - 1] = N;
  out_shape[rank - 2] = M;
  return {out_shape};
}

std::vector<paddle::DataType> FusedFp8GemmDtype(
    const paddle::DataType& x_type,
    const paddle::DataType& y_type,
    const paddle::optional<paddle::DataType>& bias_type,
    bool trans_x,
    bool trans_y,
    float scale,  // only support per-tensor quantization
    std::string output_dtype) {
  paddle::DataType data_type;
  if (output_dtype == "bfloat16")
    data_type = paddle::DataType::BFLOAT16;
  else
    PD_THROW("fused_fp8_gemm only support bfloat16 output");

  return {data_type};
}

PD_BUILD_OP(fused_fp8_gemm)
    .Inputs({"x", "y", paddle::Optional("bias")})
    .Attrs({"transpose_x: bool",
            "transpose_y: bool",
            "scale: float",
            "output_dtype: std::string",
            "act: std::string"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(FusedFp8GemmForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedFp8GemmShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedFp8GemmDtype));
