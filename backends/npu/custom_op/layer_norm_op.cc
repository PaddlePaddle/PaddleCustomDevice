// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <vector>

#include "kernels/funcs/npu_op_runner.h"
#include "paddle/extension.h"
#ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#include "acltransformer/params/norm.h"
#include "kernels/funcs/format_utils.h"
#endif

std::vector<std::vector<int64_t>> LayerNormOpInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& weight_shape,
    const std::vector<int64_t>& bias_shape) {
  return {input_shape};
}

std::vector<paddle::DataType> LayerNormOpInferDtype(
    const paddle::DataType input_type,
    const paddle::DataType weight_type,
    const paddle::DataType bias_type) {
  return {input_type};
}

std::vector<paddle::Tensor> LayerNormOp(const paddle::Tensor& input,
                                        const paddle::Tensor& weight,
                                        const paddle::Tensor& bias,
                                        int begin_norm_axis = 1,
                                        float epsilon = 1e-5) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(input.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto input_tensor = static_cast<const phi::DenseTensor*>(input.impl().get());
  auto weight_tensor =
      static_cast<const phi::DenseTensor*>(weight.impl().get());
  auto bias_tensor = static_cast<const phi::DenseTensor*>(bias.impl().get());

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  auto out_shape =
      LayerNormOpInferShape(input.shape(), weight.shape(), bias.shape()).at(0);
  out_tensor->Resize(phi::make_ddim(out_shape));
  dev_ctx->Alloc(out_tensor.get(), input_tensor->dtype());
#ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
  auto input_asd = ConvertDenseTensorToAsdTensor(*input_tensor);
  auto weight_asd = ConvertDenseTensorToAsdTensor(*weight_tensor);
  auto bias_asd = ConvertDenseTensorToAsdTensor(*bias_tensor);
  auto out_tensor_asd = ConvertDenseTensorToAsdTensor(*out_tensor);

  AclTransformer::NormParam opParam = {
      epsilon, begin_norm_axis, begin_norm_axis};
  AclTransformer::OperationCall opCall("LayerNormOperation", opParam);
  AsdOps::SVector<AsdOps::Tensor> inTensors = {input_asd, weight_asd, bias_asd};
  AsdOps::SVector<AsdOps::Tensor> outTensors = {out_tensor_asd};

  int ret = opCall.ExecuteSync(inTensors, outTensors, stream);
  VLOG(6) << "LayerNormOp run in transformer acceleration ret:" << ret;
  return {paddle::Tensor(out_tensor)};
#endif
  return {paddle::experimental::layer_norm(
      input, weight, bias, epsilon, begin_norm_axis)};
}

PD_BUILD_OP(custom_layer_norm)
    .Inputs({"X", "Scale", "Bias"})
    .Outputs({"Out"})
    .Attrs({"begin_norm_axis: int", "epsilon: float"})
    .SetKernelFn(PD_KERNEL(LayerNormOp))
    .SetInferDtypeFn(PD_INFER_DTYPE(LayerNormOpInferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(
        LayerNormOpInferShape));  // neccessary if the op has muti_inputs
