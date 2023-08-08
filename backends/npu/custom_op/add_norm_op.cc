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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "paddle/extension.h"
#ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#include "acltransformer/params/add_norm.h"
#include "kernels/funcs/format_utils.h"
#endif

std::vector<std::vector<int64_t>> AddNormOpInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y_shape,
    const std::vector<int64_t>& weight_shape,
    const std::vector<int64_t>& bias_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> AddNormOpInferDtype(
    const paddle::DataType x_type,
    const paddle::DataType y_type,
    const paddle::DataType weight_type,
    const paddle::DataType bias_type) {
  return {x_type};
}

std::vector<paddle::Tensor> AddNormOp(const paddle::Tensor& x,
                                      const paddle::Tensor& y,
                                      const paddle::Tensor& weight,
                                      const paddle::Tensor& bias,
                                      int begin_norm_axis = 1,
                                      float epsilon = 1e-5) {
  // 1. preprocessing: stream, input tensors;
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto y_tensor = static_cast<const phi::DenseTensor*>(y.impl().get());
  auto weight_tensor =
      static_cast<const phi::DenseTensor*>(weight.impl().get());
  auto bias_tensor = static_cast<const phi::DenseTensor*>(bias.impl().get());

  // 2. create output tensor: tensor, out_shape, alloc;
  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();

  auto out_shape =
      AddNormOpInferShape(x.shape(), y.shape(), weight.shape(), bias.shape())
          .at(0);
  out_tensor->Resize(phi::make_ddim(out_shape));

  dev_ctx->Alloc(out_tensor.get(), x_tensor->dtype());
#ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
  auto x_tensor_asd = ConvertDenseTensorToAsdTensor(*x_tensor);
  auto y_tensor_asd = ConvertDenseTensorToAsdTensor(*y_tensor);
  auto weight_tensor_asd = ConvertDenseTensorToAsdTensor(*weight_tensor);
  auto bias_tensor_asd = ConvertDenseTensorToAsdTensor(*bias_tensor);
  auto out_tensor_asd = ConvertDenseTensorToAsdTensor(*out_tensor);

  AclTransformer::AddNormParam opParam = {epsilon, 1.0, true}; /* 不设置scale*/
  AclTransformer::OperationCall opCall("AddNormOperation", opParam);
  AsdOps::SVector<AsdOps::Tensor> inTensors = {
      x_tensor_asd, y_tensor_asd, weight_tensor_asd, bias_tensor_asd};
  AsdOps::SVector<AsdOps::Tensor> outTensors = {out_tensor_asd};

  int ret = opCall.ExecuteSync(inTensors, outTensors, stream);
  VLOG(6) << "AddNorm Op run in transformer acceleration ret:" << ret;
  return {paddle::Tensor(out_tensor)};
#else
  const auto& runner =
      NpuOpRunner("Add", {*x_tensor, *y_tensor}, {*out_tensor}, {});
  runner.Run(stream);

  paddle::Tensor add_res(out_tensor);

  // PADDLE_API Tensor layer_norm(const Tensor& x, const
  // paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias,
  // float epsilon = 1e-5, int begin_norm_axis = 1);
  paddle::Tensor output = paddle::experimental::layer_norm(
      add_res, weight, bias, epsilon, begin_norm_axis);

  return {output};
#endif
}

PD_BUILD_OP(add_norm)
    .Inputs({"X", "Y", "Weight", "Bias"})
    .Outputs({"Out"})
    .Attrs({"begin_norm_axis: int", "epsilon: float"})
    .SetKernelFn(PD_KERNEL(AddNormOp))
    .SetInferDtypeFn(PD_INFER_DTYPE(AddNormOpInferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(
        AddNormOpInferShape));  // neccessary if the op has muti_inputs
