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
#include "acltransformer/params/add.h"
#include "kernels/funcs/format_utils.h"
#endif

std::vector<std::vector<int64_t>> AddOpInferShape(
    const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> AddOpInferDtype(const paddle::DataType x_type,
                                              const paddle::DataType y_type) {
  return {x_type};
}

std::vector<paddle::Tensor> AddOp(const paddle::Tensor& x_tensor,
                                  const paddle::Tensor& y_tensor) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          x_tensor.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto x = static_cast<const phi::DenseTensor*>(x_tensor.impl().get());
  auto y = static_cast<const phi::DenseTensor*>(y_tensor.impl().get());

#ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
  std::shared_ptr<phi::DenseTensor> out = std::make_shared<phi::DenseTensor>();
  auto out_shape = AddOpInferShape(x_tensor.shape(), y_tensor.shape()).at(0);
  out->Resize(phi::make_ddim(out_shape));
  dev_ctx->Alloc(out.get(), x->dtype());

  auto x_asd = ConvertDenseTensorToAsdTensor(*x);
  auto y_asd = ConvertDenseTensorToAsdTensor(*y);
  auto out_tensor_asd = ConvertDenseTensorToAsdTensor(*out);

  // AclTransformer::AddParam opParam = {false, true}; /* 加速库默认会转置W */
  AclTransformer::AddParam opParam = {};
  AclTransformer::OperationCall opCall("AddOperation", opParam);
  AsdOps::SVector<AsdOps::Tensor> inTensors = {x_asd, y_asd};
  AsdOps::SVector<AsdOps::Tensor> outTensors = {out_tensor_asd};

  int ret = opCall.ExecuteSync(inTensors, outTensors, stream);
  VLOG(6) << "AddOp run in transformer acceleration ret:" << ret;
  return {paddle::Tensor(out)};
#else
  return {paddle::add(x_tensor, y_tensor)};
#endif
}

PD_BUILD_OP(custom_add)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(AddOp))
    .SetInferDtypeFn(PD_INFER_DTYPE(AddOpInferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(
        AddOpInferShape));  // neccessary if the op has muti_inputs
