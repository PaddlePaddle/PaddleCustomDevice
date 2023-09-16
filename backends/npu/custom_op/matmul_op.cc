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
#include "paddle/phi/core/enforce.h"
// #ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#if 0
#include "acltransformer/params/matmul.h"
#include "kernels/funcs/format_utils.h"
#endif

std::vector<std::vector<int64_t>> MatmulOpInferShape(
    const std::vector<int64_t>& input_x_shape,
    const std::vector<int64_t>& input_y_shape,
    bool trans_x,
    bool trans_y) {
  std::vector<int64_t> x_shape = input_x_shape;
  std::vector<int64_t> y_shape = input_y_shape;

  auto ndims_x = x_shape.size();
  auto ndims_y = y_shape.size();
  PADDLE_ENFORCE_GT(ndims_x,
                    0UL,
                    phi::errors::InvalidArgument(
                        "The Input(x) dims size must be greater than 0,"
                        " but reviced dims size is 0. "));
  PADDLE_ENFORCE_GT(ndims_y,
                    0UL,
                    phi::errors::InvalidArgument(
                        "The Input(y) dims size must be greater than 0,"
                        " but reviced dims size is 0. "));

  bool x_broadcasted = false, y_broadcasted = false;
  if (ndims_x == 1) {
    x_shape.insert(x_shape.begin(), 1);
    ndims_x = 2;
    x_broadcasted = true;
  }

  if (ndims_y == 1) {
    y_shape.push_back(1);
    ndims_y = 2;
    y_broadcasted = true;
  }

  size_t M, N;
  if (trans_x) {
    M = x_shape[ndims_x - 1];
  } else {
    M = x_shape[ndims_x - 2];
  }
  if (trans_y) {
    N = y_shape[ndims_y - 2];
  } else {
    N = y_shape[ndims_y - 1];
  }

  std::vector<int64_t> new_dims;
  if (ndims_x > ndims_y) {
    new_dims.assign(x_shape.begin(), x_shape.end() - 2);
  } else if (ndims_x < ndims_y) {
    new_dims.assign(y_shape.begin(), y_shape.end() - 2);
  } else {
    new_dims.reserve(ndims_x);
    for (size_t i = 0; i < ndims_x - 2; ++i) {
      new_dims.push_back(std::max(x_shape[i], y_shape[i]));
    }
  }
  if (!x_broadcasted) {
    new_dims.push_back(M);
  }
  if (!y_broadcasted) {
    new_dims.push_back(N);
  }

  return {new_dims};
}

std::vector<paddle::DataType> MatmulOpInferDtype(
    const paddle::DataType& x_dtype, const paddle::DataType& y_dtype) {
  return {x_dtype};
}

std::vector<paddle::Tensor> MatmulOp(const paddle::Tensor& input_x,
                                     const paddle::Tensor& input_y,
                                     bool trans_x,
                                     bool trans_y) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(input_x.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto x = static_cast<const phi::DenseTensor*>(input_x.impl().get());
  auto y = static_cast<const phi::DenseTensor*>(input_y.impl().get());
// #ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#if 0
  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  auto out_shape =
      MatmulOpInferShape(input_x.shape(), input_y.shape(), trans_x, trans_y)
          .at(0);
  out_tensor->Resize(phi::make_ddim(out_shape));
  dev_ctx->Alloc(out_tensor.get(), x->dtype());

  auto x_asd = ConvertDenseTensorToAsdTensor(*x);
  auto y_asd = ConvertDenseTensorToAsdTensor(*y);
  auto out_tensor_asd = ConvertDenseTensorToAsdTensor(*out_tensor);

  /* 加速库默认会转置W */
  AclTransformer::MatmulParam opParam = {trans_x, trans_y};
  AclTransformer::OperationCall opCall("MatmulOperation", opParam);
  AsdOps::SVector<AsdOps::Tensor> inTensors = {x_asd, y_asd};
  AsdOps::SVector<AsdOps::Tensor> outTensors = {out_tensor_asd};

  int ret = opCall.ExecuteSync(inTensors, outTensors, stream);
  VLOG(6) << "MatmulOp run in transformer acceleration ret:" << ret;
  return {paddle::Tensor(out_tensor)};
#else
  return {paddle::matmul(input_x, input_y, trans_x, trans_y)};
#endif
}

PD_BUILD_OP(custom_matmul)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .Attrs({"trans_x: bool", "trans_y: bool"})
    .SetKernelFn(PD_KERNEL(MatmulOp))
    .SetInferDtypeFn(PD_INFER_DTYPE(MatmulOpInferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(
        MatmulOpInferShape));  // neccessary if the op has muti_inputs
