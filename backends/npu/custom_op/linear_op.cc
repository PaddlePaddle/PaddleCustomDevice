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
// #ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#if 0
#include "acltransformer/params/linear.h"
#include "kernels/funcs/format_utils.h"
#endif

std::vector<std::vector<int64_t>> LinearOpInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& weight_shape,
    const std::vector<int64_t>& bias_shape) {
  std::vector<int64_t> out_shape = input_shape;
  out_shape[out_shape.size() - 1] = weight_shape[weight_shape.size() - 1];
  return {out_shape};
}

std::vector<paddle::DataType> LinearOpInferDtype(
    const paddle::DataType input_type,
    const paddle::DataType weight_type,
    const paddle::DataType bias_type) {
  return {input_type};
}

std::vector<paddle::Tensor> LinearOp(const paddle::Tensor& input,
                                     const paddle::Tensor& weight,
                                     const paddle::Tensor& bias) {
  // note: only for case of [B, M, K] x [K, N] = [B, M, N].
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(input.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto x = static_cast<const phi::DenseTensor*>(input.impl().get());
  auto y = static_cast<const phi::DenseTensor*>(weight.impl().get());
  auto b = static_cast<const phi::DenseTensor*>(bias.impl().get());
// #ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#if 0
  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  auto out_shape =
      LinearOpInferShape(input.shape(), weight.shape(), bias.shape()).at(0);

  out_tensor->Resize(phi::make_ddim(out_shape));
  dev_ctx->Alloc(out_tensor.get(), x->dtype());

  auto x_asd = ConvertDenseTensorToAsdTensor(*x);
  auto y_asd = ConvertDenseTensorToAsdTensor(*y);
  auto b_asd = ConvertDenseTensorToAsdTensor(*b);
  auto out_tensor_asd = ConvertDenseTensorToAsdTensor(*out_tensor);

  AclTransformer::LinearParam opParam = {false, true}; /* 加速库默认会转置W */
  AclTransformer::OperationCall opCall("LinearOperation", opParam);
  AsdOps::SVector<AsdOps::Tensor> inTensors = {x_asd, y_asd, b_asd};
  AsdOps::SVector<AsdOps::Tensor> outTensors = {out_tensor_asd};

  int ret = opCall.ExecuteSync(inTensors, outTensors, stream);
  VLOG(6) << "Linear run in transformer acceleration ret:" << ret;
  return {paddle::Tensor(out_tensor)};
#else
  std::vector<int64_t> x_dims = phi::vectorize(x->dims());
  std::vector<int64_t> y_dims = phi::vectorize(y->dims());
  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();

  // x->x_temp : [B, M, K]->[B*M, K]
  phi::DenseTensor x_temp(*x);
  const int K = x_dims[x_ndim - 1];
  std::vector<int64_t> vec_dim = {x_temp.numel() / K, K};
  x_temp.Resize(phi::make_ddim(vec_dim));

  // out
  std::vector<int64_t> temp_dims = {x_temp.numel() / K, y_dims[y_ndim - 1]};

  std::shared_ptr<phi::DenseTensor> matmul_res =
      std::make_shared<phi::DenseTensor>();
  matmul_res->Resize(phi::make_ddim(temp_dims));
  dev_ctx->Alloc(matmul_res.get(), x->dtype());

  std::shared_ptr<phi::DenseTensor> out = std::make_shared<phi::DenseTensor>();
  out->Resize(phi::make_ddim(temp_dims));
  dev_ctx->Alloc(out.get(), x->dtype());

  // attr
  const bool transpose_x1 = false;
  const bool transpose_x2 = false;
  NPUAttributeMap attrs = {{"transpose_x1", transpose_x1},
                           {"transpose_x2", transpose_x2}};  // for MatMul

  const int64_t need_trans = 0;
  const int64_t with_bias = 1;
  const int64_t operate = 0;
  // NPUAttributeMap attrs = {{"needTrans", need_trans}, {"withBias",
  // with_bias}, {"operateType", operate}}; // for MatmulAll

  // run
  // use MatmulAll
  // const auto& matmulAllRunner =
  //     NpuOpRunner("MatmulAll", {x_temp, *y, *b, *b}, {*out}, attrs);
  // runner.Run(stream);

  // use MatMul
  const auto& matmulRunner =
      NpuOpRunner("MatMul", {x_temp, *y, *b}, {*out}, attrs);
  matmulRunner.Run(stream);

  // post
  auto out_dims =
      LinearOpInferShape(input.shape(), weight.shape(), bias.shape()).at(0);
  out->Resize(phi::make_ddim(out_dims));
  auto out_tensor = paddle::Tensor(out);

  return {out_tensor};
#endif
}

PD_BUILD_OP(linear)
    .Inputs({"Input", "Weight", "Bias"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(LinearOp))
    .SetInferDtypeFn(PD_INFER_DTYPE(LinearOpInferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(
        LinearOpInferShape));  // neccessary if the op has muti_inputs
