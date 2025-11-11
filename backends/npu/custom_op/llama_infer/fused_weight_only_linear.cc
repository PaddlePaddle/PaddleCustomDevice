// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

std::vector<std::vector<int64_t>> WeightOnlyLinearInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& weight_shape,
    const std::vector<int64_t>& scale_shape) {
  std::vector<int64_t> output_shape;
  output_shape.push_back(x_shape[0]);
  output_shape.push_back(weight_shape[1]);
  return {x_shape};
}

std::vector<paddle::Tensor> weight_only_linear_npu(
    const paddle::Tensor& x,
    const paddle::Tensor& weight,
    const paddle::Tensor& scale) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto weight_tensor =
      static_cast<const phi::DenseTensor*>(weight.impl().get());
  auto scale_tensor = static_cast<const phi::DenseTensor*>(scale.impl().get());

  std::vector<int64_t> out_dims = {x_tensor->dims()[0],
                                   weight_tensor->dims()[1]};
  // auto x_dims = x_tensor->dims();
  // for (int i = 0; i < x_dims.size()-1; i++) {
  //   out_dims.push_back(x_dims[i]);
  // }
  // out_dims.push_back(weight_tensor->dims()[1]);

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(phi::make_ddim(out_dims));
  dev_ctx->Alloc(out_tensor.get(), x_tensor->dtype());

  phi::DenseTensor* null1 = nullptr;
  phi::DenseTensor* null2 = nullptr;
  phi::DenseTensor* null3 = nullptr;
  phi::DenseTensor* null4 = nullptr;

  int64_t zero = 0;

  EXEC_NPU_CMD(aclnnWeightQuantBatchMatmulV2,
               *dev_ctx,
               *x_tensor,
               *weight_tensor,
               *scale_tensor,
               null1,
               null2,
               null3,
               null4,
               zero,
               *out_tensor);
  return {paddle::Tensor(out_tensor)};
}

PD_BUILD_OP(weight_only_linear_npu)
    .Inputs({"x", "weight", "scale"})
    .Outputs({"output"})
    .SetKernelFn(PD_KERNEL(weight_only_linear_npu))
    .SetInferShapeFn(PD_INFER_SHAPE(
        WeightOnlyLinearInferShape));  // neccessary if the op has muti_inputs
