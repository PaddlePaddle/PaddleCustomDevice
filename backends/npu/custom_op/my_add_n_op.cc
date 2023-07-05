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

std::vector<paddle::Tensor> MyAddNOp(const paddle::Tensor& x,
                                     const paddle::Tensor& y,
                                     const paddle::Tensor& z) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto y_tensor = static_cast<const phi::DenseTensor*>(y.impl().get());
  auto z_tensor = static_cast<const phi::DenseTensor*>(z.impl().get());

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(x_tensor->dims());
  dev_ctx->Alloc(out_tensor.get(), x_tensor->dtype());

  const auto& add_runner1 =
      NpuOpRunner("Add", {*x_tensor, *y_tensor}, {*out_tensor}, {});
  add_runner1.Run(stream);
  const auto& add_runner2 =
      NpuOpRunner("Add", {*out_tensor, *z_tensor}, {*out_tensor}, {});
  add_runner2.Run(stream);

  return {paddle::Tensor(out_tensor)};
}

std::vector<std::vector<int64_t>> MyAddNOpInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y_shape,
    const std::vector<int64_t>& z_shape) {
  return {x_shape};
}

PD_BUILD_OP(my_add_n)
    .Inputs({"X", "Y", "Z"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(MyAddNOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        MyAddNOpInferShape));  // neccessary if the op has muti_inputs
