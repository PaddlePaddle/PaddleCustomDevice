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

#include "funcs.h"
#include "hpu_operator.h"
#include "paddle/extension.h"

std::vector<std::vector<int64_t>> RmsNormInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& gamma_shape,
    float epsilon) {
    int x_dim = x_shape.size();
    int rstd_dim = x_dim - gamma_shape.size();
    std::vector<int64_t> rstd_shape(x_shape.begin(), x_shape.begin() +
    rstd_dim); rstd_shape.resize(x_dim, 1);
  return {x_shape, rstd_shape};
}

std::vector<paddle::Tensor> intel_hpu_rms_norm(const paddle::Tensor& x,
                                               const paddle::Tensor& gamma,
                                               float epsilon) {
  //   auto dev_ctx = static_cast<const phi::CustomContext*>(
  //       paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  //   auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  //   auto gamma_tensor = static_cast<const
  //   phi::DenseTensor*>(gamma.impl().get());

  //   std::shared_ptr<phi::DenseTensor> y_tensor =
  //       std::make_shared<phi::DenseTensor>();
  //   y_tensor->Resize(x_tensor->dims());
  //   dev_ctx->Alloc(y_tensor.get(), gamma_tensor->dtype());

  //   std::shared_ptr<phi::DenseTensor> rstd_tensor =
  //       std::make_shared<phi::DenseTensor>();
  //   auto x_shape = phi::vectorize(x_tensor->dims());
  //   auto gamma_shape = phi::vectorize(gamma_tensor->dims());
  //   int x_dim = x_shape.size();
  //   int rstd_dim = x_dim - gamma_shape.size();
  //   std::vector<int64_t> rstd_shape_vec(x_shape.begin(),
  //                                       x_shape.begin() + rstd_dim);
  //   rstd_shape_vec.resize(x_dim, 1);
  //   auto rstd_shape = phi::make_ddim(rstd_shape_vec);
  //   rstd_tensor->Resize(rstd_shape);
  //   dev_ctx->Alloc(rstd_tensor.get(), phi::DataType::FLOAT32);
  //   double eps = epsilon;

  //   EXEC_NPU_CMD(aclnnRmsNorm,
  //                *dev_ctx,
  //                *x_tensor,
  //                *gamma_tensor,
  //                eps,
  //                *y_tensor,
  //                *rstd_tensor);
  //   return {paddle::Tensor(y_tensor), paddle::Tensor(rstd_tensor)};
}

PD_BUILD_OP(rms_norm_intel_hpu)
    .Inputs({"x", "gamma"})
    .Outputs({"y", "rstd"})
    .Attrs({"epsilon: float"})
    .SetKernelFn(PD_KERNEL(intel_hpu_rms_norm));
