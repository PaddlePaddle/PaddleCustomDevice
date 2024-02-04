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

std::vector<std::vector<int64_t>> RmsNormInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& gamma_shape,
    float epsilon) {
  int x_dim = x_shape.size();
  int rstd_dim = x_dim - gamma_shape.size();
  std::vector<int64_t> rstd_shape(x_shape.begin(), x_shape.begin() + rstd_dim);
  rstd_shape.resize(x_dim, 1);
  return {x_shape, rstd_shape};
}

std::vector<std::vector<int64_t>> RmsNormGradInferShape(
    const std::vector<int64_t>& dy_shape,
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& rstd_shape,
    const std::vector<int64_t>& gamma_shape) {
  return {x_shape, gamma_shape};
}

std::vector<paddle::Tensor> npu_rms_norm(const paddle::Tensor& x,
                                         const paddle::Tensor& gamma,
                                         float epsilon) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto gamma_tensor = static_cast<const phi::DenseTensor*>(gamma.impl().get());

  std::shared_ptr<phi::DenseTensor> y_tensor =
      std::make_shared<phi::DenseTensor>();
  y_tensor->Resize(x_tensor->dims());
  dev_ctx->Alloc(y_tensor.get(), gamma_tensor->dtype());

  std::shared_ptr<phi::DenseTensor> rstd_tensor =
      std::make_shared<phi::DenseTensor>();
  auto x_shape = phi::vectorize(x_tensor->dims());
  auto gamma_shape = phi::vectorize(gamma_tensor->dims());
  int x_dim = x_shape.size();
  int rstd_dim = x_dim - gamma_shape.size();
  std::vector<int64_t> rstd_shape_vec(x_shape.begin(),
                                      x_shape.begin() + rstd_dim);
  rstd_shape_vec.resize(x_dim, 1);
  auto rstd_shape = phi::make_ddim(rstd_shape_vec);
  rstd_tensor->Resize(rstd_shape);
  dev_ctx->Alloc(rstd_tensor.get(), phi::DataType::FLOAT32);

  EXEC_NPU_CMD(aclnnRmsNorm,
               *dev_ctx,
               *x_tensor,
               *gamma_tensor,
               epsilon,
               *y_tensor,
               *rstd_tensor);
  return {paddle::Tensor(y_tensor), paddle::Tensor(rstd_tensor)};
}

std::vector<paddle::Tensor> npu_rms_norm_grad(const paddle::Tensor& dy,
                                              const paddle::Tensor& x,
                                              const paddle::Tensor& rstd,
                                              const paddle::Tensor& gamma) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto dy_tensor = static_cast<const phi::DenseTensor*>(dy.impl().get());
  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto rstd_tensor = static_cast<const phi::DenseTensor*>(rstd.impl().get());
  auto gamma_tensor = static_cast<const phi::DenseTensor*>(gamma.impl().get());

  std::shared_ptr<phi::DenseTensor> dx_tensor =
      std::make_shared<phi::DenseTensor>();
  dx_tensor->Resize(x_tensor->dims());
  dev_ctx->Alloc(dx_tensor.get(), x_tensor->dtype());

  std::shared_ptr<phi::DenseTensor> dgamma_tensor =
      std::make_shared<phi::DenseTensor>();
  dgamma_tensor->Resize(gamma_tensor->dims());
  dev_ctx->Alloc(dgamma_tensor.get(), phi::DataType::FLOAT32);

  EXEC_NPU_CMD(aclnnRmsNormGrad,
               *dev_ctx,
               *dy_tensor,
               *x_tensor,
               *rstd_tensor,
               *gamma_tensor,
               *dx_tensor,
               *dgamma_tensor);
  return {paddle::Tensor(dx_tensor), paddle::Tensor(dgamma_tensor)};
}

PD_BUILD_OP(rms_norm_npu)
    .Inputs({"x", "gamma"})
    .Outputs({"y", "rstd"})
    .Attrs({"epsilon: float"})
    .SetKernelFn(PD_KERNEL(npu_rms_norm))
    .SetInferShapeFn(PD_INFER_SHAPE(
        RmsNormInferShape));  // neccessary if the op has muti_inputs

PD_BUILD_GRAD_OP(rms_norm_npu)
    .Inputs({paddle::Grad("y"), "x", "rstd", "gamma"})
    .Outputs({paddle::Grad("x"), paddle::Grad("gamma")})
    .SetKernelFn(PD_KERNEL(npu_rms_norm_grad))
    .SetInferShapeFn(PD_INFER_SHAPE(
        RmsNormGradInferShape));  // neccessary if the op has muti_inputs
