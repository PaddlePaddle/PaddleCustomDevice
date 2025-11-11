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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"
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

std::vector<paddle::Tensor> mlu_rms_norm(const paddle::Tensor& x,
                                         const paddle::Tensor& gamma,
                                         float epsilon) {
  auto dev_ctx = static_cast<const custom_kernel::Context*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto gamma_tensor = static_cast<const phi::DenseTensor*>(gamma.impl().get());

  std::shared_ptr<phi::DenseTensor> y_tensor =
      std::make_shared<phi::DenseTensor>();
  phi::DenseTensor* y_tensor_s;
  y_tensor->Resize(x_tensor->dims());
  dev_ctx->Alloc(y_tensor.get(), gamma_tensor->dtype());
  y_tensor_s = y_tensor.get();

  std::shared_ptr<phi::DenseTensor> rstd_tensor =
      std::make_shared<phi::DenseTensor>();
  phi::DenseTensor* rstd_tensor_s;
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
  rstd_tensor_s = rstd_tensor.get();
  double eps = epsilon;

  custom_kernel::MLUCnnlTensorDesc x_desc(*x_tensor);
  custom_kernel::MLUCnnlTensorDesc gamma_desc(*gamma_tensor);
  custom_kernel::MLUCnnlTensorDesc y_desc(*y_tensor_s);
  custom_kernel::MLUCnnlTensorDesc rstd_desc(*rstd_tensor_s);
  custom_kernel::MLUCnnl::RmsNormForward(
      *dev_ctx,
      rstd_dim,
      x_desc.get(),
      custom_kernel::GetBasePtr(x_tensor),
      gamma_desc.get(),
      custom_kernel::GetBasePtr(gamma_tensor),
      nullptr,
      eps,
      y_desc.get(),
      custom_kernel::GetBasePtr(y_tensor_s),
      rstd_desc.get(),
      custom_kernel::GetBasePtr(rstd_tensor_s));
  y_tensor = std::make_unique<phi::DenseTensor>(*y_tensor_s);
  rstd_tensor = std::make_unique<phi::DenseTensor>(*rstd_tensor_s);
  return {paddle::Tensor(y_tensor), paddle::Tensor(rstd_tensor)};
}

std::vector<paddle::Tensor> mlu_rms_norm_grad(const paddle::Tensor& dy,
                                              const paddle::Tensor& x,
                                              const paddle::Tensor& rstd,
                                              const paddle::Tensor& gamma) {
  auto dev_ctx = static_cast<const custom_kernel::Context*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto dy_tensor = static_cast<const phi::DenseTensor*>(dy.impl().get());
  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto rstd_tensor = static_cast<const phi::DenseTensor*>(rstd.impl().get());
  auto gamma_tensor = static_cast<const phi::DenseTensor*>(gamma.impl().get());

  std::shared_ptr<phi::DenseTensor> dx_tensor =
      std::make_shared<phi::DenseTensor>();
  dx_tensor->Resize(x_tensor->dims());
  dev_ctx->Alloc(dx_tensor.get(), x_tensor->dtype());
  phi::DenseTensor* dx_tensor_s;
  dx_tensor_s = dx_tensor.get();

  std::shared_ptr<phi::DenseTensor> dgamma_tensor =
      std::make_shared<phi::DenseTensor>();
  dgamma_tensor->Resize(gamma_tensor->dims());
  dev_ctx->Alloc(dgamma_tensor.get(), x_tensor->dtype());
  phi::DenseTensor* dgamma_tensor_s;
  dgamma_tensor_s = dgamma_tensor.get();

  auto x_shape = phi::vectorize(x_tensor->dims());
  auto gamma_shape = phi::vectorize(gamma_tensor->dims());
  int x_dim = x_shape.size();
  int rstd_dim = x_dim - gamma_shape.size();
  custom_kernel::MLUCnnlTensorDesc dy_tensor_desc(*dy_tensor);
  custom_kernel::MLUCnnlTensorDesc x_tensor_desc(*x_tensor);
  custom_kernel::MLUCnnlTensorDesc rstd_tensor_desc(*rstd_tensor);
  custom_kernel::MLUCnnlTensorDesc gamma_tensor_desc(*gamma_tensor);
  custom_kernel::MLUCnnlTensorDesc dx_tensor_desc(*dx_tensor_s);
  custom_kernel::MLUCnnlTensorDesc dgamma_tensor_desc(*dgamma_tensor_s);

  custom_kernel::MLUCnnl::RmsNormBackward(
      *dev_ctx,
      rstd_dim,
      x_tensor_desc.get(),
      custom_kernel::GetBasePtr(x_tensor),
      dy_tensor_desc.get(),
      custom_kernel::GetBasePtr(dy_tensor),
      gamma_tensor_desc.get(),
      custom_kernel::GetBasePtr(gamma_tensor),
      rstd_tensor_desc.get(),
      custom_kernel::GetBasePtr(rstd_tensor),
      dx_tensor_desc.get(),
      custom_kernel::GetBasePtr(dx_tensor_s),
      dgamma_tensor_desc.get(),
      custom_kernel::GetBasePtr(dgamma_tensor_s),
      nullptr);

  dx_tensor = std::make_unique<phi::DenseTensor>(*dx_tensor_s);
  dgamma_tensor = std::make_unique<phi::DenseTensor>(*dgamma_tensor_s);

  return {paddle::Tensor(dx_tensor), paddle::Tensor(dgamma_tensor)};
}

PD_BUILD_OP(rms_norm_mlu)
    .Inputs({"x", "gamma"})
    .Outputs({"y", "rstd"})
    .Attrs({"epsilon: float"})
    .SetKernelFn(PD_KERNEL(mlu_rms_norm))
    .SetInferShapeFn(PD_INFER_SHAPE(
        RmsNormInferShape));  // neccessary if the op has muti_inputs

PD_BUILD_GRAD_OP(rms_norm_mlu)
    .Inputs({paddle::Grad("y"), "x", "rstd", "gamma"})
    .Outputs({paddle::Grad("x"), paddle::Grad("gamma")})
    .SetKernelFn(PD_KERNEL(mlu_rms_norm_grad))
    .SetInferShapeFn(PD_INFER_SHAPE(
        RmsNormGradInferShape));  // neccessary if the op has muti_inputs
