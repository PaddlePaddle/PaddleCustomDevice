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

phi::DenseTensor paddletensor2densortensor(const paddle::Tensor& paddletensor);

const phi::CustomContext* getcontext(const paddle::Tensor& tensor);
std::vector<std::vector<int64_t>> fusedropeInferShape(
    std::vector<int64_t> shape) {
  return {shape};
}

std::vector<paddle::Tensor> npufusedrope(const paddle::Tensor& query,
                                         const paddle::Tensor& cos,
                                         const paddle::Tensor& sin) {
  auto dev_ctx = getcontext(query);
  // query [batch_size, seq_len, num_heads, head_dim]
  auto query_tensor = paddletensor2densortensor(query);
  int64_t numel = query_tensor.numel();
  if (numel <= 0) {
    PADDLE_ENFORCE_GE(
        numel,
        0,
        phi::errors::InvalidArgument(
            "The number of the input 'query' for rope op must be a >=0 "
            "integer, but the value received is %d.",
            numel));
  }
  // query_out
  std::shared_ptr<phi::DenseTensor> query_out =
      std::make_shared<phi::DenseTensor>();
  query_out->Resize(query_tensor.dims());
  (*dev_ctx).Alloc(query_out.get(), query_tensor.dtype());
  auto head_dim = query_tensor.dims()[3];
  PADDLE_ENFORCE_EQ(head_dim % 64,
                    0,
                    phi::errors::InvalidArgument(
                        "The head_dim of input must be a multiple of 64."));
  auto sin_tensor = paddletensor2densortensor(sin);
  auto cos_tensor = paddletensor2densortensor(cos);
  // sin (1,seq_len,1,head_dim)
  auto sin_dims = sin_tensor.dims();
  auto cos_dims = cos_tensor.dims();
  int dims_size = sin_dims.size();
  PADDLE_ENFORCE_EQ(sin_dims,
                    cos_dims,
                    phi::errors::InvalidArgument(
                        "The dims of sin and cos must be the same. But "
                        "recieved sin's dims is {%s}, cos's dims is {%s}.",
                        sin_dims,
                        cos_dims));
  PADDLE_ENFORCE_EQ(
      dims_size == 4,
      true,
      phi::errors::InvalidArgument("The dims of sin and cos is expected to "
                                   "be  4, but recieved %d.",
                                   dims_size));
  PADDLE_ENFORCE_EQ(
      (sin_dims[0] == 1 && sin_dims[2] == 1),
      true,
      phi::errors::InvalidArgument(
          "The batch_size and num_heads of sin and cos must be 1."));
  auto stream = (*dev_ctx).stream();
  NpuOpRunner runner;
  runner.SetType("RotaryMul")
      .AddInput(query_tensor)
      .AddInput(cos_tensor)
      .AddInput(sin_tensor)
      .AddOutput(*(query_out.get()))
      .Run(stream);
  return {paddle::Tensor(query_out)};
}

std::vector<paddle::Tensor> npufusedropegrad(const paddle::Tensor& query,
                                             const paddle::Tensor& grad_query,
                                             const paddle::Tensor& cos,
                                             const paddle::Tensor& sin) {
  auto dev_ctx = getcontext(query);
  // query [batch_size, seq_len, num_heads, head_dim]
  auto query_tensor = paddletensor2densortensor(query);
  auto d_query = paddletensor2densortensor(grad_query);

  auto sin_tensor = paddletensor2densortensor(sin);
  auto cos_tensor = paddletensor2densortensor(cos);

  // query_grad
  std::shared_ptr<phi::DenseTensor> query_grad =
      std::make_shared<phi::DenseTensor>();
  query_grad->Resize(query_tensor.dims());
  (*dev_ctx).Alloc(query_grad.get(), query_tensor.dtype());

  // sin_grad
  std::shared_ptr<phi::DenseTensor> sin_grad =
      std::make_shared<phi::DenseTensor>();
  sin_grad->Resize(sin_tensor.dims());
  (*dev_ctx).Alloc(sin_grad.get(), sin_tensor.dtype());

  // cos_grad
  std::shared_ptr<phi::DenseTensor> cos_grad =
      std::make_shared<phi::DenseTensor>();
  cos_grad->Resize(cos_tensor.dims());
  (*dev_ctx).Alloc(cos_grad.get(), cos_tensor.dtype());

  auto stream = (*dev_ctx).stream();
  NpuOpRunner runner;
  runner.SetType("RotaryMulGrad")
      .AddInput(query_tensor)
      .AddInput(cos_tensor)
      .AddInput(sin_tensor)
      .AddInput(d_query)
      .AddOutput(*(query_grad.get()))
      .AddOutput(*(cos_grad.get()))
      .AddOutput(*(sin_grad.get()))
      .AddAttr("need_backward", false)
      .Run(stream);
  return {paddle::Tensor(query_grad)};
}

PD_BUILD_OP(fused_rope)
    .Inputs({"query", "cos", "sin"})
    .Outputs({"query_out"})
    .SetKernelFn(PD_KERNEL(npufusedrope))
    .SetInferShapeFn(PD_INFER_SHAPE(fusedropeInferShape));

PD_BUILD_GRAD_OP(fused_rope)
    .Inputs({"query", paddle::Grad("query_out"), "cos", "sin"})
    .Outputs({paddle::Grad("query")})
    .SetKernelFn(PD_KERNEL(npufusedropegrad))
    .SetInferShapeFn(PD_INFER_SHAPE(fusedropeInferShape));
