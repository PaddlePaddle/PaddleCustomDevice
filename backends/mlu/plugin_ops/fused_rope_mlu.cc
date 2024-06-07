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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"
#include "paddle/extension.h"

std::vector<std::vector<int64_t>> fusedropeInferShape(
    std::vector<int64_t> shape) {
  return {shape};
}

std::vector<paddle::Tensor> mlufusedrope(const paddle::Tensor& query,
                                         const paddle::Tensor& cos,
                                         const paddle::Tensor& sin) {
  auto dev_ctx = static_cast<const custom_kernel::Context*>(
      paddle::experimental::DeviceContextPool::Instance().Get(query.place()));

  // query [batch_size, seq_len, num_heads, head_dim]
  auto query_tensor = static_cast<const phi::DenseTensor*>(query.impl().get());
  auto batch_size = query_tensor->dims()[0];
  auto seq_len = query_tensor->dims()[1];
  auto num_heads = query_tensor->dims()[2];
  auto head_dim = query_tensor->dims()[3];

  std::vector<int> perm{1, 0, 2, 3};
  std::shared_ptr<phi::DenseTensor> trans_query_tensor =
      std::make_shared<phi::DenseTensor>();
  trans_query_tensor->Resize(
      phi::make_ddim({seq_len, batch_size, num_heads, head_dim}));
  (*dev_ctx).Alloc(trans_query_tensor.get(), query_tensor->dtype());

  custom_kernel::MLUCnnlTensorDesc query_tensor_desc(*query_tensor);
  phi::DenseTensor* trans_query_tensor_temp;
  trans_query_tensor_temp = trans_query_tensor.get();
  custom_kernel::MLUCnnlTensorDesc trans_query_tensor_desc(
      *trans_query_tensor_temp);

  // transpose query form [batch_size, seq_len, num_heads, head_dim] to
  // [seq_len, batch_size, num_heads, head_dim]
  custom_kernel::MLUCnnl::Transpose(
      *dev_ctx,
      perm,
      query_tensor->dims().size(),
      query_tensor_desc.get(),
      custom_kernel::GetBasePtr(query_tensor),
      trans_query_tensor_desc.get(),
      custom_kernel::GetBasePtr(trans_query_tensor_temp));

  int64_t numel = query_tensor->numel();
  if (numel <= 0) {
    PADDLE_ENFORCE_GE(
        numel,
        0,
        phi::errors::InvalidArgument(
            "The number of the input 'query' for rope op must be a >=0 "
            "integer, but the value received is %d.",
            numel));
  }

  // query_out_temp
  std::shared_ptr<phi::DenseTensor> query_out_temp =
      std::make_shared<phi::DenseTensor>();

  // query_out_temp (seq_len, batch_size, num_heads, head_dim)
  query_out_temp->Resize(trans_query_tensor->dims());
  (*dev_ctx).Alloc(query_out_temp.get(), query_tensor->dtype());

  PADDLE_ENFORCE_EQ(head_dim % 64,
                    0,
                    phi::errors::InvalidArgument(
                        "The head_dim of input must be a multiple of 64."));
  auto cos_tensor = static_cast<phi::DenseTensor*>(cos.impl().get());
  auto sin_tensor = static_cast<phi::DenseTensor*>(sin.impl().get());

  // sin (1,seq_len,1,head_dim)
  auto sin_dims = sin_tensor->dims();
  auto cos_dims = cos_tensor->dims();
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

  phi::DenseTensor* query_out_temp_s;
  query_out_temp_s = query_out_temp.get();

  sin_tensor->Resize(phi::make_ddim({seq_len, head_dim}));
  cos_tensor->Resize(phi::make_ddim({seq_len, head_dim}));

  VLOG(4) << "query tensor dims: " << query_tensor->dims();
  VLOG(4) << "cos tensor dims: " << cos_tensor->dims();
  VLOG(4) << "sin tensor dims: " << sin_tensor->dims();

  custom_kernel::MLUCnnlTensorDesc cos_tensor_desc(*cos_tensor);
  custom_kernel::MLUCnnlTensorDesc sin_tensor_desc(*sin_tensor);
  custom_kernel::MLUCnnlTensorDesc query_out_temp_desc(*query_out_temp_s);

  bool conj = false;  // The value false indicates forward calculation
  custom_kernel::MLUCnnl::RotaryEmbedding(
      *dev_ctx,
      conj,
      trans_query_tensor_desc.get(),
      custom_kernel::GetBasePtr(trans_query_tensor_temp),
      nullptr,
      nullptr,
      cos_tensor_desc.get(),
      custom_kernel::GetBasePtr(cos_tensor),
      sin_tensor_desc.get(),
      custom_kernel::GetBasePtr(sin_tensor),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      query_out_temp_desc.get(),
      custom_kernel::GetBasePtr(query_out_temp_s),
      nullptr,
      nullptr);
  // query_out
  std::shared_ptr<phi::DenseTensor> query_out =
      std::make_shared<phi::DenseTensor>();

  // query_out (batch_size, seq_len, num_heads, head_dim)
  query_out->Resize(query_tensor->dims());
  (*dev_ctx).Alloc(query_out.get(), query_tensor->dtype());

  phi::DenseTensor* trans_query_out;
  trans_query_out = query_out.get();
  dev_ctx->Alloc(trans_query_out, query_tensor->dtype());

  custom_kernel::MLUCnnlTensorDesc trans_query_out_tensor_desc(
      *trans_query_out);
  // transpose query_out_temp_s form [seq_len, batch_size, num_heads, head_dim]
  // to [batch_size, seq_len, num_heads, head_dim]
  custom_kernel::MLUCnnl::Transpose(*dev_ctx,
                                    perm,
                                    query_tensor->dims().size(),
                                    query_out_temp_desc.get(),
                                    custom_kernel::GetBasePtr(query_out_temp_s),
                                    trans_query_out_tensor_desc.get(),
                                    custom_kernel::GetBasePtr(trans_query_out));

  query_out = std::make_unique<phi::DenseTensor>(*trans_query_out);
  return {paddle::Tensor(query_out)};
}

std::vector<paddle::Tensor> mlufusedropegrad(const paddle::Tensor& query,
                                             const paddle::Tensor& grad_query,
                                             const paddle::Tensor& cos,
                                             const paddle::Tensor& sin) {
  auto dev_ctx = static_cast<const custom_kernel::Context*>(
      paddle::experimental::DeviceContextPool::Instance().Get(query.place()));

  // query [batch_size, seq_len, num_heads, head_dim]
  auto query_tensor = static_cast<const phi::DenseTensor*>(query.impl().get());
  auto d_query = static_cast<phi::DenseTensor*>(grad_query.impl().get());

  auto batch_size = query_tensor->dims()[0];
  auto seq_len = query_tensor->dims()[1];
  auto num_heads = query_tensor->dims()[2];
  auto head_dim = query_tensor->dims()[3];

  std::vector<int> perm{1, 0, 2, 3};
  std::shared_ptr<phi::DenseTensor> trans_d_query =
      std::make_shared<phi::DenseTensor>();
  trans_d_query->Resize(
      phi::make_ddim({seq_len, batch_size, num_heads, head_dim}));
  (*dev_ctx).Alloc(trans_d_query.get(), query_tensor->dtype());

  custom_kernel::MLUCnnlTensorDesc d_query_desc(*d_query);
  phi::DenseTensor* trans_d_query_temp;
  trans_d_query_temp = trans_d_query.get();
  custom_kernel::MLUCnnlTensorDesc trans_d_query_desc(*trans_d_query_temp);
  // transpose d_query form [batch_size, seq_len, num_heads, head_dim] to
  // [seq_len, batch_size, num_heads, head_dim]
  custom_kernel::MLUCnnl::Transpose(
      *dev_ctx,
      perm,
      query_tensor->dims().size(),
      d_query_desc.get(),
      custom_kernel::GetBasePtr(d_query),
      trans_d_query_desc.get(),
      custom_kernel::GetBasePtr(trans_d_query_temp));

  auto sin_tensor = static_cast<phi::DenseTensor*>(sin.impl().get());
  auto cos_tensor = static_cast<phi::DenseTensor*>(cos.impl().get());

  // query_grad_temp
  std::shared_ptr<phi::DenseTensor> query_grad_temp =
      std::make_shared<phi::DenseTensor>();

  // query_grad_temp (seq_len, batch_size, num_heads, head_dim)
  query_grad_temp->Resize(trans_d_query->dims());
  (*dev_ctx).Alloc(query_grad_temp.get(), query_tensor->dtype());

  phi::DenseTensor* query_grad_temp_s;
  query_grad_temp_s = query_grad_temp.get();

  sin_tensor->Resize(phi::make_ddim({seq_len, head_dim}));
  cos_tensor->Resize(phi::make_ddim({seq_len, head_dim}));

  bool conj = true;  // The value true indicates backward calculation
  custom_kernel::MLUCnnlTensorDesc sin_tensor_desc(*sin_tensor);
  custom_kernel::MLUCnnlTensorDesc cos_tensor_desc(*cos_tensor);
  custom_kernel::MLUCnnlTensorDesc query_grad_temp_desc(*query_grad_temp_s);
  custom_kernel::MLUCnnl::RotaryEmbedding(
      *dev_ctx,
      conj,
      trans_d_query_desc.get(),
      custom_kernel::GetBasePtr(trans_d_query_temp),
      nullptr,
      nullptr,
      cos_tensor_desc.get(),
      custom_kernel::GetBasePtr(cos_tensor),
      sin_tensor_desc.get(),
      custom_kernel::GetBasePtr(sin_tensor),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      query_grad_temp_desc.get(),
      custom_kernel::GetBasePtr(query_grad_temp_s),
      nullptr,
      nullptr);
  // query_grad
  std::shared_ptr<phi::DenseTensor> query_grad =
      std::make_shared<phi::DenseTensor>();
  query_grad->Resize(query_tensor->dims());
  (*dev_ctx).Alloc(query_grad.get(), query_tensor->dtype());

  phi::DenseTensor* trans_query_grad_temp;
  trans_query_grad_temp = query_grad.get();
  (*dev_ctx).Alloc(trans_query_grad_temp, query_tensor->dtype());

  custom_kernel::MLUCnnlTensorDesc trans_query_grad_temp_desc(
      *trans_query_grad_temp);
  // transpose query_grad_temp form [seq_len, batch_size, num_heads, head_dim]
  // to [batch_size, seq_len, num_heads, head_dim]
  custom_kernel::MLUCnnl::Transpose(
      *dev_ctx,
      perm,
      query_tensor->dims().size(),
      query_grad_temp_desc.get(),
      custom_kernel::GetBasePtr(query_grad_temp_s),
      trans_query_grad_temp_desc.get(),
      custom_kernel::GetBasePtr(trans_query_grad_temp));

  query_grad = std::make_unique<phi::DenseTensor>(*trans_query_grad_temp);
  return {paddle::Tensor(query_grad)};
}

PD_BUILD_OP(fused_rope)
    .Inputs({"query", "cos", "sin"})
    .Outputs({"query_out"})
    .SetKernelFn(PD_KERNEL(mlufusedrope))
    .SetInferShapeFn(PD_INFER_SHAPE(fusedropeInferShape));

PD_BUILD_GRAD_OP(fused_rope)
    .Inputs({"query", paddle::Grad("query_out"), "cos", "sin"})
    .Outputs({paddle::Grad("query")})
    .SetKernelFn(PD_KERNEL(mlufusedropegrad))
    .SetInferShapeFn(PD_INFER_SHAPE(fusedropeInferShape));
