// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {
template <typename T, typename Context>
void SquaredL2NormKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc out_desc(*out);

  // L2Loss
  MLUCnnl::L2Loss(dev_ctx, input_desc.get(), GetBasePtr(&x), GetBasePtr(out));

  // do mul
  phi::DenseTensor scale_tensor;
  scale_tensor.Resize({1});
  dev_ctx.template Alloc<T>(&scale_tensor);

  phi::DenseTensor bias_tensor;
  bias_tensor.Resize({1});
  dev_ctx.template Alloc<T>(&bias_tensor);

  MLUCnnlTensorDesc scale_desc(scale_tensor);
  MLUCnnlTensorDesc bias_desc(bias_tensor);
  FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(2.0f), &scale_tensor);
  FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0.0f), &bias_tensor);

  MLUCnnl::Scale(dev_ctx,
                 0,
                 out_desc.get(),
                 GetBasePtr(out),
                 scale_desc.get(),
                 GetBasePtr(&scale_tensor),
                 bias_desc.get(),
                 GetBasePtr(&bias_tensor),
                 out_desc.get(),
                 GetBasePtr(out));
}

template <typename T, typename Context>
void SquaredL2NormGradKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::DenseTensor& out_grad,
                             phi::DenseTensor* x_grad) {
  PADDLE_ENFORCE_EQ(
      out_grad.numel(),
      1,
      phi::errors::InvalidArgument(
          "Input(GRAD@Out) of SquaredL2NormGradOP should be a scalar."));

  // broadcast out_grad
  Tensor broadcasted_out_grad;
  broadcasted_out_grad.Resize(x_grad->dims());
  dev_ctx.template Alloc<T>(&broadcasted_out_grad);
  MLUCnnlTensorDesc broadcasted_out_grad_desc(broadcasted_out_grad);
  MLUCnnlTensorDesc out_grad_desc(out_grad);
  MLUCnnl::BroadcastTo(dev_ctx,
                       out_grad_desc.get(),
                       GetBasePtr(&out_grad),
                       broadcasted_out_grad_desc.get(),
                       GetBasePtr(&broadcasted_out_grad));

  // mul x
  Tensor tmp_x_grad;
  tmp_x_grad.Resize(x_grad->dims());
  dev_ctx.template Alloc<T>(&tmp_x_grad);

  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc tmp_x_grad_desc(tmp_x_grad);
  MLUCnnlOpTensorDesc mul_op_desc(
      CNNL_OP_TENSOR_MUL, ToCnnlDataType(x.dtype()), CNNL_NOT_PROPAGATE_NAN);
  MLUCnnl::OpTensor(dev_ctx,
                    mul_op_desc.get(),
                    x_desc.get(),
                    GetBasePtr(&x),
                    broadcasted_out_grad_desc.get(),
                    GetBasePtr(&broadcasted_out_grad),
                    tmp_x_grad_desc.get(),
                    GetBasePtr(&tmp_x_grad),
                    ToCnnlDataType(x.dtype()));

  // mul
  phi::DenseTensor scale_tensor;
  scale_tensor.Resize({1});
  dev_ctx.template Alloc<T>(&scale_tensor);

  phi::DenseTensor bias_tensor;
  bias_tensor.Resize({1});
  dev_ctx.template Alloc<T>(&bias_tensor);

  MLUCnnlTensorDesc scale_desc(scale_tensor);
  MLUCnnlTensorDesc bias_desc(bias_tensor);
  FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(2.0f), &scale_tensor);
  FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0.0f), &bias_tensor);

  dev_ctx.template Alloc<T>(x_grad);
  MLUCnnlTensorDesc x_grad_desc(*x_grad);
  MLUCnnl::Scale(dev_ctx,
                 0,
                 tmp_x_grad_desc.get(),
                 GetBasePtr(&tmp_x_grad),
                 scale_desc.get(),
                 GetBasePtr(&scale_tensor),
                 bias_desc.get(),
                 GetBasePtr(&bias_tensor),
                 x_grad_desc.get(),
                 GetBasePtr(x_grad));
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(squared_l2_norm,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SquaredL2NormKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(squared_l2_norm_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SquaredL2NormGradKernel,
                          float,
                          phi::dtype::float16) {}
