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
#include "kernels/funcs/reduce_op.h"
namespace custom_kernel {

template <typename T, typename Context>
void WhereKernel(const Context& dev_ctx,
                 const phi::DenseTensor& condition,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc y_desc(y);
  MLUCnnlTensorDesc condition_desc(condition);
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnl::Select(dev_ctx,
                  condition_desc.get(),
                  GetBasePtr(&condition),
                  x_desc.get(),
                  GetBasePtr(&x),
                  y_desc.get(),
                  GetBasePtr(&y),
                  out_desc.get(),
                  GetBasePtr(out));
}

template <typename T, typename Context>
void WhereGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& condition,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     const phi::DenseTensor& out_grad,
                     phi::DenseTensor* x_grad,
                     phi::DenseTensor* y_grad) {
  if (x_grad != nullptr) {
    dev_ctx.template Alloc<T>(x_grad);
  }
  if (y_grad != nullptr) {
    dev_ctx.template Alloc<T>(y_grad);
  }

  MLUCnnlTensorDesc condition_desc(condition);
  MLUCnnlTensorDesc out_grad_desc(out_grad);

  Tensor tensor_zeros;
  tensor_zeros.Resize(out_grad.dims());
  dev_ctx.template Alloc<T>(&tensor_zeros);
  MLUCnnlTensorDesc tensor_zeros_desc(tensor_zeros);

  auto value = static_cast<T>(0);
  MLUCnnl::Fill(dev_ctx,
                CNNL_POINTER_MODE_HOST,
                &value,
                tensor_zeros_desc.get(),
                GetBasePtr(&tensor_zeros));

  if (x_grad != nullptr) {
    MLUCnnlTensorDesc x_grad_desc(*x_grad);
    MLUCnnl::Select(dev_ctx,
                    condition_desc.get(),
                    GetBasePtr(&condition),
                    out_grad_desc.get(),
                    GetBasePtr(&out_grad),
                    tensor_zeros_desc.get(),
                    GetBasePtr(&tensor_zeros),
                    x_grad_desc.get(),
                    GetBasePtr(x_grad));
  }
  if (y_grad != nullptr) {
    MLUCnnlTensorDesc y_grad_desc(*y_grad);
    MLUCnnl::Select(dev_ctx,
                    condition_desc.get(),
                    GetBasePtr(&condition),
                    tensor_zeros_desc.get(),
                    GetBasePtr(&tensor_zeros),
                    out_grad_desc.get(),
                    GetBasePtr(&out_grad),
                    y_grad_desc.get(),
                    GetBasePtr(y_grad));
  }
}  // namespace custom_kernel
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(where,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::WhereKernel,
                          int32_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(where_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::WhereGradKernel,
                          int32_t,
                          float,
                          phi::dtype::float16) {}
