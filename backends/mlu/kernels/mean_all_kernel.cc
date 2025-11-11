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
void MeanAllKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  auto rank = x.dims().size();
  if (rank == 0) {  // scalar
    TensorCopy(dev_ctx, x, false, out);
    return;
  }

  MLUReduceOp<T>(dev_ctx, x, {}, false, true, "reduce_mean", out);
}

template <typename T, typename Context>
void MeanAllGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& out_grad,
                       phi::DenseTensor* x_grad) {
  PADDLE_ENFORCE_EQ(out_grad.numel(),
                    1,
                    phi::errors::InvalidArgument(
                        "Mean Gradient Input Tensor len should be 1. But "
                        "received Out@Grad's elements num is %d.",
                        out_grad.numel()));
  dev_ctx.template Alloc<T>(x_grad);

  auto rank = x_grad->dims().size();
  if (rank == 0) {  // scalar
    TensorCopy(dev_ctx, out_grad, false, x_grad);
    return;
  }

  // means
  Tensor mean_var;
  mean_var.Resize(x_grad->dims());
  dev_ctx.template Alloc<T>(&mean_var);
  MLUCnnlTensorDesc mean_var_desc(
      mean_var, CNNL_LAYOUT_ARRAY, ToCnnlDataType(mean_var.dtype()));
  auto value = static_cast<T>(1.0 / static_cast<float>(x_grad->numel()));
  MLUCnnl::Fill(dev_ctx,
                CNNL_POINTER_MODE_HOST,
                &value,
                mean_var_desc.get(),
                GetBasePtr(&mean_var));

  // means mul out_grad
  MLUCnnlTensorDesc in_desc(
      out_grad, CNNL_LAYOUT_ARRAY, ToCnnlDataType(out_grad.dtype()));
  MLUCnnlTensorDesc out_desc(
      *x_grad, CNNL_LAYOUT_ARRAY, ToCnnlDataType(x_grad->dtype()));

  MLUCnnlOpTensorDesc op_tensor_desc(
      CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);

  MLUCnnl::OpTensor(dev_ctx,
                    op_tensor_desc.get(),
                    in_desc.get(),
                    GetBasePtr(&out_grad),
                    mean_var_desc.get(),
                    GetBasePtr(&mean_var),
                    out_desc.get(),
                    GetBasePtr(x_grad),
                    ToCnnlDataType<T>());
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(mean_all,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::MeanAllKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(mean_all_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::MeanAllGradKernel,
                          float,
                          phi::dtype::float16) {}
