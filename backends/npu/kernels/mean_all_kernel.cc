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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "kernels/funcs/op_command.h"

namespace custom_kernel {

template <typename T, typename Context>
extern void FullLikeKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const phi::Scalar& val,
                           phi::DataType dtype,
                           phi::DenseTensor* out);

template <typename T, typename Context>
void MeanAllKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  std::vector<int> axes;
  dev_ctx.template Alloc<T>(out);
  experimental::OpCommand("ReduceMeanD")
      .Input(x)
      .Output(*out)
      .Attr("keep_dims", false)
      .Attr("axes", axes)
      .Run(dev_ctx);
}

template <typename T, typename Context>
void MeanAllGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& grad,
                       phi::DenseTensor* x_grad) {
  PADDLE_ENFORCE_EQ(
      grad.numel(),
      1,
      phi::errors::InvalidArgument(
          "Mean Gradient Input phi::DenseTensor len should be 1. But "
          "received Out@Grad's elements num is %d.",
          grad.numel()));

  dev_ctx.template Alloc<T>(x_grad);

  phi::DenseTensor mean_tensor;
  phi::DenseTensor value_tensor;
  value_tensor.Resize({1});
  dev_ctx.template HostAlloc<T>(&value_tensor);
  *(value_tensor.data<T>()) =
      static_cast<T>(1.0 / static_cast<float>(x_grad->numel()));
  custom_kernel::FullLikeKernel<T, Context>(
      dev_ctx, *x_grad, value_tensor, x_grad->dtype(), &mean_tensor);

  experimental::OpCommand("Mul")
      .Input(mean_tensor,
             experimental::TensorDescMaker("x1")
                 .FromTensor(mean_tensor)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Input(grad,
             experimental::TensorDescMaker("x2").FromTensor(grad).SetDataLayout(
                 phi::DataLayout::ANY))

      .Output(
          *x_grad,
          experimental::TensorDescMaker("y").FromTensor(*x_grad).SetDataLayout(
              phi::DataLayout::ANY))
      .Run(dev_ctx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(mean_all,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MeanAllKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(mean_all_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MeanAllGradKernel,
                          float,
                          phi::dtype::float16) {}
