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
void MeanAllKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  std::vector<int> axes;
  ACL_RUN(dev_ctx.template Alloc<T>(out));
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
  auto stream = dev_ctx.stream();

  PADDLE_ENFORCE_EQ(
      grad.numel(),
      1,
      phi::errors::InvalidArgument(
          "Mean Gradient Input phi::DenseTensor len should be 1. But "
          "received Out@Grad's elements num is %d.",
          grad.numel()));

  ACL_RUN(dev_ctx.template Alloc<T>(x_grad));
  // ones
  phi::DenseTensor ones;
  phi::DenseTensorMeta ones_meta = {grad.dtype(), x_grad->dims()};
  ones.set_meta(ones_meta);
  ACL_RUN(dev_ctx.template Alloc<T>(&ones));
  experimental::OpCommand("OnesLike").Input(*x_grad).Output(ones).Run(dev_ctx);

  // means
  phi::DenseTensor mean_tensor;
  phi::DenseTensorMeta mean_meta = {grad.dtype(), {1}};
  mean_tensor.set_meta(mean_meta);
  ACL_RUN({
    dev_ctx.template Alloc<T>(&mean_tensor);
    FillNpuTensorWithConstant<T>(
        &mean_tensor,
        dev_ctx,
        static_cast<T>(1.0 / static_cast<float>(x_grad->numel())));
  });

  // means mul ones and mul grad
  phi::DenseTensor mean_ma;
  phi::DenseTensorMeta mean_ma_meta = {grad.dtype(), x_grad->dims()};
  mean_ma.set_meta(mean_ma_meta);
  ACL_RUN(dev_ctx.template Alloc<T>(&mean_ma));
  experimental::OpCommand("Mul")
      .Input(mean_tensor)
      .Input(ones)
      .Output(mean_ma)
      .Run(dev_ctx);
  experimental::OpCommand("Mul").Input(mean_ma).Input(grad).Output(*x_grad).Run(
      dev_ctx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(mean_all,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MeanAllKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(mean_all_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MeanAllGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
