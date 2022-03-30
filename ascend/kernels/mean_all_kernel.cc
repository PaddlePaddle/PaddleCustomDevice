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

#include "npu_funcs.h"
#include "npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void MeanAllKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  std::vector<int> axes;
  NPUAttributeMap attr_input = {{"keep_dims", false}, {"axes", axes}};
  dev_ctx.template Alloc<T>(out);
  const auto& runner = NpuOpRunner("ReduceMeanD", {x}, {*out}, attr_input);
  auto stream = dev_ctx.stream();
  runner.Run(stream);
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

  dev_ctx.template Alloc<T>(x_grad);

  // ones
  phi::DenseTensor ones;
  phi::DenseTensorMeta meta_1 = {grad.dtype(), x_grad->dims()};
  ones.set_meta(meta_1);
  dev_ctx.template Alloc<T>(&ones);
  const auto& runner_ones = NpuOpRunner("OnesLike", {*x_grad}, {ones}, {});
  runner_ones.Run(stream);

  // means
  phi::DenseTensor mean_tensor;
  phi::DenseTensorMeta meta_2 = {grad.dtype(), {1}};
  mean_tensor.set_meta(meta_2);
  dev_ctx.template Alloc<T>(&mean_tensor);
  FillNpuTensorWithConstant<T>(
      &mean_tensor,
      dev_ctx,
      static_cast<T>(1.0 / static_cast<float>(x_grad->numel())));

  // means mul ones
  phi::DenseTensor mean_ma;
  phi::DenseTensorMeta meta_3 = {grad.dtype(), x_grad->dims()};
  mean_ma.set_meta(meta_3);
  dev_ctx.template Alloc<T>(&mean_ma);

  const auto& runner_mul_1 =
      NpuOpRunner("Mul", {mean_tensor, ones}, {mean_ma}, {});
  runner_mul_1.Run(stream);

  // and mul grad
  const auto& runner_mul_2 = NpuOpRunner("Mul", {mean_ma, grad}, {*x_grad}, {});
  runner_mul_2.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(mean_all,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::MeanAllKernel,
                          float,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(mean_all_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::MeanAllGradKernel,
                          float,
                          phi::dtype::bfloat16) {}
