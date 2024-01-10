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

namespace custom_kernel {
template <typename T, typename Context>
void SquaredL2NormKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         phi::DenseTensor* out) {
  std::vector<int> axis;
  for (int i = 0; i < x.dims().size(); ++i) {
    axis.push_back(i);
  }
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  const auto& runner = NpuOpRunner(
      "SquareSumV1", {x}, {*out}, {{"axis", axis}, {"keep_dims", false}});
  runner.Run(stream);
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

  // auto place = context.GetPlace();
  auto stream = dev_ctx.stream();

  // broadcast out_grad
  phi::DenseTensor broadcasted_out_grad;
  phi::DenseTensorMeta broadcasted_meta = {x_grad->dtype(), x_grad->dims()};
  broadcasted_out_grad.set_meta(broadcasted_meta);
  dev_ctx.template Alloc<T>(&broadcasted_out_grad);
  phi::DenseTensor shape_tensor;
  dev_ctx.template Alloc<T>(&shape_tensor);
  std::vector<int64_t> input_dims = phi::vectorize(x_grad->dims());
  custom_kernel::TensorFromVector(dev_ctx, input_dims, dev_ctx, &shape_tensor);
  const auto& broadcast_runner = NpuOpRunner(
      "BroadcastTo", {out_grad, shape_tensor}, {broadcasted_out_grad}, {});
  broadcast_runner.Run(stream);
  // mul x
  phi::DenseTensor tmp_x_grad;
  phi::DenseTensorMeta tmp_meta = {x_grad->dtype(), x_grad->dims()};
  tmp_x_grad.set_meta(tmp_meta);
  dev_ctx.template Alloc<T>(&tmp_x_grad);
  const auto& mul_x_runner =
      NpuOpRunner("Mul", {broadcasted_out_grad, x}, {tmp_x_grad}, {});
  mul_x_runner.Run(stream);
  // mul coefficient:2
  phi::DenseTensor coefficient;
  phi::DenseTensorMeta coefficient_meta = {x_grad->dtype(), {1}};
  coefficient.set_meta(coefficient_meta);
  dev_ctx.template Alloc<T>(&coefficient);
  FillNpuTensorWithConstant<T>(&coefficient, dev_ctx, static_cast<T>(2.0));
  dev_ctx.template Alloc<T>(x_grad);
  const auto& mul_coefficient_runner =
      NpuOpRunner("Mul", {tmp_x_grad, coefficient}, {*x_grad}, {});
  mul_coefficient_runner.Run(stream);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(squared_l2_norm,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SquaredL2NormKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(squared_l2_norm_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SquaredL2NormGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
