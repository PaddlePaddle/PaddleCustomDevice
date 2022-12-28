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
void HardshrinkKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const float lambd,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  const auto& runner =
      NpuOpRunner("HardShrink", {x}, {*out}, {{"lambd", lambd}});
  runner.Run(stream);
}

template <typename T, typename Context>
void HardshrinkGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& a,
                          const phi::DenseTensor& dout,
                          const float lambd,
                          phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  phi::DenseTensor tensor_low_bound;
  phi::DenseTensorMeta low_bound_meta = {dout.dtype(), dout.dims()};
  tensor_low_bound.set_meta(low_bound_meta);

  phi::DenseTensor tensor_up_bound;
  phi::DenseTensorMeta up_bound_meta = {dout.dtype(), dout.dims()};
  tensor_up_bound.set_meta(up_bound_meta);

  phi::DenseTensor tensor_low_bound_y;
  phi::DenseTensorMeta low_bound_y_meta = {paddle::experimental::DataType::BOOL,
                                           dout.dims()};
  tensor_low_bound_y.set_meta(low_bound_y_meta);

  phi::DenseTensor tensor_up_bound_y;
  phi::DenseTensorMeta up_bound_y_meta = {paddle::experimental::DataType::BOOL,
                                          dout.dims()};
  tensor_up_bound_y.set_meta(up_bound_y_meta);

  phi::DenseTensor tmp_out;
  phi::DenseTensorMeta out_meta = {paddle::experimental::DataType::BOOL,
                                   dout.dims()};
  tmp_out.set_meta(out_meta);
  dev_ctx.template Alloc<T>(&tensor_low_bound);
  dev_ctx.template Alloc<T>(&tensor_up_bound);
  dev_ctx.template Alloc<bool>(&tensor_low_bound_y);
  dev_ctx.template Alloc<bool>(&tensor_up_bound_y);

  const auto& runner = NpuOpRunner("OnesLike", {dout}, {tensor_low_bound}, {});
  runner.Run(stream);
  const auto& runner_1 = NpuOpRunner("OnesLike", {dout}, {tensor_up_bound}, {});
  runner_1.Run(stream);

  float neg_lambd = -lambd;
  const auto& runner_mul = NpuOpRunner(
      "Muls", {tensor_low_bound}, {tensor_low_bound}, {{"value", neg_lambd}});
  runner_mul.Run(stream);
  const auto& runner_mul_1 = NpuOpRunner(
      "Muls", {tensor_up_bound}, {tensor_up_bound}, {{"value", lambd}});
  runner_mul_1.Run(stream);

  const auto& runner_less =
      NpuOpRunner("Less", {a, tensor_low_bound}, {tensor_low_bound_y}, {});
  runner_less.Run(stream);
  const auto& runner_greater =
      NpuOpRunner("Greater", {a, tensor_up_bound}, {tensor_up_bound_y}, {});
  runner_greater.Run(stream);

  const auto& runner_or = NpuOpRunner("LogicalOr",
                                      {tensor_low_bound_y, tensor_up_bound_y},
                                      {tensor_low_bound_y},
                                      {});
  runner_or.Run(stream);

  const auto& runner_cast = NpuOpRunner(
      "Cast",
      {tensor_low_bound_y},
      {*dx},
      {{"dst_type", static_cast<int>(ConvertToNpuDtype(dx->dtype()))}});
  runner_cast.Run(stream);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(hard_shrink,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::HardshrinkKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(hard_shrink_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::HardshrinkGradKernel,
                          float,
                          phi::dtype::float16) {}
