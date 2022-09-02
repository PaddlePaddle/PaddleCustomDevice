/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void ArgMinKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  int dtype,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<int32_t>(out);

  NpuOpRunner runner;
  runner.SetType("ArgMin")
      .AddInput(x)
      .AddInput(dev_ctx, std::vector<int64_t>({axis.to<int64_t>()}))
      .AddOutput(*out)
      .AddAttr("dtype", dtype);

  auto stream = dev_ctx.stream();
  runner.Run(stream);
}

template <typename T, typename Context>
void ArgMaxKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  int dtype,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<int32_t>(out);

  phi::DenseTensor transformed_x(x);
  if (flatten) {
    transformed_x.Resize(phi::make_ddim({x.numel()}));
  }

  auto stream = dev_ctx.stream();
  NpuOpRunner runner;
  runner.SetType("ArgMaxD")
      .AddInput(transformed_x)
      .AddOutput(*out)
      .AddAttr("dimension", axis.to<int64_t>())
      .AddAttrDataType("dtype", dtype)
      .Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(arg_min,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::ArgMinKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(arg_max,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::ArgMaxKernel,
                          float,
                          phi::dtype::float16) {}
