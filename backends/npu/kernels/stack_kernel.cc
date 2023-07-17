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
void StackKernel(const Context& dev_ctx,
                 const std::vector<const phi::DenseTensor*>& x,
                 int axis,
                 phi::DenseTensor* y) {
  dev_ctx.template Alloc<T>(y);
  auto stream = dev_ctx.stream();

  if (axis < 0) axis += (x[0]->dims().size() + 1);
  int num = 0;
  std::vector<std::string> names;

  std::vector<phi::DenseTensor> x_list;
  for (int i = 0; i < x.size(); i++) {
    if (x[i] != nullptr) {
      x_list.push_back(*x[i]);
      names.push_back("x" + std::to_string(i));
      num++;
    } else {
      continue;
    }
  }

  PADDLE_ENFORCE_GT(
      num, 0, phi::errors::InvalidArgument("number of input Tensor <= 0"));

  NpuOpRunner runner;
  runner.SetType("Pack")
      .AddInputs(x_list)
      .AddOutput(*y)
      .AddAttr("axis", static_cast<int>(axis))
      .AddAttr("N", static_cast<int>(num));
  runner.AddInputNames(names);
  runner.Run(stream);
}

template <typename T, typename Context>
void StackGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& dy,
                     int axis,
                     std::vector<phi::DenseTensor*> dx) {
  auto stream = dev_ctx.stream();

  if (axis < 0) axis += dy.dims().size();
  int num = dy.dims()[axis];

  PADDLE_ENFORCE_GT(
      num, 0, phi::errors::InvalidArgument("number of input Tensor <= 0"));

  std::vector<phi::DenseTensor> dx_list;
  for (int i = 0; i < num; i++) {
    dev_ctx.template Alloc<T>(dx[i]);
    dx_list.push_back(*dx[i]);
  }

  const auto& runner =
      NpuOpRunner("Unpack", {dy}, {dx_list}, {{"axis", axis}, {"num", num}});
  runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(stack,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::StackKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(stack_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::StackGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}
