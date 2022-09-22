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
void UnStackKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int axis,
                   int num,
                   std::vector<phi::DenseTensor*> outs) {
  auto stream = dev_ctx.stream();

  if (axis < 0) axis += x.dims().size();
  num = x.dims()[axis];

  std::vector<phi::DenseTensor> dx_list;
  for (int i = 0; i < num; i++) {
    dev_ctx.template Alloc<T>(outs[i]);
    dx_list.push_back(*outs[i]);
  }

  const auto& runner =
      NpuOpRunner("Unpack", {x}, {dx_list}, {{"axis", axis}, {"num", num}});
  runner.Run(stream);
}

template <typename T, typename Context>
void UnStackGradKernel(const Context& dev_ctx,
                       const std::vector<const phi::DenseTensor*>& x,
                       int axis,
                       phi::DenseTensor* outs) {
  dev_ctx.template Alloc<T>(outs);
  auto stream = dev_ctx.stream();

  if (axis < 0) axis += (x[0]->dims().size() + 1);
  int num = static_cast<int>(x.size());

  std::vector<phi::DenseTensor> x_list;
  for (int i = 0; i < num; i++) {
    x_list.push_back(*x[i]);
  }
  const auto& runner =
      NpuOpRunner("Pack", {x_list}, {*outs}, {{"axis", axis}, {"N", num}});
  runner.Run(stream);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    unstack, ascend, ALL_LAYOUT, custom_kernel::UnStackKernel, float, double) {}

PD_REGISTER_PLUGIN_KERNEL(unstack_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::UnStackGradKernel,
                          float,
                          double) {}
