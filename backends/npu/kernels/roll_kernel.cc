// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
void RollKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& shifts,
                const std::vector<int64_t>& axis,
                phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();
  auto shifts_data = shifts.GetData();
  dev_ctx.template Alloc<T>(out);

  phi::DenseTensor shifts_v, axis_v;
  TensorFromVector<int64_t>(dev_ctx, shifts_data, dev_ctx, &shifts_v);
  TensorFromVector<int64_t>(dev_ctx, axis, dev_ctx, &axis_v);

  const auto& runner = NpuOpRunner("RollV2", {x, shifts_v, axis_v}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void RollGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x UNUSED,
                    const phi::DenseTensor& out_grad,
                    const phi::IntArray& shifts,
                    const std::vector<int64_t>& axis,
                    phi::DenseTensor* x_grad) {
  auto stream = dev_ctx.stream();
  std::vector<int64_t> shifts_data = shifts.GetData();
  dev_ctx.template Alloc<T>(x_grad);

  for (int i = 0; i < shifts_data.size(); ++i) {
    shifts_data[i] = 0 - shifts_data[i];
  }

  phi::DenseTensor shifts_v, axis_v;
  TensorFromVector<int64_t>(dev_ctx, shifts_data, dev_ctx, &shifts_v);
  TensorFromVector<int64_t>(dev_ctx, axis, dev_ctx, &axis_v);

  const auto& runner =
      NpuOpRunner("RollV2", {out_grad, shifts_v, axis_v}, {*x_grad}, {});
  runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(roll,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::RollKernel,
                          float,
                          phi::dtype::float16,
                          int) {}

PD_REGISTER_PLUGIN_KERNEL(roll_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::RollGradKernel,
                          float,
                          phi::dtype::float16,
                          int) {}
