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
void RollKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& shifts,
                const std::vector<int64_t>& axis,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  auto shifts_data = shifts.GetData();
  size_t nums = shifts_data.size();
  auto input_dim = x.dims();
  auto dims = axis;

  // axis = none, reshape to 1-D tensor
  if (dims.size() == 0) {
    dims.push_back(0l);
    input_dim = phi::Dim<1>(x.numel());
  }
  
  // handle negative dim, shift
  std::vector<int> shifts_in;
  std::vector<int> axis_in;

  for (auto i = 0; i < nums; ++i) {
    int a = dims[i];
    if (a < 0) {
      a += (input_dim.size());
    }
    axis_in.emplace_back(a);
    int sh = shifts_data[i] % input_dim[a]; 
    if (sh < 0) {
      sh += input_dim[a];
    }
    shifts_in.emplace_back(sh);
  }

  const auto& runner =
      NpuOpRunner("Roll", {x}, {*out}, {{"shifts", shifts_in}, {"dims", axis_in}});
  runner.Run(stream);
}

template <typename T, typename Context>
void RollGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& out_grad,
                    const phi::IntArray& shifts,
                    const std::vector<int64_t>& axis,
                    phi::DenseTensor* x_grad) {
  auto stream = dev_ctx.stream();

  auto shifts_data = shifts.GetData();
  dev_ctx.template Alloc<T>(x_grad);
  auto input_dim = x.dims();
  std::vector<int> xshape;
  size_t nums = shifts_data.size();
  for (int i = 0; i < input_dim.size(); ++i) {
    xshape.emplace_back(input_dim[i]);
  }

  auto dims = axis;

  // axis = none, reshape to 1-D tensor
  if (dims.size() == 0) {
    dims.push_back(0l);
    input_dim = phi::Dim<1>(x.numel());
  }
  std::vector<int> shifts_in;
  std::vector<int> axis_in;

  for (size_t i = 0; i < nums; ++i) {
    int a = dims[i];
    if (a < 0) {
      a += (input_dim.size());
    }
    axis_in.emplace_back(a);
    int sh = (0 - shifts_data[i]) % input_dim[a];
    if (sh < 0) {
      sh += input_dim[a];
    }
    shifts_in.emplace_back(sh);
  }

  const auto& runner =
      NpuOpRunner("Roll", {out_grad}, {*x_grad}, {{"shifts", shifts_in}, {"dims", axis_in}});
  runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(roll,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::RollKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(roll_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::RollGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}
