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

void CheckAxis(int axis, int rank) {
  // check the axis is in [-rank, rank-1]
  PADDLE_ENFORCE_EQ(axis >= -rank && axis <= rank - 1,
                    true,
                    phi::errors::InvalidArgument(
                        "axis in norm operator must between (%d) and (%d)"
                        "but got (%d)",
                        -rank,
                        rank - 1,
                        axis));
}

template <typename T, typename Context>
void NormKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                int axis,
                float eps,
                bool is_test,
                phi::DenseTensor* out_y,
                phi::DenseTensor* out_norm) {
  dev_ctx.template Alloc<T>(out_y);
  dev_ctx.template Alloc<T>(out_norm);
  auto xdim = x.dims();
  CheckAxis(axis, xdim.size());
  axis = axis < 0 ? axis + xdim.size() : axis;

  NPUAttributeMap attr_input_norm;
  attr_input_norm["axes"] = std::vector<int>({axis});
  attr_input_norm["p"] = 2;
  attr_input_norm["keepdim"] = true;
  attr_input_norm["epsilon"] = eps;
  const auto& runner = NpuOpRunner("LpNorm", {x}, {*out_norm}, attr_input_norm);
  auto stream = dev_ctx.stream();
  runner.Run(stream);
  NpuOpRunner("Div", {x, *out_norm}, {*out_y}, {}).Run(stream);
}

template <typename T, typename Context>
void NormGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    const phi::DenseTensor& dy,
                    int axis,
                    float eps,
                    bool is_test,
                    phi::DenseTensor* dx) {
  auto xdim = x.dims();
  CheckAxis(axis, xdim.size());
  axis = axis < 0 ? axis + xdim.size() : axis;

  dev_ctx.template Alloc<T>(dx);

  auto stream = dev_ctx.stream();
  NPUAttributeMap attr_input_norm;
  attr_input_norm["dim"] = std::vector<int>({axis});
  attr_input_norm["eps"] = eps;

  phi::DenseTensor tmp;
  tmp.Resize(x.dims());
  dev_ctx.template Alloc<T>(&tmp);
  NpuOpRunner("Div", {x, y}, {tmp}, {}).Run(stream);
  const auto& runner =
      NpuOpRunner("L2NormalizeGrad", {x, tmp, dy}, {*dx}, attr_input_norm);
  runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(norm,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::NormKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(norm_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::NormGradKernel,
                          float,
                          phi::dtype::float16) {}
