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
void UnbindKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  int axis,
                  std::vector<phi::DenseTensor*> outs) {
  auto stream = dev_ctx.stream();

  auto x_dims = x.dims();
  axis = axis < 0 ? x_dims.size() + axis : axis;
  int num = x_dims[axis];
  auto split_dims = x.dims();
  split_dims[axis] = 1;

  std::vector<phi::DenseTensor> outs_t;
  for (size_t i = 0; i < outs.size(); ++i) {
    dev_ctx.template Alloc<T>(outs[i]);
    phi::DenseTensor out_tmp(*outs[i]);
    out_tmp.Resize(split_dims);
    outs_t.emplace_back(out_tmp);
  }
  NpuOpRunner runner;
  runner.SetType("Split")
      .AddInput(dev_ctx, std::vector<int>({axis}))
      .AddInput(x)
      .AddOutputs(outs_t)
      .AddAttr("num_split", num);
  runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(unbind,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::UnbindKernel,
                          float,
                          phi::dtype::float16,
                          int,
                          int64_t,
                          bool) {}
