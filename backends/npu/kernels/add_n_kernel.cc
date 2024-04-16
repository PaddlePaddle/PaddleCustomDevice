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
void AddNKernel(const Context& dev_ctx,
                const std::vector<const phi::DenseTensor*>& x,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  int n = static_cast<int>(x.size());
  if (n == 1) {
    TensorCopy(dev_ctx, *x[0], false, out);
    return;
  }

  std::vector<phi::DenseTensor> inputs;
  std::vector<std::string> names;
  int actual_n = 0;
  for (int i = 0; i < n; ++i) {
    if (x[i] && x[i]->numel() > 0) {
      inputs.push_back(*x[i]);
      names.push_back("x" + std::to_string(i));
      ++actual_n;
    }
  }

  NpuOpRunner runner{"AddN", {inputs}, {*out}, {{"N", actual_n}}};
  runner.AddInputNames(names);
  runner.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(add_n,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AddNKernel,
                          float,
                          phi::dtype::float16,
                          double) {}
