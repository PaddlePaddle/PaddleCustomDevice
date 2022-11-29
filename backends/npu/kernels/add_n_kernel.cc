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
#include "kernels/funcs/op_command.h"

namespace custom_kernel {

template <typename T, typename Context>
void AddNKernel(const Context& dev_ctx,
                const std::vector<const phi::DenseTensor*>& x,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  int n = static_cast<int>(x.size());
  PADDLE_ENFORCE_GT(n,
                    0,
                    phi::errors::InvalidArgument(
                        "Expected size of Input(x) must greater than 0."));
  phi::DenseTensor tmp;
  tmp = *x[0];
  for (int i = 1; i < n; ++i) {
    phi::DenseTensor tmp_out;
    tmp_out.Resize(out->dims());
    dev_ctx.template Alloc<T>(&tmp_out);
    experimental::OpCommand("Add").Input(*x[i]).Input(tmp).Output(tmp_out).Run(
        dev_ctx);
    tmp = tmp_out;
  }
  *out = tmp;
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(add_n,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AddNKernel,
                          float,
                          phi::dtype::float16,
                          double) {}
