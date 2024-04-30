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
void NPUIdentityKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const int format,
                       phi::DenseTensor* out) {
  if (format < 0) {
    dev_ctx.template Alloc<T>(out);
  } else {
    AllocNPUTensor<T>(dev_ctx, aclFormat(format), out);
  }

  auto stream = dev_ctx.stream();
  NpuOpRunner runner_identity;
  runner_identity.SetType("Identity").AddInput(x).AddOutput(*out).Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(npu_identity,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::NPUIdentityKernel,
                          float,
                          double,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
