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
void ShapeKernel(const Context& dev_ctx,
                 const phi::DenseTensor& input,
                 phi::DenseTensor* out) {
  auto& in_dims = input.dims();
  out->Resize({in_dims.size()});
  auto out_data = dev_ctx.template HostAlloc<int32_t>(out);
  for (int i = 0; i < in_dims.size(); ++i) {
    out_data[i] = in_dims[i];
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(shape,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ShapeKernel,
                          bool,
                          int,
                          int8_t,
                          uint8_t,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
