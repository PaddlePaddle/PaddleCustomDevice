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

#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void FlipKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const std::vector<int>& axis,
                phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  if (axis.size() == 0) {
    TensorCopy(dev_ctx, x, false, out);
  } else {
    MLUCnnlTensorDesc input_desc(x);
    MLUCnnlTensorDesc output_desc(*out);
    MLUCnnl::Flip(dev_ctx,
                  axis.data(),
                  axis.size(),
                  input_desc.get(),
                  GetBasePtr(&x),
                  output_desc.get(),
                  GetBasePtr(out));
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(flip,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::FlipKernel,
                          phi::dtype::float16,
                          float,
                          int32_t,
                          bool) {}
