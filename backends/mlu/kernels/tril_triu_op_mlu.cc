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

#include "kernels/funcs/mlu_baseop.h"

namespace custom_kernel {

template <typename T, typename Context>
void TrilTriuKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    int diagonal,
                    bool lower,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  bool upper;
  if (lower) {
    upper = 0;
  } else {
    upper = 1;
  }

  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnl::TrilTriu(dev_ctx,
                    diagonal,
                    upper,
                    x_desc.get(),
                    GetBasePtr(&x),
                    out_desc.get(),
                    GetBasePtr(out));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(tril_triu,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::TrilTriuKernel,
                          float,
                          int,
                          phi::dtype::float16) {}
