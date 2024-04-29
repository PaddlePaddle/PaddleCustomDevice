// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void ArangeRawKernel(const Context& dev_ctx,
                     const T start_value,
                     const T end_value,
                     const T step_value,
                     phi::DenseTensor* out) {
  int64_t size = 0;
  GetSize(start_value, end_value, step_value, &size);

  out->Resize(phi::make_ddim({size}));
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc output_desc(*out);
  MLUCnnl::Range(dev_ctx,
                 &start_value,
                 &end_value,
                 &step_value,
                 output_desc.get(),
                 GetBasePtr(out));
}

}  // namespace custom_kernel
