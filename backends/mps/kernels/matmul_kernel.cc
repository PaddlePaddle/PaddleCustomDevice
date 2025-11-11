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

#include "kernels/matmul_impl.h"
#include "paddle/phi/capi/all.h"

namespace custom_kernel {

template <typename T>
void MatmulKernel(const phi::Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  bool transpose_x,
                  bool transpose_y,
                  phi::DenseTensor* out) {
  T* out_data = dev_ctx.template Alloc<T>(out);
  mps_kernel::Matmul(x.data<T>(),
                     y.data<T>(),
                     out->data<T>(),
                     x.dims(),
                     y.dims(),
                     transpose_x,
                     transpose_y);
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(
    matmul, mps, ALL_LAYOUT, custom_kernel::MatmulKernel, float) {}
