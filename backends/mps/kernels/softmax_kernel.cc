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

#include "kernels/softmax_impl.h"
#include "paddle/phi/capi/all.h"

namespace custom_kernel {

template <typename T>
void SoftmaxKernel(const phi::Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int axis,
                   phi::DenseTensor* out) {
  const int rank = x.dims().size();
  // allocate memory on device.
  T* out_data = dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  if (rank == 0) {
    out_data[0] = static_cast<T>(1);
    return;
  }

  mps_kernel::Softmax(x.data<T>(), out->data<T>(), x.dims(), out->dims(), axis);
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(
    softmax, mps, ALL_LAYOUT, custom_kernel::SoftmaxKernel, float) {}
