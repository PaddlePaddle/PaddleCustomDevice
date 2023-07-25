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

#include "paddle/phi/capi/all.h"
#include "phi_funcs.h"  //NOLINT

namespace custom_kernel {

template <typename T>
void ConcatKernel(const phi::Context& dev_ctx,
                  const std::vector<const phi::DenseTensor*>& x,
                  const phi::Scalar& axis_scalar,
                  phi::DenseTensor* out) {
  int64_t axis = axis_scalar.to<int64_t>();
  if (axis < 0) {
    axis = axis + x[0]->dims().size();
  }
  std::vector<int64_t> out_dims(x[0]->dims().size(), 0);
  for (auto i = 0; i < x[0]->dims().size(); ++i) {
    if (axis == i) {
      for (auto j = 0; j < x.size(); ++j) {
        out_dims[i] += x[j]->dims()[i];
      }
    } else {
      out_dims[i] = x[0]->dims()[i];
    }
  }
  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);

  auto M = std::accumulate(
      out_dims.cbegin(), out_dims.cbegin() + axis, 1, std::multiplies<int>());
  auto N = std::accumulate(
      out_dims.cbegin() + axis + 1, out_dims.cend(), 1, std::multiplies<int>());
  auto out_offset = 0;
  for (auto i = 0; i < M; ++i) {
    for (auto j = 0; j < x.size(); ++j) {
      memcpy(out->data<T>() + out_offset,
             x[j]->data<T>() + i * N * x[j]->dims()[axis],
             (N * x[j]->dims()[axis]) * sizeof(T));
      out_offset += N * x[j]->dims()[axis];
    }
  }
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(concat,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::ConcatKernel,
                    float,
                    double,
                    int,
                    int64_t,
                    bool,
                    int8_t,
                    uint8_t,
                    phi::dtype::float16,
                    phi::dtype::bfloat16) {}
