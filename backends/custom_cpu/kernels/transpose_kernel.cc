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
void TransposeKernel(const phi::Context& ctx,
                     const phi::DenseTensor& x,
                     const std::vector<int>& axis,
                     phi::DenseTensor* out) {
  auto x_dims = x.dims();
  auto out_dims = out->dims();

  auto x_data = x.data<T>();
  auto out_data = ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }
  auto rank = x_dims.size();
  if (rank == 1) {
    memcpy(out_data, x_data, x.numel() * sizeof(T));
  }
  PD_CHECK(axis.size() == rank,
           "axis.size (%d) must be equal the rank of input (%d).",
           axis.size(),
           rank);

  std::vector<size_t> step(out_dims.size(), 1);
  for (auto i = out_dims.size() - 1; i > 0; --i) {
    step[i - 1] = step[i] * out_dims[i];
  }

  std::vector<size_t> index(rank, 0);
  for (auto i = 0; i < x.numel(); ++i) {
    std::vector<size_t> dst_index(rank, 0);
    for (auto j = 0; j < rank; ++j) {
      dst_index[j] = index[axis[j]];
    }
    out_data[phi::vec_product(dst_index, step)] = x_data[i];

    index.back()++;
    for (auto j = rank - 1; j > 0; --j) {
      if (index[j] >= x_dims[j]) {
        index[j - 1]++;
        index[j] = 0;
      } else {
        break;
      }
    }
  }
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(transpose,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::TransposeKernel,
                    bool,
                    float,
                    double,
                    uint8_t,
                    int8_t,
                    int16_t,
                    int32_t,
                    int64_t) {}
