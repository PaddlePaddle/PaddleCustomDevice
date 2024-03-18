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

#include "kernels/phi_funcs.h"
#include "paddle/phi/capi/all.h"

namespace custom_kernel {

template <typename T>
void ContiguousKernel(const phi::Context& dev_ctx,
                      const phi::DenseTensor& input,
                      phi::DenseTensor* out) {
  out->set_strides(phi::CalcStrides(input.dims()));
  out->set_offset(0);

  const T* input_data = input.data<T>();
  T* output_data = dev_ctx.template Alloc<T>(out);
  int rank = input.dims().size();
  auto dims = input.dims();
  auto input_stride = input.strides();
  auto numel = input.numel();

  for (int64_t i = 0; i < numel; i++) {
    int64_t input_offset = 0;
    int64_t index_tmp = i;
    for (int dim = rank - 1; dim >= 0; --dim) {
      int64_t mod = index_tmp % dims[dim];
      index_tmp = index_tmp / dims[dim];
      input_offset += mod * input_stride[dim];
    }

    output_data[i] = input_data[input_offset];
  }
}
}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(contiguous,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::ContiguousKernel,
                    bool,
                    uint8_t,
                    int8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    float,
                    double,
                    phi::dtype::float16,
                    phi::dtype::bfloat16,
                    phi::dtype::complex<float>,
                    phi::dtype::complex<double>) {}
