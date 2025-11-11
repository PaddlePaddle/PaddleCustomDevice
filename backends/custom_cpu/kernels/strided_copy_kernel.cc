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
void StridedCopyKernel(const phi::Context& dev_ctx,
                       const phi::DenseTensor& input,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& out_stride,
                       int64_t offset,
                       phi::DenseTensor* out) {
  out->Resize(dims);
  out->set_strides(out_stride);
  out->set_offset(offset);

  PD_CHECK(input.dims().size() == out->dims().size(),
           "Input shape(%d) must be equal with out shape(%d).",
           input.dims().size(),
           out->dims().size());

  PD_CHECK(input.numel() == out->numel(),
           "Input numel(%d) must be equal with out numel(%d).",
           input.numel(),
           out->numel());

  if (input.numel() <= 0) {
    return;
  }

  const T* input_data = input.data<T>();
  int input_rank = input.dims().size();
  const int64_t* input_dims = input.dims().data();
  const int64_t* input_stride = input.strides().data();

  T* output_data = out->data<T>();
  PD_CHECK(output_data != nullptr,
           "StridedCopyKernel's out tensor must complete "
           "mutable data before call kernel.");

  int output_rank = input.dims().size();
  const int64_t* output_dims = input.dims().data();
  const int64_t* output_stride = input.dims().data();

  auto numel = input.numel();

  for (int64_t i = 0; i < numel; i++) {
    int64_t input_offset = 0;
    int64_t index_tmp = i;
    for (int dim = input_rank - 1; dim >= 0; --dim) {
      input_offset += (index_tmp % input_dims[dim]) * input_stride[dim];
      index_tmp = index_tmp / input_dims[dim];
    }
    int64_t output_offset = 0;
    index_tmp = i;
    for (int dim = output_rank - 1; dim >= 0; --dim) {
      output_offset += (index_tmp % output_dims[dim]) * output_stride[dim];
      index_tmp = index_tmp / output_dims[dim];
    }
    output_data[output_offset] = input_data[input_offset];
  }
}
}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(strided_copy,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::StridedCopyKernel,
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
