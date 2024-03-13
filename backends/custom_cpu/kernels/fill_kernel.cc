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
void FillKernel(const phi::Context& dev_ctx,
                const phi::Scalar& value,
                phi::DenseTensor* out) {
  double fill_var = value.to<double>();
  PD_CHECK(std::isnan(fill_var) == false,
           "fill value should not be NaN, but received NaN");

  auto t = dev_ctx.template Alloc<T>(out);
  T val = value.to<T>();
  for (auto i = 0; i < out->numel(); ++i) {
    t[i] = val;
  }
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(fill,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::FillKernel,
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
