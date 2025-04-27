/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/shape_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ShapeKernel(const Context& ctx,
                 const DenseTensor& input,
                 DenseTensor* out) {
  auto& in_dims = input.dims();
  out->Resize({in_dims.size()});
  auto out_data = ctx.template HostAlloc<int32_t>(out);
  for (int i = 0; i < in_dims.size(); ++i) {
    out_data[i] = in_dims[i];
  }
}

template <typename T, typename Context>
void Shape64Kernel(const Context& ctx,
                   const DenseTensor& input,
                   DenseTensor* out) {
  auto& in_dims = input.dims();
  out->Resize({in_dims.size()});
  auto out_data = ctx.template HostAlloc<int64_t>(out);
  for (int i = 0; i < in_dims.size(); ++i) {
    out_data[i] = in_dims[i];
  }
}

}  // namespace phi


PD_CUSTOM_KERNEL_REGISTER(shape,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ShapeKernel,
                          bool,
                          int,
                          int8_t,
                          uint8_t,
                          int64_t,
                          float,
                          phi::dtype::complex<float>,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
}

PD_CUSTOM_KERNEL_REGISTER(shape64,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::Shape64Kernel,
                          bool,
                          int,
                          int8_t,
                          uint8_t,
                          int64_t,
                          float,
                          phi::dtype::complex<float>,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}