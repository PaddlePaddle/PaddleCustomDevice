// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
// Reserved.
/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
// clang-format off
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/matmul_kernel_impl.h"


#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#if CUDA_VERSION >= 12010 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
PD_CUSTOM_KERNEL_REGISTER(matmul,
                   metax_gpu,
                   ALL_LAYOUT,
                   phi::MatmulKernel,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   phi::dtype::float8_e4m3fn,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   int8_t) {
#else
PD_CUSTOM_KERNEL_REGISTER(matmul,
  metax_gpu,
                   ALL_LAYOUT,
                   phi::MatmulKernel,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   int8_t) {
#endif
  if (kernel_key.dtype() == phi::DataType::INT8) {
    kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
  }
  if (kernel_key.dtype() == phi::DataType::FLOAT8_E4M3FN) {
    kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT16);
  }
}
#else
PD_CUSTOM_KERNEL_REGISTER(matmul,
  metax_gpu,
                   ALL_LAYOUT,
                   phi::MatmulKernel,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  if (kernel_key.dtype() == phi::DataType::INT8) {
    kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
  }
}
#endif
