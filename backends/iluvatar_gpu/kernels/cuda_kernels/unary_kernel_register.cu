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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/sparse/impl/unary_kernel_impl.h"
#include "paddle/phi/kernels/sparse/unary_kernel.h"

#define PD_REGISTER_SPARSE_UNARY_GPU_KERNEL(name, prefix)          \
  PD_CUSTOM_KERNEL_REGISTER(name##_coo,                            \
                            iluvatar_gpu,                          \
                            ALL_LAYOUT,                            \
                            phi::sparse::prefix##CooKernel,        \
                            phi::dtype::float16,                   \
                            float,                                 \
                            double) {                              \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO); \
  }                                                                \
                                                                   \
  PD_CUSTOM_KERNEL_REGISTER(name##_csr,                            \
                            iluvatar_gpu,                          \
                            ALL_LAYOUT,                            \
                            phi::sparse::prefix##CsrKernel,        \
                            phi::dtype::float16,                   \
                            float,                                 \
                            double) {                              \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR); \
  }

#define PD_REGISTER_SPARSE_UNARY_GPU_KERNEL_WITH_COMPLEX(name, prefix) \
  PD_CUSTOM_KERNEL_REGISTER(name##_coo,                                \
                            iluvatar_gpu,                              \
                            ALL_LAYOUT,                                \
                            phi::sparse::prefix##CooKernel,            \
                            phi::dtype::float16,                       \
                            float,                                     \
                            double,                                    \
                            phi::dtype::complex<float>,                \
                            phi::dtype::complex<double>) {             \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);     \
  }                                                                    \
                                                                       \
  PD_CUSTOM_KERNEL_REGISTER(name##_csr,                                \
                            iluvatar_gpu,                              \
                            ALL_LAYOUT,                                \
                            phi::sparse::prefix##CsrKernel,            \
                            phi::dtype::float16,                       \
                            float,                                     \
                            double,                                    \
                            phi::dtype::complex<float>,                \
                            phi::dtype::complex<double>) {             \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);     \
  }

PD_CUSTOM_KERNEL_REGISTER(divide_scalar_coo,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::sparse::DivScalarCooKernel,
                          float,
                          double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_CUSTOM_KERNEL_REGISTER(divide_scalar_csr,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::sparse::DivScalarCsrKernel,
                          float,
                          double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_CUSTOM_KERNEL_REGISTER(cast_coo,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::sparse::CastCooKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          bool) {}

PD_CUSTOM_KERNEL_REGISTER(cast_csr,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::sparse::CastCsrKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          bool) {}

PD_CUSTOM_KERNEL_REGISTER(isnan_coo,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::sparse::IsnanCooKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          int,
                          int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_CUSTOM_KERNEL_REGISTER(isnan_csr,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::sparse::IsnanCsrKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          int,
                          int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
