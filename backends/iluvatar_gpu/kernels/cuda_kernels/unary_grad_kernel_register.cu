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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/sparse/impl/unary_grad_kernel_impl.h"
#include "paddle/phi/kernels/sparse/unary_grad_kernel.h"

#define PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL(name, prefix)     \
  PD_CUSTOM_KERNEL_REGISTER(name##_coo_grad,                       \
                            iluvatar_gpu,                          \
                            ALL_LAYOUT,                            \
                            phi::sparse::prefix##CooGradKernel,    \
                            phi::dtype::float16,                   \
                            float,                                 \
                            double) {                              \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO); \
  }                                                                \
                                                                   \
  PD_CUSTOM_KERNEL_REGISTER(name##_csr_grad,                       \
                            iluvatar_gpu,                          \
                            ALL_LAYOUT,                            \
                            phi::sparse::prefix##CsrGradKernel,    \
                            phi::dtype::float16,                   \
                            float,                                 \
                            double) {                              \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR); \
  }

#define PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL_WITH_COMPLEX(name, prefix) \
  PD_CUSTOM_KERNEL_REGISTER(name##_coo_grad,                                \
                            iluvatar_gpu,                                   \
                            ALL_LAYOUT,                                     \
                            phi::sparse::prefix##CooGradKernel,             \
                            phi::dtype::float16,                            \
                            float,                                          \
                            double,                                         \
                            phi::dtype::complex<float>,                     \
                            phi::dtype::complex<double>) {                  \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);          \
  }                                                                         \
                                                                            \
  PD_CUSTOM_KERNEL_REGISTER(name##_csr_grad,                                \
                            iluvatar_gpu,                                   \
                            ALL_LAYOUT,                                     \
                            phi::sparse::prefix##CsrGradKernel,             \
                            phi::dtype::float16,                            \
                            float,                                          \
                            double,                                         \
                            phi::dtype::complex<float>,                     \
                            phi::dtype::complex<double>) {                  \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);          \
  }

PD_CUSTOM_KERNEL_REGISTER(cast_coo_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::sparse::CastCooGradKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          bool) {}

PD_CUSTOM_KERNEL_REGISTER(cast_csr_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::sparse::CastCsrGradKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          bool) {}
