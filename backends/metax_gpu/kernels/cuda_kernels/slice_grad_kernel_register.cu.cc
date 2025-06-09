// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
// Reserved. Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/impl/slice_grad_kernel_impl.h"
#include "paddle/phi/kernels/slice_grad_kernel.h"

PD_CUSTOM_KERNEL_REGISTER(slice_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SliceGradKernel,
                          bool,
                          int,
                          uint8_t,
                          int64_t,
                          float,
                          double,
                          int16_t,
                          int8_t,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>,
                          phi::dtype::bfloat16,
                          phi::dtype::float16) {}

PD_CUSTOM_KERNEL_REGISTER(slice_array_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SliceArrayGradKernel,
                          bool,
                          int,
                          uint8_t,
                          int64_t,
                          float,
                          double,
                          int16_t,
                          int8_t,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>,
                          phi::dtype::bfloat16,
                          phi::dtype::float16) {}

PD_CUSTOM_KERNEL_REGISTER(slice_array_dense_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SliceArrayDenseGradKernel,
                          bool,
                          int,
                          uint8_t,
                          int64_t,
                          float,
                          double,
                          int16_t,
                          int8_t,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>,
                          phi::dtype::bfloat16,
                          phi::dtype::float16) {}
