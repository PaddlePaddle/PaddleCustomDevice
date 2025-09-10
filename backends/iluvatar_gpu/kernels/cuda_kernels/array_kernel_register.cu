

// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/array_kernel.h"

PD_CUSTOM_KERNEL_REGISTER(create_array,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::CreateArrayKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}

PD_CUSTOM_KERNEL_REGISTER(create_array_like,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::CreateArrayLikeKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}

PD_CUSTOM_KERNEL_REGISTER(array_read,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ArrayReadKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}

PD_CUSTOM_KERNEL_REGISTER(array_write,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ArrayWriteKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}

PD_CUSTOM_KERNEL_REGISTER(array_to_tensor,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ArrayToTensorKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}

PD_CUSTOM_KERNEL_REGISTER(array_pop,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ArrayPopKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
