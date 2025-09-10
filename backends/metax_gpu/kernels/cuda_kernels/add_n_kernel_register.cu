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
#include "paddle/phi/kernels/add_n_kernel.h"

PD_CUSTOM_KERNEL_REGISTER(add_n,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::AddNKernel,
                          float,
                          double,
                          int,
                          phi::dtype::bfloat16,
                          phi::dtype::float16,
                          int64_t,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}

PD_CUSTOM_KERNEL_REGISTER(add_n_array,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::AddNArrayKernel,
                          float,
                          double,
                          int,
                          phi::dtype::bfloat16,
                          phi::dtype::float16,
                          int64_t,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
