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

#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"

PD_CUSTOM_KERNEL_REGISTER(empty,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::EmptyKernel,
                          float,
                          double,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::float8_e4m3fn,
                          phi::dtype::float8_e5m2,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}

PD_CUSTOM_KERNEL_REGISTER(empty_like,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::EmptyLikeKernel,
                          float,
                          double,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
