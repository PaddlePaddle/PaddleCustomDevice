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
// clang-format off
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/set_value_kernel.cu" // NOLINT
#include "paddle/phi/kernels/set_value_kernel.h"
// clang-format on
PD_CUSTOM_KERNEL_REGISTER(set_value,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SetValueKernelV2,
                          float,
                          double,
                          int,
                          int64_t,
                          bool,
                          int16_t,
                          uint8_t,
                          int8_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
PD_CUSTOM_KERNEL_REGISTER(set_value_with_tensor,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SetTensorValueKernelV2,
                          float,
                          double,
                          int,
                          int64_t,
                          bool,
                          int16_t,
                          uint8_t,
                          int8_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
