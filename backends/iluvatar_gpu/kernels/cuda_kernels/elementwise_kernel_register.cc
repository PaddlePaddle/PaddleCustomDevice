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
#include "paddle/phi/kernels/kps/elementwise_kernel.cu"  // NOLINT

using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;
using complex64 = ::phi::dtype::complex<float>;

PD_CUSTOM_KERNEL_REGISTER(maximum,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::MaximumKernel,
                          float,
                          int,
                          int64_t,
                          float16,
                          bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(minimum,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::MinimumKernel,
                          float,
                          int,
                          int64_t,
                          float16,
                          bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(remainder,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::RemainderKernel,
                          float,
                          int,
                          int64_t,
                          float16,
                          bfloat16,
                          complex64) {}

PD_CUSTOM_KERNEL_REGISTER(floor_divide,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FloorDivideKernel,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          float16,
                          bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(elementwise_pow,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ElementwisePowKernel,
                          float,
                          int,
                          int64_t,
                          float16,
                          bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(copysign,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::CopySignKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          float16,
                          bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(fmax,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FMaxKernel,
                          float,
                          int,
                          float16,
                          bfloat16,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(fmin,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FMinKernel,
                          float,
                          int,
                          float16,
                          bfloat16,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(heaviside,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::HeavisideKernel,
                          float,
                          int,
                          float16,
                          bfloat16,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(add,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::AddKernel,
                          float,
                          int16_t,
                          int,
                          bool,
                          uint8_t,
                          int8_t,
                          int64_t,
                          float16,
                          bfloat16,
                          complex64) {}

PD_CUSTOM_KERNEL_REGISTER(grad_add,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::GradAddKernel,
                          float,
                          int16_t,
                          int,
                          bool,
                          uint8_t,
                          int8_t,
                          int64_t,
                          float16,
                          bfloat16,
                          complex64) {}

PD_CUSTOM_KERNEL_REGISTER(divide,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::DivideKernel,
                          float,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          bool,
                          float16,
                          bfloat16,
                          complex64) {}

PD_CUSTOM_KERNEL_REGISTER(multiply,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::MultiplyKernel,
                          float,
                          int,
                          int64_t,
                          bool,
                          float16,
                          complex64,
                          bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(subtract,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::SubtractKernel,
                          float,
                          int16_t,
                          int,
                          int64_t,
                          float16,
                          bfloat16,
                          complex64) {}
