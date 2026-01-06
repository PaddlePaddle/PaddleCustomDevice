// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
// Reserved. Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

PD_CUSTOM_KERNEL_REGISTER(maximum,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::MaximumKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
PD_CUSTOM_KERNEL_REGISTER(minimum,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::MinimumKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
PD_CUSTOM_KERNEL_REGISTER(remainder,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::RemainderKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>) {}
PD_CUSTOM_KERNEL_REGISTER(floor_divide,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::FloorDivideKernel,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
PD_CUSTOM_KERNEL_REGISTER(elementwise_pow,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::ElementwisePowKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
PD_CUSTOM_KERNEL_REGISTER(copysign,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::CopySignKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;
using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_CUSTOM_KERNEL_REGISTER(fmax,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::FMaxKernel,
                          float,
                          double,
                          int,
                          float16,
                          bfloat16,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(fmin,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::FMinKernel,
                          float,
                          double,
                          int,
                          float16,
                          bfloat16,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(heaviside,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::HeavisideKernel,
                          float,
                          double,
                          int,
                          float16,
                          bfloat16,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(add,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::AddKernel,
                          float,
                          double,
                          int16_t,
                          int,
                          bool,
                          uint8_t,
                          int8_t,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          complex64,
                          complex128) {}

PD_CUSTOM_KERNEL_REGISTER(grad_add,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::GradAddKernel,
                          float,
                          double,
                          int16_t,
                          int,
                          bool,
                          uint8_t,
                          int8_t,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          complex64,
                          complex128) {}

PD_CUSTOM_KERNEL_REGISTER(divide,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::DivideKernel,
                          float,
                          double,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          bool,
                          float16,
                          bfloat16,
                          complex64,
                          complex128) {}

PD_CUSTOM_KERNEL_REGISTER(multiply,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::MultiplyKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          bool,
                          float16,
                          complex64,
                          complex128,
                          bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(subtract,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SubtractKernel,
                          float,
                          double,
                          int16_t,
                          int,
                          int64_t,
                          float16,
                          bfloat16,
                          complex64,
                          complex128) {}
