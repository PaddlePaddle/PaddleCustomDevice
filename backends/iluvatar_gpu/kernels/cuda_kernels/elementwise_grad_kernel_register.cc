//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/elementwise_add_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_grad_kernel.h"

PD_CUSTOM_KERNEL_REGISTER(fmax_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ElementwiseFMaxGradKernel,
                          float,
                          int,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(fmin_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ElementwiseFMinGradKernel,
                          float,
                          int,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(maximum_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::MaximumGradKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(minimum_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::MinimumGradKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(remainder_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::RemainderGradKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(heaviside_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::HeavisideGradKernel,
                          float,
                          int,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(elementwise_pow_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ElementwisePowGradKernel,
                          float,
                          int,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(add_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::AddGradKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>) {}

PD_CUSTOM_KERNEL_REGISTER(add_double_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::AddDoubleGradKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>) {}

PD_CUSTOM_KERNEL_REGISTER(add_triple_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::AddTripleGradKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>) {}

PD_CUSTOM_KERNEL_REGISTER(divide_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::DivideGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::complex<float>) {}

PD_CUSTOM_KERNEL_REGISTER(divide_double_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::DivideDoubleGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::complex<float>) {}

PD_CUSTOM_KERNEL_REGISTER(multiply_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::MultiplyGradKernel,
                          float,
                          phi::dtype::float16,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>) {}

PD_CUSTOM_KERNEL_REGISTER(multiply_double_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::MultiplyDoubleGradKernel,
                          float,
                          phi::dtype::float16,
                          double,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>) {}

PD_CUSTOM_KERNEL_REGISTER(multiply_triple_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::MultiplyTripleGradKernel,
                          float,
                          phi::dtype::float16,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>) {}

PD_CUSTOM_KERNEL_REGISTER(subtract_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::SubtractGradKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>) {}

PD_CUSTOM_KERNEL_REGISTER(subtract_double_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::SubtractDoubleGradKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>) {}

PD_CUSTOM_KERNEL_REGISTER(copysign_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::CopySignGradKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
