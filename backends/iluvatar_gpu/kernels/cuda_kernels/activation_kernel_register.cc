/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/activation_kernel.h"

PD_CUSTOM_KERNEL_REGISTER(relu,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ReluKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(sin,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::SinKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(cos,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::CosKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>) {}

PD_CUSTOM_KERNEL_REGISTER(tan,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::TanKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(acos,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::AcosKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(asin,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::AsinKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(atan,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::AtanKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(sinh,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::SinhKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(cosh,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::CoshKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(asinh,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::AsinhKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(acosh,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::AcoshKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(atanh,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::AtanhKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(tanh,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::TanhKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(hardtanh,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::HardTanhKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(thresholded_relu,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ThresholdedReluKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(relu6,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::Relu6Kernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(leaky_relu,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::LeakyReluKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(mish,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::MishKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(stanh,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::StanhKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(reciprocal,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ReciprocalKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(sqrt,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::SqrtKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(rsqrt,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::RsqrtKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(softplus,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::SoftplusKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(exp,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ExpKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(expm1,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::Expm1Kernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(square,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::SquareKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(hard_shrink,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::HardShrinkKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(softshrink,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::SoftShrinkKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(tanh_shrink,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::TanhShrinkKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(elu,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::EluKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(silu,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::SiluKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(softsign,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::SoftsignKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(sigmoid,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::SigmoidKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(logsigmoid,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::LogSigmoidKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(hardsigmoid,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::HardSigmoidKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(hardswish,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::HardSwishKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(swish,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::SwishKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(round,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::RoundKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(floor,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FloorKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(ceil,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::CeilKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(celu,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::CeluKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(log,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::LogKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(log2,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::Log2Kernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(log10,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::Log10Kernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(log1p,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::Log1pKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(pow,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::PowKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
