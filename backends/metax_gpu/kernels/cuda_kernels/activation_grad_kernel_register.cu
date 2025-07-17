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
#include "paddle/phi/kernels/activation_grad_kernel.h"

PD_CUSTOM_KERNEL_REGISTER(relu_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::ReluGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(sin_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SinGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(cos_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::CosGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(tan_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::TanGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(acos_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::AcosGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(asin_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::AsinGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(atan_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::AtanGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(sinh_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SinhGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(cosh_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::CoshGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(asinh_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::AsinhGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(acosh_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::AcoshGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(atanh_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::AtanhGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(tanh_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::TanhGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(hardtanh_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::HardTanhGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(thresholded_relu_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::ThresholdedReluGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(relu6_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::Relu6GradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(leaky_relu_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::LeakyReluGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(mish_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::MishGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(stanh_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::STanhGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(reciprocal_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::ReciprocalGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(sqrt_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SqrtGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(rsqrt_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::RsqrtGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(softplus_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SoftplusGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(exp_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::ExpGradKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(expm1_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::Expm1GradKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(square_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SquareGradKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(hard_shrink_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::HardShrinkGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(softshrink_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SoftShrinkGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(tanh_shrink_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::TanhShrinkGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(elu_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::EluGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(silu_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SiluGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(softsign_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SoftsignGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(sigmoid_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SigmoidGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(logsigmoid_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::LogSigmoidGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(hardsigmoid_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::HardSigmoidGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(hardswish_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::HardSwishGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(swish_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SwishGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(round_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::RoundGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(floor_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::FloorGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(ceil_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::CeilGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(celu_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::CeluGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(log_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::LogGradKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(log2_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::Log2GradKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(log10_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::Log10GradKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(log1p_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::Log1pGradKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(pow_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::PowGradKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
