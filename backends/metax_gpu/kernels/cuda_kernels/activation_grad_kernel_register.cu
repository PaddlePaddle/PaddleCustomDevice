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
#include "paddle/phi/kernels/activation_grad_kernel.h"

PD_CUSTOM_KERNEL_REGISTER(relu_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::ReluGradKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
PD_CUSTOM_KERNEL_REGISTER(relu_double_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::ReluDoubleGradKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

#define PD_REGISTER_ACTIVATION_GRAD_KERNEL(name, func) \
  PD_CUSTOM_KERNEL_REGISTER(name,                      \
                            metax_gpu,                 \
                            ALL_LAYOUT,                \
                            phi::func,                 \
                            float,                     \
                            double,                    \
                            phi::dtype::float16,       \
                            phi::dtype::bfloat16) {}

#define PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(name, func) \
  PD_CUSTOM_KERNEL_REGISTER(name,                                   \
                            metax_gpu,                              \
                            ALL_LAYOUT,                             \
                            phi::func,                              \
                            float,                                  \
                            double,                                 \
                            phi::dtype::float16,                    \
                            phi::dtype::bfloat16,                   \
                            phi::dtype::complex<float>,             \
                            phi::dtype::complex<double>) {}

PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(sin_grad, SinGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(cos_grad, CosGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(tan_grad, TanGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(acos_grad, AcosGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(asin_grad, AsinGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(atan_grad, AtanGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(sinh_grad, SinhGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(cosh_grad, CoshGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(asinh_grad, AsinhGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(acosh_grad, AcoshGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(atanh_grad, AtanhGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(tanh_grad, TanhGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(tanh_double_grad,
                                                TanhDoubleGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(tanh_triple_grad,
                                                TanhTripleGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(hardtanh_grad, HardTanhGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(leaky_relu_grad, LeakyReluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(leaky_relu_double_grad,
                                   LeakyReluDoubleGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(thresholded_relu_grad,
                                   ThresholdedReluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(relu6_grad, Relu6GradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(mish_grad, MishGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(stanh_grad, STanhGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(reciprocal_grad,
                                                ReciprocalGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(softplus_grad,
                                                SoftplusGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(softplus_double_grad,
                                                SoftplusDoubleGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(sqrt_grad, SqrtGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(sqrt_double_grad, SqrtDoubleGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(rsqrt_grad, RsqrtGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(rsqrt_double_grad, RsqrtDoubleGradKernel)

PD_CUSTOM_KERNEL_REGISTER(exp_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::ExpGradKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}

PD_REGISTER_ACTIVATION_GRAD_KERNEL(softshrink_grad, SoftShrinkGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(hard_shrink_grad, HardShrinkGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(tanh_shrink_grad, TanhShrinkGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(silu_grad, SiluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(elu_grad, EluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(elu_double_grad, EluDoubleGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(logit_grad, LogitCUDAGradKernel)

PD_CUSTOM_KERNEL_REGISTER(expm1_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::Expm1GradKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}

PD_CUSTOM_KERNEL_REGISTER(square_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SquareGradKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
PD_CUSTOM_KERNEL_REGISTER(square_double_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SquareDoubleGradKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}

PD_CUSTOM_KERNEL_REGISTER(sin_double_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SinDoubleGradKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}

PD_CUSTOM_KERNEL_REGISTER(sin_triple_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SinTripleGradKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}

PD_CUSTOM_KERNEL_REGISTER(cos_double_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::CosDoubleGradKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}

PD_CUSTOM_KERNEL_REGISTER(cos_triple_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::CosTripleGradKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}

PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(softsign_grad,
                                                SoftsignGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(sigmoid_grad, SigmoidGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(sigmoid_double_grad,
                                                SigmoidDoubleGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(sigmoid_triple_grad,
                                                SigmoidTripleGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(hardsigmoid_grad, HardSigmoidGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(logsigmoid_grad,
                                                LogSigmoidGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(log_grad, LogGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(log2_grad, Log2GradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(log10_grad, Log10GradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(log1p_grad, Log1pGradKernel)
PD_CUSTOM_KERNEL_REGISTER(log_double_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::LogDoubleGradKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
PD_REGISTER_ACTIVATION_GRAD_KERNEL_WITH_COMPLEX(hardswish_grad,
                                                HardSwishGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(swish_grad, SwishGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(floor_grad, FloorGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(ceil_grad, CeilGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(celu_grad, CeluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(celu_double_grad, CeluDoubleGradKernel)

PD_CUSTOM_KERNEL_REGISTER(round_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::RoundGradKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
PD_CUSTOM_KERNEL_REGISTER(pow_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::PowGradKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
PD_CUSTOM_KERNEL_REGISTER(pow_double_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::PowDoubleGradKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
PD_CUSTOM_KERNEL_REGISTER(pow_triple_grad,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::PowTripleGradKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
