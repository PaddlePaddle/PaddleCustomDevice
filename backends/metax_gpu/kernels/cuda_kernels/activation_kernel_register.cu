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
#include "paddle/phi/kernels/gpu/activation_kernel.cu"  // NOLINT

PD_CUSTOM_KERNEL_REGISTER(relu,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::ReluKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

#define PD_REGISTER_ACTIVATION_KERNEL(name, func) \
  PD_CUSTOM_KERNEL_REGISTER(name,                 \
                            metax_gpu,            \
                            ALL_LAYOUT,           \
                            phi::func,            \
                            float,                \
                            double,               \
                            phi::dtype::float16,  \
                            phi::dtype::bfloat16) {}

#define PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(name, func) \
  PD_CUSTOM_KERNEL_REGISTER(name,                              \
                            metax_gpu,                         \
                            ALL_LAYOUT,                        \
                            phi::func,                         \
                            float,                             \
                            double,                            \
                            phi::dtype::float16,               \
                            phi::dtype::bfloat16,              \
                            phi::dtype::complex<float>,        \
                            phi::dtype::complex<double>) {}

PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(sin, SinKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(cos, CosKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(tan, TanKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(acos, AcosKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(asin, AsinKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(atan, AtanKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(sinh, SinhKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(cosh, CoshKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(asinh, AsinhKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(acosh, AcoshKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(atanh, AtanhKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(tanh, TanhKernel)
PD_REGISTER_ACTIVATION_KERNEL(hardtanh, HardTanhKernel)
PD_REGISTER_ACTIVATION_KERNEL(thresholded_relu, ThresholdedReluKernel)
PD_REGISTER_ACTIVATION_KERNEL(relu6, Relu6Kernel)
PD_REGISTER_ACTIVATION_KERNEL(leaky_relu, LeakyReluKernel)
PD_REGISTER_ACTIVATION_KERNEL(mish, MishKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(stanh, StanhKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(reciprocal, ReciprocalKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(sqrt, SqrtKernel)
PD_REGISTER_ACTIVATION_KERNEL(rsqrt, RsqrtKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(softplus, SoftplusKernel)

PD_CUSTOM_KERNEL_REGISTER(exp,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::ExpKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
PD_CUSTOM_KERNEL_REGISTER(expm1,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::Expm1Kernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
PD_CUSTOM_KERNEL_REGISTER(square,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SquareKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}

PD_REGISTER_ACTIVATION_KERNEL(hard_shrink, HardShrinkKernel)
PD_REGISTER_ACTIVATION_KERNEL(softshrink, SoftShrinkKernel)
PD_REGISTER_ACTIVATION_KERNEL(tanh_shrink, TanhShrinkKernel)
PD_REGISTER_ACTIVATION_KERNEL(elu, EluKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(silu, SiluKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(softsign, SoftsignKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(sigmoid, SigmoidKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(logsigmoid, LogSigmoidKernel)
PD_REGISTER_ACTIVATION_KERNEL(hardsigmoid, HardSigmoidKernel)
PD_REGISTER_ACTIVATION_KERNEL_WITH_COMPLEX(hardswish, HardSwishKernel)
PD_REGISTER_ACTIVATION_KERNEL(swish, SwishKernel)
PD_REGISTER_ACTIVATION_KERNEL(floor, FloorKernel)
PD_REGISTER_ACTIVATION_KERNEL(ceil, CeilKernel)
PD_REGISTER_ACTIVATION_KERNEL(celu, CeluKernel)
PD_REGISTER_ACTIVATION_KERNEL(selu, SeluKernel)
PD_REGISTER_ACTIVATION_KERNEL(logit, LogitCUDAKernel)

PD_CUSTOM_KERNEL_REGISTER(round,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::RoundKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
PD_CUSTOM_KERNEL_REGISTER(log,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::LogKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
PD_CUSTOM_KERNEL_REGISTER(log2,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::Log2Kernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
PD_CUSTOM_KERNEL_REGISTER(log10,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::Log10Kernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
PD_CUSTOM_KERNEL_REGISTER(log1p,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::Log1pKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
PD_CUSTOM_KERNEL_REGISTER(pow,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::PowKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
