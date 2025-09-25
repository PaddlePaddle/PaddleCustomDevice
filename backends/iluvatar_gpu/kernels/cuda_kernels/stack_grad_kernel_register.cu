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

#include "paddle/phi/kernels/gpu/stack_grad_kernel.cu"  // NOLINT

PD_CUSTOM_KERNEL_REGISTER(stack_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::StackGradKernel,
                          bool,
                          float,
                          double,
                          int,
                          int8_t,
                          int64_t,
                          uint8_t,
                          int16_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::float8_e4m3fn,
                          phi::dtype::float8_e5m2,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
