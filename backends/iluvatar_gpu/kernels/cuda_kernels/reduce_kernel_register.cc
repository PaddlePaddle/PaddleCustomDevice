// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/gpu/reduce_kernel.cu"  // NOLINT

PD_CUSTOM_KERNEL_REGISTER(reduce,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ReduceKernel,
                          float,
                          int,
                          bool,
                          int8_t,
                          uint8_t,
                          int64_t,
                          phi::dtype::bfloat16,
                          phi::dtype::float16) {}

PD_CUSTOM_KERNEL_REGISTER(amax_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ReduceAMaxGradKernel,
                          float,
                          int,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(amin_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ReduceAMinGradKernel,
                          float,
                          double,
                          int,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(max_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ReduceMaxGradKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(mean_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ReduceMeanGradKernel,
                          bool,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>,
                          int,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(min_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ReduceMinGradKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(sum_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ReduceSumGradKernel,
                          bool,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          phi::dtype::complex<float>) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
