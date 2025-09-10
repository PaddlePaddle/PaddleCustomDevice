// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
// Reserved. Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/reduce_sum_kernel.h"

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_CUSTOM_KERNEL_REGISTER(sum,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::SumKernel,
                          bool,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int16_t,
                          int,
                          int64_t,
                          uint8_t,
                          int8_t,
                          complex64,
                          complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
