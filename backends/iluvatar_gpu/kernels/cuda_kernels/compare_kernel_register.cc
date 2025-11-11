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
#include "paddle/phi/kernels/compare_kernel.h"

PD_CUSTOM_KERNEL_REGISTER(equal_all,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::EqualAllKernel,
                          bool,
                          int,
                          int64_t,
                          float) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

#define PD_REGISTER_COMPARE_KERNEL(name, func)            \
  PD_CUSTOM_KERNEL_REGISTER(name,                         \
                            iluvatar_gpu,                 \
                            ALL_LAYOUT,                   \
                            phi::func##Kernel,            \
                            bool,                         \
                            int,                          \
                            uint8_t,                      \
                            int8_t,                       \
                            int16_t,                      \
                            int64_t,                      \
                            float,                        \
                            phi::dtype::float16,          \
                            phi::dtype::bfloat16) {       \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL); \
  }

#define PD_REGISTER_COMPLEX_COMPARE_KERNEL(name, func)    \
  PD_CUSTOM_KERNEL_REGISTER(name,                         \
                            iluvatar_gpu,                 \
                            ALL_LAYOUT,                   \
                            phi::func##Kernel,            \
                            bool,                         \
                            int,                          \
                            uint8_t,                      \
                            int8_t,                       \
                            int16_t,                      \
                            int64_t,                      \
                            phi::dtype::complex<float>,   \
                            float,                        \
                            phi::dtype::float16,          \
                            phi::dtype::bfloat16) {       \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL); \
  }

PD_REGISTER_COMPARE_KERNEL(less_than, LessThan)
PD_REGISTER_COMPARE_KERNEL(less_equal, LessEqual)
PD_REGISTER_COMPARE_KERNEL(greater_than, GreaterThan)
PD_REGISTER_COMPARE_KERNEL(greater_equal, GreaterEqual)

PD_REGISTER_COMPLEX_COMPARE_KERNEL(equal, Equal)
PD_REGISTER_COMPLEX_COMPARE_KERNEL(not_equal, NotEqual)
