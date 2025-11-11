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
// limitation

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/logical_kernel.h"

#define REGISTER_LOGICAL_CUDA_KERNEL_ILUVATAR(logical_and, func_type) \
  PD_CUSTOM_KERNEL_REGISTER(logical_and,                              \
                            iluvatar_gpu,                             \
                            ALL_LAYOUT,                               \
                            phi::Logical##func_type##Kernel,          \
                            float,                                    \
                            phi::dtype::float16,                      \
                            phi::dtype::bfloat16,                     \
                            bool,                                     \
                            int64_t,                                  \
                            int,                                      \
                            int8_t,                                   \
                            phi::dtype::complex<float>,               \
                            int16_t) {                                \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);             \
  }

REGISTER_LOGICAL_CUDA_KERNEL_ILUVATAR(logical_and, And)
REGISTER_LOGICAL_CUDA_KERNEL_ILUVATAR(logical_or, Or)
REGISTER_LOGICAL_CUDA_KERNEL_ILUVATAR(logical_not, Not)
REGISTER_LOGICAL_CUDA_KERNEL_ILUVATAR(logical_xor, Xor)
