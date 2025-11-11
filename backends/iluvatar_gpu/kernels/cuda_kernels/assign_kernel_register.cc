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
#include "paddle/phi/kernels/assign_kernel.h"

PD_CUSTOM_KERNEL_REGISTER_FOR_ALL_DTYPE(assign,
                                        iluvatar_gpu,
                                        ALL_LAYOUT,
                                        phi::AssignKernel) {}

PD_CUSTOM_KERNEL_REGISTER_FOR_ALL_DTYPE(assign_raw,
                                        iluvatar_gpu,
                                        ALL_LAYOUT,
                                        phi::AssignRawKernel) {}

PD_CUSTOM_KERNEL_REGISTER_FOR_ALL_DTYPE(assign_array,
                                        iluvatar_gpu,
                                        ALL_LAYOUT,
                                        phi::AssignArrayKernel) {}

PD_CUSTOM_KERNEL_REGISTER(assign_value,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::AssignValueKernel,
                          bool,
                          int,
                          float,
                          int8_t,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          phi::dtype::complex<float>) {}
