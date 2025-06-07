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

// clang-format will try to sort headers according to google c++ style,
// and that cause compiling problems.
// clang-format off
#include "paddle/phi/kernels/gelu_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
// clang-format on

PD_CUSTOM_KERNEL_REGISTER(gelu,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::GeluKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
