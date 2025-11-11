//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/flatten_kernel.h"

PD_CUSTOM_KERNEL_REGISTER(flatten,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FlattenKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(flatten_with_xshape,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FlattenWithXShapeKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t) {}
