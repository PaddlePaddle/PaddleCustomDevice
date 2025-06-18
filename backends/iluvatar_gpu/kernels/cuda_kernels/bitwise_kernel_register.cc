/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/bitwise_kernel.h"
#include "paddle/phi/kernels/funcs/bitwise_functors.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"

PD_CUSTOM_KERNEL_REGISTER(bitwise_and,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::BitwiseAndKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(bitwise_or,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::BitwiseOrKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(bitwise_xor,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::BitwiseXorKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(bitwise_not,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::BitwiseNotKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(bitwise_left_shift,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::BitwiseLeftShiftKernel,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(bitwise_right_shift,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::BitwiseRightShiftKernel,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t) {}
