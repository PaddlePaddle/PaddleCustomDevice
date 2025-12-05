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
#include "paddle/phi/kernels/gpu/partial_send_kernel.h"
#if (NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000) || \
    defined(PADDLE_WITH_HIP)
PD_CUSTOM_KERNEL_REGISTER(partial_send,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::PartialSendKernel,
                          float,
                          double,
                          phi::dtype::bfloat16,
                          int,
                          int64_t,
                          phi::dtype::float16) {}
#else
PD_CUSTOM_KERNEL_REGISTER(partial_send,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::PartialSendKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16) {}
#endif
