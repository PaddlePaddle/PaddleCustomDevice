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
#include "paddle/phi/kernels/all_reduce_kernel.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#endif

#if NCCL_VERSION_CODE >= 21000
PD_CUSTOM_KERNEL_REGISTER(all_reduce,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::AllReduceKernel,
                          float,
                          double,
                          int,
                          bool,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int64_t,
                          phi::dtype::bfloat16,
                          phi::dtype::float16) {}
#else
PD_CUSTOM_KERNEL_REGISTER(all_reduce,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::AllReduceKernel,
                          float,
                          double,
                          int,
                          bool,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int64_t,
                          phi::dtype::float16) {}
#endif
