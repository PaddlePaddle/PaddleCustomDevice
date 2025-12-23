// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/mp_allreduce_sum_kernel.h"  //NOLINT
PD_CUSTOM_KERNEL_REGISTER(mp_allreduce_sum,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::MpAllReduceSumKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16) {}
