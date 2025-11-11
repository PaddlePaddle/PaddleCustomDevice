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

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <algorithm>
#include <vector>

#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/graph_send_recv_funcs.h"
#include "paddle/phi/kernels/send_u_recv_kernel.h"

PD_CUSTOM_KERNEL_REGISTER(send_u_recv,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::SendURecvKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
}
