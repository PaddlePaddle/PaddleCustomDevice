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
#include "paddle/phi/kernels/gpu/group_norm_kernel.cu"  // NOLINT

PD_CUSTOM_KERNEL_REGISTER(group_norm,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::GroupNormKernel,
                          float,
                          double,
                          phi::dtype::bfloat16,
                          phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::BFLOAT16 ||
      kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}

PD_CUSTOM_KERNEL_REGISTER(add_group_norm_silu,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::GroupNormNDHWCKernel,
                          phi::dtype::bfloat16,
                          phi::dtype::float16) {
  kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
}
