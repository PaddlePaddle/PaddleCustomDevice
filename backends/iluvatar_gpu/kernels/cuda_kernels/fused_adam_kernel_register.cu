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
#include "paddle/phi/kernels/fused_adam_kernel.h"

PD_CUSTOM_KERNEL_REGISTER(fused_adam,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FusedAdamKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          double) {
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(9).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(3).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(4).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(5).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(6).SetDataType(phi::DataType::UNDEFINED);
}
