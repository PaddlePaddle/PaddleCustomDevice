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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_helper.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/mixed_vector.h"
#include "paddle/phi/kernels/rprop_kernel.h"

#ifdef PADDLE_WITH_CUDA
PD_CUSTOM_KERNEL_REGISTER(rprop,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::RpropKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          float,
                          double) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16 ||
      kernel_key.dtype() == phi::DataType::BFLOAT16) {
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
  }
}
#endif

#ifdef PADDLE_WITH_HIP
PD_CUSTOM_KERNEL_REGISTER(rprop,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::RpropKernel,
                          phi::dtype::float16,
                          float,
                          double) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
  }
}
#endif
