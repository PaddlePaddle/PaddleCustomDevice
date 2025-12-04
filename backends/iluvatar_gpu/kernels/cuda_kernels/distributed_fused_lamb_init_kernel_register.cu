// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/distributed_fused_lamb_init_kernel.h"
#include "paddle/phi/kernels/funcs/algorithm.h"
#include "paddle/phi/kernels/fusion/gpu/cast_with_ptr.h"

PD_CUSTOM_KERNEL_REGISTER(distributed_fused_lamb_init,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::fusion::DistributedFusedLambInitOpKernel,
                          float) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT16);
  kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT16);
  kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(6).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(7).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(8).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(9).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(10).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(11).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(12).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(13).SetDataType(kernel_key.dtype());
  kernel->OutputAt(14).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(15).SetDataType(kernel_key.dtype());
  kernel->OutputAt(16).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(17).SetDataType(phi::DataType::INT64);
}
