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
#include "paddle/phi/kernels/fusion/gpu/fused_transpose_split_quant_kernel.h"
#include "paddle/phi/kernels/fusion/gpu/quant_utils.h"

PD_CUSTOM_KERNEL_REGISTER(fused_transpose_split_quant,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FusedTransposeSplitQuantKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT8_E4M3FN);
  kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
}
