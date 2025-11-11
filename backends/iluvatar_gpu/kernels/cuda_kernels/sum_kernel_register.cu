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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/cum_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/index_select_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"
#include "paddle/phi/kernels/sparse/unary_kernel.h"

PD_CUSTOM_KERNEL_REGISTER(sum_coo,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::sparse::SumCooKernel,
                          float,
                          double,
                          int,
                          int64_t) {
  kernel->OutputAt(0).SetDataType(paddle::DataType::UNDEFINED);
}

PD_CUSTOM_KERNEL_REGISTER(sum_csr,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::sparse::SumCsrKernel,
                          float,
                          double,
                          int,
                          int64_t) {
  kernel->OutputAt(0).SetDataType(paddle::DataType::UNDEFINED);
}
