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

#include <vector>

#include "common/common.h"
#include "common/utils.h"
#include "kernels/funcs/gcu_name_list.h"
#include "paddle/phi/core/dense_tensor.h"

#pragma once
namespace custom_kernel {

#define DECLARE_REDUCTION_OP(op)                                   \
  void op##_compute(const phi::CustomContext& dev_ctx,             \
                    const phi::DenseTensor& data,                  \
                    bool keep_dims,                                \
                    std::vector<int64_t> axes,                     \
                    phi::DenseTensor& output);                     \
                                                                   \
  phi::DenseTensor op##_compute(const phi::CustomContext& dev_ctx, \
                                const phi::DenseTensor& data,      \
                                bool keep_dims,                    \
                                std::vector<int64_t> axes);

DECLARE_REDUCTION_OP(reduce_sum)
DECLARE_REDUCTION_OP(reduce_mean)
DECLARE_REDUCTION_OP(reduce_max)
DECLARE_REDUCTION_OP(reduce_min)
DECLARE_REDUCTION_OP(reduce_prod)

#undef DECLARE_REDUCTION_OP
}  // namespace custom_kernel
