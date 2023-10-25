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

#include "common/common.h"
#include "common/utils.h"
#include "kernels/funcs/gcu_name_list.h"
#include "paddle/phi/core/dense_tensor.h"

#pragma once
namespace custom_kernel {

#define DECLARE_UNARY_OP(op)                                       \
  void op##_compute(const phi::CustomContext& dev_ctx,             \
                    const phi::DenseTensor& input,                 \
                    phi::DenseTensor* output);                     \
  phi::DenseTensor op##_compute(const phi::CustomContext& dev_ctx, \
                                const phi::DenseTensor& input);

DECLARE_UNARY_OP(abs)
DECLARE_UNARY_OP(bitwise_not)
DECLARE_UNARY_OP(exp)
DECLARE_UNARY_OP(floor)
DECLARE_UNARY_OP(log)
DECLARE_UNARY_OP(relu)
DECLARE_UNARY_OP(sigmoid)
DECLARE_UNARY_OP(neg)

#undef DECLARE_UNARY_OP
}  // namespace custom_kernel
