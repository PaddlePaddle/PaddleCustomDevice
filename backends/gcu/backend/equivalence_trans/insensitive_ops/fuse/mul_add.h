/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "backend/equivalence_trans/insensitive_ops/fuse/utility.h"
#include "backend/register/register.h"

namespace backend {
const char *const kMulAdd = "mul_add";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, op, map_inputs, running_mode, MulAddEquivalenceTrans) {
  auto x = *(map_inputs["X"].at(0));
  auto y = *(map_inputs["Y"].at(0));
  auto axis1 =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis1")));
  auto axis2 =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis2")));

  auto mul_out = add_or_mul_op(x, y, axis1, false, running_mode);
  auto add_out = add_or_mul_op(x, mul_out, axis2, true, running_mode);

  return std::make_shared<GcuOp>(add_out);
}

EQUIVALENCE_TRANS_FUNC_REG(kMulAdd, INSENSITIVE, MulAddEquivalenceTrans);

}  // namespace backend
