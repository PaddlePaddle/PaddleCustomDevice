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

#include <memory>

#include "backend/register/register.h"

namespace backend {
const char *const kAbs = "abs";
const char *const kAbsGrad = "abs_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, op, map_inputs, running_mode, AbsEquivalenceTrans) {
  return std::make_shared<GcuOp>(builder::Abs(*(map_inputs["X"].at(0))));
}

//           /  1, x > 0
// dy / dx = -  0, x = 0
//           \ -1, x < 0
IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, op, map_inputs, running_mode, AbsGradEquivalenceTrans) {
  GcuOp x = *(map_inputs["X"].at(0));
  GcuOp out_grad = *(map_inputs["Out@GRAD"].at(0));
  GcuOp zero = builder::ZerosLike(x);
  auto pred_negative = builder::Less(x, zero);
  auto temp = builder::Select(pred_negative, -out_grad, out_grad);
  auto pred_positive = builder::Equal(x, zero);
  return std::make_shared<GcuOp>(builder::Select(pred_positive, zero, temp));
}

EQUIVALENCE_TRANS_FUNC_REG(kAbs, INSENSITIVE, AbsEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kAbsGrad, INSENSITIVE, AbsGradEquivalenceTrans);
}  // namespace backend
