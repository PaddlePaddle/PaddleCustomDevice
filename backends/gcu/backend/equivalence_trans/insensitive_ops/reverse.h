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
#include <vector>

#include "backend/register/register.h"

namespace backend {
const char *const kReverse = "reverse";
const char *const kReverseGrad = "reverse_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, op, map_inputs, running_mode, ReverseEquivalenceTrans) {
  builder::Op input = *(map_inputs["X"].at(0));

  auto axis = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("axis"));
  auto rank = input.GetType().GetRank();
  std::vector<int64_t> new_axis;
  for (size_t i = 0; i < axis.size(); i++) {
    if (axis[i] < 0) axis[i] += rank;
    new_axis.emplace_back(static_cast<int64_t>(axis[i]));
  }
  auto output = builder::Reverse(input, new_axis);
  return std::make_shared<GcuOp>(output);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, op, map_inputs, running_mode, ReverseGradEquivalenceTrans) {
  builder::Op out_grad = *(map_inputs["Out@GRAD"].at(0));

  auto axis = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("axis"));
  auto rank = out_grad.GetType().GetRank();
  std::vector<int64_t> new_axis;
  for (size_t i = 0; i < axis.size(); i++) {
    if (axis[i] < 0) axis[i] += rank;
    new_axis.emplace_back(static_cast<int64_t>(axis[i]));
  }
  auto x_grad = builder::Reverse(out_grad, new_axis);
  return std::make_shared<GcuOp>(x_grad);
}

EQUIVALENCE_TRANS_FUNC_REG(kReverse, INSENSITIVE, ReverseEquivalenceTrans);

EQUIVALENCE_TRANS_FUNC_REG(kReverseGrad,
                           INSENSITIVE,
                           ReverseGradEquivalenceTrans);

}  // namespace backend
