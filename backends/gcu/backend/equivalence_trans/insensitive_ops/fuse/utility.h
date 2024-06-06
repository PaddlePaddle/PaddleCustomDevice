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

#include "backend/register/register.h"

namespace backend {
namespace {  // NOLINT
GcuOp add_or_mul_op(const GcuOp& input_op,
                    const GcuOp& input2_op,
                    int64_t axis,
                    bool is_add,
                    const std::string& running_mode = "serial") {
  std::vector<GcuOp> inputs;
  inputs.push_back(input_op);
  inputs.push_back(input2_op);

  auto lhs_shape = inputs[0].GetType().GetShape();
  auto rhs_shape = inputs[1].GetType().GetShape();
  print_shape(lhs_shape, "lhs");
  print_shape(rhs_shape, "rhs");
  if (lhs_shape == rhs_shape) {
    if (lhs_shape.size() != 4 ||
        (lhs_shape.size() == 4 && running_mode != RunningMode::ADAPTIVE)) {
      if (is_add) {
        return builder::Add(inputs[0], inputs[1]);
      } else {
        return builder::Mul(inputs[0], inputs[1]);
      }
    }
    auto lv = builder::Transpose(inputs[0], {0, 2, 3, 1});
    auto rv = builder::Transpose(inputs[1], {0, 2, 3, 1});
    if (is_add) {
      return builder::Transpose(builder::Add(lv, rv), {0, 3, 1, 2});
    } else {
      return builder::Transpose(builder::Mul(lv, rv), {0, 3, 1, 2});
    }
  }

  auto lhs_rank = inputs[0].GetType().GetRank();
  auto rhs_rank = inputs[1].GetType().GetRank();
  std::map<std::string, GcuOp> op_map{{"X", inputs[0]}, {"Y", inputs[1]}};
  auto low = lhs_rank < rhs_rank ? "X" : "Y";
  std::vector<int64_t> new_shape;
  int64_t iter = 0;
  if (lhs_rank < rhs_rank) {
    new_shape.assign(rhs_rank, 1);
    axis = axis > 0 ? axis : rhs_rank - lhs_rank;
    for (int64_t i = axis; i < axis + lhs_rank; ++i) {
      new_shape[i] = lhs_shape[iter++];
    }
  } else {
    new_shape.assign(lhs_rank, 1);
    axis = axis > 0 ? axis : lhs_rank - rhs_rank;
    for (int64_t i = axis; i < axis + rhs_rank; ++i) {
      new_shape[i] = rhs_shape[iter++];
    }
  }
  op_map[low] = builder::Reshape(op_map[low], new_shape);
  if (op_map["X"].GetType().GetShape().size() == 4 &&
      running_mode == RunningMode::ADAPTIVE) {
    auto lv = builder::Transpose(op_map["X"], {0, 2, 3, 1});
    auto rv = builder::Transpose(op_map["Y"], {0, 2, 3, 1});
    if (is_add) {
      return builder::Transpose(builder::Add(lv, rv), {0, 3, 1, 2});
    } else {
      return builder::Transpose(builder::Mul(lv, rv), {0, 3, 1, 2});
    }
  }

  if (is_add) {
    return builder::Add(op_map["X"], op_map["Y"]);
  } else {
    return builder::Mul(op_map["X"], op_map["Y"]);
  }
}
}  // namespace
}  // namespace backend
