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
const char* const kAssignValue = "assign_value";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, op, map_inputs, running_mode, AssignValueEquivalenceTrans) {
  auto shape = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("shape"));
  auto dtype = PADDLE_GET_CONST(int, op->GetAttr("dtype"));
  auto bool_values =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("bool_values"));
  auto f32_values =
      PADDLE_GET_CONST(std::vector<float>, op->GetAttr("fp32_values"));
  auto i32_values =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("int32_values"));
  auto i64_values =
      PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("int64_values"));

  GcuOp out;
  std::vector<int64_t> shapes(shape.begin(), shape.end());
  auto ptype = builder::PrimitiveType::NONE();
  if (static_cast<phi::DataType>(dtype) == phi::DataType::BOOL) {
    ptype = builder::PrimitiveType::PRED();
    std::vector<int8_t> b_values(bool_values.begin(), bool_values.end());
    out = builder::Const(gcu_builder, b_values, builder::Type(shapes, ptype));
  } else if (static_cast<phi::DataType>(dtype) == phi::DataType::FLOAT32) {
    ptype = builder::PrimitiveType::F32();
    out = builder::Const(gcu_builder, f32_values, builder::Type(shapes, ptype));
  } else if (static_cast<phi::DataType>(dtype) == phi::DataType::INT32) {
    ptype = builder::PrimitiveType::S32();
    out = builder::Const(gcu_builder, i32_values, builder::Type(shapes, ptype));
  } else if (static_cast<phi::DataType>(dtype) == phi::DataType::INT64) {
    ptype = builder::PrimitiveType::S64();
    out = builder::Const(gcu_builder, i64_values, builder::Type(shapes, ptype));
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported data type(code %d) for AssignValue, only supports bool, "
        "int32, float32 and int64.",
        dtype));
  }

  return std::make_shared<GcuOp>(out);
}

EQUIVALENCE_TRANS_FUNC_REG(kAssignValue,
                           INSENSITIVE,
                           AssignValueEquivalenceTrans);

}  // namespace backend
