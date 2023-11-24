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
#include <string>
#include <vector>

#include "backend/register/register.h"

namespace backend {
const char *const kArgMax = "arg_max";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, op, map_inputs, running_mode, ArgMaxEquivalenceTrans) {
  int64_t axis = PADDLE_GET_CONST(int64_t, op->GetAttr("axis"));
  auto keepdims = PADDLE_GET_CONST(bool, op->GetAttr("keepdims"));
  auto flatten = PADDLE_GET_CONST(bool, op->GetAttr("flatten"));
  GcuOp data = *(map_inputs["X"].at(0));
  int64_t rank = data.GetType().GetRank();
  GcuOp result;
  if (flatten) {
    auto data_shape = data.GetType().GetShape();
    int64_t new_shape = 1;
    for (auto dim : data_shape) {
      new_shape *= dim;
    }
    builder::Type output_type(
        {
            new_shape,
        },
        data.GetType().GetPrimitiveType());
    auto out = builder::Reshape(data, output_type);
    result = builder::ArgMax(out, /*axis*/ 0, keepdims);
  } else {
    if (axis < 0) {
      axis = axis + rank;
    }
    result = builder::ArgMax(data, axis, keepdims);
  }
  auto dtype = PADDLE_GET_CONST(int, op->GetAttr("dtype"));
  auto ptype = builder::PrimitiveType::NONE();
  if (dtype == 2) {  //  framework::proto::VarType::INT32
    ptype = builder::PrimitiveType::S32();
  } else if (dtype == 3) {  //  framework::proto::VarType::INT64
    ptype = builder::PrimitiveType::S64();
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported data type(code %d) for ArgMax, only supports int32 and "
        "int64.",
        dtype));
  }
  if (ptype != data.GetType().GetPrimitiveType()) {
    auto shape = result.GetType().GetShape();
    result = builder::Convert(result, builder::Type(shape, ptype));
  }
  return std::make_shared<GcuOp>(result);
}

EQUIVALENCE_TRANS_FUNC_REG(kArgMax, INSENSITIVE, ArgMaxEquivalenceTrans);

}  // namespace backend
