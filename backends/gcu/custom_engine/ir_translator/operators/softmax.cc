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

#include <vector>

#include "custom_engine/ir_translator/translator_registry.h"

namespace custom_engine {

static GcuOpPtr TranslateSoftmax(
    GcuBuilderPtr gcu_builder,
    const pir::Operation* op,
    const std::vector<std::vector<GcuOpPtr>>& gcu_op_inputs) {
  auto input = *(gcu_op_inputs[0][0]);

  // Get attributes
  const auto& attributes = op->attributes();
  int64_t axis = static_cast<int64_t>(
      attributes.at("axis").dyn_cast<pir::Int32Attribute>().data());

  if (!(input.GetType().GetPrimitiveType() == builder::PrimitiveType::F32()) &&
      !(input.GetType().GetPrimitiveType() == builder::PrimitiveType::F64())) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "GCU softmax only support FP32/FP64 datatype so far as now!"));
  }

  // to avoid 0
  double max_value_d = 1.0;
  double min_value_d = 1e-16;
  float max_value_f = 1.0;
  float min_value_f = 1e-7;
  void* max_ptr = nullptr;
  void* min_ptr = nullptr;
  auto scalar_type = builder::Type(input.GetType().GetPrimitiveType());
  if (input.GetType().GetPrimitiveType() == builder::PrimitiveType::F32()) {
    max_ptr = static_cast<void*>(&max_value_f);
    min_ptr = static_cast<void*>(&min_value_f);
  } else if (input.GetType().GetPrimitiveType() ==
             builder::PrimitiveType::F64()) {
    max_ptr = static_cast<void*>(&max_value_d);
    min_ptr = static_cast<void*>(&min_value_d);
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument("Unsupported datatype"));
  }

  auto max_op = builder::Const(gcu_builder, max_ptr, scalar_type);
  auto min_op = builder::Const(gcu_builder, min_ptr, scalar_type);
  auto softmax = builder::Softmax(input, axis, true, false, 0.0);
  auto res = builder::Clamp(min_op, softmax, max_op);
  return std::make_shared<GcuOp>(res);
}

}  // namespace custom_engine

REGISTER_OP_TRANSLATOR(pd_op_softmax, custom_engine::TranslateSoftmax)
REGISTER_OP_TRANSLATOR(pd_op_softmax_, custom_engine::TranslateSoftmax)
