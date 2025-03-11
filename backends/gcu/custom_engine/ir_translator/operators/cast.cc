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

static GcuOpPtr TranslateCast(
    GcuBuilderPtr gcu_builder,
    const pir::Operation *op,
    const std::vector<std::vector<GcuOpPtr>> &gcu_op_inputs) {
  auto in = *(gcu_op_inputs[0][0]);
  const auto &attrs = op->attributes();
  phi::DataType dtype =
      attrs.at("dtype").dyn_cast<paddle::dialect::DataTypeAttribute>().data();
  auto ptype = custom_engine::ConvertFromPhiDataType(dtype);
  auto input_ptype = in.GetType().GetPrimitiveType();
  if (ptype == input_ptype) {
    return std::make_shared<GcuOp>(builder::Reshape(in, in.GetType()));
  }

  builder::Type output_type(in.GetType().GetShape(), ptype);
  return std::make_shared<GcuOp>(builder::Convert(in, output_type));
}

}  // namespace custom_engine

REGISTER_OP_TRANSLATOR(pd_op_cast, custom_engine::TranslateCast)
