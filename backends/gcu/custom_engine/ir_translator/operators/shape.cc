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

static GcuOpPtr TranslateShape(
    GcuBuilderPtr gcu_builder,
    const pir::Operation *op,
    const std::vector<std::vector<GcuOpPtr>> &gcu_op_inputs) {
  auto x = *(gcu_op_inputs[0][0]);
  auto out = builder::Shape(x);
  out = builder::Convert(
      out, {out.GetType().GetShape(), builder::PrimitiveType::S32()});
  return std::make_shared<GcuOp>(out);
}

}  // namespace custom_engine

REGISTER_OP_TRANSLATOR(pd_op_shape, custom_engine::TranslateShape)
