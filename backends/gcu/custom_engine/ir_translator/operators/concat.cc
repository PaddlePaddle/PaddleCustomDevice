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

static GcuOpPtr TranslateConcat(
    GcuBuilderPtr gcu_builder,
    const pir::Operation *op,
    const std::vector<std::vector<GcuOpPtr>> &gcu_op_inputs) {
  auto in_tensors = gcu_op_inputs[0];
  auto axis_tensor = *(gcu_op_inputs[1][0]);
  PADDLE_ENFORCE_EQ(axis_tensor.IsConstant(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Input[1] axis_tensor is not a Constant."));
  int64_t axis = axis_tensor.GetConstData<int64_t>().at(0);
  std::vector<builder::Op> ops;
  auto input_num = in_tensors.size();
  for (size_t i = 0; i < input_num; ++i) {
    ops.emplace_back(*(in_tensors.at(i)));
  }
  if (axis < 0) {
    axis += ops[0].GetType().GetRank();
  }
  return std::make_shared<GcuOp>(builder::Concatenate(ops, axis));
}

}  // namespace custom_engine

REGISTER_OP_TRANSLATOR(pd_op_concat, custom_engine::TranslateConcat)
