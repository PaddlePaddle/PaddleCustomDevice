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

static GcuOpPtr TranslateAbs(
    GcuBuilderPtr gcu_builder,
    const pir::Operation *op,
    const std::vector<std::vector<GcuOpPtr>> &gcu_op_inputs) {
  auto input = *(gcu_op_inputs[0][0]);
  return std::make_shared<GcuOp>(builder::Abs(input));
}

static GcuOpPtr TranslateHardsigmoid(
    GcuBuilderPtr gcu_builder,
    const pir::Operation *op,
    const std::vector<std::vector<GcuOpPtr>> &gcu_op_inputs) {
  auto input = *(gcu_op_inputs[0][0]);
  const auto &attrs = op->attributes();
  float slope = attrs.at("slope").dyn_cast<pir::FloatAttribute>().data();
  float offset = attrs.at("offset").dyn_cast<pir::FloatAttribute>().data();
  return std::make_shared<GcuOp>(builder::HardSigmoid(input, slope, offset));
}

static GcuOpPtr TranslateHardswish(
    GcuBuilderPtr gcu_builder,
    const pir::Operation *op,
    const std::vector<std::vector<GcuOpPtr>> &gcu_op_inputs) {
  auto input = *(gcu_op_inputs[0][0]);
  return std::make_shared<GcuOp>(builder::HardSwish(input));
}

static GcuOpPtr TranslateRelu(
    GcuBuilderPtr gcu_builder,
    const pir::Operation *op,
    const std::vector<std::vector<GcuOpPtr>> &gcu_op_inputs) {
  auto input = *(gcu_op_inputs[0][0]);
  return std::make_shared<GcuOp>(builder::Relu(input));
}

static GcuOpPtr TranslateSigmoid(
    GcuBuilderPtr gcu_builder,
    const pir::Operation *op,
    const std::vector<std::vector<GcuOpPtr>> &gcu_op_inputs) {
  auto input = *(gcu_op_inputs[0][0]);
  return std::make_shared<GcuOp>(builder::Sigmoid(input));
}

static GcuOpPtr TranslateSqrt(
    GcuBuilderPtr gcu_builder,
    const pir::Operation *op,
    const std::vector<std::vector<GcuOpPtr>> &gcu_op_inputs) {
  auto input = *(gcu_op_inputs[0][0]);
  return std::make_shared<GcuOp>(builder::Sqrt(input));
}
}  // namespace custom_engine

REGISTER_OP_TRANSLATOR(pd_op_abs, custom_engine::TranslateAbs)
REGISTER_OP_TRANSLATOR(pd_op_hardsigmoid, custom_engine::TranslateHardsigmoid)
REGISTER_OP_TRANSLATOR(pd_op_hardswish, custom_engine::TranslateHardswish)
REGISTER_OP_TRANSLATOR(pd_op_relu, custom_engine::TranslateRelu)
REGISTER_OP_TRANSLATOR(pd_op_relu_, custom_engine::TranslateRelu)
REGISTER_OP_TRANSLATOR(pd_op_sigmoid, custom_engine::TranslateSigmoid)
REGISTER_OP_TRANSLATOR(pd_op_sqrt, custom_engine::TranslateSqrt)
