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
const char *const kFloor = "floor";
const char *const kFloorGrad = "floor_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, op, map_inputs, running_mode, FloorEquivalenceTrans) {
  auto res = builder::Floor(*(map_inputs["X"].at(0)));
  return std::make_shared<builder::Op>(res);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, op, map_inputs, running_mode, FloorGradEquivalenceTrans) {
  // x_grad = 0 * out_grad
  auto res = builder::ZerosLike(*(map_inputs["Out@GRAD"].at(0)));
  return std::make_shared<builder::Op>(res);
}

EQUIVALENCE_TRANS_FUNC_REG(kFloor, INSENSITIVE, FloorEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kFloorGrad, INSENSITIVE, FloorGradEquivalenceTrans);

}  // namespace backend
