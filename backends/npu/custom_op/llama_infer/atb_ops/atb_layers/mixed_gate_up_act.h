// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#ifdef PADDLE_WITH_ATB

#include "atb/atb_infer.h"

namespace atb_layers {

struct MixedGateUpActParam {
  bool trans_weight;
  atb::infer::ActivationType act;
  float scale{1.0};  // for swish
  bool linear_quant{false};
  bool input_quant{false};
  float input_quant_scale{1.0};
  int input_quant_offset{0};
  bool input_smooth_quant{false};
  bool has_dequant_offset{false};
};

void CreateMixedGateUpAct(const MixedGateUpActParam& param,
                          atb::Operation** operation);
}  // namespace atb_layers

namespace atb {
template <>
inline Status CreateOperation(const atb_layers::MixedGateUpActParam& opParam,
                              Operation** operation) {
  atb_layers::CreateMixedGateUpAct(opParam, operation);
  return ErrorType::NO_ERROR;
}
}  // namespace atb

#endif
