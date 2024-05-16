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

#include "paddle/extension.h"

void RotaryQK(const paddle::Tensor& q,
              const paddle::Tensor& kv,
              const paddle::Tensor& rotary_emb,
              const paddle::Tensor& seq_lens,
              const int32_t rotary_emb_dims,
              bool use_neox) {}

PD_BUILD_OP(encode_rotary_qk)
    .Inputs({"q", "kv", "rotary_emb", "seq_lens"})
    .Outputs({"rotary_q_out", "rotary_kv_out"})
    .SetInplaceMap({{"q", "rotary_q_out"}, {"kv", "rotary_kv_out"}})
    .Attrs({"rotary_emb_dims: int", "use_neox: bool"})
    .SetKernelFn(PD_KERNEL(RotaryQK));
