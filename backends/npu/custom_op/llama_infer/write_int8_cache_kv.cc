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

void WriteInt8CacheKV(const paddle::Tensor& input_k,
                      const paddle::Tensor& input_v,
                      const paddle::Tensor& cache_kv,
                      const paddle::Tensor& k_quant_scales,
                      const paddle::Tensor& v_quant_scales,
                      const paddle::Tensor& k_dequant_scales,
                      const paddle::Tensor& v_dequant_scales) {}

PD_BUILD_OP(write_int8_cache_kv)
    .Inputs({"input_k",
             "input_v",
             "cache_kv",
             "k_quant_scales",
             "v_quant_scales",
             "q_dequant_scales",
             "v_dequant_scales"})
    .Outputs({"cache_kv_out"})
    .SetInplaceMap({{"cache_kv", "cache_kv_out"}})
    .SetKernelFn(PD_KERNEL(WriteInt8CacheKV));
