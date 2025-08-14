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

#include "pattern_passes.h"  // NOLINT

static const std::vector<std::string> KPirMetaxGpuPasses{
    // Functional pass
    "add_shadow_output_after_dead_parameter_pass",
    "delete_quant_dequant_linear_op_pass",
    "delete_weight_dequant_linear_op_pass",
    "map_op_to_another_pass",
    "identity_op_clean_pass",
    // Operator fusion pass
    "silu_fuse_pass",
    "conv2d_bn_fuse_pass",
    "conv2d_add_act_fuse_pass",
    "conv2d_add_fuse_pass",
    "embedding_eltwise_layernorm_fuse_pass",
    "fused_rotary_position_embedding_pass",
    "fused_flash_attn_pass",
    "multihead_matmul_fuse_pass",
    "fused_weight_only_linear_pass",
    "matmul_add_act_fuse_pass",
    "fc_elementwise_layernorm_fuse_pass",
    // "add_norm_fuse_pass",
    "group_norm_silu_fuse_pass",
    "matmul_scale_fuse_pass",
    "matmul_transpose_fuse_pass",
    "transpose_flatten_concat_fuse_pass",
    "remove_redundant_transpose_pass",
    "horizontal_fuse_pass",
};

const std::vector<std::string>* GetPirMetaxGpuPasses() {
  return &KPirMetaxGpuPasses;
}
