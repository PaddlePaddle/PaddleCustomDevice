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

#include "passes/gcu_pass_pipeline.h"

static const std::vector<std::string> kPirGcuPasses{
    // Functional pass
    "add_shadow_output_after_dead_parameter_pass",
    "delete_quant_dequant_linear_op_pass",
    "delete_weight_dequant_linear_op_pass",
    "map_op_to_another_pass",
    "identity_op_clean_pass",
    // Operator exchange pass
    "matmul_scale_fuse_pass",
    // Operator fusion pass
    "silu_fuse_pass",
    "gcu_conv2d_bn_fuse_pass",
    "gcu_conv2d_add_fuse_pass",
    "fused_conv2d_add_act_append_pass",
    // "matmul_transpose_fuse_pass",
    // "matmul_add_act_fuse_pass",
    // "multihead_matmul_fuse_pass",
    // "embedding_eltwise_layernorm_fuse_pass",
    // "fc_elementwise_layernorm_fuse_pass",
    // "fused_dot_product_attention_pass",
    // "fused_dropout_add_pass",
    // "fused_flash_attn_pass",
    // "fused_rotary_position_embedding_pass",
    // "horizontal_fuse_pass",
    // "transpose_flatten_concat_fuse_pass",
};

const std::vector<std::string>* GetPirGcuPasses() { return &kPirGcuPasses; }
