# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddlenlp_ops
import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)
paddle.device.set_device(f"intel_hpu:{intel_hpus_module_id}")

paddle.seed(102)

batch_size = 8
q_head = 32
head_dim = 128
hidden_size = q_head * head_dim
total_block_num = 40
block_size = 64
num_of_block = 12
scaling_factor = head_dim**-0.5

query = paddle.rand([batch_size, 1, q_head, head_dim], dtype=paddle.bfloat16)
block_list = paddle.rand([num_of_block], dtype=paddle.int32)
block_groups = paddle.rand([num_of_block], dtype=paddle.int32)
block_mapping = paddle.rand([num_of_block, batch_size], dtype=paddle.bfloat16)
attn_bias = paddle.rand([num_of_block, block_size], dtype=paddle.bfloat16)
linear_weights = paddle.rand([hidden_size, hidden_size], dtype=paddle.bfloat16)


def test_fused_flatpa_proj(kv_head, testcase):
    key_cache = paddle.rand(
        [total_block_num, block_size, kv_head, head_dim], dtype=paddle.bfloat16
    )
    value_cache = paddle.rand(
        [total_block_num, block_size, kv_head, head_dim], dtype=paddle.bfloat16
    )

    out_linear_ref = paddlenlp_ops.fused_flatpa_proj_ref(
        query,
        key_cache,
        value_cache,
        block_groups,
        block_list,
        block_mapping,
        attn_bias,
        linear_weights,
        scaling_factor=scaling_factor,
    )

    out_linear_out = paddlenlp_ops.fused_flatpa_proj(
        query,
        key_cache,
        value_cache,
        block_groups,
        block_list,
        block_mapping,
        attn_bias,
        linear_weights,
        scaling_factor=scaling_factor,
    )

    assert (
        (out_linear_out == out_linear_ref).all().item()
    ), f"Test failed for kv_head={kv_head}"
    print(f"Test Pass for {testcase} testcase")


test_fused_flatpa_proj(kv_head=32, testcase="MHA")
test_fused_flatpa_proj(kv_head=8, testcase="GQA")
