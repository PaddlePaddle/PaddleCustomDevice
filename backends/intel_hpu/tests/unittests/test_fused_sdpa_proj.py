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

import unittest

import numpy as np
import paddle
import paddlenlp_ops

import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)


def fused_sdpa_proj(
    query_states,
    key_states,
    value_states,
    attention_mask,
    linear_weights,
    scaling_factor,
):
    bsz, q_len, num_heads, head_dim = query_states.shape
    attn_output = paddle.incubate.nn.functional.fused_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
        0.0,
        attention_mask is None,
        scaling_factor,
        False,
    )
    attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])

    out_linear_out = paddle.matmul(attn_output, linear_weights)

    return out_linear_out


class TestSdpa_Proj_OpFP32(unittest.TestCase):
    def setUp(self):
        self.init_dtype()
        self.batch_size = 5
        self.num_heads = 32
        self.seq_length = 1
        self.head_dim = 128

    def set_hpu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))
        paddle.seed(105)

    def init_dtype(self):
        self.dtype = "float32"

    def prepare_input(
        self,
        batch_size=5,
        num_heads=32,
        seq_length=1,
        head_dim=128,
        kv_seq_len=25,
        max_seq_length=2048,
    ):
        kv_num_heads = num_heads
        hidden_size = num_heads * head_dim

        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.head_dim = head_dim
        self.kv_seq_len = kv_seq_len

        query_states = paddle.rand(
            [batch_size, num_heads, seq_length, head_dim], dtype=paddle.float32
        ).to(paddle.bfloat16)
        key_states = paddle.rand(
            [batch_size, kv_num_heads, kv_seq_len, head_dim], dtype=paddle.float32
        ).to(paddle.bfloat16)
        value_states = paddle.rand(
            [batch_size, kv_num_heads, kv_seq_len, head_dim], dtype=paddle.float32
        ).to(paddle.bfloat16)

        attn_mask = paddle.ones(
            [1, 1, max_seq_length, max_seq_length], dtype=paddle.bfloat16
        )
        attn_mask = paddle.tril(attn_mask)
        attn_mask = (1.0 - attn_mask) * -10000.0

        linear_weights = paddle.rand(
            [hidden_size, hidden_size], dtype=paddle.float32
        ).to(paddle.bfloat16)

        return query_states, key_states, value_states, attn_mask, linear_weights

    def fused_sdpa_proj_op_custom(
        self, query_states, key_states, value_states, attn_mask, linear_weights
    ):
        attention_mask = attn_mask[..., : self.seq_length, : self.kv_seq_len]
        attention_mask = attention_mask.astype(query_states.dtype)

        out_fused_sdpa_proj_tensor = paddlenlp_ops.fused_sdpa_proj(
            query_states,
            key_states,
            value_states,
            attention_mask,
            linear_weights,
            scaling_factor=self.head_dim**-0.5,
        )
        return out_fused_sdpa_proj_tensor

    def check_result(self, torch_result, ops_result):
        np.testing.assert_allclose(torch_result, ops_result)

    def test_fused_sdpa_proj(self):
        batch_size = 5
        num_heads = 32
        seq_length = 1
        head_dim = 128
        kv_seq_len = 25
        max_seq_length = 2048
        (
            query_states,
            key_states,
            value_states,
            attn_mask,
            linear_weights,
        ) = self.prepare_input(
            batch_size, num_heads, seq_length, head_dim, kv_seq_len, max_seq_length
        )

        custom_op_result = self.fused_sdpa_proj_op_custom(
            query_states, key_states, value_states, attn_mask, linear_weights
        )

        attention_mask = attn_mask[..., : self.seq_length, : self.kv_seq_len]
        attention_mask = attention_mask.astype(query_states.dtype)
        torch_result = fused_sdpa_proj(
            query_states.transpose([0, 2, 1, 3]),
            key_states.transpose([0, 2, 1, 3]),
            value_states.transpose([0, 2, 1, 3]),
            attention_mask,
            linear_weights,
            scaling_factor=head_dim**-0.5,
        )

        self.check_result(torch_result.numpy(), custom_op_result)


if __name__ == "__main__":
    unittest.main()
