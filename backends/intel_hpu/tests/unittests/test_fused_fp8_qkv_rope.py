# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddlenlp_ops
import numpy as np

import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 4)

paddle.seed(2025)


class TestFusedFp8QkvRope(unittest.TestCase):
    def __init__(self, with_bias=False):
        self.head_dim = 128
        self.num_head = 32
        self.kv_num_heads = 32
        self.hidden_size = 4096
        self.kv_hidden_size = self.head_dim * self.kv_num_heads

        self.epsilon = 1e-06

        self.use_neox = True
        self.position_offset = 0
        self.rope_theta = 10000

        self.with_bias = with_bias

        self.init_block_prefill_params()
        self.create_tensors()

    def init_block_prefill_params(self):
        self.batch_size = 4
        self.seq_len = 34
        position_id = paddle.arange(self.seq_len, dtype=paddle.int64).to(paddle.int64)
        self.position_ids = paddle.expand(
            position_id, shape=[self.batch_size, self.seq_len]
        )

    def create_tensors(self):
        device = paddle.get_device()
        self.input_ids = paddle.zeros(
            [self.batch_size, self.seq_len], dtype=paddle.bfloat16
        )
        self.src = paddle.rand(
            [self.batch_size * self.seq_len, self.hidden_size], dtype=paddle.bfloat16
        )

        self.qkv_weights = paddle.rand(
            [self.hidden_size + 2 * self.kv_hidden_size, self.hidden_size],
            dtype=paddle.bfloat16,
        )

        if self.with_bias:
            np_qkv_biases = np.random.rand(
                self.hidden_size + 2 * self.kv_hidden_size
            ).astype("float32")
            self.qkv_biases = (
                paddle.to_tensor(np_qkv_biases, place=paddle.CPUPlace())
                .to(paddle.bfloat16)
                .to(device)
            )
        else:
            self.qkv_biases = None

        self.head_dim_shape_tensor = paddle.ones(self.head_dim, dtype="int8")

        self.new_rope = paddlenlp_ops.fused_get_rotary_embedding(
            self.input_ids,
            self.position_ids,
            self.head_dim_shape_tensor,
            self.position_offset,
            self.rope_theta,
            self.use_neox,
        ).to(paddle.bfloat16)

    def get_similarity(self, x, y):
        x = x.cpu().to("float32")
        y = y.cpu().to("float32")
        return paddle.nn.functional.cosine_similarity(
            x.flatten(), y.flatten(), axis=0
        ).item()

    def check_result(self):
        ref_query_states, ref_key_value_states = paddlenlp_ops.fused_qkv_rope_bf16(
            self.src,
            self.qkv_weights,
            self.qkv_biases,
            self.new_rope.transpose([0, 1, 3, 2, 4]),
            None,
            None,
            self.head_dim,
            self.num_head,
            self.batch_size,
            True,
            False,
            1e-6,
        )

        _, de_src_scale = paddlenlp_ops.fused_quant(self.src)
        src_scale = 1.0 / de_src_scale
        qkv_weights_fp8, d_qkv_weights_scale = paddlenlp_ops.fused_quant(
            self.qkv_weights
        )

        qkv_weights_scale = 1.0 / d_qkv_weights_scale
        ref_key_states = ref_key_value_states[0]
        ref_value_states = ref_key_value_states[1]
        _, d_out_q_scale = paddlenlp_ops.fused_quant(ref_query_states)
        _, d_out_k_scale = paddlenlp_ops.fused_quant(ref_key_states)
        _, d_out_v_scale = paddlenlp_ops.fused_quant(ref_value_states)
        out_q_scale = 1.0 / d_out_q_scale
        out_k_scale = 1.0 / d_out_k_scale
        out_v_scale = 1.0 / d_out_v_scale
        query_states_fp8, key_value_states_fp8 = paddlenlp_ops.fused_qkv_rope(
            self.src,
            qkv_weights_fp8,
            self.qkv_biases,
            self.new_rope.transpose([0, 1, 3, 2, 4]),
            src_scale,
            d_qkv_weights_scale,
            out_q_scale,
            out_k_scale,
            out_v_scale,
            None,
            None,
            self.head_dim,
            self.num_head,
            self.batch_size,
            True,
            False,
            1e-6,
        )
        key_states_fp8 = key_value_states_fp8[0]
        value_states_fp8 = key_value_states_fp8[1]
        query_states = query_states_fp8.to(paddle.bfloat16) * d_out_q_scale.item()
        key_states = key_states_fp8.to(paddle.bfloat16) * d_out_k_scale.item()
        value_states = value_states_fp8.to(paddle.bfloat16) * d_out_v_scale.item()
        key_value_states = paddle.stack([key_states, value_states], axis=0)

        similarity_query = self.get_similarity(ref_query_states, query_states)
        similarity_key_value = self.get_similarity(
            ref_key_value_states, key_value_states
        )

        assert not paddle.any(paddle.isnan(query_states)).item()
        assert not paddle.any(paddle.isnan(key_value_states)).item()

        required_similarity = 0.99
        if (
            similarity_query < required_similarity
            or similarity_key_value < required_similarity
        ):
            print(
                f"TestFusedFp8QkvRope fp8 out failed! Similarities are {similarity_query} and {similarity_key_value}."
            )
            # print("ref_query_states:", ref_query_states)
            # print("query_states_fp8:", query_states)
            # print("ref_key_value_states:", ref_key_value_states)
            # print("value_states_fp8:", key_value_states)
        else:
            print(
                f"TestFusedFp8QkvRope fp8 out passed! Similarities are {similarity_query} and {similarity_key_value}."
            )

        query_states_bf16, key_value_states_bf16 = paddlenlp_ops.fused_qkv_rope(
            self.src,
            qkv_weights_fp8,
            self.qkv_biases,
            self.new_rope.transpose([0, 1, 3, 2, 4]),
            src_scale,
            d_qkv_weights_scale,
            None,
            None,
            None,
            None,
            None,
            self.head_dim,
            self.num_head,
            self.batch_size,
            True,
            False,
            1e-6,
        )
        similarity_query = self.get_similarity(ref_query_states, query_states_bf16)
        similarity_key_value = self.get_similarity(
            ref_key_value_states, key_value_states_bf16
        )
        required_similarity = 0.99
        if (
            similarity_query < required_similarity
            or similarity_key_value < required_similarity
        ):
            print(
                f"TestFusedFp8QkvRope bf16 out failed! Similarities are {similarity_query} and {similarity_key_value}."
            )
            # print("ref_query_states:", ref_query_states)
            # print("query_states_bf16:", query_states_bf16)
            # print("ref_key_value_states:", ref_key_value_states)
            # print("key_value_states_bf16:", key_value_states_bf16)
        else:
            print(
                f"TestFusedFp8QkvRope bf16 out passed! Similarities are {similarity_query} and {similarity_key_value}."
            )

        (
            query_states_full_bf16,
            key_value_states_full_bf16,
        ) = paddlenlp_ops.fused_qkv_rope(
            self.src,
            self.qkv_weights,
            self.qkv_biases,
            self.new_rope.transpose([0, 1, 3, 2, 4]),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            self.head_dim,
            self.num_head,
            self.batch_size,
            True,
            False,
            1e-6,
        )
        similarity_query = self.get_similarity(ref_query_states, query_states_full_bf16)
        similarity_key_value = self.get_similarity(
            ref_key_value_states, key_value_states_full_bf16
        )
        required_similarity = 0.99
        if (
            similarity_query < required_similarity
            or similarity_key_value < required_similarity
        ):
            print(
                f"TestFusedFp8QkvRope _full_bf16 failed! Similarities are {similarity_query} and {similarity_key_value}."
            )
            # print("ref_query_states:", ref_query_states)
            # print("query_states_bf16:", query_states_bf16)
            # print("ref_key_value_states:", ref_key_value_states)
            # print("key_value_states_bf16:", key_value_states_bf16)
        else:
            print(
                f"TestFusedFp8QkvRope _full_bf16 passed! Similarities are {similarity_query} and {similarity_key_value}."
            )


if __name__ == "__main__":
    test = TestFusedFp8QkvRope()
    test.check_result()

    test_with_bias = TestFusedFp8QkvRope(with_bias=True)
    test_with_bias.check_result()
