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

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)

paddle.seed(2025)


class TestFusedFp8RmsQkvRopeT(unittest.TestCase):
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
        self.batch_size = 1
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
        self.src = (
            240
            * paddle.rand(
                [self.batch_size, self.seq_len, self.hidden_size], dtype=paddle.float32
            )
        ).to(paddle.bfloat16)
        self.residual = paddle.zeros_like(self.src, dtype=paddle.bfloat16)

        self.ln_scales = paddle.rand([self.hidden_size], dtype=paddle.bfloat16)
        self.qkv_weights = (
            240
            * paddle.rand(
                [self.hidden_size * 3, self.hidden_size], dtype=paddle.float32
            )
        ).to(paddle.bfloat16)

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
        ref_query_states, ref_key_value_states = paddlenlp_ops.fused_rms_qkv_rope_t(
            self.src,
            self.ln_scales,
            self.qkv_weights,
            self.qkv_biases,
            self.new_rope.transpose([0, 1, 3, 2, 4]),
            self.residual,
            None,
            None,
            self.epsilon,
            self.head_dim,
            self.num_head,
        )

        qkv_weights_fp8 = self.qkv_weights.to(paddle.float8_e4m3fn)
        query_states, key_value_states = paddlenlp_ops.fused_fp8_rms_qkv_rope_t(
            self.src,
            self.ln_scales,
            qkv_weights_fp8,
            self.qkv_biases,
            self.new_rope.transpose([0, 1, 3, 2, 4]),
            self.residual,
            paddle.to_tensor([1.0]),
            paddle.to_tensor([1.0]),
            None,
            None,
            self.epsilon,
            self.head_dim,
            self.num_head,
        )

        similarity_query = self.get_similarity(ref_query_states, query_states)
        similarity_key_value = self.get_similarity(
            ref_key_value_states, key_value_states
        )

        assert not paddle.any(paddle.isnan(query_states)).item()
        assert not paddle.any(paddle.isnan(key_value_states)).item()
        assert not paddle.any(paddle.isinf(query_states)).item()
        assert not paddle.any(paddle.isinf(key_value_states)).item()

        assert abs(1 - similarity_query) < 1e-4
        assert abs(1 - similarity_key_value) < 1e-4

        print(
            f"TestFusedFp8RmsQkvRopeT passed! Similarities are {similarity_query} and {similarity_key_value}."
        )


if __name__ == "__main__":
    test = TestFusedFp8RmsQkvRopeT()
    test.check_result()

    test_with_bias = TestFusedFp8RmsQkvRopeT(with_bias=True)
    test_with_bias.check_result()
