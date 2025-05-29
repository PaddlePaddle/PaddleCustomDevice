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

paddle.device.set_device("intel_hpu:1")

# paddle.seed(102)


class TestFusedBlockAttention:
    def __init__(self):
        self.head_dim = 128
        self.num_head = 32
        self.kv_num_heads = 32
        self.hidden_size = self.num_head * self.head_dim

        self.epsilon = 1e-06

        self.use_neox = True
        self.position_offset = 0
        self.rope_theta = 10000

    def init_decode_params(self):
        self.test_name = "TestFusedBlockAttentionDecode"
        self.batch_size = 16
        self.seq_len = 1
        self.block_size = 128
        self.num_of_block = 32
        self.total_block_num = 20
        position_id = paddle.to_tensor([80])
        self.position_ids = paddle.expand(
            position_id, shape=[self.batch_size, self.seq_len]
        )

    def create_tensors(self):
        self.k_cache = (
            paddle.rand(
                [
                    self.total_block_num,
                    self.block_size,
                    self.kv_num_heads,
                    self.head_dim,
                ],
                dtype=paddle.float32,
            )
            * 1000
        )
        self.k_cache = self.k_cache.to(paddle.bfloat16)
        self.k_cache_test = self.k_cache.clone()
        self.v_cache = (
            paddle.rand(
                [
                    self.total_block_num,
                    self.block_size,
                    self.kv_num_heads,
                    self.head_dim,
                ],
                dtype=paddle.float32,
            )
            * 1000
        )
        self.v_cache = self.v_cache.to(paddle.bfloat16)
        self.v_cache_test = self.v_cache.clone()

        self.input_ids = paddle.zeros(
            [self.batch_size, self.seq_len], dtype=paddle.bfloat16
        )
        self.src = paddle.rand(
            [self.batch_size, self.seq_len, self.hidden_size], dtype=paddle.float32
        ).to(paddle.bfloat16)
        self.residual = paddle.rand(
            [self.batch_size, self.seq_len, self.hidden_size], dtype=paddle.float32
        ).to(paddle.bfloat16)
        self.residual_test = self.residual.clone()

        self.ln_scales = paddle.rand([self.hidden_size], dtype=paddle.bfloat16)
        self.qkv_weights = paddle.rand(
            [self.hidden_size * 3, self.hidden_size], dtype=paddle.float32
        )
        self.qkv_weights = self.qkv_weights.to(paddle.bfloat16)

        self.linear_weights = paddle.rand(
            [self.hidden_size, self.hidden_size], dtype=paddle.float32
        ).to(paddle.bfloat16)

        self.head_dim_shape_tensor = paddle.ones(self.head_dim, dtype="int8")
        self.new_rope = paddlenlp_ops.fused_get_rotary_embedding(
            self.input_ids,
            self.position_ids,
            self.head_dim_shape_tensor,
            self.position_offset,
            self.rope_theta,
            self.use_neox,
        ).to(paddle.bfloat16)

        self.block_indices = paddle.randint(
            0,
            self.total_block_num,
            [
                self.batch_size,
            ],
            dtype=paddle.int32,
        )
        self.block_offsets = paddle.randint(
            0,
            self.block_size,
            [
                self.batch_size,
            ],
            dtype=paddle.int32,
        )

        self.block_groups = paddle.randint(
            0,
            self.batch_size,
            [
                self.num_of_block,
            ],
            dtype=paddle.int32,
        )
        self.block_list = paddle.randint(
            0,
            self.num_of_block,
            [
                self.num_of_block,
            ],
            dtype=paddle.int32,
        )
        self.block_mapping = paddle.randint(
            0, 2, [self.num_of_block, self.batch_size], dtype=paddle.int32
        ).to(paddle.bfloat16)
        self.block_bias = paddle.rand(
            [self.num_of_block, self.block_size], dtype=paddle.bfloat16
        )

    def run_test(self):
        query_states, key_value_states = paddlenlp_ops.fused_rms_qkv_rope_t(
            self.src,
            self.ln_scales,
            self.qkv_weights,
            self.new_rope.transpose([0, 1, 3, 2, 4]),
            self.residual,
            self.epsilon,
            self.head_dim,
            self.num_head,
        )
        key_states = key_value_states[0].squeeze(1)
        value_states = key_value_states[1].squeeze(1)

        self.k_cache.index_put_((self.block_indices, self.block_offsets), key_states)
        self.v_cache.index_put_((self.block_indices, self.block_offsets), value_states)

        out_linear_out_ref = paddlenlp_ops.fused_flatpa_proj(
            query_states,
            self.k_cache,
            self.v_cache,
            self.block_groups,
            self.block_list,
            self.block_mapping,
            self.block_bias,
            self.linear_weights,
            scaling_factor=self.head_dim**-0.5,
        )

        out_linear_out = paddlenlp_ops.fused_block_attention(
            self.src,
            self.residual_test,
            self.new_rope.transpose([0, 1, 3, 2, 4]),
            self.k_cache_test,
            self.v_cache_test,
            self.block_groups,
            self.block_list,
            self.block_mapping,
            self.block_bias,
            self.block_indices,
            self.block_offsets,
            self.ln_scales,
            self.qkv_weights,
            self.linear_weights,
            self.epsilon,
            self.head_dim,
            self.num_head,
            scaling_factor=self.head_dim**-0.5,
        )

        assert (
            (out_linear_out_ref == out_linear_out).all().item()
        ), f"Test failed for {self.test_name} fused_block_attention out_linear_out"
        assert (
            (self.k_cache == self.k_cache_test).all().item()
        ), f"Test failed for {self.test_name} fused_block_attention k_cache"
        assert (
            (self.v_cache == self.v_cache_test).all().item()
        ), f"Test failed for {self.test_name} fused_block_attention v_cache"
        assert (
            (self.residual == self.residual_test).all().item()
        ), f"Test failed for {self.test_name} fused_block_attention residual"

        # ===============summary==============
        print(f"Test Pass for {self.test_name} testcase")


class test_case_decode(TestFusedBlockAttention):
    def __init__(self):
        super().__init__()
        self.init_decode_params()
        self.create_tensors()


if __name__ == "__main__":
    test_1 = test_case_decode()
    test_1.run_test()
