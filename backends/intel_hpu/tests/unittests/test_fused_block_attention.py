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

import numpy as np

import paddle
import paddlenlp_ops
import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)
paddle.device.set_device(f"intel_hpu:{intel_hpus_module_id}")

seed = 102
paddle.seed(seed)
np.random.seed(seed)


class TestFusedBlockAttention:
    def __init__(self):
        self.head_dim = 128
        self.num_head = 32
        self.hidden_size = self.num_head * self.head_dim

        self.epsilon = 1e-06

        self.use_neox = True
        self.position_offset = 0
        self.rope_theta = 10000

    def init_decode_MHA_params(self):
        self.test_name = "Test_MHA_FusedBlockAttentionDecode"
        self.kv_num_heads = 32
        self.kv_hidden_size = self.head_dim * self.kv_num_heads
        self.qkv_biases = None

        self.batch_size = 16
        self.seq_len = 1
        self.block_size = 128
        self.num_of_block = 32
        self.total_block_num = 20
        position_id = paddle.to_tensor([80])
        self.position_ids = paddle.expand(
            position_id, shape=[self.batch_size, self.seq_len]
        )

    def init_decode_GQA_params(self):
        self.test_name = "Test_GQA_FusedBlockAttentionDecode"
        self.kv_num_heads = 4
        self.kv_hidden_size = self.head_dim * self.kv_num_heads
        self.qkv_biases = None

        self.batch_size = 16
        self.seq_len = 1
        self.block_size = 128
        self.num_of_block = 32
        self.total_block_num = 20
        position_id = paddle.to_tensor([80])
        self.position_ids = paddle.expand(
            position_id, shape=[self.batch_size, self.seq_len]
        )

    def init_decode_MHA_QKVbias_params(self):
        self.test_name = "Test_MHA_QKVbias_FusedBlockAttentionDecode"
        self.kv_num_heads = 32
        self.kv_hidden_size = self.head_dim * self.kv_num_heads
        self.qkv_biases = 1

        self.batch_size = 16
        self.seq_len = 1
        self.block_size = 128
        self.num_of_block = 32
        self.total_block_num = 20
        position_id = paddle.to_tensor([80])
        self.position_ids = paddle.expand(
            position_id, shape=[self.batch_size, self.seq_len]
        )

    def init_decode_GQA_QKVbias_params(self):
        self.test_name = "Test_GQA_QKVbias_FusedBlockAttentionDecode"
        self.kv_num_heads = 4
        self.kv_hidden_size = self.head_dim * self.kv_num_heads
        self.qkv_biases = 1

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
        device = paddle.get_device()

        np_k_cache = np.random.rand(
            self.total_block_num, self.block_size, self.kv_num_heads, self.head_dim
        ).astype("float32")
        self.k_cache = (
            paddle.to_tensor(np_k_cache, place=paddle.CPUPlace())
            .to(paddle.bfloat16)
            .to(device)
        )
        self.k_cache_test = self.k_cache.clone()

        np_v_cache = np.random.rand(
            self.total_block_num, self.block_size, self.kv_num_heads, self.head_dim
        ).astype("float32")
        self.v_cache = (
            paddle.to_tensor(np_v_cache, place=paddle.CPUPlace())
            .to(paddle.bfloat16)
            .to(device)
        )
        self.v_cache_test = self.v_cache.clone()

        self.input_ids = paddle.zeros(
            [self.batch_size, self.seq_len], dtype=paddle.bfloat16
        )

        np_src = np.random.rand(self.batch_size, self.seq_len, self.hidden_size).astype(
            "float32"
        )
        self.src = (
            paddle.to_tensor(np_src, place=paddle.CPUPlace())
            .to(paddle.bfloat16)
            .to(device)
        )

        np_residual = np.random.rand(
            self.batch_size, self.seq_len, self.hidden_size
        ).astype("float32")
        self.residual = (
            paddle.to_tensor(np_residual, place=paddle.CPUPlace())
            .to(paddle.bfloat16)
            .to(device)
        )
        self.residual_test = self.residual.clone()

        np_ln_scales = np.random.rand(self.hidden_size).astype("float32")
        self.ln_scales = (
            paddle.to_tensor(np_ln_scales, place=paddle.CPUPlace())
            .to(paddle.bfloat16)
            .to(device)
        )

        np_qkv_weights = np.random.rand(
            self.hidden_size + 2 * self.kv_hidden_size, self.hidden_size
        ).astype("float32")
        self.qkv_weights = (
            paddle.to_tensor(np_qkv_weights, place=paddle.CPUPlace())
            .to(paddle.bfloat16)
            .to(device)
        )

        if self.qkv_biases is not None:
            np_qkv_biases = np.random.rand(
                self.hidden_size + 2 * self.kv_hidden_size
            ).astype("float32")
            self.qkv_biases = (
                paddle.to_tensor(np_qkv_biases, place=paddle.CPUPlace())
                .to(paddle.bfloat16)
                .to(device)
            )

        np_linear_weights = np.random.rand(self.hidden_size, self.hidden_size).astype(
            "float32"
        )
        self.linear_weights = (
            paddle.to_tensor(np_linear_weights, place=paddle.CPUPlace())
            .to(paddle.bfloat16)
            .to(device)
        )

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

        np_block_bias = np.random.rand(self.num_of_block, self.block_size).astype(
            "float32"
        )
        self.block_bias = (
            paddle.to_tensor(np_block_bias, place=paddle.CPUPlace())
            .to(paddle.bfloat16)
            .to(device)
        )
        self.q_rmsnorm_gamma = paddle.randint(
            0, 1, [self.head_dim], dtype=paddle.int32
        ).to(paddle.bfloat16)
        self.k_rmsnorm_gamma = paddle.randint(
            0, 1, [self.head_dim], dtype=paddle.int32
        ).to(paddle.bfloat16)

    def run_test(self):
        query_states, key_value_states = paddlenlp_ops.fused_rms_qkv_rope_t(
            self.src,
            self.ln_scales,
            self.qkv_weights,
            self.qkv_biases,
            self.new_rope.transpose([0, 1, 3, 2, 4]),
            self.residual,
            self.q_rmsnorm_gamma,
            self.k_rmsnorm_gamma,
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

        src, self.residual_test = paddle.incubate.nn.functional.fused_rms_norm(
            self.src,
            norm_weight=self.ln_scales,
            norm_bias=None,
            epsilon=self.epsilon,
            begin_norm_axis=2,
            bias=None,
            residual=self.residual_test,
        )

        b, s, h = src.shape
        src = src.reshape([-1, h])
        out_linear_out = paddlenlp_ops.fused_block_attention(
            src,
            self.new_rope.transpose([0, 1, 3, 2, 4]).squeeze(2),
            self.k_cache_test,
            self.v_cache_test,
            self.block_groups,
            self.block_list,
            self.block_mapping,
            self.block_bias,
            self.block_indices,
            self.block_offsets,
            self.qkv_weights,
            self.qkv_biases,
            self.linear_weights,
            self.q_rmsnorm_gamma,
            self.k_rmsnorm_gamma,
            self.head_dim,
            self.num_head,
            scaling_factor=self.head_dim**-0.5,
            transpose=True,
            use_neox_style=True,
            epsilon=self.epsilon,
        ).reshape([b, -1, h])

        assert paddle.allclose(
            out_linear_out_ref.to("cpu").to("float32"),
            out_linear_out.to("cpu").to("float32"),
            rtol=1e-2,
        ), f"Test failed for {self.test_name} fused_block_attention out_linear_out"

        close_mask = paddle.isclose(
            self.k_cache.to("cpu").to("float32"),
            self.k_cache_test.to("cpu").to("float32"),
            rtol=1e-1,
        )
        mismatch_count = paddle.sum(~close_mask).item()
        mismatch_percentage = mismatch_count / close_mask.numel() * 100.0
        assert (
            mismatch_percentage <= 1e-3
        ), f"k_cache Mismatched elements percentage: {mismatch_percentage:}% > {1e-3}% threshold\n"

        close_mask = paddle.isclose(
            self.v_cache.to("cpu").to("float32"),
            self.v_cache_test.to("cpu").to("float32"),
            rtol=1e-2,
        )
        mismatch_count = paddle.sum(~close_mask).item()
        mismatch_percentage = mismatch_count / close_mask.numel() * 100.0
        assert (
            mismatch_percentage <= 1e-3
        ), f"v_cache Mismatched elements percentage: {mismatch_percentage:}% > {1e-3}% threshold\n"

        assert paddle.allclose(
            self.residual.to("cpu").to("float32"),
            self.residual_test.to("cpu").to("float32"),
            rtol=1e-2,
        ), f"Test failed for {self.test_name} fused_block_attention residual"

        # ===============summary==============
        print(f"Test Pass for {self.test_name} testcase")


class test_case_decode_MHA(TestFusedBlockAttention):
    def __init__(self):
        super().__init__()
        self.init_decode_MHA_params()
        self.create_tensors()


class test_case_decode_GQA(TestFusedBlockAttention):
    def __init__(self):
        super().__init__()
        self.init_decode_GQA_params()
        self.create_tensors()


class test_case_decode_MHA_QKVbias(TestFusedBlockAttention):
    def __init__(self):
        super().__init__()
        self.init_decode_MHA_QKVbias_params()
        self.create_tensors()


class test_case_decode_GQA_QKVbias(TestFusedBlockAttention):
    def __init__(self):
        super().__init__()
        self.init_decode_GQA_QKVbias_params()
        self.create_tensors()


if __name__ == "__main__":
    test_1 = test_case_decode_MHA()
    test_1.run_test()

    test_2 = test_case_decode_GQA()
    test_2.run_test()

    test_3 = test_case_decode_MHA_QKVbias()
    test_3.run_test()

    test_4 = test_case_decode_GQA_QKVbias()
    test_4.run_test()
