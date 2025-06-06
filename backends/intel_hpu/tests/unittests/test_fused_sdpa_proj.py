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
import math
import paddle.nn.functional as F

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


class TestSdpa_Proj(unittest.TestCase):
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
            [batch_size, num_heads, max_seq_length, max_seq_length],
            dtype=paddle.bfloat16,
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

    def check_result(self, torch_result, ops_result, rtol=1e-07):
        np.testing.assert_allclose(torch_result, ops_result, rtol)

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


class Sdpa_Proj_Fp8(TestSdpa_Proj):
    def get_scale_values(self, t, is_t_amax=False):
        FP8_MAX_143 = 240 * 0.9
        if is_t_amax is False:
            maxT = paddle.max(paddle.abs(t)).to(paddle.float32).item()
        else:
            maxT = t.item()
        scaleT = FP8_MAX_143 / maxT

        lg2 = math.log2(scaleT)
        lg2_int = int(lg2)

        scaleT_pow2 = 2.0**lg2_int
        scaleTInv = 1.0 / scaleT_pow2

        return scaleT_pow2, scaleTInv

    def get_max_weight(
        self,
        query,
        key,
        scale=None,
    ):
        sqrt_dim_head = query.shape[-1] ** 0.5
        scores = paddle.matmul(
            query,
            key,
            transpose_x=False,
            transpose_y=True,
        )
        if scale is None:
            scores = scores / sqrt_dim_head
        else:
            scores = scores * scale

        weight = F.softmax(scores, axis=-1)

        return paddle.max(paddle.abs(weight)).to(paddle.float32)

    def fused_sdpa_proj_op_custom(
        self, query_states, key_states, value_states, attn_mask, linear_weights
    ):
        attention_mask = attn_mask[..., : self.seq_length, : self.kv_seq_len]
        attention_mask = attention_mask.astype(paddle.bfloat16)

        scaleQ, scaleQInv = self.get_scale_values(query_states)
        scaleK, scaleKInv = self.get_scale_values(key_states)
        scaleV, scaleVInv = self.get_scale_values(value_states)

        with paddle.amp.auto_cast(dtype="bfloat16", enable=True):
            amax_s_ref = self.get_max_weight(query_states, key_states, scale=None)

        scaleS, scaleSInv = self.get_scale_values(amax_s_ref, is_t_amax=True)

        q_fp8 = (scaleQ * query_states).astype(paddle.float8_e4m3fn)
        k_fp8 = (scaleK * key_states).astype(paddle.float8_e4m3fn)
        v_fp8 = (scaleV * value_states).astype(paddle.float8_e4m3fn)
        linear_weights_fp8 = linear_weights.astype(paddle.float8_e4m3fn)

        d_scale_q = paddle.to_tensor([scaleQInv])
        d_scale_k = paddle.to_tensor([scaleKInv])
        d_scale_v = paddle.to_tensor([scaleVInv])
        q_scale_s = paddle.to_tensor([scaleS])
        q_scale_o = None
        d_scale_s = paddle.to_tensor([scaleSInv])

        out_fused_sdpa_proj_tensor = paddlenlp_ops.fused_fp8_sdpa_proj(
            q_fp8,
            k_fp8,
            v_fp8,
            attention_mask,
            linear_weights_fp8,
            scaling_factor=self.head_dim**-0.5,
            d_scale_q=d_scale_q,
            d_scale_k=d_scale_k,
            d_scale_v=d_scale_v,
            q_scale_s=q_scale_s,
            q_scale_o=q_scale_o,
            d_scale_s=d_scale_s,
        )

        return out_fused_sdpa_proj_tensor

    def test_fused_sdpa_proj(self):
        batch_size = 5
        num_heads = 32
        seq_length = 1
        head_dim = 128
        kv_seq_len = 5
        max_seq_length = 100
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

        self.check_result(torch_result.numpy(), custom_op_result, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
