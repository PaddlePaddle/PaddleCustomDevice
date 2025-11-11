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

import numpy as np
import paddle
import paddlenlp_ops

import os
import math
import paddle.nn.functional as F

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)


def fused_sdpa(
    query_states,
    key_states,
    value_states,
    attention_mask,
    scaling_factor,
):
    attn_output = paddle.incubate.nn.functional.fused_dot_product_attention(
        query_states.transpose([0, 2, 1, 3]),
        key_states.transpose([0, 2, 1, 3]),
        value_states.transpose([0, 2, 1, 3]),
        attention_mask,
        0.0,
        False,
        scaling_factor,
        False,
    )

    return attn_output.transpose([0, 2, 1, 3])


class TestSdpaFp8(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_heads = 4
        self.seq_length = 128
        self.head_dim = 8

    def set_hpu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))
        paddle.seed(105)

    def prepare_input(
        self,
        batch_size=2,
        num_heads=4,
        seq_length=128,
        head_dim=8,
        kv_seq_len=128,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.head_dim = head_dim
        self.kv_seq_len = kv_seq_len

        query_states = 240.0 * paddle.rand(
            [batch_size, num_heads, seq_length, head_dim], dtype=paddle.float32
        ).to(paddle.bfloat16)
        key_states = 240.0 * paddle.rand(
            [batch_size, num_heads, kv_seq_len, head_dim], dtype=paddle.float32
        ).to(paddle.bfloat16)
        value_states = 240.0 * paddle.rand(
            [batch_size, num_heads, kv_seq_len, head_dim], dtype=paddle.float32
        ).to(paddle.bfloat16)

        return query_states, key_states, value_states

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

    def fused_sdpa_op_custom(self, query_states, key_states, value_states, attn_mask):
        scaleQ_hpu, scaleQInv_hpu = self.get_scale_values(query_states)
        scaleK_hpu, scaleKInv_hpu = self.get_scale_values(key_states)
        scaleV_hpu, scaleVInv_hpu = self.get_scale_values(value_states)

        with paddle.amp.auto_cast(dtype="bfloat16", enable=True):
            amax_s_ref = self.get_max_weight(query_states, key_states, scale=None)

        scaleS_hpu, scaleSInv_hpu = self.get_scale_values(amax_s_ref, is_t_amax=True)
        q_scale_s = paddle.to_tensor([scaleS_hpu])
        q_scale_o = None

        q_fp8 = (scaleQ_hpu * query_states).astype(paddle.float8_e4m3fn)
        k_fp8 = (scaleK_hpu * key_states).astype(paddle.float8_e4m3fn)
        v_fp8 = (scaleV_hpu * value_states).astype(paddle.float8_e4m3fn)

        out_fused_sdpa_tensor = paddlenlp_ops.fused_fp8_sdpa(
            q_fp8,
            k_fp8,
            v_fp8,
            attn_mask=None,
            scaling_factor=self.head_dim**-0.5,
            causal=False,
            d_scale_q=paddle.to_tensor([scaleQInv_hpu]),
            d_scale_k=paddle.to_tensor([scaleKInv_hpu]),
            d_scale_v=paddle.to_tensor([scaleVInv_hpu]),
            q_scale_s=q_scale_s,
            q_scale_o=q_scale_o,
            d_scale_s=paddle.to_tensor([scaleSInv_hpu]),
        )
        return out_fused_sdpa_tensor

    def check_result(self, torch_result, ops_result):
        np.testing.assert_allclose(torch_result, ops_result, rtol=1e-3)

    def test_fused_sdpa(self):
        batch_size = 5
        num_heads = 32
        seq_length = 1
        head_dim = 4
        kv_seq_len = 25
        (query_states, key_states, value_states) = self.prepare_input(
            batch_size, num_heads, seq_length, head_dim, kv_seq_len
        )

        custom_op_result = self.fused_sdpa_op_custom(
            query_states, key_states, value_states, None
        )

        torch_result = fused_sdpa(
            query_states.to(paddle.float8_e4m3fn).to(paddle.bfloat16),
            key_states.to(paddle.float8_e4m3fn).to(paddle.bfloat16),
            value_states.to(paddle.float8_e4m3fn).to(paddle.bfloat16),
            None,
            scaling_factor=head_dim**-0.5,
        )

        self.check_result(torch_result, custom_op_result)


if __name__ == "__main__":
    unittest.main()
