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
import unittest
from parameterized import parameterized

import os
import math
import numpy as np
import paddle.nn.functional as F


intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 1)
paddle.device.set_device(f"intel_hpu:{intel_hpus_module_id}")

paddle.seed(105)


def get_scale_values(t, is_t_amax=False):
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


def ref_result(
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
    attn_output = attn_output.reshape([bsz * q_len, head_dim * num_heads])

    out_linear_out = paddle.matmul(attn_output, linear_weights)

    return out_linear_out


HEAD_DIM = [32]
NUM_HEAD = [8]
BATCH_SIZE = [4, 8, 16]
SEQ_LEN = [16]
KV_SEQ_LEN = [16]
MAX_SEQ_LENGTH = [2048]


class FP8_SDPA_Proj_T_Test(unittest.TestCase):
    @parameterized.expand(
        [
            (
                head_dim,
                num_head,
                batch_size,
                seq_len,
                kv_seq_len,
                max_seq_length,
            )
            for head_dim in HEAD_DIM
            for num_head in NUM_HEAD
            for batch_size in BATCH_SIZE
            for seq_len in SEQ_LEN
            for kv_seq_len in KV_SEQ_LEN
            for max_seq_length in MAX_SEQ_LENGTH
        ]
    )
    def test(
        self,
        head_dim,
        num_head,
        batch_size,
        seq_len,
        kv_seq_len,
        max_seq_length,
    ):
        kv_num_head = num_head
        hidden_size = num_head * head_dim
        scaling_factor = head_dim**-0.5

        query_states = paddle.rand(
            [batch_size, seq_len, num_head, head_dim], dtype=paddle.float32
        ).to(paddle.bfloat16)
        key_states = paddle.rand(
            [batch_size, kv_seq_len, kv_num_head, head_dim], dtype=paddle.float32
        ).to(paddle.bfloat16)
        value_states = paddle.rand(
            [batch_size, kv_seq_len, kv_num_head, head_dim], dtype=paddle.float32
        ).to(paddle.bfloat16)

        linear_weights = paddle.rand(
            [hidden_size, hidden_size], dtype=paddle.float32
        ).to(paddle.bfloat16)

        scaleQ, scaleQInv = get_scale_values(query_states)
        scaleK, scaleKInv = get_scale_values(key_states)
        scaleV, scaleVInv = get_scale_values(value_states)

        with paddle.amp.auto_cast(dtype="bfloat16", enable=True):
            amax_s_ref = get_max_weight(query_states, key_states, scale=None)

        scaleS, scaleSInv = get_scale_values(amax_s_ref, is_t_amax=True)

        q_fp8 = (scaleQ * query_states).astype(paddle.float8_e4m3fn)
        kv_fp8 = paddle.stack(
            [scaleK * key_states, scaleV * value_states], axis=0
        ).astype(paddle.float8_e4m3fn)

        scale_one = paddle.to_tensor([1.0], dtype=paddle.float32)
        linear_weights_fp8 = linear_weights.transpose([1, 0]).astype(
            paddle.float8_e4m3fn
        )

        d_scale_q = paddle.to_tensor([scaleQInv])
        d_scale_k = paddle.to_tensor([scaleKInv])
        d_scale_v = paddle.to_tensor([scaleVInv])
        q_scale_s = paddle.to_tensor([scaleS])
        q_scale_o = None
        d_scale_s = paddle.to_tensor([scaleSInv])

        out_linear_out_ref = ref_result(
            query_states,
            key_states,
            value_states,
            None,
            linear_weights,
            scaling_factor,
        )

        out_linear_t_op = paddlenlp_ops.fused_fp8_sdpa_proj_t(
            q_fp8,
            kv_fp8,
            None,
            None,
            linear_weights_fp8,
            d_scale_q,
            d_scale_k,
            d_scale_v,
            q_scale_s,
            q_scale_o,
            d_scale_s,
            scale_one,
            scale_one,
            scaling_factor,
            causal=True,
        )

        np.testing.assert_allclose(out_linear_out_ref, out_linear_t_op, rtol=1e-2)


if __name__ == "__main__":
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(FP8_SDPA_Proj_T_Test)

    # Create a test runner with the desired verbosity level
    runner = unittest.TextTestRunner(
        verbosity=2
    )  # Set verbosity to 2 for detailed output

    # Run the test suite
    runner.run(suite)
