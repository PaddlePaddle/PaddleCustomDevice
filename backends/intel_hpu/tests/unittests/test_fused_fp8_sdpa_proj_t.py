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
import os

# os.environ['ENABLE_EXPERIMENTAL_FLAGS'] = '1'
# os.environ['VISUALIZATION_MODE'] = '0'
# os.environ['GRAPH_VISUALIZATION'] = '1'
# os.environ['HABANA_LOGS'] = 'logs'
# os.environ['LOG_LEVEL_ALL'] = '0'
# os.environ['LOG_LEVEL_PERF_LIB'] = '0'

import paddle
import paddlenlp_ops
import unittest
from parameterized import parameterized

import numpy as np
import paddle.nn.functional as F


intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 1)
paddle.device.set_device(f"intel_hpu:{intel_hpus_module_id}")

paddle.seed(105)


def get_scale_values(t, is_t_amax=False):
    FP8_MAX_143 = 240
    if is_t_amax is False:
        maxT = paddle.max(paddle.abs(t)).to(paddle.float32).item()
    else:
        maxT = t.item()
    scaleT = FP8_MAX_143 / maxT
    scaleTInv = 1.0 / scaleT

    return scaleT, scaleTInv


def get_max_weight(
    query,
    key,
    scale=None,
):
    sqrt_dim_head = query.shape[-1] ** 0.5

    if is_gqa(query, key):
        key, _ = gqa_input_reshape_fwd(query, key, key)
    scores = paddle.matmul(
        query.transpose([0, 2, 1, 3]),
        key.transpose([0, 2, 1, 3]),
        transpose_x=False,
        transpose_y=True,
    )
    if scale is None:
        scores = scores / sqrt_dim_head
    else:
        scores = scores * scale

    weight = F.softmax(scores, axis=-1)

    return paddle.max(paddle.abs(weight)).to(paddle.float32)


def is_gqa(q, k):
    gqa = False
    dims = q.dim()
    if dims == 4:
        q_heads = q.shape[2]
        kv_heads = k.shape[2]
        gqa = (q_heads != kv_heads) and kv_heads != 1
    return gqa


def gqa_input_reshape_fwd(q, k, v):
    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    q_heads_per_group = q_heads // kv_heads

    k = k.repeat_interleave(q_heads_per_group, axis=2)
    v = v.repeat_interleave(q_heads_per_group, axis=2)

    return k, v


def check_using_cosine_similarity(final_states, final_states_ref):
    vec1 = final_states.reshape(-1)
    vec2 = final_states_ref.reshape(-1)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        cos_sim = 1.0 if np.array_equal(vec1, vec2) else 0.0
    else:
        cos_sim = np.dot(vec1, vec2) / (norm1 * norm2)

    # print(f"Cosine similarity: {cos_sim}")
    return cos_sim


def ref_result(
    query_states,
    key_states,
    value_states,
    attention_mask,
    linear_weights,
    scaling_factor,
):
    bsz, q_len, num_heads, head_dim = query_states.shape

    if is_gqa(query_states, key_states):
        key_states, value_states = gqa_input_reshape_fwd(
            query_states, key_states, value_states
        )

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

    return out_linear_out, attn_output


BATCH_SIZE = [1, 4]
SEQ_LEN = [128]
KV_SEQ_LEN = [128, 1024]
NUM_HEAD = [64]
KV_NUM_HEAD = [8, 64]
HEAD_DIM = [128]
MAX_SEQ_LENGTH = [2048]
SCALE_O = [None, paddle.to_tensor([1.0], dtype=paddle.float32)]
BF16_FP8_MODE = ["ALL_BF16", "BF16_SDPA_FP8_PROJ", "ALL_FP8"]
MULTI_CARD = [1, 4]
IS_CAUSAL = [True, False]

"""
BATCH_SIZE = [4]
KV_NUM_HEAD = [64]
SEQ_LEN = [128]
KV_SEQ_LEN = [1024]
BF16_FP8_MODE = ["ALL_BF16"]
SCALE_O = [None]
IS_CAUSAL = [False]
MULTI_CARD = [1]
"""


class FP8_SDPA_Proj_T_Test(unittest.TestCase):
    @parameterized.expand(
        [
            (
                head_dim,
                num_head,
                kv_num_head,
                batch_size,
                seq_len,
                kv_seq_len,
                max_seq_length,
                scale_o,
                bf16_fp8_mode,
                tp_size,
                is_causal,
            )
            for head_dim in HEAD_DIM
            for num_head in NUM_HEAD
            for kv_num_head in KV_NUM_HEAD
            for batch_size in BATCH_SIZE
            for seq_len in SEQ_LEN
            for kv_seq_len in KV_SEQ_LEN
            for max_seq_length in MAX_SEQ_LENGTH
            for scale_o in SCALE_O
            for bf16_fp8_mode in BF16_FP8_MODE
            for tp_size in MULTI_CARD
            for is_causal in IS_CAUSAL
        ]
    )
    def test(
        self,
        head_dim,
        num_head,
        kv_num_head,
        batch_size,
        seq_len,
        kv_seq_len,
        max_seq_length,
        scale_o,
        bf16_fp8_mode,
        tp_size,
        is_causal,
    ):
        # print(
        #     f"Test for head_dim={head_dim}, num_head={num_head}, kv_num_head={kv_num_head}, batch_size={batch_size}, seq_len={seq_len}, kv_seq_len={kv_seq_len}, max_seq_length={max_seq_length}, scale_o={scale_o}, bf16_fp8_mode={bf16_fp8_mode}, tp_size={tp_size}, is_causal={is_causal}"
        # )
        hidden_size = num_head * head_dim
        scaling_factor = head_dim**-0.5

        num_head = (int)(num_head / tp_size)
        kv_num_head = (int)(kv_num_head / tp_size)

        query_states = (
            paddle.rand(
                [batch_size, seq_len, num_head, head_dim], dtype=paddle.float32
            ).to(paddle.bfloat16)
            * 10
            - 5
        )
        key_states = (
            paddle.rand(
                [batch_size, kv_seq_len, kv_num_head, head_dim], dtype=paddle.float32
            ).to(paddle.bfloat16)
            * 10
            - 5
        )
        value_states = (
            paddle.rand(
                [batch_size, kv_seq_len, kv_num_head, head_dim], dtype=paddle.float32
            ).to(paddle.bfloat16)
            * 10
            - 5
        )

        linear_weights = (
            paddle.rand(
                [(int)(hidden_size / tp_size), hidden_size], dtype=paddle.float32
            ).to(paddle.bfloat16)
            * 0.6
            - 0.3
        )
        if not is_causal:
            attn_mask = paddle.full(
                [batch_size, 1, seq_len, kv_seq_len],
                float("-inf"),
                dtype=paddle.bfloat16,
            )
            mask = paddle.tril(
                paddle.ones([seq_len, kv_seq_len], dtype="bool"),
                diagonal=kv_seq_len - seq_len,
            )
            attn_mask[:, :, :, :] = paddle.where(
                mask, paddle.zeros_like(attn_mask), attn_mask
            )
            if num_head != kv_num_head:
                attn_mask = attn_mask.unsqueeze(1)
        else:
            attn_mask = None

        out_linear_out_ref, attn_output_ref = ref_result(
            query_states,
            key_states,
            value_states,
            None,
            linear_weights,
            scaling_factor,
        )

        scaleO, scaleOInv = get_scale_values(attn_output_ref)
        scaleQ, scaleQInv = get_scale_values(query_states)
        scaleK, scaleKInv = get_scale_values(key_states)
        scaleV, scaleVInv = get_scale_values(value_states)

        with paddle.amp.auto_cast(dtype="bfloat16", enable=True):
            amax_s_ref = get_max_weight(query_states, key_states, scale=None)

        scaleS, scaleSInv = get_scale_values(amax_s_ref, is_t_amax=True)

        q_fp8 = (scaleQ * query_states).astype(paddle.float8_e4m3fn)
        key_value_states = paddle.stack([key_states, value_states], axis=0)
        kv_fp8 = paddle.stack(
            [scaleK * key_states, scaleV * value_states], axis=0
        ).astype(paddle.float8_e4m3fn)

        weight_scale, weight_scaleInv = get_scale_values(linear_weights)
        linear_weights_fp8 = (weight_scale * linear_weights).astype(
            paddle.float8_e4m3fn
        )

        d_scale_q = paddle.to_tensor([scaleQInv])
        d_scale_k = paddle.to_tensor([scaleKInv])
        d_scale_v = paddle.to_tensor([scaleVInv])
        q_scale_s = paddle.to_tensor([scaleS])
        q_scale_o = None if scale_o is None else paddle.to_tensor([scaleO])
        d_scale_s = paddle.to_tensor([scaleSInv])

        linear_in_scale = (
            paddle.to_tensor([scaleO], dtype=paddle.bfloat16)
            if scale_o is None
            else paddle.to_tensor([scaleOInv], dtype=paddle.bfloat16)
        )
        scale_weight = paddle.to_tensor([weight_scaleInv], dtype=paddle.bfloat16)

        if bf16_fp8_mode == "ALL_BF16":
            out_linear_t_op = paddlenlp_ops.fused_sdpa_proj(
                query_states,
                key_value_states,
                attn_mask,
                None,
                linear_weights,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                scaling_factor,
                causal=attn_mask is None,
                softmax_mode=0,
            )
        elif bf16_fp8_mode == "BF16_SDPA_FP8_PROJ":
            out_linear_t_op = paddlenlp_ops.fused_sdpa_proj(
                query_states,
                key_value_states,
                attn_mask,
                None,
                linear_weights_fp8,
                None,
                None,
                None,
                None,
                None,
                None,
                linear_in_scale,
                scale_weight,
                scaling_factor,
                causal=attn_mask is None,
                softmax_mode=0,
            )
        else:  # "ALL_FP8"
            out_linear_t_op = paddlenlp_ops.fused_sdpa_proj(
                q_fp8,
                kv_fp8,
                attn_mask,
                None,
                linear_weights_fp8,
                d_scale_q,
                d_scale_k,
                d_scale_v,
                q_scale_s,
                q_scale_o,
                d_scale_s,
                linear_in_scale,
                scale_weight,
                scaling_factor,
                causal=attn_mask is None,
                softmax_mode=0,
            )
        # print(f"\nout_linear_t_op.shape: {out_linear_t_op.shape}")
        # print(f"out_linear_out_ref.shape: {out_linear_out_ref.shape}")
        similar = check_using_cosine_similarity(
            out_linear_t_op.to("float32").cpu().numpy(),
            out_linear_out_ref.to("float32").cpu().numpy(),
        )

        return similar >= 0.99


if __name__ == "__main__":
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(FP8_SDPA_Proj_T_Test)

    # Create a test runner with the desired verbosity level
    runner = unittest.TextTestRunner(
        verbosity=2
    )  # Set verbosity to 2 for detailed output

    # Run the test suite
    runner.run(suite)
