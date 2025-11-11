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

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 1)
paddle.device.set_device(f"intel_hpu:{intel_hpus_module_id}")

paddle.seed(105)


def repeat_kv(query_states, key_states, value_states, attention_mask):
    # query_states: [B, Q, H, D]
    # key_states, value_states: [B, K, H_kv, D]
    # attention_mask: [1, 1, Q, K]
    # Repeat key/value heads to match query heads
    bsz, q_len, num_heads, head_dim = query_states.shape
    _, kv_seq_len, kv_num_heads, _ = key_states.shape
    num_key_value_groups = num_head // kv_num_heads

    key_states = (
        key_states.unsqueeze(2)
        .expand([-1, -1, num_key_value_groups, -1, -1])
        .reshape([bsz, kv_seq_len, num_heads, head_dim])
    )
    value_states = (
        value_states.unsqueeze(2)
        .expand([-1, -1, num_key_value_groups, -1, -1])
        .reshape([bsz, kv_seq_len, num_heads, head_dim])
    )

    return query_states, key_states, value_states, attention_mask


def ref_result(
    query_states,  # B,T,M,H
    key_states,  # B,T,M,H
    value_states,  # B,T,M,H
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


def ref_fused_sdpa_proj_t(
    query_states,  # B,T,M,H
    key_value_states,  # 2, B,T,M,H
    attn_mask,
    valid_seq_len,
    linear_weights,
    scaling_factor,
    causal=True,
):
    query_states = query_states.transpose([0, 2, 1, 3]).clone()
    key_states = key_value_states[0].transpose([0, 2, 1, 3]).clone()
    value_states = key_value_states[1].transpose([0, 2, 1, 3]).clone()

    batch, num_heads, q_len, head_dim = query_states.shape
    batch, num_key_value_heads, kv_len, head_dim = key_states.shape
    n_rep = num_heads // num_key_value_heads
    if n_rep != 1 and num_key_value_heads != 1:
        new_kv_shape = (batch, num_key_value_heads, 1, kv_len, head_dim)
        key_states = key_states.reshape(new_kv_shape)
        value_states = value_states.reshape(new_kv_shape)

        new_q_shape = (batch, num_key_value_heads, n_rep, q_len, head_dim)
        query_states = query_states.reshape(new_q_shape)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)

        attn_weights = (
            paddle.matmul(query_states, key_states.transpose([0, 1, 2, 4, 3]))
            * scaling_factor
        )
    else:
        attn_weights = (
            paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2]))
            * scaling_factor
        )

    if attn_mask is not None:
        causal_mask = attn_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    elif causal is True:
        # Generate lower triangular causal mask if attention_mask is None
        causal_mask = paddle.full(
            (q_len, kv_len), float("-inf"), dtype=attn_weights.dtype
        )
        causal_mask = paddle.triu(causal_mask, diagonal=1)
        attn_weights = attn_weights + causal_mask

    attn_weights = paddle.nn.functional.softmax(attn_weights, axis=-1)
    # attn_weights = paddle.nn.functional.dropout(attn_weights, p=0.0, training=False)
    attn_output = paddle.matmul(attn_weights, value_states)
    attn_output = attn_output.reshape([batch, -1, q_len, head_dim])

    attn_output = attn_output.transpose([0, 2, 1, 3])
    attn_output = attn_output.reshape([batch, q_len, -1])
    attn_output = paddle.matmul(attn_output, linear_weights)

    return attn_output


head_dim = 128
num_head = 32
# kv_num_head = num_head
kv_num_head = 4
hidden_size = num_head * head_dim


batch_size = 4
seq_len = 16
kv_seq_len = 16
max_seq_length = 2048

scaling_factor = head_dim**-0.5


def main():

    query_states = paddle.rand(
        [batch_size, seq_len, num_head, head_dim], dtype=paddle.float32
    ).to(paddle.bfloat16)
    key_states = paddle.rand(
        [batch_size, kv_seq_len, kv_num_head, head_dim], dtype=paddle.float32
    ).to(paddle.bfloat16)
    value_states = paddle.rand(
        [batch_size, kv_seq_len, kv_num_head, head_dim], dtype=paddle.float32
    ).to(paddle.bfloat16)
    key_value_states = paddle.stack([key_states, value_states], axis=0)

    attn_mask = paddle.ones(
        [1, 1, max_seq_length, max_seq_length], dtype=paddle.bfloat16
    )
    attn_mask = paddle.tril(attn_mask)
    attn_mask = (1.0 - attn_mask) * -10000.0

    linear_weights = paddle.rand([hidden_size, hidden_size], dtype=paddle.float32).to(
        paddle.bfloat16
    )

    attention_mask = attn_mask[..., :seq_len, :kv_seq_len]
    attention_mask = attention_mask.astype(query_states.dtype)

    key_states_t = key_states
    value_states_t = value_states
    key_value_states_t = key_value_states

    if kv_num_head != num_head:
        query_states, key_states_t, value_states_t, attention_mask = repeat_kv(
            query_states, key_states, value_states, attention_mask
        )
        key_value_states_t = paddle.stack([key_states_t, value_states_t], axis=0)

    out_linear_out_ref = ref_result(
        query_states,
        key_states_t,
        value_states_t,
        attention_mask,
        linear_weights,
        scaling_factor,
    )

    out_linear_out_op = paddlenlp_ops.fused_sdpa_proj(
        query_states.transpose([0, 2, 1, 3]),
        key_states_t.transpose([0, 2, 1, 3]),
        value_states_t.transpose([0, 2, 1, 3]),
        attention_mask,
        linear_weights,
        scaling_factor,
    )

    out_linear_v2_op_mask = paddlenlp_ops.fused_sdpa_proj_v2(
        query_states.transpose([0, 2, 1, 3]),
        key_value_states_t.transpose([0, 1, 3, 2, 4]),
        attention_mask,
        linear_weights,
        scaling_factor,
        causal=False,
    )

    out_linear_v2_op_causal = paddlenlp_ops.fused_sdpa_proj_v2(
        query_states.transpose([0, 2, 1, 3]),
        key_value_states_t.transpose([0, 1, 3, 2, 4]),
        None,
        linear_weights,
        scaling_factor,
        causal=True,
    )

    out_linear_separate_ref = ref_fused_sdpa_proj_t(
        query_states,
        key_value_states,
        None,
        None,
        linear_weights,
        scaling_factor,
        causal=True,
    )

    out_linear_t_op = paddlenlp_ops.fused_sdpa_proj_t(
        query_states,
        key_value_states,
        None,
        None,
        linear_weights,
        scaling_factor,
        causal=True,
        softmax_mode=0,
    )

    if (out_linear_out_ref == out_linear_out_op).all():
        print("fused_sdpa_proj PASS")
    else:
        print("fused_sdpa_proj FAILED")

    if (out_linear_out_ref == out_linear_v2_op_mask).all():
        print("fused_sdpa_proj_v2 with mask PASS")
    else:
        print("fused_sdpa_proj_v2 with mask FAILED")

    if paddle.allclose(
        out_linear_out_ref.to("cpu").to("float32"),
        out_linear_v2_op_causal.to("cpu").to("float32"),
        rtol=1e-1,
    ):
        print("fused_sdpa_proj_v2 causal PASS")
    else:
        print("fused_sdpa_proj_v2 causal FAILED")

    if (out_linear_v2_op_causal == out_linear_t_op).all():
        print("fused_sdpa_proj_t PASS")
    else:
        print("fused_sdpa_proj_t FAILED")

    if paddle.allclose(
        out_linear_separate_ref.to("cpu").to("float32"),
        out_linear_t_op.to("cpu").to("float32"),
        rtol=1e-2,
    ):
        print("fused_sdpa_proj_t PASS")
    else:
        print("fused_sdpa_proj_t FAILED")
        print(f"out_linear_separate_ref is {out_linear_separate_ref}")
        print(f"out_linear_t_op is {out_linear_t_op}")

    abs_error = paddle.abs(
        out_linear_t_op.to("cpu").to("float32")
        - out_linear_out_ref.to("cpu").to("float32")
    )
    print("Max absolute error:", abs_error.max().item())


if __name__ == "__main__":
    main()
