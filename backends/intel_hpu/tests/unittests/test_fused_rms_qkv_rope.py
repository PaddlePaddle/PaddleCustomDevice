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
from paddle import nn


import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)

paddle.device.set_device("intel_hpu")


class LlamaRotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # [dim / 2]
        self.inv_freq = 1.0 / (
            self.base
            ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32") / self.dim)
        )
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, dim/2]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, dim]
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.cos_cached.dtype != x.dtype:
            self.cos_cached = self.cos_cached.cast(x.dtype)
            self.sin_cached = self.sin_cached.cast(x.dtype)
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]
        return (
            cos.cast(x.dtype) if cos.dtype != x.dtype else cos,
            sin.cast(x.dtype) if sin.dtype != x.dtype else sin,
        )


def fused_rms_qkv_rope(
    src, ln_scales, qkv_weights, cos, sin, position_ids, epsilon, head_dim, num_head
):
    hidden_states = paddle.incubate.nn.functional.fused_rms_norm(
        src, ln_scales, None, epsilon, 2
    )[0]

    qkv_out = paddle.matmul(hidden_states, qkv_weights, False, True)

    fused_hidden_size = qkv_out.shape[2]
    kv_num_heads = (fused_hidden_size - num_head * head_dim) // head_dim // 2
    num_groups = num_head // kv_num_heads
    target_shape = [0, 0, (num_groups + 2) * kv_num_heads, head_dim]

    qkv_out = paddle.reshape_(qkv_out, target_shape)

    qkv_out = paddle.transpose(qkv_out, [0, 2, 1, 3])

    query_states, key_states, value_states = paddle.split(
        qkv_out,
        num_or_sections=[num_head, kv_num_heads, kv_num_heads],
        axis=1,
    )

    cos = cos.squeeze().unsqueeze(0).unsqueeze(0)
    sin = sin.squeeze().unsqueeze(0).unsqueeze(0)
    query_states, _, _ = paddle.incubate.nn.functional.fused_rotary_position_embedding(
        query_states, None, None, sin=sin, cos=cos, position_ids=position_ids
    )
    key_states, _, _ = paddle.incubate.nn.functional.fused_rotary_position_embedding(
        key_states, None, None, sin=sin, cos=cos, position_ids=position_ids
    )
    return (
        query_states.to(paddle.float32),
        key_states.to(paddle.float32),
        value_states.to(paddle.float32),
    )


class TestFused_RMS_QKV_Rope_OpFP16(unittest.TestCase):
    def setUp(self):
        self.init_dtype()
        self.batch_size = 8
        self.num_heads = 32
        self.seq_length = 34
        self.head_dim = 128
        self.epsilon = 1e-06

    def set_hpu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))
        paddle.seed(102)

    def init_dtype(self):
        self.dtype = "float32"

    def prepare_input(
        self,
        kv_num_heads,
        batch_size,
        num_heads,
        seq_length,
        head_dim,
        kv_seq_len,
    ):
        hidden_size = num_heads * head_dim

        src = paddle.rand([batch_size, seq_length, hidden_size], dtype=paddle.bfloat16)
        ln_scales = paddle.randn([hidden_size], dtype=paddle.bfloat16)
        qkv_weights = paddle.rand(
            [hidden_size * 3, hidden_size], dtype=paddle.float32
        ).to(paddle.bfloat16)

        position_id = paddle.arange(seq_length, dtype=paddle.int64).to(paddle.int64)
        new_rope = paddle.expand(position_id, shape=[batch_size, seq_length])

        rotary_emb = LlamaRotaryEmbedding(head_dim)
        cos, sin = rotary_emb(src, seq_len=new_rope[0][-1].item() + 1)

        return src, ln_scales, qkv_weights, cos, sin, new_rope

    def fused_rms_qkv_rope_op_custom(
        self,
        src,
        ln_scales,
        qkv_weights,
        cos,
        sin,
        new_rope,
        epsilon,
        head_dim,
        num_heads,
    ):
        (
            query_states_op,
            key_states_op,
            value_states_op,
        ) = paddlenlp_ops.fused_rms_qkv_rope(
            src,
            ln_scales,
            qkv_weights,
            cos,
            sin,
            new_rope,
            epsilon,
            head_dim,
            num_heads,
        )
        return query_states_op, key_states_op, value_states_op

    def check_result(
        self,
        torch_result_query_states,
        torch_result_key_states,
        torch_result_value_states,
        op_result_query_states,
        op_result_key_states,
        op_result_value_states,
    ):
        if self.dtype == "float32":
            rtol = 1e-6
            atol = 1e1
        elif self.dtype == "float16":
            rtol = 1e-3
            atol = 1e-4
        elif self.dtype == "bfloat16":
            rtol = 1e-2
            atol = 1e-3
        else:
            self.assertTrue(
                False,
                msg="test_fused_rms_qkv_rope input dtype only supports bfloat16, \
                     float16 and float32, but got "
                + self.dtype,
            )

        np.testing.assert_allclose(
            torch_result_query_states, op_result_query_states, rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            torch_result_key_states, op_result_key_states, rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            torch_result_value_states, op_result_value_states, rtol=rtol, atol=atol
        )

    def test_fused_rms_qkv_rope(self):
        kv_num_heads = 32
        kv_seq_len = 25
        src, ln_scales, qkv_weights, cos, sin, new_rope = self.prepare_input(
            kv_num_heads,
            self.batch_size,
            self.num_heads,
            self.seq_length,
            self.head_dim,
            kv_seq_len,
        )

        (
            query_states_op,
            key_states_op,
            value_states_op,
        ) = paddlenlp_ops.fused_rms_qkv_rope(
            src,
            ln_scales,
            qkv_weights,
            cos,
            sin,
            new_rope,
            self.epsilon,
            self.head_dim,
            self.num_heads,
        )

        query_states_ref, key_states_ref, value_states_ref = fused_rms_qkv_rope(
            src,
            ln_scales,
            qkv_weights,
            cos,
            sin,
            new_rope,
            self.epsilon,
            self.head_dim,
            self.num_heads,
        )

        self.check_result(
            query_states_ref.numpy(),
            key_states_ref.numpy(),
            value_states_ref.numpy(),
            query_states_op.to(paddle.float32),
            key_states_op.to(paddle.float32),
            value_states_op.to(paddle.float32),
        )


if __name__ == "__main__":
    unittest.main()
