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

import paddle
import numpy as np
import math
import unittest
from ddt import ddt, data, unpack
from api_base import TestAPIBase
from typing import List

from paddle_custom_device.gcu import ops as gcu_ops


# The table retains its original format for better comparison of parameter settings.
# fmt: off
ATTENTION_CASE = [
    # for llama2
    # topsvllmMemoryEfficientAttention only support float16
    {"batch_size": 4, "q_seq_len": 512, "kv_seq_len": 512, "num_heads": 40, "num_kv_heads": 40, "head_dim": 128, "dtype": np.float16, "var_len": False},
    {"batch_size": 4, "q_seq_len": 512, "kv_seq_len": 512, "num_heads": 40, "num_kv_heads": 40, "head_dim": 512, "dtype": np.float16, "var_len": False},
    {"batch_size": 4, "q_seq_len": 512, "kv_seq_len": 256, "num_heads": 40, "num_kv_heads": 40, "head_dim": 128, "dtype": np.float16, "var_len": False},

    # GQA
    {"batch_size": 4, "q_seq_len": 512, "kv_seq_len": 256, "num_heads": 40, "num_kv_heads": 20, "head_dim": 128, "dtype": np.float16, "var_len": False},

    # var len
    {"batch_size": 4, "q_seq_len": [512, 256, 125, 96], "kv_seq_len": [512, 256, 125, 96], "num_heads": 40, "num_kv_heads": 20, "head_dim": 128, "dtype": np.float16, "var_len": True},

]
# fmt: on

NATIVE_IMPL_DEV = "gcu"


def get_triangle_upper_mask(shape, dtype):
    #  [batch_size, 1, q_seq_len, kv_seq_len]
    shape[1] = 1
    paddle_dtype = dtype  # paddle.base.data_feeder.convert_dtype(dtype)
    mask = paddle.full(shape, paddle.finfo(paddle_dtype).min, dtype=paddle_dtype)
    mask = paddle.triu(mask, diagonal=1)
    return mask


def create_attn_mask_var_len(
    mask_type,
    batch_size,
    seq_lens,
):
    max_seq_len = max(seq_lens)
    mask = paddle.zeros([batch_size, 1, max_seq_len, max_seq_len], dtype=mask_type)
    for i in range(batch_size):
        seq_len = seq_lens[i]
        mask[i, 0, :seq_len, :seq_len] = (
            paddle.tril(paddle.ones(shape=(seq_len, seq_len), dtype=mask_type)) - 1
        ) * 1e4
    return mask


def run_sdpa(q, k, v):
    # print(f"TESTCASE_DEBUG In run_sdpa, q.shape:{q.shape}, k.shape:{k.shape}, v.shape:{v.shape}", flush=True)
    batch, q_seq_len, heads, head_dim = q.shape
    kv_seq_len = k.shape[1]

    #  [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
    q = paddle.transpose(q, [0, 2, 1, 3])
    k = paddle.transpose(k, [0, 2, 1, 3])
    v = paddle.transpose(v, [0, 2, 1, 3])

    # GQA
    if q.shape[1] != k.shape[1]:
        kv_head = k.shape[1]

        k = k.reshape([batch, kv_head, 1, kv_seq_len, head_dim])
        k = paddle.tile(k, [1, 1, heads // kv_head, 1, 1])
        k = k.reshape([batch, heads, kv_seq_len, head_dim])

        v = v.reshape([batch, kv_head, 1, kv_seq_len, head_dim])
        v = paddle.tile(v, [1, 1, heads // kv_head, 1, 1])
        v = v.reshape([batch, heads, kv_seq_len, head_dim])

    # matmul and devide by sqrt(head_dim)
    attn_weights = paddle.matmul(q / math.sqrt(head_dim), k.transpose([0, 1, 3, 2]))

    attention_mask = get_triangle_upper_mask([batch, 1, q_seq_len, kv_seq_len], q.dtype)
    attn_weights = attn_weights + attention_mask
    attn_weights = paddle.nn.functional.softmax(
        attn_weights, axis=-1, dtype="float32"
    ).astype(q.dtype)

    attn_output = paddle.matmul(attn_weights, v)
    attn_output = attn_output.transpose([0, 2, 1, 3])
    return attn_output


def native_var_len_attention_impl(
    query: paddle.Tensor,
    key_cache: paddle.Tensor,
    value_cache: paddle.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
):
    """Run the self attention by using paddle native op.

    Args:
        query:        [num_tokens, num_heads, head_size]
        key_cache:    [num_tokens, num_heads, head_size]
        value_cache:  [num_tokens, num_heads, head_size]
        query_lens:   [num_seqs]
        kv_lens:      [num_seqs]

    Returns:
        output:       [num_tokens, num_heads, head_size]
    """
    paddle.set_device(NATIVE_IMPL_DEV)
    num_seqs = len(query_lens)

    outputs = paddle.empty_like(query)
    start_idx = 0
    kv_start_idx = 0
    # print(f"TESTCASE_DEBUG In native_var_len_attention_impl, query.shape:{query.shape}, k.shape:{key_cache.shape}, value_cache.shape:{value_cache.shape}", flush=True)
    # print(f"TESTCASE_DEBUG In native_var_len_attention_impl, query_lens:{query_lens}, kv_lens:{kv_lens}", flush=True)

    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]

        q = query[:, start_idx : start_idx + query_len, :, :]
        k = key_cache[:, kv_start_idx : kv_start_idx + kv_len, :, :]
        v = value_cache[:, kv_start_idx : kv_start_idx + kv_len, :, :]

        # inputs: [num_tokens, num_heads, head_size] -> [1, num_tokens, num_heads, head_size]
        out = run_sdpa(q, k, v)
        outputs[:, start_idx : start_idx + query_len, :, :] = out
        start_idx += query_len
        kv_start_idx += kv_len

    return outputs


@ddt
class TestMemEfficientAttention(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.batch_size = 4
        self.q_seq_len = 512
        self.kv_seq_len = 512
        self.num_heads = 40
        self.num_kv_heads = 40
        self.head_dim = 128
        self.dtype = np.float16

        self.dropout = 0.0
        self.var_len = False

    def prepare_data(self):
        self.data_q = self.generate_data(self.q_shape, self.dtype)
        self.data_k = self.generate_data(self.k_shape, self.dtype)
        self.data_v = self.generate_data(self.v_shape, self.dtype)

    def forward(self):
        q = paddle.to_tensor(self.data_q, dtype=self.dtype)
        k = paddle.to_tensor(self.data_k, dtype=self.dtype)
        v = paddle.to_tensor(self.data_v, dtype=self.dtype)
        mask_mode = 1
        softmax_scale = 1.0 / math.sqrt(self.head_dim)
        out = gcu_ops.mem_efficient_attention(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout=self.dropout,
            mask_mode=mask_mode,
            seqlens=self.seqlens,
        )
        return out

    def sdp_flash_attention_impl(self, dtype):
        paddle.set_device(NATIVE_IMPL_DEV)
        query_states = paddle.to_tensor(self.data_q, dtype=dtype)
        key_states = paddle.to_tensor(self.data_k, dtype=dtype)
        value_states = paddle.to_tensor(self.data_v, dtype=dtype)
        if self.var_len:
            out = native_var_len_attention_impl(
                query=query_states,
                key_cache=key_states,
                value_cache=value_states,
                query_lens=self.seqlens,
                kv_lens=self.kv_seqlens,
            )
            out = out.reshape(self.q_shape)
        else:
            out = run_sdpa(query_states, key_states, value_states)
        if dtype != self.dtype:
            return out.astype(self.dtype)
        return out

    def expect_output(self):
        if NATIVE_IMPL_DEV == "cpu" and self.dtype == np.float16:
            out = self.sdp_flash_attention_impl(np.float32)
        else:
            out = self.sdp_flash_attention_impl(self.dtype)
        return out

    @data(*ATTENTION_CASE)
    @unpack
    def test_check_output(
        self,
        batch_size,
        q_seq_len,
        kv_seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        dtype,
        var_len,
    ):
        self.batch_size = batch_size
        self.q_seq_len = q_seq_len
        self.kv_seq_len = kv_seq_len
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.var_len = var_len

        if self.var_len:
            assert isinstance(
                self.q_seq_len, List
            ), "q_seq_len should be a List of int when set var_len=True."
            assert isinstance(
                self.kv_seq_len, List
            ), "kv_seq_len should be a List of int when set var_len=True."
            assert self.batch_size == len(
                self.q_seq_len
            ), "len mismatch when set var_len=True."
            assert self.batch_size == len(
                self.kv_seq_len
            ), "len mismatch when set var_len=True."

            self.q_shape = [
                1,
                np.sum(self.q_seq_len),
                self.num_heads,
                self.head_dim,
            ]
            self.k_shape = [
                1,
                np.sum(self.kv_seq_len),
                self.num_kv_heads,
                self.head_dim,
            ]
            self.v_shape = [
                1,
                np.sum(self.kv_seq_len),
                self.num_kv_heads,
                self.head_dim,
            ]
            self.seqlens = self.q_seq_len
            self.kv_seqlens = self.kv_seq_len
        else:
            self.q_shape = [
                self.batch_size,
                self.q_seq_len,
                self.num_heads,
                self.head_dim,
            ]
            self.k_shape = [
                self.batch_size,
                self.kv_seq_len,
                self.num_kv_heads,
                self.head_dim,
            ]
            self.v_shape = [
                self.batch_size,
                self.kv_seq_len,
                self.num_kv_heads,
                self.head_dim,
            ]
            self.seqlens = None
            self.kv_seqlens = None

        rtol = 1e-5
        atol = 1e-5
        if dtype == np.float16:
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    unittest.main()
