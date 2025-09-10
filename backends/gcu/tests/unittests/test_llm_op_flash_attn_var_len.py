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
import math
from typing import List

from paddle_custom_device.gcu import ops as gcu_ops


# The table retains its original format for better comparison of parameter settings.
# fmt: off
FLASH_ATTENTION_CASE = [

    {"query_lens": [1, 5, 129], "kv_lens": [1328, 18, 463], "num_heads": 4, "num_kv_heads": 4, "head_size": 128, "block_size": 16, "num_blocks": 2048, "dtype": np.float16, "with_cache": True},
    {"query_lens": [1, 5, 129], "kv_lens": [1328, 18, 463], "num_heads": 4, "num_kv_heads": 4, "head_size": 128, "block_size": 16, "num_blocks": 2048, "dtype": np.float16, "with_cache": False},
    {"query_lens": [5, 10, 80], "kv_lens": [50, 60, 100], "num_heads": 4, "num_kv_heads": 4, "head_size": 128, "block_size": 16, "num_blocks": 2048, "dtype": np.float16, "with_cache": True},
    {"query_lens": [5, 10, 60], "kv_lens": [50, 60, 100], "num_heads": 4, "num_kv_heads": 4, "head_size": 128, "block_size": 16, "num_blocks": 2048, "dtype": np.float16, "with_cache": False},

]
# fmt: on

NATIVE_IMPL_DEV = "cpu"


def native_attention_impl(
    query: paddle.Tensor,
    key_cache: paddle.Tensor,
    value_cache: paddle.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: paddle.Tensor,
):
    """Run the self attention by using paddle native op.

    Args:
        query:        [num_tokens, num_heads, head_size]
        key_cache:    [num_tokens, num_heads, head_size] or
                      [max_block_num, block_size, kv_num_head, head_size]
        value_cache:  [num_tokens, num_heads, head_size] or
                      [max_block_num, block_size, kv_num_head, head_size]
        query_lens:   [num_seqs]
        kv_lens:      [num_seqs]
        block_tables: [num_seqs, max_num_blocks_per_seq]

    Returns:
        output:       [num_tokens, num_heads, head_size]
    """
    paddle.set_device(NATIVE_IMPL_DEV)

    def get_triangle_upper_mask(shape, dtype):
        #  [batch_size, 1, q_seq_len, kv_seq_len]
        shape[1] = 1
        q_seq_len = shape[2]
        kv_seq_len = shape[3]
        paddle_dtype = dtype  # paddle.base.data_feeder.convert_dtype(dtype)
        mask = paddle.full(shape, paddle.finfo(paddle_dtype).min, dtype=paddle_dtype)
        mask = paddle.triu(mask, diagonal=kv_seq_len - q_seq_len + 1)
        return mask

    def run_sdpa(q, k, v):
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

        attention_mask = get_triangle_upper_mask(
            [batch, 1, q_seq_len, kv_seq_len], q.dtype
        )
        # print(
        #     f"TESTCASE_DEBUG In run_sdpa, q.shape:{q.shape}, k.shape:{k.shape}, attention_mask.shape:{attention_mask.shape}",
        #     flush=True,
        # )
        # print(f"TESTCASE_DEBUG In run_sdpa, attention_mask:{attention_mask}", flush=True)
        attn_weights = attn_weights + attention_mask
        attn_weights = paddle.nn.functional.softmax(
            attn_weights, axis=-1, dtype="float32"
        ).astype(q.dtype)

        attn_output = paddle.matmul(attn_weights, v)
        attn_output = attn_output.transpose([0, 2, 1, 3])
        return attn_output

    num_seqs = len(query_lens)
    with_cache = False if block_tables is None else True
    if with_cache:
        max_block_num, block_size, num_kv_heads, head_size = key_cache.shape

    outputs = paddle.empty_like(query)
    start_idx = 0
    kv_start_idx = 0
    # print(
    #     f"TESTCASE_DEBUG In native_attention_impl, query_lens:{query_lens}, kv_lens:{kv_lens}",
    #     flush=True,
    # )
    # print(
    #     f"TESTCASE_DEBUG In native_attention_impl, query.shape:{query.shape}, key_cache.shape:{key_cache.shape}, value_cache.shape:{value_cache.shape}",
    #     flush=True,
    # )
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]

        q = query[start_idx : start_idx + query_len, :, :]

        if with_cache:
            num_kv_blocks = (kv_len + block_size - 1) // block_size
            # [num_seqs, max_num_blocks_per_seq]
            block_indices = block_tables[i, :num_kv_blocks]

            k = key_cache[block_indices].reshape([-1, num_kv_heads, head_size])
            k = k[:kv_len]
            v = value_cache[block_indices].reshape([-1, num_kv_heads, head_size])
            v = v[:kv_len]
        else:
            k = key_cache[kv_start_idx : kv_start_idx + kv_len, :, :]
            v = value_cache[kv_start_idx : kv_start_idx + kv_len, :, :]

        # inputs: [num_tokens, num_heads, head_size] -> [1, num_tokens, num_heads, head_size]
        out = run_sdpa(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
        outputs[start_idx : start_idx + query_len, :, :] = out.squeeze(0)
        start_idx += query_len
        kv_start_idx += kv_len

    return outputs


@ddt
class TestFlashAttnVarLenAttention(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.query_lens = [1, 5, 129]
        self.kv_lens = [1328, 18, 463]
        self.num_heads = 4
        self.num_kv_heads = 4
        self.head_size = 128
        self.block_size = 16
        self.num_blocks = 2048
        self.dtype = np.float16

        self.dropout = 0.0

    def prepare_data(self):
        assert self.num_heads % self.num_kv_heads == 0
        self.max_query_len = np.max(self.query_lens)
        self.max_kv_len = np.max(self.kv_lens)
        self.num_seqs = len(self.query_lens)

        self.query = self.generate_data(
            shape=[np.sum(self.query_lens), self.num_heads, self.head_size],
            dtype=self.dtype,
        )

        if self.with_cache:
            self.key_cache = self.generate_data(
                shape=[
                    self.num_blocks,
                    self.block_size,
                    self.num_kv_heads,
                    self.head_size,
                ],
                dtype=self.dtype,
            )

            self.value_cache = self.generate_data(
                shape=[
                    self.num_blocks,
                    self.block_size,
                    self.num_kv_heads,
                    self.head_size,
                ],
                dtype=self.dtype,
            )

            max_num_blocks_per_seq = (
                self.max_kv_len + self.block_size - 1
            ) // self.block_size
            self.block_tables = np.random.randint(
                0,
                high=self.num_blocks,
                size=(self.num_seqs, max_num_blocks_per_seq),
                dtype=np.int32,
            )
        else:
            self.key_cache = self.generate_data(
                shape=[np.sum(self.kv_lens), self.num_kv_heads, self.head_size],
                dtype=self.dtype,
            )
            self.value_cache = self.generate_data(
                shape=[np.sum(self.kv_lens), self.num_kv_heads, self.head_size],
                dtype=self.dtype,
            )

        cu_query_lens_data = [0] + self.query_lens
        self.cu_query_lens = np.array(cu_query_lens_data, dtype=np.int32).cumsum(axis=0)
        cu_key_lens_data = [0] + self.kv_lens
        self.cu_key_lens = np.array(cu_key_lens_data, dtype=np.int32).cumsum(axis=0)
        self.seqused_k = np.array(self.kv_lens, dtype=np.int32)

        # print(f"TESTCASE_DEBUG cu_query_lens:{self.cu_query_lens}", flush=True)
        # print(f"TESTCASE_DEBUG cu_key_lens:{self.cu_key_lens}", flush=True)
        # print(f"TESTCASE_DEBUG seqused_k:{self.seqused_k}", flush=True)

    def forward(self):
        query = paddle.to_tensor(self.query, dtype=self.dtype)
        key_cache = paddle.to_tensor(self.key_cache, dtype=self.dtype)
        value_cache = paddle.to_tensor(self.value_cache, dtype=self.dtype)
        cu_query_lens = paddle.to_tensor(self.cu_query_lens, dtype="int32")
        cu_key_lens = paddle.to_tensor(self.cu_key_lens, dtype="int32")
        seqused_k = paddle.to_tensor(self.seqused_k, dtype="int32")
        if self.with_cache:
            block_tables = paddle.to_tensor(self.block_tables, dtype="int32")

        softmax_scale = 1.0 / math.sqrt(self.head_size)
        out = gcu_ops.flash_attn_var_len(
            query,
            key_cache,
            value_cache,
            max_seqlen_q=self.max_query_len,
            cu_seqlens_q=cu_query_lens,
            max_seqlen_k=self.max_kv_len,
            cu_seqlens_k=None,
            seqused_k=seqused_k,
            leftpad_k=None,
            block_table=(block_tables if self.with_cache else None),
            alibi_slopes=None,
            p_dropout=0.0,
            softmax_scale=softmax_scale,
            zero_tensors=False,
            is_causal=True,
        )
        return out

    def ref_attention_impl(self, dtype):
        paddle.set_device(NATIVE_IMPL_DEV)
        query = paddle.to_tensor(self.query, dtype=dtype)
        key_cache = paddle.to_tensor(self.key_cache, dtype=dtype)
        value_cache = paddle.to_tensor(self.value_cache, dtype=dtype)
        if self.with_cache:
            block_tables = paddle.to_tensor(self.block_tables, dtype="int32")
        out = native_attention_impl(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=self.query_lens,
            kv_lens=self.kv_lens,
            block_tables=(block_tables if self.with_cache else None),
        )
        if dtype != self.dtype:
            return out.astype(self.dtype)
        return out

    def expect_output(self):
        if NATIVE_IMPL_DEV == "cpu" and self.dtype == np.float16:
            out = self.ref_attention_impl(np.float32)
        else:
            out = self.ref_attention_impl(self.dtype)
        return out

    @data(*FLASH_ATTENTION_CASE)
    @unpack
    def test_check_output(
        self,
        query_lens,
        kv_lens,
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        num_blocks,
        dtype,
        with_cache,
    ):
        self.query_lens = query_lens
        self.kv_lens = kv_lens
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.with_cache = with_cache

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
