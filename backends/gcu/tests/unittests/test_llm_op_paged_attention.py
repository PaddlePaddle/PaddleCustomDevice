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
PAGED_ATTENTION_V1_CASE = [

    {"num_seqs": 1, "num_heads": 4, "num_kv_heads": 4, "head_size": 128, "block_size": 16, "max_tokens_size": 512, "dtype": np.float16},
    {"num_seqs": 10, "num_heads": 4, "num_kv_heads": 4, "head_size": 128, "block_size": 16, "max_tokens_size": 512, "dtype": np.float16},
    {"num_seqs": 16, "num_heads": 4, "num_kv_heads": 4, "head_size": 128, "block_size": 16, "max_tokens_size": 512, "dtype": np.float16},

    {"num_seqs": 1, "num_heads": 64, "num_kv_heads": 8, "head_size": 128, "block_size": 32, "max_tokens_size": 512, "dtype": np.float16},
    {"num_seqs": 10, "num_heads": 64, "num_kv_heads": 8, "head_size": 128, "block_size": 32, "max_tokens_size": 512, "dtype": np.float16},
    {"num_seqs": 16, "num_heads": 64, "num_kv_heads": 8, "head_size": 128, "block_size": 32, "max_tokens_size": 512, "dtype": np.float16},

]
# fmt: on

NATIVE_IMPL_DEV = "cpu"


def native_attention_impl(
    query: paddle.Tensor,
    key_cache: paddle.Tensor,
    value_cache: paddle.Tensor,
    seq_lens: List[int],
    block_tables: List[List[int]],
):
    """Run the self attention by using paddle native op.

    Args:
        query:        [num_seqs, num_heads, head_size]
        key_cache:    [num_blocks, num_kv_heads, head_size/x, block_size, x]
        value_cache:  [num_blocks, num_kv_heads, head_size, block_size]
        seq_lens:     [num_seqs]
        block_tables: [num_seqs, max_num_blocks_per_seq]

    Returns:
        output:       [num_seqs, num_heads, head_size]
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
        attn_weights = attn_weights + attention_mask
        attn_weights = paddle.nn.functional.softmax(
            attn_weights, axis=-1, dtype="float32"
        ).astype(q.dtype)

        attn_output = paddle.matmul(attn_weights, v)
        attn_output = attn_output.transpose([0, 2, 1, 3])
        # if dtype != self.dtype:
        #     return attn_output.astype(self.dtype)
        return attn_output

    # query:        [num_seqs, num_heads, head_size]
    # key_cache:    [num_blocks, num_kv_heads, head_size/x, block_size, x]
    # value_cache:  [num_blocks, num_kv_heads, head_size, block_size]
    # seq_lens:     [num_seqs]
    # block_tables: [num_seqs, max_num_blocks_per_seq]
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    num_seqs = query.shape[0]

    outputs: List[paddle.Tensor] = []
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        seq_len = int(seq_lens[i])

        keys_list: List[paddle.Tensor] = []
        values_list: List[paddle.Tensor] = []

        for j in range(seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape([num_kv_heads, head_size])
            keys_list.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values_list.append(v)

        keys = paddle.stack(keys_list, axis=0)
        values = paddle.stack(values_list, axis=0)
        # print(f"TESTCASE_DEBUG In native_attention_impl, q.shape:{q.shape}, keys.shape:{keys.shape}, values.shape:{values.shape}", flush=True)

        # inputs: [num_tokens, num_heads, head_size] -> [1, num_tokens, num_heads, head_size]
        out = run_sdpa(q.unsqueeze(0), keys.unsqueeze(0), values.unsqueeze(0))
        # print(f"TESTCASE_DEBUG In native_attention_impl, out.shape:{out.shape}", flush=True)
        outputs.append(out.reshape([-1, num_query_heads, head_size]))
    return paddle.concat(outputs, axis=0)


@ddt
class TestPagedAttentionV1(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.num_seqs = 1
        self.num_heads = 4
        self.num_kv_heads = 4
        self.head_size = 128
        self.block_size = 16
        self.max_tokens_size = 2048
        self.dtype = np.float16

    def prepare_data(self):
        assert self.num_heads % self.num_kv_heads == 0

        self.MAX_SEQ_LEN = 50

        self.seq_lens = [
            np.random.randint(1, self.MAX_SEQ_LEN) for _ in range(self.num_seqs)
        ]
        self.max_seq_len = np.max(self.seq_lens)

        self.query = self.generate_data(
            shape=[self.num_seqs, self.num_heads, self.head_size], dtype=self.dtype
        )

        tmp = paddle.to_tensor(np.array([1]), dtype=self.dtype)
        x = self.block_size // tmp.element_size()
        num_blocks = self.max_tokens_size // self.block_size

        key_cache_shape = [
            num_blocks,
            self.num_kv_heads,
            self.head_size // x,
            self.block_size,
            x,
        ]
        value_cache_shape = [
            num_blocks,
            self.num_kv_heads,
            self.head_size,
            self.block_size,
        ]

        self.key_cache = np.zeros(shape=key_cache_shape, dtype=self.dtype)

        self.value_cache = np.zeros(shape=value_cache_shape, dtype=self.dtype)

        max_block_size = (self.max_tokens_size + self.block_size - 1) // self.block_size
        max_num_blocks_per_seq = (
            self.max_seq_len + self.block_size - 1
        ) // self.block_size
        self.block_tables = np.random.randint(
            0,
            high=max_block_size,
            size=(self.num_seqs, max_num_blocks_per_seq),
            dtype=np.int32,
        )

        # print(f"TESTCASE_DEBUG In prepare_data, self.block_tables shape:{self.block_tables.shape}", flush=True)
        # print(f"TESTCASE_DEBUG In prepare_data, self.block_tables:{self.block_tables}", flush=True)

    def forward(self):
        query = paddle.to_tensor(self.query, dtype=self.dtype)

        key_cache = paddle.to_tensor(self.key_cache, dtype=self.dtype)
        value_cache = paddle.to_tensor(self.value_cache, dtype=self.dtype)

        context_lens = paddle.to_tensor(self.seq_lens, dtype="int32")
        block_tables = paddle.to_tensor(self.block_tables, dtype="int32")
        softmax_scale = 1.0 / math.sqrt(self.head_size)

        out = gcu_ops.paged_attention(
            q=query,
            k_cache=key_cache,
            v_cache=value_cache,
            num_kv_heads=self.num_kv_heads,
            scale=softmax_scale,
            block_tables=block_tables,
            seq_lens=context_lens,
            block_size=self.block_size,
            max_seq_len=self.max_seq_len,
            kv_cache_dtype="auto",
            k_scale=1.0,
            k_zero=0.0,
            v_scale=1.0,
            v_zero=0.0,
            alibi_slopes=None,
            out_scales=None,
        )

        return out

    def ref_attention_impl(self, dtype):
        paddle.set_device(NATIVE_IMPL_DEV)
        query = paddle.to_tensor(self.query, dtype=dtype)
        key_cache = paddle.to_tensor(self.key_cache, dtype=dtype)
        value_cache = paddle.to_tensor(self.value_cache, dtype=dtype)

        out = native_attention_impl(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            seq_lens=self.seq_lens,
            block_tables=self.block_tables.tolist(),
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

    @data(*PAGED_ATTENTION_V1_CASE)
    @unpack
    def test_check_output(
        self,
        num_seqs,
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        max_tokens_size,
        dtype,
    ):
        self.num_seqs = num_seqs
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_size = block_size
        self.max_tokens_size = max_tokens_size
        self.dtype = dtype

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
