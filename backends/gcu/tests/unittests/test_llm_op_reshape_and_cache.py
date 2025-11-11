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
import unittest
from ddt import ddt, data, unpack
from api_base import TestAPIBase
from typing import List

from paddle_custom_device.gcu import ops as gcu_ops


# The table retains its original format for better comparison of parameter settings.
# fmt: off
RESHAPE_AND_CACHE_CASE = [

    {"num_blocks": 32, "num_tokens": 5, "num_kv_heads": 4, "head_size": 128, "block_size": 16, "x": 8, "dtype": np.float16},
    # {"num_blocks": 32, "num_tokens": 8, "num_kv_heads": 4, "head_size": 128, "block_size": 16, "x": 8, "dtype": np.float16},
    # {"num_blocks": 64, "num_tokens": 5, "num_kv_heads": 8, "head_size": 128, "block_size": 16, "x": 8, "dtype": np.float16},
    # {"num_blocks": 32, "num_tokens": 5, "num_kv_heads": 64, "head_size": 128, "block_size": 16, "x": 8, "dtype": np.float16},
    # {"num_blocks": 32, "num_tokens": 5, "num_kv_heads": 4, "head_size": 256, "block_size": 16, "x": 8, "dtype": np.float16},
    # {"num_blocks": 32, "num_tokens": 5, "num_kv_heads": 4, "head_size": 128, "block_size": 32, "x": 8, "dtype": np.float16},

]
# fmt: on

NATIVE_IMPL_DEV = "cpu"


def native_reshape_and_cache_impl(
    key: paddle.Tensor,
    value: paddle.Tensor,
    key_cache: paddle.Tensor,
    value_cache: paddle.Tensor,
    slot_mapping: List[int],
    key_stride0,
    value_stride0,
):
    """Run the reshape_and_cache by using paddle native op.

    Args:
        key:            [num_tokens, num_kv_heads, head_size]
        value:          [num_tokens, num_kv_heads, head_size]
        key_cache:      [num_blocks, num_kv_heads, head_size/x, block_size, x]
        value_cache:    [num_blocks, num_kv_heads, head_size, block_size]
        slot_mapping:   [num_tokens]
    Returns:
        None
    """
    paddle.set_device(NATIVE_IMPL_DEV)

    key_cache_shape = key_cache.shape
    value_cache_shape = value_cache.shape

    num_tokens = key.shape[0]
    num_heads = value_cache_shape[1]
    head_size = value_cache_shape[2]
    block_size = value_cache_shape[3]
    n = num_heads * head_size
    thread_num = 512

    block_size = value_cache_shape[3]

    key_cache_h = key_cache.flatten()
    key_c = key.flatten()
    value_cache_h = value_cache.flatten()
    value_c = value.flatten()

    for token_idx in range(num_tokens):
        slot_idx = slot_mapping[token_idx]

        if slot_idx < 0:
            continue

        block_idx = slot_idx // block_size
        block_offset = slot_idx % block_size

        for j in range(thread_num):
            for i in range(j, n, thread_num):
                src_key_idx = token_idx * key_stride0 + i
                src_value_idx = token_idx * value_stride0 + i

                head_idx = i // head_size
                head_offset = i % head_size
                tgt_key_idx = (
                    block_idx * num_heads * block_size * head_size
                    + head_idx * block_size * head_size
                    + block_offset * head_size
                    + head_offset
                )

                tgt_value_idx = (
                    block_idx * num_heads * block_size * head_size
                    + head_idx * block_size * head_size
                    + block_offset * head_size
                    + head_offset
                )

                key_cache_h[tgt_key_idx] = key_c[src_key_idx]
                value_cache_h[tgt_value_idx] = value_c[src_value_idx]

    key_cache_h = key_cache_h.reshape(key_cache_shape)
    value_cache_h = value_cache_h.reshape(value_cache_shape)


def get_kv_from_kv_cache(
    key_cache: paddle.Tensor,
    value_cache: paddle.Tensor,
    block_tables,
    seq_lens,
) -> None:
    # [num_blocks, num_kv_heads, head_size, block_size]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    num_seqs = len(seq_lens)
    # print(f"TESTCASE_DEBUG In get_kv_from_kv_cache, key_cache:{key_cache.to('cpu').tolist()}", flush=True)

    keys_lst: List[paddle.Tensor] = []
    values_lst: List[paddle.Tensor] = []

    for i in range(num_seqs):
        block_table = block_tables[i]
        seq_len = int(seq_lens[i])

        for j in range(seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            # k = key_cache[block_number, :, :, block_offset, :]
            k = key_cache[block_number, :, block_offset, :, :]
            k = k.reshape([num_kv_heads, head_size])
            keys_lst.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values_lst.append(v)
            # print(f"TESTCASE_DEBUG In get_kv_from_kv_cache, k:{k}, v:{v}", flush=True)

    keys = paddle.stack(keys_lst, axis=0)
    values = paddle.stack(values_lst, axis=0)
    return keys, values


@ddt
class TestReshapeAndCache(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.num_blocks = 32
        self.num_tokens = 1
        self.num_kv_heads = 4
        self.head_size = 128
        self.block_size = 16
        self.x = 8
        self.dtype = np.float16

    def prepare_data(self):
        # self.key = self.generate_data(
        #     shape=[self.num_tokens, self.num_kv_heads, self.head_size], dtype=self.dtype
        # )

        # self.value = self.generate_data(
        #     shape=[self.num_tokens, self.num_kv_heads, self.head_size], dtype=self.dtype
        # )

        self.key_cache = np.zeros(
            shape=[
                self.num_blocks,
                self.num_kv_heads,
                self.head_size // self.x,
                self.block_size,
                self.x,
            ],
            dtype=self.dtype,
        )

        self.value_cache = np.zeros(
            shape=[self.num_blocks, self.num_kv_heads, self.head_size, self.block_size],
            dtype=self.dtype,
        )

        all_nums = self.num_tokens * self.num_kv_heads * self.head_size
        key_value = np.arange(0, all_nums, dtype=np.int32)
        self.key = key_value.reshape(
            [self.num_tokens, self.num_kv_heads, self.head_size]
        ).astype(self.dtype)

        self.value = key_value.reshape(
            [self.num_tokens, self.num_kv_heads, self.head_size]
        ).astype(self.dtype)

        # self.key_cache = np.zeros(
        #     shape=[
        #         self.num_blocks,
        #         self.num_kv_heads,
        #         self.block_size,
        #         self.head_size,
        #     ],
        #     dtype=self.dtype,
        # )

        # self.value_cache = np.zeros(
        #     shape=[
        #         self.num_blocks,
        #         self.num_kv_heads,
        #         self.block_size,
        #         self.head_size,
        #     ],
        #     dtype=self.dtype,
        # )

        self.slot_mapping = np.arange(0, self.num_tokens)

        # print(f"TESTCASE_DEBUG In prepare_data, self.slot_mapping:{self.slot_mapping}", flush=True)

    def forward(self):
        key = paddle.to_tensor(self.key, dtype=self.dtype)
        value = paddle.to_tensor(self.value, dtype=self.dtype)

        key_cache = paddle.to_tensor(self.key_cache, dtype=self.dtype)
        value_cache = paddle.to_tensor(self.value_cache, dtype=self.dtype)

        slot_mapping = paddle.to_tensor(self.slot_mapping, dtype="int32")

        _, _ = gcu_ops.reshape_and_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            kv_cache_dtype="auto",
            k_scale=1.0,
            k_zero=0.0,
            v_scale=1.0,
            v_zero=0.0,
        )

        # print(f"TESTCASE_DEBUG In prepare_data, origin key_cache:\n{key_cache.to('cpu').tolist()}", flush=True)

        # key_cache_v = key_cache.transpose([0, 2, 1, 3, 4])
        # key_cache_v_reshape = key_cache_v.reshape([self.num_blocks * self.block_size, self.num_kv_heads, self.head_size])
        # key_cache_v_reshape_index = key_cache_v_reshape[0:self.num_tokens, :, :]

        # print(f"TESTCASE_DEBUG In prepare_data, self.key_cache_v_reshape_index:\n{key_cache_v_reshape_index.to('cpu').tolist()}", flush=True)
        # print(f"TESTCASE_DEBUG In prepare_data, self.key:\n{key.to('cpu').tolist()}", flush=True)

        return [key_cache, value_cache]

    def ref_attention_impl(self, dtype):
        paddle.set_device(NATIVE_IMPL_DEV)
        key = paddle.to_tensor(self.key, dtype=dtype)
        value = paddle.to_tensor(self.value, dtype=dtype)
        key_cache = paddle.to_tensor(self.key_cache, dtype=dtype)
        value_cache = paddle.to_tensor(self.value_cache, dtype=dtype)
        slot_mapping = self.slot_mapping.tolist()

        native_reshape_and_cache_impl(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            key_stride0=key.strides[0],
            value_stride0=value.strides[0],
        )

        if dtype != self.dtype:
            return [key_cache.astype(self.dtype), value_cache.astype(self.dtype)]
        return [key_cache, value_cache]

    def expect_output(self):
        if NATIVE_IMPL_DEV == "cpu" and self.dtype == np.float16:
            out = self.ref_attention_impl(np.float32)
        else:
            out = self.ref_attention_impl(self.dtype)
        return out

    @data(*RESHAPE_AND_CACHE_CASE)
    @unpack
    def test_check_output(
        self,
        num_blocks,
        num_tokens,
        num_kv_heads,
        head_size,
        block_size,
        x,
        dtype,
    ):
        self.num_blocks = num_blocks
        self.num_tokens = num_tokens
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.block_size = block_size
        self.x = x
        self.dtype = np.float16

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
