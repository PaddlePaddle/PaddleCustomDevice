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

from paddle_custom_device.gcu import ops as gcu_ops


# The table retains its original format for better comparison of parameter settings.
# fmt: off
RESHAPE_AND_CACHE_FLASH_CASE = [

    {"num_blocks": 32, "block_size": 16, "num_tokens": 5, "num_kv_heads": 4, "head_size": 128, "dtype": np.float16, "kv_dims_combine": False},
    {"num_blocks": 32, "block_size": 16, "num_tokens": 5, "num_kv_heads": 4, "head_size": 128, "dtype": np.float16, "kv_dims_combine": True},

    {"num_blocks": 32, "block_size": 64, "num_tokens": 512, "num_kv_heads": 20, "head_size": 128, "dtype": np.float16, "kv_dims_combine": False},
    {"num_blocks": 32, "block_size": 64, "num_tokens": 512, "num_kv_heads": 20, "head_size": 128, "dtype": np.float16, "kv_dims_combine": True},

    {"num_blocks": 32, "block_size": 64, "num_tokens": 512, "num_kv_heads": 20, "head_size": 128, "dtype": np.float32, "kv_dims_combine": False},
    {"num_blocks": 32, "block_size": 64, "num_tokens": 512, "num_kv_heads": 20, "head_size": 128, "dtype": np.float32, "kv_dims_combine": True},

]
# fmt: on

NATIVE_IMPL_DEV = "gcu"


def native_reshape_and_cache_flash_impl(
    key_cache: paddle.Tensor,
    value_cache: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    slot_mapping: paddle.Tensor,
    kv_dims_combine: bool,
):
    """Run the reshape_and_cache_flash using paddle native op.

    Args:
        key:            [num_tokens, num_kv_heads, head_size] or [num_tokens, hidden_size_kv]
        value:          [num_tokens, num_kv_heads, head_size] or [num_tokens, hidden_size_kv]
        key_cache:      [num_blocks, block_size, num_kv_heads, head_size]
        value_cache:    [num_blocks, block_size, num_kv_heads, head_size]
        slot_mapping:   [num_tokens]
    Returns:
        None
    """
    paddle.set_device(NATIVE_IMPL_DEV)
    kv_cache_shape = key_cache.shape
    num_kv_heads = kv_cache_shape[2]
    head_size = kv_cache_shape[3]

    if kv_dims_combine:
        key = key.reshape_((-1, num_kv_heads, head_size))
        value = value.reshape_((-1, num_kv_heads, head_size))

    key_cache = key_cache.reshape_((-1, num_kv_heads, head_size))
    value_cache = value_cache.reshape_((-1, num_kv_heads, head_size))

    key_cache[slot_mapping, :, :] = key
    value_cache[slot_mapping, :, :] = value

    key_cache = key_cache.reshape_(kv_cache_shape)
    value_cache = value_cache.reshape_(kv_cache_shape)


@ddt
class TestReshapeAndCacheFlash(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.num_blocks = 32
        self.block_size = 16
        self.num_tokens = 1
        self.num_kv_heads = 4
        self.head_size = 128
        self.dtype = np.float16
        self.kv_dims_combine = False

    def prepare_data(self):
        self.key = self.generate_data(
            shape=[self.num_tokens, self.num_kv_heads, self.head_size], dtype=self.dtype
        )

        self.value = self.generate_data(
            shape=[self.num_tokens, self.num_kv_heads, self.head_size], dtype=self.dtype
        )

        if self.kv_dims_combine:
            self.key = self.key.reshape((-1, self.num_kv_heads * self.head_size))
            self.value = self.value.reshape((-1, self.num_kv_heads * self.head_size))

        self.key_cache = np.zeros(
            shape=[
                self.num_blocks,
                self.block_size,
                self.num_kv_heads,
                self.head_size,
            ],
            dtype=self.dtype,
        )

        self.value_cache = np.zeros(
            shape=[
                self.num_blocks,
                self.block_size,
                self.num_kv_heads,
                self.head_size,
            ],
            dtype=self.dtype,
        )

        self.slot_mapping = np.arange(0, self.num_tokens)

        # print(f"TESTCASE_DEBUG In prepare_data, self.slot_mapping:{self.slot_mapping}", flush=True)

    def forward(self):
        key = paddle.to_tensor(self.key, dtype=self.dtype)
        value = paddle.to_tensor(self.value, dtype=self.dtype)

        key_cache = paddle.to_tensor(self.key_cache, dtype=self.dtype)
        value_cache = paddle.to_tensor(self.value_cache, dtype=self.dtype)

        slot_mapping = paddle.to_tensor(self.slot_mapping, dtype="int32")

        gcu_ops.reshape_and_cache_flash(
            key_cache=key_cache,
            value_cache=value_cache,
            key=key,
            value=value,
            slot_mapping=slot_mapping,
        )

        return [key_cache, value_cache]

    def ref_attention_impl(self, dtype):
        paddle.set_device(NATIVE_IMPL_DEV)
        key = paddle.to_tensor(self.key, dtype=dtype)
        value = paddle.to_tensor(self.value, dtype=dtype)
        key_cache = paddle.to_tensor(self.key_cache, dtype=dtype)
        value_cache = paddle.to_tensor(self.value_cache, dtype=dtype)
        slot_mapping = paddle.to_tensor(self.slot_mapping, dtype="int32")

        native_reshape_and_cache_flash_impl(
            key_cache=key_cache,
            value_cache=value_cache,
            key=key,
            value=value,
            slot_mapping=slot_mapping,
            kv_dims_combine=self.kv_dims_combine,
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

    @data(*RESHAPE_AND_CACHE_FLASH_CASE)
    @unpack
    def test_check_output(
        self,
        num_blocks,
        block_size,
        num_tokens,
        num_kv_heads,
        head_size,
        dtype,
        kv_dims_combine,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_tokens = num_tokens
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.dtype = dtype
        self.kv_dims_combine = kv_dims_combine

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
