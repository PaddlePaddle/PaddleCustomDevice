# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import os
import unittest
import paddle.nn.functional as F
import numpy as np
import paddle
from paddle.base import core

for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
    if lib.endswith(".so"):
        paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
            lib
        )
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")


def attention_naive(q, k, v, causal=False, datetype=paddle.float16):
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(q, paddle.transpose(k, [0, 1, 3, 2]))
    s = paddle.scale(s, scale)
    if causal:
        attn_mask = paddle.ones((q.shape[2], q.shape[2]), dtype=paddle.bool)
        attn_mask = ~paddle.tril(attn_mask)
        scale1 = paddle.to_tensor([-20000], dtype=datetype)
        attn_mask = attn_mask.to(datetype)
        attn_mask = attn_mask.multiply(scale1)
        s = s + attn_mask
    p = F.softmax(s)
    o = paddle.matmul(p, v)
    return paddle.transpose(o, [0, 2, 1, 3])


def attention_naive_withmask(q, k, v, mask, datetype=paddle.float16):
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(q, paddle.transpose(k, [0, 1, 3, 2]))
    s = paddle.scale(s, scale)
    scale1 = paddle.to_tensor([-20000], dtype=datetype)
    # mask = mask.cast(paddle.float16)
    mask = mask.multiply(scale1)
    s = s + mask
    p = F.softmax(s)
    o = paddle.matmul(p, v)
    return paddle.transpose(o, [0, 2, 1, 3])


class TestFlashAttentionAPI(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CustomPlace("npu", 0)
        # (B,N,S,D)
        self.shape = (2, 5, 2048, 128)
        self.dtype = "float16"
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False

    def test_fa_fp16(self):
        # profiler.start()
        self.causal = False
        self.dtype = "float16"
        print(
            f"Test fa case shape {self.shape} dtype {self.dtype} causal {self.causal}"
        )
        paddle.disable_static()
        query_raw = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        key_raw = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        value_raw = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        qurey_naive = np.copy(query_raw)
        key_naive = np.copy(key_raw)
        value_naive = np.copy(value_raw)
        query_fa = np.copy(query_raw)
        key_fa = np.copy(key_raw)
        value_fa = np.copy(value_raw)
        # (B,N,S,D) ->(B,N,S,S) ->(B,N,S,D) ->(B,S,N,D)
        qurey_naive_input = paddle.to_tensor(
            qurey_naive, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        key_naive_input = paddle.to_tensor(
            key_naive, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        value_naive_input = paddle.to_tensor(
            value_naive, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        query_fa_input = paddle.to_tensor(
            query_fa, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        key_fa_input = paddle.to_tensor(
            key_fa, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        value_fa_input = paddle.to_tensor(
            value_fa, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        fa_naive = attention_naive(
            qurey_naive_input,
            key_naive_input,
            value_naive_input,
            self.causal,
            self.dtype,
        )

        fixed_seed_offset = None
        attn_mask = None
        dropout = 0.0
        causal = self.causal
        return_softmax = True
        is_test = False
        # (B，S, N, D)->(B,S,N,D)
        query_fa_input_ = query_fa_input.transpose((0, 2, 1, 3))
        key_fa_input_ = key_fa_input.transpose((0, 2, 1, 3))
        value_fa_input_ = value_fa_input.transpose((0, 2, 1, 3))
        fa_fusion = core.eager._run_custom_op(
            "flash_attention_npu",
            query_fa_input_,
            key_fa_input_,
            value_fa_input_,
            fixed_seed_offset,
            attn_mask,
            dropout,
            False,
            return_softmax,
            is_test,
        )
        np.testing.assert_allclose(
            fa_fusion[0].numpy(), fa_naive, rtol=5e-03, atol=5e-03
        )
        fa_fusion[0].backward()
        fa_naive.backward()
        np.testing.assert_allclose(
            qurey_naive_input.grad.numpy(),
            query_fa_input.grad.numpy(),
            rtol=5e-03,
            atol=5e-03,
        )
        np.testing.assert_allclose(
            key_naive_input.grad.numpy(),
            key_fa_input.grad.numpy(),
            rtol=5e-03,
            atol=5e-03,
        )
        np.testing.assert_allclose(
            value_naive_input.grad.numpy(),
            value_fa_input.grad.numpy(),
            rtol=5e-03,
            atol=5e-03,
        )
        print(
            f"Test fa case shape {self.shape} dtype {self.dtype} causal {self.causal} passed"
        )

    def test_fa_casual_fp16(self):
        self.causal = False
        self.dtype = "float16"
        print(
            f"Test fa case shape {self.shape} dtype {self.dtype} causal {self.causal}"
        )
        paddle.disable_static()
        query_raw = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        key_raw = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        value_raw = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        qurey_naive = np.copy(query_raw)
        key_naive = np.copy(key_raw)
        value_naive = np.copy(value_raw)
        query_fa = np.copy(query_raw)
        key_fa = np.copy(key_raw)
        value_fa = np.copy(value_raw)
        # (B,N,S,D) ->(B,N,S,S) ->(B,N,S,D) ->(B,S,N,D)
        qurey_naive_input = paddle.to_tensor(
            qurey_naive, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        key_naive_input = paddle.to_tensor(
            key_naive, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        value_naive_input = paddle.to_tensor(
            value_naive, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        query_fa_input = paddle.to_tensor(
            query_fa, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        key_fa_input = paddle.to_tensor(
            key_fa, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        value_fa_input = paddle.to_tensor(
            value_fa, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        fixed_seed_offset = None
        attn_mask = None
        dropout = 0.0
        return_softmax = True
        is_test = False
        # (B，S, N, D)->(B,S,N,D)
        query_fa_input_ = query_fa_input.transpose((0, 2, 1, 3))
        key_fa_input_ = key_fa_input.transpose((0, 2, 1, 3))
        value_fa_input_ = value_fa_input.transpose((0, 2, 1, 3))
        fa_fusion = core.eager._run_custom_op(
            "flash_attention_npu",
            query_fa_input_,
            key_fa_input_,
            value_fa_input_,
            fixed_seed_offset,
            attn_mask,
            dropout,
            self.causal,
            return_softmax,
            is_test,
        )
        fa_naive = attention_naive(
            qurey_naive_input,
            key_naive_input,
            value_naive_input,
            self.causal,
            self.dtype,
        )
        np.testing.assert_allclose(
            fa_fusion[0].numpy(), fa_naive, rtol=5e-03, atol=5e-03
        )
        fa_fusion[0].backward()
        fa_naive.backward()
        np.testing.assert_allclose(
            qurey_naive_input.grad.numpy(),
            query_fa_input.grad.numpy(),
            rtol=5e-03,
            atol=5e-03,
        )
        np.testing.assert_allclose(
            key_naive_input.grad.numpy(),
            key_fa_input.grad.numpy(),
            rtol=5e-03,
            atol=5e-03,
        )
        np.testing.assert_allclose(
            value_naive_input.grad.numpy(),
            value_fa_input.grad.numpy(),
            rtol=5e-03,
            atol=5e-03,
        )
        print(
            f"Test fa case shape {self.shape} dtype {self.dtype} causal {self.causal} passed"
        )

    def test_fa_withmask_fp16(self):
        self.causal = False
        self.dtype = "float16"
        print(
            f"Test fa case shape {self.shape} dtype {self.dtype} causal {self.causal} with defined mask"
        )
        paddle.disable_static()
        query_raw = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        key_raw = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        value_raw = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        qurey_naive = np.copy(query_raw)
        key_naive = np.copy(key_raw)
        value_naive = np.copy(value_raw)
        query_fa = np.copy(query_raw)
        key_fa = np.copy(key_raw)
        value_fa = np.copy(value_raw)
        # (B,N,S,D) ->(B,N,S,S) ->(B,N,S,D) ->(B,S,N,D)
        qurey_naive_input = paddle.to_tensor(
            qurey_naive, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        key_naive_input = paddle.to_tensor(
            key_naive, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        value_naive_input = paddle.to_tensor(
            value_naive, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        query_fa_input = paddle.to_tensor(
            query_fa, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        key_fa_input = paddle.to_tensor(
            key_fa, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        value_fa_input = paddle.to_tensor(
            value_fa, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        attn_mask = np.ones((self.shape[2], self.shape[2]))
        truncation_masknum = np.random.randint(0, self.shape[2])
        attn_mask[:, :truncation_masknum] = 0
        attn_mask_naive = np.copy(attn_mask)
        attn_mask_fa = np.copy(attn_mask)
        attn_mask_naive_input = paddle.to_tensor(
            attn_mask_naive, dtype=self.dtype, place=self.place
        )
        attn_mask_fa_input = paddle.to_tensor(
            attn_mask, dtype=paddle.bool, place=self.place
        )

        fixed_seed_offset = None
        dropout = 0.0
        return_softmax = True
        is_test = False
        query_fa_input_ = query_fa_input.transpose((0, 2, 1, 3))
        key_fa_input_ = key_fa_input.transpose((0, 2, 1, 3))
        value_fa_input_ = value_fa_input.transpose((0, 2, 1, 3))
        fa_fusion = core.eager._run_custom_op(
            "flash_attention_npu",
            query_fa_input_,
            key_fa_input_,
            value_fa_input_,
            fixed_seed_offset,
            attn_mask_fa_input,
            dropout,
            self.causal,
            return_softmax,
            is_test,
        )
        fa_naive = attention_naive_withmask(
            qurey_naive_input,
            key_naive_input,
            value_naive_input,
            attn_mask_naive_input,
            self.dtype,
        )

        np.testing.assert_allclose(
            fa_fusion[0].numpy(), fa_naive, rtol=5e-03, atol=5e-03
        )
        fa_fusion[0].backward()
        fa_naive.backward()
        np.testing.assert_allclose(
            qurey_naive_input.grad.numpy(),
            query_fa_input.grad.numpy(),
            rtol=5e-03,
            atol=5e-03,
        )
        np.testing.assert_allclose(
            key_naive_input.grad.numpy(),
            key_fa_input.grad.numpy(),
            rtol=5e-03,
            atol=5e-03,
        )
        np.testing.assert_allclose(
            value_naive_input.grad.numpy(),
            value_fa_input.grad.numpy(),
            rtol=5e-03,
            atol=5e-03,
        )
        print(
            f"Test fa case shape {self.shape} dtype {self.dtype} causal {self.causal} with defined mask passed"
        )

    def test_fa_bf16(self):
        self.causal = False
        self.dtype = "bfloat16"
        print(
            f"Test fa case shape {self.shape} dtype {self.dtype} causal {self.causal}"
        )
        paddle.disable_static()
        query_raw = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        key_raw = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        value_raw = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        qurey_naive = np.copy(query_raw)
        key_naive = np.copy(key_raw)
        value_naive = np.copy(value_raw)
        query_fa = np.copy(query_raw)
        key_fa = np.copy(key_raw)
        value_fa = np.copy(value_raw)
        # (B,N,S,D) ->(B,N,S,S) ->(B,N,S,D) ->(B,S,N,D)
        qurey_naive_input = paddle.to_tensor(
            qurey_naive, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        key_naive_input = paddle.to_tensor(
            key_naive, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        value_naive_input = paddle.to_tensor(
            value_naive, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        query_fa_input = paddle.to_tensor(
            query_fa, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        key_fa_input = paddle.to_tensor(
            key_fa, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        value_fa_input = paddle.to_tensor(
            value_fa, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        fa_naive = attention_naive(
            qurey_naive_input,
            key_naive_input,
            value_naive_input,
            self.causal,
            self.dtype,
        )

        fixed_seed_offset = None
        attn_mask = None
        dropout = 0.0
        causal = self.causal
        return_softmax = True
        is_test = False
        # (B，S, N, D)->(B,S,N,D)
        query_fa_input_ = query_fa_input.transpose((0, 2, 1, 3))
        key_fa_input_ = key_fa_input.transpose((0, 2, 1, 3))
        value_fa_input_ = value_fa_input.transpose((0, 2, 1, 3))
        fa_fusion = core.eager._run_custom_op(
            "flash_attention_npu",
            query_fa_input_,
            key_fa_input_,
            value_fa_input_,
            fixed_seed_offset,
            attn_mask,
            dropout,
            False,
            return_softmax,
            is_test,
        )
        fa_result = fa_fusion[0].cast(paddle.float16).numpy()
        fa_naive_result = fa_naive.cast(paddle.float16).numpy()
        np.testing.assert_allclose(fa_result, fa_naive_result, rtol=5e-03, atol=5e-03)
        fa_fusion[0].backward()
        fa_naive.backward()
        qurey_naive_grad = qurey_naive_input.grad.cast(paddle.float16).numpy()
        key_naive_grad = key_naive_input.grad.cast(paddle.float16).numpy()
        value_naive_grad = value_naive_input.grad.cast(paddle.float16).numpy()
        query_fa_grad = query_fa_input.grad.cast(paddle.float16).numpy()
        key_fa_grad = key_fa_input.grad.cast(paddle.float16).numpy()
        value_fa_grad = value_fa_input.grad.cast(paddle.float16).numpy()
        np.testing.assert_allclose(
            qurey_naive_grad, query_fa_grad, rtol=8e-03, atol=8e-03
        )
        np.testing.assert_allclose(key_naive_grad, key_fa_grad, rtol=8e-03, atol=8e-03)
        np.testing.assert_allclose(
            value_naive_grad, value_fa_grad, rtol=8e-03, atol=8e-03
        )
        print(
            f"Test fa case shape {self.shape} dtype {self.dtype} causal {self.causal} passed"
        )

    def test_fa_casual_bf16(self):
        self.causal = False
        self.dtype = "bfloat16"
        print(
            f"Test fa case shape {self.shape} dtype {self.dtype} causal {self.causal}"
        )
        paddle.disable_static()
        query_raw = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        key_raw = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        value_raw = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        qurey_naive = np.copy(query_raw)
        key_naive = np.copy(key_raw)
        value_naive = np.copy(value_raw)
        query_fa = np.copy(query_raw)
        key_fa = np.copy(key_raw)
        value_fa = np.copy(value_raw)
        # (B,N,S,D) ->(B,N,S,S) ->(B,N,S,D) ->(B,S,N,D)
        qurey_naive_input = paddle.to_tensor(
            qurey_naive, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        key_naive_input = paddle.to_tensor(
            key_naive, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        value_naive_input = paddle.to_tensor(
            value_naive, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        query_fa_input = paddle.to_tensor(
            query_fa, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        key_fa_input = paddle.to_tensor(
            key_fa, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        value_fa_input = paddle.to_tensor(
            value_fa, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        fa_naive = attention_naive(
            qurey_naive_input,
            key_naive_input,
            value_naive_input,
            self.causal,
            self.dtype,
        )
        fixed_seed_offset = None
        attn_mask = None
        dropout = 0.0
        return_softmax = True
        is_test = False
        # (B，S, N, D)->(B,S,N,D)
        query_fa_input_ = query_fa_input.transpose((0, 2, 1, 3))
        key_fa_input_ = key_fa_input.transpose((0, 2, 1, 3))
        value_fa_input_ = value_fa_input.transpose((0, 2, 1, 3))
        fa_fusion = core.eager._run_custom_op(
            "flash_attention_npu",
            query_fa_input_,
            key_fa_input_,
            value_fa_input_,
            fixed_seed_offset,
            attn_mask,
            dropout,
            self.causal,
            return_softmax,
            is_test,
        )
        fa_result = fa_fusion[0].cast(paddle.float16).numpy()
        fa_naive_result = fa_naive.cast(paddle.float16).numpy()
        np.testing.assert_allclose(fa_result, fa_naive_result, rtol=8e-03, atol=8e-03)
        fa_fusion[0].backward()
        fa_naive.backward()
        qurey_naive_grad = qurey_naive_input.grad.cast(paddle.float16).numpy()
        key_naive_grad = key_naive_input.grad.cast(paddle.float16).numpy()
        value_naive_grad = value_naive_input.grad.cast(paddle.float16).numpy()
        query_fa_grad = query_fa_input.grad.cast(paddle.float16).numpy()
        key_fa_grad = key_fa_input.grad.cast(paddle.float16).numpy()
        value_fa_grad = value_fa_input.grad.cast(paddle.float16).numpy()

        np.testing.assert_allclose(
            qurey_naive_grad, query_fa_grad, rtol=8e-03, atol=8e-03
        )
        np.testing.assert_allclose(key_naive_grad, key_fa_grad, rtol=8e-03, atol=8e-03)
        np.testing.assert_allclose(
            value_naive_grad, value_fa_grad, rtol=8e-03, atol=8e-03
        )
        print(
            f"Test fa case shape {self.shape} dtype {self.dtype} causal {self.causal} passed"
        )

    def test_fa_withmask_bf16(self):
        self.causal = False
        self.dtype = "bfloat16"
        print(
            f"Test fa case shape {self.shape} dtype {self.dtype} causal {self.causal} with defined mask"
        )
        paddle.disable_static()
        query_raw = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        key_raw = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        value_raw = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        qurey_naive = np.copy(query_raw)
        key_naive = np.copy(key_raw)
        value_naive = np.copy(value_raw)
        query_fa = np.copy(query_raw)
        key_fa = np.copy(key_raw)
        value_fa = np.copy(value_raw)
        # (B,N,S,D) ->(B,N,S,S) ->(B,N,S,D) ->(B,S,N,D)
        qurey_naive_input = paddle.to_tensor(
            qurey_naive, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        key_naive_input = paddle.to_tensor(
            key_naive, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        value_naive_input = paddle.to_tensor(
            value_naive, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        query_fa_input = paddle.to_tensor(
            query_fa, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        key_fa_input = paddle.to_tensor(
            key_fa, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        value_fa_input = paddle.to_tensor(
            value_fa, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        attn_mask = np.ones((self.shape[2], self.shape[2]))
        truncation_masknum = np.random.randint(0, self.shape[2])
        attn_mask[:, :truncation_masknum] = 0
        attn_mask_naive = np.copy(attn_mask)
        attn_mask_fa = np.copy(attn_mask)
        attn_mask_naive_input = paddle.to_tensor(
            attn_mask_naive, dtype=self.dtype, place=self.place
        )
        attn_mask_fa_input = paddle.to_tensor(
            attn_mask, dtype=paddle.bool, place=self.place
        )

        fixed_seed_offset = None
        dropout = 0.0
        return_softmax = True
        is_test = False
        query_fa_input_ = query_fa_input.transpose((0, 2, 1, 3))
        key_fa_input_ = key_fa_input.transpose((0, 2, 1, 3))
        value_fa_input_ = value_fa_input.transpose((0, 2, 1, 3))
        fa_fusion = core.eager._run_custom_op(
            "flash_attention_npu",
            query_fa_input_,
            key_fa_input_,
            value_fa_input_,
            fixed_seed_offset,
            attn_mask_fa_input,
            dropout,
            self.causal,
            return_softmax,
            is_test,
        )
        fa_naive = attention_naive_withmask(
            qurey_naive_input,
            key_naive_input,
            value_naive_input,
            attn_mask_naive_input,
            self.dtype,
        )
        fa_result = fa_fusion[0].cast(paddle.float16).numpy()
        fa_naive_result = fa_naive.cast(paddle.float16).numpy()
        np.testing.assert_allclose(fa_result, fa_naive_result, rtol=5e-03, atol=5e-03)
        fa_fusion[0].backward()
        fa_naive.backward()
        qurey_naive_grad = qurey_naive_input.grad.cast(paddle.float16).numpy()
        key_naive_grad = key_naive_input.grad.cast(paddle.float16).numpy()
        value_naive_grad = value_naive_input.grad.cast(paddle.float16).numpy()
        query_fa_grad = query_fa_input.grad.cast(paddle.float16).numpy()
        key_fa_grad = key_fa_input.grad.cast(paddle.float16).numpy()
        value_fa_grad = value_fa_input.grad.cast(paddle.float16).numpy()

        np.testing.assert_allclose(
            qurey_naive_grad, query_fa_grad, rtol=8e-03, atol=8e-03
        )
        np.testing.assert_allclose(key_naive_grad, key_fa_grad, rtol=8e-03, atol=8e-03)
        np.testing.assert_allclose(
            value_naive_grad, value_fa_grad, rtol=8e-03, atol=8e-03
        )
        print(
            f"Test fa case shape {self.shape} dtype {self.dtype} causal {self.causal} with defined mask passed"
        )


if __name__ == "__main__":
    unittest.main()
