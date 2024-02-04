#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.base import core
from tests.op_test import convert_float_to_uint16, convert_uint16_to_float
from npu_utils import check_soc_version
for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
    if lib.endswith(".so"):
        paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
            lib
        )
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")


def attention_naive(q, k, v):
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(q, paddle.transpose(k, [0, 1, 3, 2]))
    s = paddle.scale(s, scale)
    p = F.softmax(s)
    o = paddle.matmul(p, v)
    return paddle.transpose(o, [0, 2, 1, 3])

class TestNPUFAFP16(unittest.TestCase):
    def setUp(self):
        self.npu_place = paddle.CustomPlace("npu", 0)
        self.cpu_place = paddle.CPUPlace()
        # (B,S,N,D)
        self.shape = (1, 5, 16, 64)
        self.dropout = 0.0
        self.fixed_seed_offset = None
        self.attn_mask = None
        self.causal = False
        self.return_softmax = False
        self.is_test = False
        self.init_dtype()

    def init_dtype(self):
        self.dtype = "float16"

    def check_result(self, golden_res, fused_res):
        if self.dtype == "float32":
            rtol = 5e-3
            atol = 5e-3
        elif self.dtype == "float16":
            rtol = 5e-3
            atol = 5e-3
        elif self.dtype == "bfloat16":
            rtol = 8e-3
            atol = 8e-3
        else:
            self.assertTrue(
                False,
                msg="NPURMSNorm input dtype only supports bfloat16, \
                     float16 and float32, but got "
                + self.dtype,
            )
        golden_y, golden_dx = golden_res
        fused_y, fused_dx = fused_res
        np.testing.assert_allclose(golden_y, fused_y, rtol=rtol, atol=atol)
        np.testing.assert_allclose(golden_dx, fused_dx, rtol=rtol, atol=atol)

    def golden_fa(self, query_, key_, value_):
        query = query_.cast("float32")
        key = key_.cast("float32")
        value = value_.cast("float32")
        y = attention_naive(
            query,
            key,
            value,
        )
        y.backward()
        dx = query_.grad.cast("float32")
        return y.numpy(), dx.numpy()

    def fused_fa(self, query_, key_, value_):
        query = query_.transpose((0, 2, 1, 3))
        key = key_.transpose((0, 2, 1, 3))
        value = value_.transpose((0, 2, 1, 3))
        y = core.eager._run_custom_op("flash_attention_npu", 
                                      query, 
                                      key, 
                                      value,
                                      self.fixed_seed_offset,
                                      self.attn_mask,
                                      self.dropout,
                                      self.causal,
                                      self.return_softmax,
                                      self.is_test
                                      )[0]
        y.backward()
        dx = query_.grad
        if self.dtype == "bfloat16":
            y = convert_uint16_to_float(y.numpy())
            dx = convert_uint16_to_float(dx.numpy())
            return y,  dx
        return y.numpy(),dx.numpy()

    def gen_input(self):
        np_query = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        np_key = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        np_value = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        return np_query, np_key, np_value

    @check_soc_version
    def test_fa(self):
        np_query, np_key, np_value = self.gen_input()
        if self.dtype == "float16":
            golden_query = paddle.to_tensor(
                np_query, place=self.cpu_place, dtype=self.dtype, stop_gradient=False
            )
            golden_key = paddle.to_tensor(
                np_key, place=self.cpu_place, dtype=self.dtype, stop_gradient=True
            )
            golden_value = paddle.to_tensor(
                np_value, place=self.cpu_place, dtype=self.dtype, stop_gradient=True
            )
        elif self.dtype == "bfloat16":
            golden_query = paddle.to_tensor(
                np_query, place=self.npu_place, dtype=self.dtype, stop_gradient=False
            )
            golden_key = paddle.to_tensor(
                np_key, place=self.npu_place, dtype=self.dtype, stop_gradient=True
            )
            golden_value = paddle.to_tensor(
                np_value, place=self.npu_place, dtype=self.dtype, stop_gradient=True
            )
        fused_query = paddle.to_tensor(
            np_query, place=self.npu_place, dtype=self.dtype, stop_gradient=False
        )
        fused_key = paddle.to_tensor(
            np_key, place=self.npu_place, dtype=self.dtype, stop_gradient=True
        )
        fused_value = paddle.to_tensor(
            np_value, place=self.npu_place, dtype=self.dtype, stop_gradient=True
        )

        golden_res = self.golden_fa(golden_query, golden_key, golden_value)
        fused_res = self.fused_fa(fused_query, fused_key, fused_value)

        self.check_result(golden_res, fused_res)


class TestNPUFABF16(TestNPUFAFP16):
    def init_dtype(self):
        self.dtype = "bfloat16"

    def gen_input(self):
        np_query = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        np_key = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        np_value = np.random.randn(
            self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        )
        np_uint16_query = convert_float_to_uint16(np_query)
        np_uint16_key = convert_float_to_uint16(np_key)
        np_uint16_value = convert_float_to_uint16(np_value)
        return np_uint16_query, np_uint16_key, np_uint16_value


if __name__ == "__main__":
    np.random.seed(2024)
    unittest.main()
