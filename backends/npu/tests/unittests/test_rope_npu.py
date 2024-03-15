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
from paddle.base import core
from tests.op_test import convert_float_to_uint16, convert_uint16_to_float

from npu_utils import check_soc_version


for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
    if lib.endswith(".so"):
        paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
            lib
        )
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")


def rope_naive(data, cos, sin):
    r1, r2 = paddle.chunk(data, 2, -1)
    data_new = paddle.concat((-r2, r1), axis=-1)
    output = cos * data + sin * data_new
    return output


class TestNPUROPEFP32(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CustomPlace("npu", 0)
        # (B,S,N,D)
        self.x_shape = (1, 64, 2, 64)
        # (1,S,1,D)
        self.cos_shape = (1, 64, 1, 64)
        # (1,S,1,D)
        self.sin_shape = (1, 64, 1, 64)
        self.init_dtype()

    def init_dtype(self):
        self.dtype = "float32"

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

    def golden_rope(self, x_, cos_, sin_):
        x = x_.cast("float32")
        cos = cos_.cast("float32")
        sin = sin_.cast("float32")
        y = rope_naive(x, cos, sin)
        y.backward()
        dx = x_.grad.cast("float32")
        return y.numpy(), dx.numpy()

    def fused_rope(self, x, cos, sin):
        y = core.eager._run_custom_op("fused_rope", x, cos, sin)[0]
        y.backward()
        dx = x.grad
        if self.dtype == "bfloat16":
            y = convert_uint16_to_float(y.numpy())
            dx = convert_uint16_to_float(dx.numpy())
            return y, dx
        return y.numpy(), dx.numpy()

    def gen_input(self):
        np_x = np.random.uniform(1, 10, self.x_shape).astype(np.float32)
        np_cos = np.random.uniform(0, 1, self.cos_shape).astype(np.float32)
        np_sin = np.random.uniform(0, 1, self.cos_shape).astype(np.float32)
        return np_x, np_cos, np_sin

    @check_soc_version
    def test_rope(self):
        np_x, np_cos, np_sin = self.gen_input()
        golden_x = paddle.to_tensor(
            np_x, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        golden_cos = paddle.to_tensor(
            np_cos, place=self.place, dtype=self.dtype, stop_gradient=True
        )
        golden_sin = paddle.to_tensor(
            np_sin, place=self.place, dtype=self.dtype, stop_gradient=True
        )
        fused_x = paddle.to_tensor(
            np_x, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        fused_cos = paddle.to_tensor(
            np_cos, place=self.place, dtype=self.dtype, stop_gradient=True
        )
        fused_sin = paddle.to_tensor(
            np_sin, place=self.place, dtype=self.dtype, stop_gradient=True
        )

        golden_res = self.golden_rope(golden_x, golden_cos, golden_sin)
        fused_res = self.fused_rope(fused_x, fused_cos, fused_sin)

        self.check_result(golden_res, fused_res)


class TestNPUROPEFP16(TestNPUROPEFP32):
    def init_dtype(self):
        self.dtype = "float16"


class TestNPUROPEBF16(TestNPUROPEFP32):
    def init_dtype(self):
        self.dtype = "bfloat16"

    def gen_input(self):
        np_x = np.random.uniform(1, 10, self.x_shape).astype(np.float32)
        np_cos = np.random.uniform(0, 1, self.cos_shape).astype(np.float32)

        np_sin = np.random.uniform(0, 1, self.cos_shape).astype(np.float32)
        np_uint16_x = convert_float_to_uint16(np_x)
        np_uint16_cos = convert_float_to_uint16(np_cos)
        np_uint16_sin = convert_float_to_uint16(np_cos)
        return np_uint16_x, np_uint16_cos, np_uint16_sin


if __name__ == "__main__":
    unittest.main()
