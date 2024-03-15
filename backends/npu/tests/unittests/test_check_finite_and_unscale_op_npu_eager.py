#  Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
from paddle import _legacy_C_ops
from tests.op_test import (
    OpTest,
    convert_float_to_uint16,
    convert_uint16_to_float,
)
import numpy as np
import paddle

from npu_utils import check_soc_version


# bf16
class TestCheckFiniteAndUnscaleOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "check_finite_and_unscale"
        self.init_test_case()

    def init_test_case(self):
        x_fp32 = np.random.random((129, 129)).astype(np.float32)
        x = convert_float_to_uint16(x_fp32)
        scale = np.random.random((1)).astype(np.float32)

        self.inputs = {"X": [("x0", x)], "Scale": scale}
        self.outputs = {
            "FoundInfinite": np.array([0]),
            "Out": [("out0", convert_uint16_to_float(x) / scale)],
        }

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestCheckFiniteAndUnscaleOpWithNan(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32

    @check_soc_version
    def test_with_nan(self):
        x = np.random.random((129, 129)).astype(self.dtype)
        x[128][128] = np.nan
        y = np.random.random(1).astype(np.float32)
        npu_x = paddle.to_tensor(x)
        scale = paddle.to_tensor(y)
        found_inf = paddle.to_tensor(np.array([0]).astype(np.bool_))
        _legacy_C_ops.check_finite_and_unscale([npu_x], scale, [npu_x], found_inf)

        k = x / y
        np.testing.assert_allclose(npu_x.numpy()[:-1], k[:-1], rtol=1e-03)
        np.testing.assert_allclose(npu_x.numpy()[-1][:-1], k[-1][:-1], rtol=1e-03)
        np.testing.assert_equal(found_inf.numpy(), 1)


class TestCheckFiniteAndUnscaleOpWithNanFP16(TestCheckFiniteAndUnscaleOpWithNan):
    def setUp(self):
        self.dtype = np.float16


class TestCheckFiniteAndUnscaleOp2(unittest.TestCase):
    @check_soc_version
    def test_with_add(self):
        a = np.array([1, 3e38, 1]).astype(np.float32)
        b = np.array([1, 3e38, 1]).astype(np.float32)
        x = a + b

        y = np.random.random(1).astype(np.float32)
        npu_x = paddle.to_tensor(x)
        scale = paddle.to_tensor(y)
        found_inf = paddle.to_tensor(np.array([0]).astype(np.bool_))
        _legacy_C_ops.check_finite_and_unscale([npu_x], scale, [npu_x], found_inf)

        np.testing.assert_equal(found_inf.numpy(), 1)

    @check_soc_version
    def test_with_div(self):
        a = np.ones((32, 32)).astype(np.float32)
        b = np.zeros((32, 32)).astype(np.float32)
        x = a / b

        y = np.random.random(1).astype(np.float32)
        npu_x = paddle.to_tensor(x)
        scale = paddle.to_tensor(y)
        found_inf = paddle.to_tensor(np.array([0]).astype(np.bool_))
        _legacy_C_ops.check_finite_and_unscale([npu_x], scale, [npu_x], found_inf)

        np.testing.assert_equal(found_inf.numpy(), 1)


class TestCheckFiniteAndUnscaleOpWithInf(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32

    @check_soc_version
    def test_with_inf(self):
        x = np.random.random((129, 129)).astype(self.dtype)
        x[128][128] = np.inf
        y = np.random.random(1).astype(np.float32)
        npu_x = paddle.to_tensor(x)
        scale = paddle.to_tensor(y)
        found_inf = paddle.to_tensor(np.array([0]).astype(np.bool_))
        _legacy_C_ops.check_finite_and_unscale([npu_x], scale, [npu_x], found_inf)

        k = x / y
        np.testing.assert_allclose(npu_x.numpy()[:-1], k[:-1], rtol=1e-03)
        np.testing.assert_allclose(npu_x.numpy()[-1][:-1], k[-1][:-1], rtol=1e-03)
        np.testing.assert_equal(found_inf.numpy(), 1)


class TestCheckFiniteAndUnscaleOpWithInfFP16(TestCheckFiniteAndUnscaleOpWithInf):
    def setUp(self):
        self.dtype = np.float16


if __name__ == "__main__":
    unittest.main()
