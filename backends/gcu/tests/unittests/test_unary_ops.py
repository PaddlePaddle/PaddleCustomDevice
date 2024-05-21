# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


UNARY_CASE = [
    {"x_shape": [2, 3, 32, 32], "x_dtype": np.float32},
    {"x_shape": [2, 3, 32, 32], "x_dtype": np.int32},
    {"x_shape": [2, 4, 4], "x_dtype": np.float32},
    {"x_shape": [2, 4, 4], "x_dtype": np.int32},
    {"x_shape": [2, 3, 32, 32], "x_dtype": np.float16},
    {"x_shape": [2, 4, 4], "x_dtype": np.float16},
]


@ddt
class TestUnary(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3, 32, 32]
        self.x_dtype = np.float32
        self.support_dtype = [np.float32, np.float16, np.int32]

    def prepare_datas(self):
        self.init_api_and_data()

    def init_api_and_data(self):
        self.unary_api = paddle.log
        self.data_x = np.random.uniform(1, 2, self.x_shape).astype(self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return self.unary_api(x)

    def get_numpy_out(self):
        return np.log(self.data_x)

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.get_numpy_out()
        return out

    @data(*UNARY_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        if x_dtype not in self.support_dtype:
            return
        rtol = 1e-5
        atol = 1e-5
        if x_dtype == np.float16:
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


class TestCos(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.cos
        self.data_x = np.random.uniform(0.1, 1, self.x_shape).astype(self.x_dtype)

    def get_numpy_out(self):
        return np.cos(self.data_x)


class TestRsqrt(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float64]

    def init_api_and_data(self):
        self.unary_api = paddle.rsqrt
        self.data_x = np.random.uniform(0.1, 1, self.x_shape).astype(self.x_dtype)

    def get_numpy_out(self):
        return 1.0 / np.sqrt(self.data_x)


class TestSilu(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.silu
        self.data_x = np.random.uniform(-1, 1, self.x_shape).astype(self.x_dtype)

    def get_numpy_out(self):
        return self.data_x / (np.exp(-self.data_x) + 1)


class TestSin(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.sin
        self.data_x = np.random.uniform(-1, 1, self.x_shape).astype(self.x_dtype)

    def get_numpy_out(self):
        return np.sin(self.data_x)


class TestTril(TestUnary):
    def init_api_and_data(self):
        self.unary_api = paddle.tril
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.tril(self.data_x)


class TestTriu(TestUnary):
    def init_api_and_data(self):
        self.unary_api = paddle.triu
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.triu(self.data_x)


class TestBitwiseNot(TestUnary):
    def init_attrs(self):
        self.support_dtype = [bool, np.uint8, np.int8, np.int16, np.int32, np.int64]

    def init_api_and_data(self):
        self.unary_api = paddle.bitwise_not
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.bitwise_not(self.data_x)


class TestPow(TestUnary):
    def init_api_and_data(self):
        self.unary_api = paddle.pow
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return self.unary_api(x, 3)

    def get_numpy_out(self):
        return np.power(self.data_x, 3)


class TestRelu(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.relu
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.maximum(0, self.data_x)


class TestSwish(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.swish
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return self.data_x / (np.exp(-self.data_x) + 1)


class TestSigmoid(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.sigmoid
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return 1 / (np.exp(-self.data_x) + 1)


class TestHardSigmoid(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.hardsigmoid
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        return np.clip(self.data_x / 6.0 + 0.5, 0, 1)


class TestHardSwish(TestUnary):
    def init_attrs(self):
        self.support_dtype = [np.float32, np.float16]

    def init_api_and_data(self):
        self.unary_api = paddle.nn.functional.hardswish
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def get_numpy_out(self):
        r = np.array(self.data_x <= -3)
        tmp = self.data_x * (self.data_x + 3) / 6.0
        zeros = np.zeros(self.data_x.shape, self.data_x.dtype)
        tmp_1 = np.select(r, zeros, tmp)
        tmp_2 = np.select(np.array(self.data_x >= 3), self.data_x, tmp_1)
        return tmp_2


if __name__ == "__main__":
    unittest.main()
