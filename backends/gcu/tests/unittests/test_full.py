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


FULL_CASE = [
    {"shape": [1], "dtype": np.float32, "fill_value": 6.6},
    {"shape": [9], "dtype": np.float32, "fill_value": 6},
    {"shape": [3, 6], "dtype": np.float32, "fill_value": 6.8},
    {"shape": [2, 3, 6], "dtype": np.float32, "fill_value": 6},
    {"shape": [2, 2, 3, 6], "dtype": np.float32, "fill_value": 6},
    {"shape": [1], "dtype": np.float16, "fill_value": 6.8},
    {"shape": [9], "dtype": np.float16, "fill_value": 6},
    {"shape": [3, 6], "dtype": np.float16, "fill_value": 6.6},
    {"shape": [2, 3, 6], "dtype": np.float16, "fill_value": 6},
    {"shape": [2, 2, 3, 6], "dtype": np.float16, "fill_value": 6},
    {"shape": [2, 2, 3, 6], "dtype": bool, "fill_value": True},
    {"shape": [2, 2, 3, 6], "dtype": bool, "fill_value": False},
    {"shape": [2, 2, 3, 6], "dtype": np.int32, "fill_value": 6},
    {"shape": [2, 2, 3, 6], "dtype": np.int64, "fill_value": 6},
    {"shape": [1], "dtype": np.float64, "fill_value": 6.6},
    {"shape": [9], "dtype": np.float64, "fill_value": 6},
    {"shape": [3, 6], "dtype": np.float64, "fill_value": 6.8},
    {"shape": [2, 3, 6], "dtype": np.float64, "fill_value": 6},
    {"shape": [2, 2, 3, 6], "dtype": np.float64, "fill_value": 6},
]

FULL_LIKE_CASE = [
    {"x_shape": [3, 6], "x_dtype": np.float32, "fill_value": 6, "dtype": np.float32},
    {"x_shape": [2, 3, 6], "x_dtype": np.float32, "fill_value": 6, "dtype": np.float32},
    {
        "x_shape": [1, 5, 3, 6],
        "x_dtype": np.float32,
        "fill_value": 6,
        "dtype": np.float32,
    },
    {"x_shape": [3, 6], "x_dtype": np.float16, "fill_value": 6, "dtype": np.float16},
    {"x_shape": [2, 3, 6], "x_dtype": np.float16, "fill_value": 6, "dtype": np.float16},
    {
        "x_shape": [1, 5, 3, 6],
        "x_dtype": np.float16,
        "fill_value": 6,
        "dtype": np.float16,
    },
    {"x_shape": [3, 6], "x_dtype": np.float32, "fill_value": 6, "dtype": np.float16},
    {"x_shape": [2, 3, 6], "x_dtype": np.float32, "fill_value": 6, "dtype": np.float16},
    {
        "x_shape": [1, 5, 3, 6],
        "x_dtype": np.float32,
        "fill_value": 6,
        "dtype": np.float16,
    },
    {"x_shape": [3, 6], "x_dtype": np.float16, "fill_value": 6, "dtype": None},
    {"x_shape": [2, 3, 6], "x_dtype": np.float16, "fill_value": 6, "dtype": None},
    {"x_shape": [1, 5, 3, 6], "x_dtype": np.float16, "fill_value": 6, "dtype": None},
    {"x_shape": [1, 5, 3, 6], "x_dtype": bool, "fill_value": True, "dtype": None},
    {"x_shape": [1, 5, 3, 6], "x_dtype": bool, "fill_value": False, "dtype": None},
    {"x_shape": [1, 5, 3, 6], "x_dtype": np.int32, "fill_value": 6, "dtype": None},
    {"x_shape": [1, 5, 3, 6], "x_dtype": np.int64, "fill_value": 6, "dtype": None},
    {"x_shape": [3, 6], "x_dtype": np.float64, "fill_value": 6, "dtype": np.float32},
    {"x_shape": [2, 3, 6], "x_dtype": np.float64, "fill_value": 6, "dtype": np.float32},
    {
        "x_shape": [1, 5, 3, 6],
        "x_dtype": np.float64,
        "fill_value": 6,
        "dtype": np.float32,
    },
]


@ddt
class TestFull(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.shape = [3, 6]
        self.fill_value = 1
        self.dtype = np.float32

    def prepare_datas(self):
        pass

    def forward(self):
        return paddle.full(self.shape, self.fill_value, self.dtype)

    def expect_output(self):
        if self.dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.fill_value * np.ones(self.shape, self.dtype)
        return out

    @data(*FULL_CASE)
    @unpack
    def test_check_output(self, shape, fill_value, dtype):
        self.shape = shape
        self.fill_value = fill_value
        self.dtype = dtype
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


@ddt
class TestFullLike(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [3, 6]
        self.x_dtype = np.float32
        self.fill_value = 1
        self.dtype = np.float32

    def prepare_datas(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return paddle.full_like(x, self.fill_value, dtype=self.dtype)

    def expect_output(self):
        if self.x_dtype != np.float16 and self.dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.fill_value * np.ones_like(self.data_x, self.dtype)
        return out

    @data(*FULL_LIKE_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, fill_value, dtype):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.fill_value = fill_value
        self.dtype = dtype
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
