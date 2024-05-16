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


BINARY_CASE = [
    {"binary_api": paddle.add, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"binary_api": paddle.add, "shape": [2, 3, 32, 32], "dtype": np.float64},
    {"binary_api": paddle.add, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"binary_api": paddle.add, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"binary_api": paddle.add, "shape": [2, 4, 4], "dtype": np.float32},
    {"binary_api": paddle.add, "shape": [2, 4, 4], "dtype": np.float64},
    {"binary_api": paddle.add, "shape": [2, 4, 4], "dtype": np.int32},
    {"binary_api": paddle.add, "shape": [2, 4, 4], "dtype": np.int64},
    {"binary_api": paddle.subtract, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"binary_api": paddle.subtract, "shape": [2, 3, 32, 32], "dtype": np.float64},
    {"binary_api": paddle.subtract, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"binary_api": paddle.subtract, "shape": [2, 4, 4], "dtype": np.float32},
    {"binary_api": paddle.subtract, "shape": [2, 4, 4], "dtype": np.float64},
    {"binary_api": paddle.subtract, "shape": [2, 4, 4], "dtype": np.int32},
    {"binary_api": paddle.multiply, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"binary_api": paddle.multiply, "shape": [2, 3, 32, 32], "dtype": np.float64},
    {"binary_api": paddle.multiply, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"binary_api": paddle.multiply, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"binary_api": paddle.multiply, "shape": [2, 4, 4], "dtype": np.float32},
    {"binary_api": paddle.multiply, "shape": [2, 4, 4], "dtype": np.float64},
    {"binary_api": paddle.multiply, "shape": [2, 4, 4], "dtype": np.int32},
    {"binary_api": paddle.multiply, "shape": [2, 4, 4], "dtype": np.int64},
    {"binary_api": paddle.divide, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"binary_api": paddle.divide, "shape": [2, 3, 32, 32], "dtype": np.float64},
    {"binary_api": paddle.divide, "shape": [2, 4, 4], "dtype": np.float32},
    {"binary_api": paddle.divide, "shape": [2, 4, 4], "dtype": np.float64},
    {"binary_api": paddle.bitwise_and, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"binary_api": paddle.bitwise_and, "shape": [2, 4, 4], "dtype": np.int32},
    {"binary_api": paddle.bitwise_and, "shape": [2, 3, 32, 32], "dtype": bool},
    {"binary_api": paddle.bitwise_and, "shape": [2, 4, 4], "dtype": bool},
    {"binary_api": paddle.logical_and, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"binary_api": paddle.logical_and, "shape": [2, 3, 32, 32], "dtype": np.float64},
    {"binary_api": paddle.logical_and, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"binary_api": paddle.logical_and, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"binary_api": paddle.logical_and, "shape": [2, 4, 4], "dtype": np.float32},
    {"binary_api": paddle.logical_and, "shape": [2, 4, 4], "dtype": np.float64},
    {"binary_api": paddle.logical_and, "shape": [2, 3, 32, 32], "dtype": bool},
    {"binary_api": paddle.logical_and, "shape": [2, 4, 4], "dtype": bool},
    {"binary_api": paddle.greater_than, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"binary_api": paddle.greater_than, "shape": [2, 3, 32, 32], "dtype": np.float64},
    {"binary_api": paddle.greater_than, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"binary_api": paddle.greater_than, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"binary_api": paddle.greater_than, "shape": [2, 4, 4], "dtype": np.float32},
    {"binary_api": paddle.greater_than, "shape": [2, 4, 4], "dtype": np.float64},
    {"binary_api": paddle.greater_than, "shape": [2, 3, 32, 32], "dtype": bool},
    {"binary_api": paddle.greater_than, "shape": [2, 4, 4], "dtype": bool},
    {"binary_api": paddle.greater_equal, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"binary_api": paddle.greater_equal, "shape": [2, 3, 32, 32], "dtype": np.float64},
    {"binary_api": paddle.greater_equal, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"binary_api": paddle.greater_equal, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"binary_api": paddle.greater_equal, "shape": [2, 4, 4], "dtype": np.float32},
    {"binary_api": paddle.greater_equal, "shape": [2, 4, 4], "dtype": np.float64},
    {"binary_api": paddle.greater_equal, "shape": [2, 3, 32, 32], "dtype": bool},
    {"binary_api": paddle.greater_equal, "shape": [2, 4, 4], "dtype": bool},
    {"binary_api": paddle.not_equal, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"binary_api": paddle.not_equal, "shape": [2, 3, 32, 32], "dtype": np.float64},
    {"binary_api": paddle.not_equal, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"binary_api": paddle.not_equal, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"binary_api": paddle.not_equal, "shape": [2, 4, 4], "dtype": np.float32},
    {"binary_api": paddle.not_equal, "shape": [2, 4, 4], "dtype": np.float64},
    {"binary_api": paddle.not_equal, "shape": [2, 3, 32, 32], "dtype": bool},
    {"binary_api": paddle.not_equal, "shape": [2, 4, 4], "dtype": bool},
    {"binary_api": paddle.equal, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"binary_api": paddle.equal, "shape": [2, 3, 32, 32], "dtype": np.float64},
    {"binary_api": paddle.equal, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"binary_api": paddle.equal, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"binary_api": paddle.equal, "shape": [2, 4, 4], "dtype": np.float32},
    {"binary_api": paddle.equal, "shape": [2, 4, 4], "dtype": np.float64},
    {"binary_api": paddle.equal, "shape": [2, 3, 32, 32], "dtype": bool},
    {"binary_api": paddle.equal, "shape": [2, 4, 4], "dtype": bool},
    {"binary_api": paddle.pow, "shape": [2, 3, 32, 32], "dtype": np.float32},
    {"binary_api": paddle.pow, "shape": [2, 3, 32, 32], "dtype": np.float64},
    {"binary_api": paddle.pow, "shape": [2, 3, 32, 32], "dtype": np.int32},
    {"binary_api": paddle.pow, "shape": [2, 3, 32, 32], "dtype": np.int64},
    {"binary_api": paddle.pow, "shape": [2, 4, 4], "dtype": np.float32},
    {"binary_api": paddle.pow, "shape": [2, 4, 4], "dtype": np.float64},
    {"binary_api": paddle.pow, "shape": [2, 4, 4], "dtype": np.int32},
    {"binary_api": paddle.pow, "shape": [2, 4, 4], "dtype": np.int64},
    {"binary_api": paddle.pow, "shape": [64], "dtype": np.float32},
]

BINARY_F16_CASE = [
    {
        "binary_api": paddle.add,
        "numpy_api": np.add,
        "shape": [2, 3, 32, 32],
        "dtype": np.float16,
    },
    {
        "binary_api": paddle.subtract,
        "numpy_api": np.subtract,
        "shape": [2, 3, 32, 32],
        "dtype": np.float16,
    },
    {
        "binary_api": paddle.multiply,
        "numpy_api": np.multiply,
        "shape": [2, 3, 32, 32],
        "dtype": np.float16,
    },
    {
        "binary_api": paddle.divide,
        "numpy_api": np.divide,
        "shape": [2, 3, 32, 32],
        "dtype": np.float16,
    },
    {
        "binary_api": paddle.logical_and,
        "numpy_api": np.logical_and,
        "shape": [2, 3, 32, 32],
        "dtype": np.float16,
    },
    {
        "binary_api": paddle.greater_than,
        "numpy_api": np.greater,
        "shape": [2, 3, 32, 32],
        "dtype": np.float16,
    },
    {
        "binary_api": paddle.greater_equal,
        "numpy_api": np.greater_equal,
        "shape": [2, 3, 32, 32],
        "dtype": np.float16,
    },
    {
        "binary_api": paddle.not_equal,
        "numpy_api": np.not_equal,
        "shape": [2, 3, 32, 32],
        "dtype": np.float16,
    },
    {
        "binary_api": paddle.equal,
        "numpy_api": np.equal,
        "shape": [2, 3, 32, 32],
        "dtype": np.float16,
    },
    # {"binary_api": paddle.pow, "numpy_api": np.power, "shape": [2, 3, 32, 32], "dtype": np.float16},
]


class TestBinary(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.binary_api = paddle.add
        self.shape = [4]
        self.dtype = np.float32

    def prepare_datas(self):
        self.data_x = self.generate_data(self.shape, self.dtype)
        self.data_y = self.generate_data(self.shape, self.dtype)
        if self.binary_api == paddle.pow and (
            self.dtype == np.int64 or self.dtype == np.int32
        ):
            self.data_x = self.generate_integer_data(self.shape, self.dtype, -6, 6)
            self.data_y = self.generate_integer_data(self.shape, self.dtype, -6, 6)
        elif self.dtype == np.int64:
            self.data_x = self.generate_integer_data(self.shape, self.dtype, -200, 200)
            self.data_y = self.generate_integer_data(self.shape, self.dtype, -200, 200)

    def binary_forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.dtype)
        y = paddle.to_tensor(self.data_y, dtype=self.dtype)
        return self.binary_api(x, y)


@ddt
class TestBinaryComputation(TestBinary):
    @data(*BINARY_CASE)
    @unpack
    def test_check_output(self, binary_api, shape, dtype):
        self.binary_api = binary_api
        self.shape = shape
        self.dtype = dtype
        self.check_output_gcu_with_cpu(self.binary_forward)


@ddt
class TestBinaryComputationF16(TestBinary):
    def numpy_binary(self):
        return self.numpy_api(self.data_x, self.data_y)

    @data(*BINARY_F16_CASE)
    @unpack
    def test_check_output(self, binary_api, numpy_api, shape, dtype):
        self.binary_api = binary_api
        self.numpy_api = numpy_api
        self.shape = shape
        self.dtype = dtype
        self.check_output_gcu_with_customized(self.binary_forward, self.numpy_binary)


if __name__ == "__main__":
    unittest.main()
