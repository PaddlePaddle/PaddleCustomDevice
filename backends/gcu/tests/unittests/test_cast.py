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


CAST_CASE = [
    {"x_shape": [3, 2], "x_dtype": np.float32, "out_dtype": np.uint8},
    {"x_shape": [3, 2], "x_dtype": np.float32, "out_dtype": np.int8},
    {"x_shape": [3, 2], "x_dtype": np.float32, "out_dtype": np.int16},
    {"x_shape": [3, 2], "x_dtype": np.float32, "out_dtype": np.int32},
    {"x_shape": [3, 2], "x_dtype": np.float32, "out_dtype": np.int64},
    {"x_shape": [3, 2], "x_dtype": np.float32, "out_dtype": bool},
    {"x_shape": [3, 2], "x_dtype": np.int32, "out_dtype": np.uint8},
    {"x_shape": [3, 2], "x_dtype": np.int32, "out_dtype": np.int8},
    {"x_shape": [3, 2], "x_dtype": np.int32, "out_dtype": np.int16},
    {"x_shape": [3, 2], "x_dtype": np.int32, "out_dtype": np.int64},
    {"x_shape": [3, 2], "x_dtype": np.int32, "out_dtype": np.float16},
    {"x_shape": [3, 2], "x_dtype": np.int32, "out_dtype": np.float32},
    {"x_shape": [3, 2], "x_dtype": np.int32, "out_dtype": bool},
    {"x_shape": [3, 2], "x_dtype": np.int64, "out_dtype": np.uint8},
    {"x_shape": [3, 2], "x_dtype": np.int64, "out_dtype": np.int8},
    {"x_shape": [3, 2], "x_dtype": np.int64, "out_dtype": np.int16},
    {"x_shape": [3, 2], "x_dtype": np.int64, "out_dtype": np.int32},
    {"x_shape": [3, 2], "x_dtype": np.int64, "out_dtype": np.float16},
    {"x_shape": [3, 2], "x_dtype": np.int64, "out_dtype": np.float32},
    {"x_shape": [3, 2], "x_dtype": np.int64, "out_dtype": bool},
    {"x_shape": [3, 2], "x_dtype": np.float16, "out_dtype": np.float32},
    {"x_shape": [3, 2], "x_dtype": bool, "out_dtype": np.float32},
    {"x_shape": [], "x_dtype": np.float32, "out_dtype": np.int32},
    {"x_shape": [5], "x_dtype": np.float32, "out_dtype": np.int32},
    {"x_shape": [2, 3], "x_dtype": np.float32, "out_dtype": np.int32},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "out_dtype": np.int32},
    {"x_shape": [2, 3, 4, 5], "x_dtype": np.float32, "out_dtype": np.int32},
    {"x_shape": [2, 3, 4, 5, 6], "x_dtype": np.float32, "out_dtype": np.int32},
]

CAST_F16_CASE = [
    {"x_shape": [], "x_dtype": np.float32, "out_dtype": np.float16},
    {"x_shape": [5], "x_dtype": np.float32, "out_dtype": np.float16},
    {"x_shape": [2, 3], "x_dtype": np.float32, "out_dtype": np.float16},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32, "out_dtype": np.float16},
    {"x_shape": [2, 3, 4, 5], "x_dtype": np.float32, "out_dtype": np.float16},
    {"x_shape": [2, 3, 4, 5, 6], "x_dtype": np.float32, "out_dtype": np.float16},
]


class TestCast(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [3, 2]
        self.x_dtype = np.float32
        self.out_dtype = np.int32

    def prepare_datas(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)
        if self.x_dtype in [np.int8, np.int16, np.int32, np.int64]:
            self.data_x = self.generate_integer_data(
                self.x_shape, self.x_dtype, -100, 100
            )

    def cast_forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return paddle.cast(x, dtype=self.out_dtype)


@ddt
class TestCastCommon(TestCast):
    @data(*CAST_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, out_dtype):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.out_dtype = out_dtype
        self.check_output_gcu_with_cpu(self.cast_forward)


@ddt
class TestCastF16Common(TestCast):
    @data(*CAST_F16_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, out_dtype):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.out_dtype = out_dtype
        self.check_output_gcu_with_cpu(self.cast_forward, 1e-2)


if __name__ == "__main__":
    unittest.main()
