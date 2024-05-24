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


ASSIGN_CASE = [
    {"x_shape": [2, 3], "x_dtype": np.float32},
    {"x_shape": [2, 3], "x_dtype": np.int32},
    {"x_shape": [2, 3], "x_dtype": bool},
    {"x_shape": [2, 3, 4], "x_dtype": np.float32},
    {"x_shape": [2, 3, 4, 5], "x_dtype": np.float32},
]


@ddt
class TestAssign(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [2, 3]
        self.x_dtype = np.float32

    def prepare_datas(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        return paddle.assign(x)

    def get_numpy_output(self):
        return self.data_x

    @data(*ASSIGN_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.check_output_gcu_with_customized(self.forward, self.get_numpy_output)


if __name__ == "__main__":
    unittest.main()
