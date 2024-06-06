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


GATHER_CASE = [
    {
        "x_shape": [3, 2],
        "x_dtype": np.float32,
        "index": [1, 2],
        "index_dtype": np.int32,
        "axis": 0,
    },
    {
        "x_shape": [3, 2],
        "x_dtype": np.float32,
        "index": [1, 2],
        "index_dtype": np.int64,
        "axis": 0,
    },
    {
        "x_shape": [3, 4],
        "x_dtype": np.int32,
        "index": [1, 3],
        "index_dtype": np.int32,
        "axis": 1,
    },
    {
        "x_shape": [3, 4],
        "x_dtype": np.int64,
        "index": [1, 3],
        "index_dtype": np.int32,
        "axis": 1,
    },
    {
        "x_shape": [3, 2],
        "x_dtype": np.float32,
        "index": 1,
        "index_dtype": np.int32,
        "axis": 0,
    },
    {
        "x_shape": [3, 2],
        "x_dtype": np.float32,
        "index": 1,
        "index_dtype": np.int64,
        "axis": 0,
    },
    {
        "x_shape": [3, 4, 5],
        "x_dtype": np.float32,
        "index": [1, 2],
        "index_dtype": np.int32,
        "axis": 1,
    },
    {
        "x_shape": [3, 4, 5, 6],
        "x_dtype": np.float32,
        "index": [1, 2, 3],
        "index_dtype": np.int32,
        "axis": 2,
    },
    {
        "x_shape": [3, 4, 5, 6, 7],
        "x_dtype": np.float32,
        "index": [1, 2, 3],
        "index_dtype": np.int32,
        "axis": 3,
    },
    {
        "x_shape": [3, 4, 5],
        "x_dtype": np.float16,
        "index": [1, 2, 3],
        "index_dtype": np.int32,
        "axis": 2,
    },
    {
        "x_shape": [3, 4, 5, 6],
        "x_dtype": np.float16,
        "index": [1, 2, 3],
        "index_dtype": np.int32,
        "axis": 3,
    },
]


@ddt
class TestGather(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.x_shape = [3, 2]
        self.x_dtype = np.float32
        self.index = [1, 2]
        self.index_dtype = np.int32
        self.axis = 0

    def prepare_datas(self):
        self.data_x = self.generate_data(self.x_shape, self.x_dtype)

    def forward(self):
        x = paddle.to_tensor(self.data_x, dtype=self.x_dtype)
        index = paddle.to_tensor(self.index, dtype=self.index_dtype)
        return paddle.gather(x, index=index, axis=self.axis)

    def gather_numpy(self):
        x_transpose = np.swapaxes(self.data_x, 0, self.axis)
        gather_transpose = x_transpose[self.index, ...]
        gather = np.swapaxes(gather_transpose, 0, self.axis)
        return gather

    def expect_output(self):
        if self.x_dtype != np.float16:
            out = self.calc_result(self.forward, "cpu")
        else:
            out = self.gather_numpy()
        return out

    @data(*GATHER_CASE)
    @unpack
    def test_check_output(self, x_shape, x_dtype, index, index_dtype, axis):
        self.x_shape = x_shape
        self.x_dtype = x_dtype
        self.index = index
        self.index_dtype = index_dtype
        self.axis = axis
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
