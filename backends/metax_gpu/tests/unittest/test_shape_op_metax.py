# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
import paddle
from unittest import TestCase


class TestShapeOp(TestCase):
    def setUp(self):
        self.device = "metax_gpu:0"
        self.place = paddle.CustomPlace("metax_gpu", 0)
        self.init()

    def init(self):
        self.dtypes = ["bool", "int8", "uint8", "int32", "int64", "float32", "float64"]
        self.shapes = [[2], [1, 2, 3], [1, 2, 3, 4, 1024]]

    def test_static_api(self):
        paddle.enable_static()
        paddle.set_device(self.device)
        for dtype in self.dtypes:
            for shape in self.shapes:
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    data_np = np.random.random_integers(0, 100, size=shape)
                    data_pd = paddle.to_tensor(data_np, dtype=dtype)
                    out = paddle.shape(data_pd)
                    exe = paddle.static.Executor(self.place)
                    out_actual = exe.run(
                        paddle.static.default_main_program(), fetch_list=[out]
                    )[0]
                    out_expected = shape
                    np.testing.assert_allclose(out_actual, out_expected)

    def test_dygraph_api(self):
        paddle.disable_static()
        paddle.set_device(self.device)
        for dtype in self.dtypes:
            for shape in self.shapes:
                data_np = np.random.random_integers(0, 100, size=shape)
                data_pd = paddle.to_tensor(data_np, dtype=dtype)
                out_actual = paddle.shape(data_pd)
                out_expected = shape
                np.testing.assert_allclose(out_actual, out_expected)


if __name__ == "__main__":
    unittest.main()
