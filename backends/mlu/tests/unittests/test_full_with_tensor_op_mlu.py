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
import paddle
import numpy as np


class TestFullWithTensor(unittest.TestCase):
    def setUp(self):
        self.dtypes = ["bool", "uint8", "int16", "int32", "int64", "float32", "float16"]
        self.shape = [5, 6]
        self.place = paddle.CustomPlace("mlu", 0)
        self.value = 2

    def test_static(self):
        paddle.enable_static()
        shapes = [
            self.shape,
            tuple(self.shape),
            [paddle.to_tensor(shape_i) for shape_i in self.shape],
            paddle.to_tensor(self.shape),
        ]
        for dtype in self.dtypes:
            for shape in shapes:
                out = paddle.full(
                    shape=shape, fill_value=paddle.to_tensor(self.value, dtype=dtype)
                )
                exe = paddle.static.Executor(self.place)
                exe.run(paddle.static.default_startup_program())
                out_actual = exe.run(
                    paddle.static.default_main_program(), feed={}, fetch_list=[out]
                )[0]
                out_expected = np.full(self.shape, 2, dtype=dtype)
                np.testing.assert_allclose(out_actual, out_expected)

    def test_python_api(self):
        paddle.disable_static()
        shapes = [
            self.shape,
            tuple(self.shape),
            [paddle.to_tensor(shape_i) for shape_i in self.shape],
            paddle.to_tensor(self.shape),
        ]
        for dtype in self.dtypes:
            for shape in shapes:
                out = paddle.full(
                    shape=shape, fill_value=paddle.to_tensor(self.value, dtype=dtype)
                )
                out_expected = np.full(self.shape, 2, dtype=dtype)
                np.testing.assert_allclose(out.numpy(), out_expected)


if __name__ == "__main__":
    unittest.main()
