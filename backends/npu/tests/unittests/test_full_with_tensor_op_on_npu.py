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
from tests.op_test import convert_uint16_to_float


class TestFullWithTensor(unittest.TestCase):
    def setUp(self):
        self.dtypes = [
            "bool",
            "int32",
            "int64",
            "float32",
            "float64",
            "float16",
            "bfloat16",
        ]
        self.shape = [5, 6]
        self.place = paddle.CustomPlace("npu", 0)
        self.value = 2

    def get_expected_output(self, dtype):
        dtype = dtype if dtype != "bfloat16" else "float32"
        return np.full(self.shape, 2, dtype=dtype)

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
                    shape=shape,
                    fill_value=paddle.to_tensor(self.value, dtype=dtype),
                    dtype=dtype,
                )
                exe = paddle.static.Executor(self.place)
                exe.run(paddle.static.default_startup_program())
                out_actual = exe.run(
                    paddle.static.default_main_program(), feed={}, fetch_list=[out]
                )[0]

                if dtype == "bfloat16":
                    out_actual = convert_uint16_to_float(out_actual)
                out_expected = self.get_expected_output(dtype)
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
                    shape=shape,
                    fill_value=paddle.to_tensor(self.value, dtype=dtype),
                    dtype=dtype,
                )
                out_actual = (
                    convert_uint16_to_float(out.numpy())
                    if dtype == "bfloat16"
                    else out.numpy()
                )
                out_expected = self.get_expected_output(dtype)
                np.testing.assert_allclose(out_actual, out_expected)


if __name__ == "__main__":
    unittest.main()
