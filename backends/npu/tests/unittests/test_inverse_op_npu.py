# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import os

select_npu = os.environ.get("FLAGS_selected_npus", 0)

import numpy as np
from tests.op_test import OpTest

import paddle
from paddle import base


class TestInverseOp(OpTest):
    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", select_npu)

    def config(self):
        self.matrix_shape = [10, 10]
        self.dtype = "float64"
        self.python_api = paddle.tensor.math.inverse

    def setUp(self):
        self.set_npu()
        self.op_type = "inverse"
        self.config()

        np.random.seed(123)
        mat = np.random.random(self.matrix_shape).astype(self.dtype)
        inverse = np.linalg.inv(mat)

        self.inputs = {"Input": mat}
        self.outputs = {"Output": inverse}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_grad(self):
        pass


class TestInverseOpBatched(TestInverseOp):
    def config(self):
        self.matrix_shape = [8, 4, 4]
        self.dtype = "float64"
        self.python_api = paddle.tensor.math.inverse


class TestInverseOpLarge(TestInverseOp):
    def config(self):
        self.matrix_shape = [32, 32]
        self.dtype = "float64"
        self.python_api = paddle.tensor.math.inverse


class TestInverseOpFP32(TestInverseOp):
    def config(self):
        self.matrix_shape = [10, 10]
        self.dtype = "float32"
        self.python_api = paddle.tensor.math.inverse

    def test_check_output(self):
        self.check_output_with_place(self.place, rtol=1e-3, atol=1e-4)


class TestInverseOpBatchedFP32(TestInverseOpFP32):
    def config(self):
        self.matrix_shape = [8, 4, 4]
        self.dtype = "float32"
        self.python_api = paddle.tensor.math.inverse


class TestInverseOpLargeFP32(TestInverseOpFP32):
    def config(self):
        self.matrix_shape = [32, 32]
        self.dtype = "float32"
        self.python_api = paddle.tensor.math.inverse


class TestInverseAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [paddle.CustomPlace("npu", select_npu)]

    def check_static_result(self, place):
        paddle.enable_static()
        with base.program_guard(base.Program(), base.Program()):
            input = paddle.static.data(name="input", shape=[4, 4], dtype="float64")
            result = paddle.inverse(x=input)
            input_np = np.random.random([4, 4]).astype("float64")
            result_np = np.linalg.inv(input_np)

            exe = base.Executor(place)
            fetches = exe.run(
                base.default_main_program(),
                feed={"input": input_np},
                fetch_list=[result],
            )
            np.testing.assert_allclose(fetches[0], np.linalg.inv(input_np), rtol=1e-05)

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                input_np = np.random.random([4, 4]).astype("float64")
                input = base.dygraph.to_variable(input_np)
                result = paddle.inverse(input)
                np.testing.assert_allclose(
                    result.numpy(), np.linalg.inv(input_np), rtol=1e-05
                )


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
