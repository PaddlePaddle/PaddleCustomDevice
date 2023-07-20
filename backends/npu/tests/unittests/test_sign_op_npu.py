#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from tests.op_test import OpTest

import paddle
from paddle import fluid


class TestSignOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "sign"
        self.python_api = paddle.sign
        self.inputs = {"X": np.random.uniform(-10, 10, (10, 10)).astype("float64")}
        self.outputs = {"Out": np.sign(self.inputs["X"])}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def test_check_grad(self):
        pass


class TestSignFP16Op(TestSignOp):
    def setUp(self):
        self.set_npu()
        self.op_type = "sign"
        self.python_api = paddle.sign
        self.inputs = {"X": np.random.uniform(-10, 10, (10, 10)).astype("float16")}
        self.outputs = {"Out": np.sign(self.inputs["X"])}


class TestSignFP32Op(TestSignOp):
    def setUp(self):
        self.set_npu()
        self.op_type = "sign"
        self.python_api = paddle.sign
        self.inputs = {"X": np.random.uniform(-10, 10, (10, 10)).astype("float32")}
        self.outputs = {"Out": np.sign(self.inputs["X"])}


class TestSignAPI(unittest.TestCase):
    def test_dygraph(self):
        with fluid.dygraph.guard(paddle.CustomPlace("npu", 0)):
            np_x = np.array([-1.0, 0.0, -0.0, 1.2, 1.5], dtype="float64")
            x = paddle.to_tensor(np_x)
            z = paddle.sign(x)
            np_z = z.numpy()
            z_expected = np.sign(np_x)
            self.assertEqual((np_z == z_expected).all(), True)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
