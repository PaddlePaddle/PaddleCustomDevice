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

from op_test import OpTest
import paddle

paddle.enable_static()


class TestReshapeOp(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"
        self.set_device()
        self.inputs = {"X": np.random.random(self.ori_shape).astype("float32")}
        self.attrs = {"shape": self.new_shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.infered_shape),
            "XShape": np.random.random(self.ori_shape).astype("float32"),
        }

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def init_data(self):
        self.ori_shape = (2, 60)
        self.new_shape = (12, 10)
        self.infered_shape = (12, 10)

    def test_check_output(self):
        self.check_output_with_place(self.place, no_check_set=["XShape"])

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestReshapeOpDimInfer1(TestReshapeOp):
    def init_data(self):
        self.ori_shape = (5, 25)
        self.new_shape = (5, -1, 5)
        self.infered_shape = (5, -1, 5)


class TestReshapeOpDimInfer2(TestReshapeOp):
    def init_data(self):
        self.ori_shape = (10, 2, 6)
        self.new_shape = (10, 0, 3, -1)
        self.infered_shape = (10, 2, 3, -1)


class TestReshapeOpBool(TestReshapeOp):
    def setUp(self):
        self.init_data()
        self.op_type = "reshape2"
        self.set_device()
        self.inputs = {"X": np.random.choice([True, False], size=self.ori_shape)}
        self.attrs = {"shape": self.new_shape}
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.infered_shape),
            "XShape": np.random.random(self.ori_shape).astype("float32"),
        }

    def test_check_grad(self):
        pass


class TestDygraphReshapeAPI(unittest.TestCase):
    def setUp(self):
        self.executed_api()

    def executed_api(self):
        self.reshape = paddle.reshape

    def test_out(self):
        paddle.disable_static(paddle.CustomPlace("gcu", 0))
        input_1 = np.random.random([5, 1, 10]).astype("int32")
        input = paddle.to_tensor(input_1)
        output = self.reshape(x=input, shape=[5, 10])
        out_np = output.numpy()
        expected_out = np.reshape(input_1, newshape=[5, 10])
        self.assertTrue(np.allclose(expected_out, out_np))

    def test_out_uint8(self):
        paddle.disable_static(paddle.CustomPlace("gcu", 0))
        input_1 = np.random.random([5, 1, 10]).astype("uint8")
        input = paddle.to_tensor(input_1)
        output = self.reshape(x=input, shape=[5, 10])
        out_np = output.numpy()
        expected_out = np.reshape(input_1, newshape=[5, 10])
        self.assertTrue(np.allclose(expected_out, out_np))

    def test_out_float32(self):
        paddle.disable_static(paddle.CustomPlace("gcu", 0))
        input_1 = np.random.random([5, 1, 10]).astype("float32")
        input = paddle.to_tensor(input_1)
        output = self.reshape(x=input, shape=[5, 10])
        out_np = output.numpy()
        expected_out = np.reshape(input_1, newshape=[5, 10])
        self.assertTrue(np.allclose(expected_out, out_np))


if __name__ == "__main__":
    unittest.main()
