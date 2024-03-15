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

import numpy as np
import unittest
from tests.op_test import OpTest
import paddle.base as base
import paddle

paddle.enable_static()


class TestStackOpBase(OpTest):
    def initDefaultParameters(self):
        self.num_inputs = 4
        self.input_dim = (5, 6, 7)
        self.axis = 0

    def initParameters(self):
        pass

    def get_x_names(self):
        x_names = []
        for i in range(self.num_inputs):
            x_names.append("x{}".format(i))
        return x_names

    def setUp(self):
        self.initDefaultParameters()
        self.initParameters()
        self.op_type = "stack"
        self.set_device()
        self.init_dtype()
        self.x = []
        for i in range(self.num_inputs):
            self.x.append(np.random.random(size=self.input_dim).astype(self.dtype))

        tmp = []
        x_names = self.get_x_names()
        for i in range(self.num_inputs):
            tmp.append((x_names[i], self.x[i]))

        self.inputs = {"X": tmp}
        self.outputs = {"Y": np.stack(self.x, axis=self.axis)}
        self.attrs = {"axis": self.axis}

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        if self.dtype == np.int32 or self.dtype == np.int64:
            return
        self.check_grad_with_place(self.place, self.get_x_names(), "Y")


class TestStackOp1(TestStackOpBase):
    def initParameters(self):
        self.num_inputs = 16


class TestStackOp2(TestStackOpBase):
    def initParameters(self):
        self.num_inputs = 20


class TestStackOp3(TestStackOpBase):
    def initParameters(self):
        self.axis = -1


class TestStackOp4(TestStackOpBase):
    def initParameters(self):
        self.axis = -4


class TestStackOp5(TestStackOpBase):
    def initParameters(self):
        self.axis = 1


class TestStackOp6(TestStackOpBase):
    def initParameters(self):
        self.axis = 3


class TestStackOpINT32(TestStackOpBase):
    def init_dtype(self):
        self.dtype = np.int32


class TestStackOpINT64(TestStackOpBase):
    def init_dtype(self):
        self.dtype = np.int64


class TestStackOpFP16(TestStackOpBase):
    def init_dtype(self):
        self.dtype = np.float16


class API_test(unittest.TestCase):
    def test_out(self):
        paddle.enable_static()
        with base.program_guard(base.Program(), base.Program()):
            data1 = paddle.static.data("data1", shape=[1, 2], dtype="float32")
            data2 = paddle.static.data("data2", shape=[1, 2], dtype="float32")
            data3 = paddle.static.data("data3", shape=[1, 2], dtype="float32")
            result_stack = paddle.stack([data1, data2, data3], axis=0)
            place = paddle.CustomPlace("gcu", 0)
            exe = base.Executor(place)
            input1 = np.random.random([1, 2]).astype("float32")
            input2 = np.random.random([1, 2]).astype("float32")
            input3 = np.random.random([1, 2]).astype("float32")
            (result,) = exe.run(
                feed={"data1": input1, "data2": input2, "data3": input3},
                fetch_list=[result_stack],
            )
            expected_result = np.stack([input1, input2, input3], axis=0)
            self.assertTrue(np.allclose(expected_result, result))

    def test_single_tensor_error(self):
        paddle.enable_static()
        with base.program_guard(base.Program(), base.Program()):
            x = paddle.rand([2, 3])
            self.assertRaises(TypeError, paddle.stack, x)


class API_DygraphTest(unittest.TestCase):
    def test_out(self):
        paddle.disable_static()
        data1 = np.array([[1.0, 2.0]])
        data2 = np.array([[3.0, 4.0]])
        data3 = np.array([[5.0, 6.0]])
        with base.dygraph.guard(place=paddle.CustomPlace("gcu", 0)):
            x1 = paddle.to_tensor(data1)
            x2 = paddle.to_tensor(data2)
            x3 = paddle.to_tensor(data3)
            result = paddle.stack([x1, x2, x3])
            result_np = result.numpy()
        expected_result = np.stack([data1, data2, data3])
        self.assertTrue(np.allclose(expected_result, result_np))

        with base.dygraph.guard(place=paddle.CustomPlace("gcu", 0)):
            y1 = paddle.to_tensor(data1)
            result = paddle.stack([y1], axis=0)
            result_np_2 = result.numpy()
        expected_result_2 = np.stack([data1], axis=0)
        self.assertTrue(np.allclose(expected_result_2, result_np_2))
        paddle.enable_static()

    def test_single_tensor_error(self):
        paddle.disable_static()
        with base.dygraph.guard(place=paddle.CustomPlace("gcu", 0)):
            x = paddle.to_tensor([1, 2, 3])
            self.assertRaises(Exception, paddle.stack, x)
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
