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

import numpy as np
from tests.op_test import OpTest

import paddle
from paddle import fluid
from paddle.fluid import Program, program_guard


class TestRollOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.python_api = paddle.roll
        self.op_type = "roll"
        self.public_python_api = paddle.roll
        self.init_dtype_type()
        self.attrs = {"shifts": self.shifts, "axis": self.axis}
        x = np.random.random(self.x_shape).astype(self.dtype)
        out = np.roll(x, self.attrs["shifts"], self.attrs["axis"])
        self.inputs = {"X": x}
        self.outputs = {"Out": out}

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def init_dtype_type(self):
        self.dtype = np.float32
        self.x_shape = (100, 4, 5)
        self.shifts = [101, -1]
        self.axis = [0, -2]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestRollOpCase2(TestRollOp):
    def init_dtype_type(self):
        self.dtype = np.float32
        self.x_shape = (100, 10, 5)
        self.shifts = [8, -1]
        self.axis = [-1, -2]


class TestRollOpCase3(TestRollOp):
    def init_dtype_type(self):
        self.dtype = np.float32
        self.x_shape = (11, 11)
        self.shifts = [1, 1]
        self.axis = [-1, 1]


class TestRollFP16OP(TestRollOp):
    def init_dtype_type(self):
        self.dtype = np.float16
        self.x_shape = (100, 4, 5)
        self.shifts = [101, -1]
        self.axis = [0, -2]


class TestRollFP16OpCase2(TestRollOp):
    def init_dtype_type(self):
        self.dtype = np.float16
        self.x_shape = (100, 10, 5)
        self.shifts = [8, -1]
        self.axis = [-1, -2]


class TestRollFP16OpCase3(TestRollOp):
    def init_dtype_type(self):
        self.dtype = np.float16
        self.x_shape = (11, 11)
        self.shifts = [1, 1]
        self.axis = [-1, 1]


class TestRollInt32(TestRollOp):
    def setUp(self):
        self.set_npu()
        self.python_api = paddle.roll
        self.op_type = "roll"
        self.public_python_api = paddle.roll
        self.init_dtype_type()
        self.attrs = {"shifts": self.shifts, "axis": self.axis}
        x = np.random.randint(0, 100, self.x_shape, self.dtype)
        out = np.roll(x, self.attrs["shifts"], self.attrs["axis"])
        self.inputs = {"X": x}
        self.outputs = {"Out": out}

    def init_dtype_type(self):
        self.dtype = np.int32
        self.x_shape = (100, 4, 5)
        self.shifts = [101, -1]
        self.axis = [0, -2]

    def test_check_grad(self):
        pass


class TestRollAPI(unittest.TestCase):
    def input_data(self):
        self.data_x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    def test_roll_op_api(self):
        self.input_data()

        paddle.enable_static()
        # case 1:
        with program_guard(Program(), Program()):
            x = paddle.static.data(name="x", shape=[-1, 3], dtype="float32")
            x.desc.set_need_check_feed(False)
            z = paddle.roll(x, shifts=1)
            exe = fluid.Executor(paddle.CustomPlace("npu", 0))
            (res,) = exe.run(
                feed={"x": self.data_x}, fetch_list=[z.name], return_numpy=False
            )
            expect_out = np.array([[9.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
            np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

        # case 2:
        with program_guard(Program(), Program()):
            x = paddle.static.data(name="x", shape=[-1, 3], dtype="float32")
            x.desc.set_need_check_feed(False)
            z = paddle.roll(x, shifts=1, axis=0)
            exe = fluid.Executor(paddle.CustomPlace("npu", 0))
            (res,) = exe.run(
                feed={"x": self.data_x}, fetch_list=[z.name], return_numpy=False
            )
        expect_out = np.array([[7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(paddle.CustomPlace("npu", 0))
        self.input_data()
        # case 1:
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(self.data_x)
            z = paddle.roll(x, shifts=1)
            np_z = z.numpy()
        expect_out = np.array([[9.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)

        # case 2:
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(self.data_x)
            z = paddle.roll(x, shifts=1, axis=0)
            np_z = z.numpy()
        expect_out = np.array([[7.0, 8.0, 9.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)
        paddle.enable_static()

    def test_shifts_as_tensor_dygraph(self):
        paddle.disable_static(paddle.CustomPlace("npu", 0))
        with fluid.dygraph.guard():
            x = paddle.arange(9).reshape([3, 3])
            shape = paddle.shape(x)
            shifts = shape // 2
            axes = [0, 1]
            out = paddle.roll(x, shifts=shifts, axis=axes).numpy()
            expected_out = np.array([[8, 6, 7], [2, 0, 1], [5, 3, 4]])
            np.testing.assert_allclose(out, expected_out, rtol=1e-05)
        paddle.enable_static()

    def test_shifts_as_tensor_static(self):
        with program_guard(Program(), Program()):
            x = paddle.arange(9).reshape([3, 3]).astype("float32")
            shape = paddle.shape(x)
            shifts = shape // 2
            axes = [0, 1]
            out = paddle.roll(x, shifts=shifts, axis=axes)
            expected_out = np.array([[8, 6, 7], [2, 0, 1], [5, 3, 4]])

            exe = fluid.Executor(paddle.CustomPlace("npu", 0))
            [out_np] = exe.run(fetch_list=[out])
            np.testing.assert_allclose(out_np, expected_out, rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
