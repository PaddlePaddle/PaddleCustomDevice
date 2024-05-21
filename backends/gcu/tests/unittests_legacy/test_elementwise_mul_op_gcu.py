#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from tests.op_test import OpTest, skip_check_grad_ci

import paddle


class ElementwiseMulOp(OpTest):
    def init_kernel_type(self):
        self.use_mkldnn = False

    def setUp(self):
        self.op_type = "elementwise_mul"
        self.prim_op_type = "prim"
        self.python_api = paddle.multiply
        self.public_python_api = paddle.multiply
        self.init_dtype()
        self.init_axis()
        self.set_device()
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()
        self.if_enable_cinn()

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(self.x),
            "Y": OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.outputs = {"Out": self.out}
        self.attrs = {"axis": self.axis, "use_mkldnn": self.use_mkldnn}

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ["X", "Y"], "Out")

    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(self.place, ["Y"], "Out", no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(self.place, ["X"], "Out", no_grad_set=set("Y"))

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    def init_dtype(self):
        self.dtype = np.float32

    def init_axis(self):
        self.axis = -1

    def if_enable_cinn(self):
        pass


@skip_check_grad_ci(reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestElementwiseMulOp_scalar(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.set_device()
        self.prim_op_type = "prim"
        self.python_api = paddle.multiply
        self.public_python_api = paddle.multiply
        self.inputs = {
            "X": np.random.rand(10, 3, 4).astype(np.float64),
            "Y": np.random.rand(1).astype(np.float64),
        }
        self.outputs = {"Out": self.inputs["X"] * self.inputs["Y"]}
        self.init_kernel_type()


class TestElementwiseMulOp_Vector(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.set_device()
        self.prim_op_type = "prim"
        self.python_api = paddle.multiply
        self.public_python_api = paddle.multiply
        self.inputs = {
            "X": np.random.random((100,)).astype("float64"),
            "Y": np.random.random((100,)).astype("float64"),
        }
        self.outputs = {"Out": np.multiply(self.inputs["X"], self.inputs["Y"])}
        self.init_kernel_type()


class TestElementwiseMulOp_broadcast_0(ElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.rand(100, 2, 3).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x * self.y.reshape(100, 1, 1)

    def init_axis(self):
        self.axis = 0


class TestElementwiseMulOp_broadcast_1(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.set_device()
        self.inputs = {
            "X": np.random.rand(2, 100, 3).astype(np.float32),
            "Y": np.random.rand(100).astype(np.float32),
        }

        self.attrs = {"axis": 1}
        self.outputs = {"Out": self.inputs["X"] * self.inputs["Y"].reshape(1, 100, 1)}
        self.init_kernel_type()


class TestElementwiseMulOp_broadcast_2(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.set_device()
        self.inputs = {
            "X": np.random.rand(2, 3, 100).astype(np.float32),
            "Y": np.random.rand(100).astype(np.float32),
        }

        self.outputs = {"Out": self.inputs["X"] * self.inputs["Y"].reshape(1, 1, 100)}
        self.init_kernel_type()


class TestElementwiseMulOp_broadcast_3(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.set_device()
        self.inputs = {
            "X": np.random.rand(2, 10, 12, 3).astype(np.float32),
            "Y": np.random.rand(10, 12).astype(np.float32),
        }

        self.attrs = {"axis": 1}
        self.outputs = {
            "Out": self.inputs["X"] * self.inputs["Y"].reshape(1, 10, 12, 1)
        }
        self.init_kernel_type()


class TestElementwiseMulOp_broadcast_4(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.set_device()
        self.inputs = {
            "X": np.random.rand(10, 2, 11).astype(np.float32),
            "Y": np.random.rand(10, 1, 11).astype(np.float32),
        }
        self.outputs = {"Out": self.inputs["X"] * self.inputs["Y"]}
        self.init_kernel_type()


class TestElementwiseMulOp_broadcast_5(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.set_device()
        self.inputs = {
            "X": np.random.rand(10, 4, 2, 3).astype(np.float32),
            "Y": np.random.rand(10, 4, 1, 3).astype(np.float32),
        }
        self.outputs = {"Out": self.inputs["X"] * self.inputs["Y"]}
        self.init_kernel_type()


class TestElementwiseMulOp_commonuse_1(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.set_device()
        self.prim_op_type = "prim"
        self.python_api = paddle.multiply
        self.public_python_api = paddle.multiply
        self.inputs = {
            "X": np.random.rand(2, 3, 100).astype(np.float64),
            "Y": np.random.rand(1, 1, 100).astype(np.float64),
        }
        self.outputs = {"Out": self.inputs["X"] * self.inputs["Y"]}
        self.init_kernel_type()


class TestElementwiseMulOp_commonuse_2(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.set_device()
        self.prim_op_type = "prim"
        self.python_api = paddle.multiply
        self.public_python_api = paddle.multiply
        self.inputs = {
            "X": np.random.rand(30, 3, 1, 5).astype(np.float64),
            "Y": np.random.rand(30, 1, 4, 1).astype(np.float64),
        }
        self.outputs = {"Out": self.inputs["X"] * self.inputs["Y"]}
        self.init_kernel_type()


class TestElementwiseMulOp_xsize_lessthan_ysize(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.set_device()
        self.prim_op_type = "prim"
        self.python_api = paddle.multiply
        self.public_python_api = paddle.multiply
        self.inputs = {
            "X": np.random.rand(10, 10).astype(np.float64),
            "Y": np.random.rand(2, 2, 10, 10).astype(np.float64),
        }

        self.attrs = {"axis": 2}

        self.outputs = {
            "Out": self.inputs["X"].reshape(1, 1, 10, 10) * self.inputs["Y"]
        }
        self.init_kernel_type()


class TestElementwiseMulop(unittest.TestCase):
    def test_dygraph_mul(self):
        paddle.disable_static()
        paddle.set_device("gcu")

        np_a = np.random.random((2, 3, 4)).astype(np.float32)
        np_b = np.random.random((2, 3, 4)).astype(np.float32)

        tensor_a = paddle.to_tensor(np_a, dtype="float32")
        tensor_b = paddle.to_tensor(np_b, dtype="float32")

        # normal case: nparray * tenor
        expect_out = np_a * np_b
        actual_out = np_a * tensor_b
        np.testing.assert_allclose(actual_out, expect_out)

        # normal case: tensor * nparray
        actual_out = tensor_a * np_b
        np.testing.assert_allclose(actual_out, expect_out)

        paddle.enable_static()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
