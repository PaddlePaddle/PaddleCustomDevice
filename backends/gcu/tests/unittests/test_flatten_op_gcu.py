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

paddle.enable_static()


class TestFlattenOp(OpTest):
    def setUp(self):
        self.op_type = "flatten_contiguous_range"
        self.python_api = paddle.flatten
        self.public_python_api = paddle.flatten
        self.python_out_sig = ["Out"]
        self.start_axis = 0
        self.stop_axis = -1
        self.set_device()
        self.init_test_case()
        self.init_test_dtype()
        self.init_input_data()
        self.init_attrs()
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.new_shape),
            "XShape": np.random.random(self.in_shape).astype("float32"),
        }

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place, no_check_set=["XShape"])

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")

    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 0
        self.stop_axis = -1
        self.new_shape = 120

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis,
        }

    def init_test_dtype(self):
        self.dtype = "float32"

    def init_input_data(self):
        x = np.random.random(self.in_shape).astype(self.dtype)
        self.inputs = {"X": x}


class TestFlattenFP64Op(TestFlattenOp):
    def init_test_dtype(self):
        self.dtype = "float64"


class TestFlattenOp_1(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 1
        self.stop_axis = 2
        self.new_shape = (3, 10, 4)

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis,
        }


class TestFlattenOp_2(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 0
        self.stop_axis = 1
        self.new_shape = (6, 5, 4)

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis,
        }


class TestFlattenOp_3(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 0
        self.stop_axis = 2
        self.new_shape = (30, 4)

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis,
        }


class TestFlattenOp_4(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = -2
        self.stop_axis = -1
        self.new_shape = (3, 2, 20)

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis,
        }


class TestFlattenOp_5(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 5, 4)
        self.start_axis = 2
        self.stop_axis = 2
        self.new_shape = (3, 2, 5, 4)

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis,
        }


class TestFlattenOpSixDims(TestFlattenOp):
    def init_test_case(self):
        self.in_shape = (3, 2, 3, 2, 4, 4)
        self.start_axis = 3
        self.stop_axis = 5
        self.new_shape = (3, 2, 3, 32)

    def init_attrs(self):
        self.attrs = {
            "start_axis": self.start_axis,
            "stop_axis": self.stop_axis,
        }


class TestFlattenFP64OpSixDims(TestFlattenOpSixDims):
    def init_test_dtype(self):
        self.dtype = "float64"


class TestFlatten2OpError(unittest.TestCase):
    def test_errors(self):
        paddle.set_device("gcu")
        image_shape = (2, 3, 4, 4)
        x = (
            np.arange(
                image_shape[0] * image_shape[1] * image_shape[2] * image_shape[3]
            ).reshape(image_shape)
            / 100.0
        )
        x = x.astype("float32")

        def test_ValueError1():
            x_var = paddle.static.data(name="x", shape=image_shape, dtype="float32")
            out = paddle.flatten(x_var, start_axis=3, stop_axis=1)

        self.assertRaises(ValueError, test_ValueError1)

        def test_ValueError2():
            x_var = paddle.static.data(name="x", shape=image_shape, dtype="float32")
            paddle.flatten(x_var, start_axis=10, stop_axis=1)

        self.assertRaises(ValueError, test_ValueError2)

        def test_ValueError3():
            x_var = paddle.static.data(name="x", shape=image_shape, dtype="float32")
            paddle.flatten(x_var, start_axis=2, stop_axis=10)

        self.assertRaises(ValueError, test_ValueError3)

        def test_ValueError4():
            x_var = paddle.static.data(name="x", shape=image_shape, dtype="float32")
            paddle.flatten(x_var, start_axis=2.0, stop_axis=10)

        self.assertRaises(ValueError, test_ValueError4)

        def test_ValueError5():
            x_var = paddle.static.data(name="x", shape=image_shape, dtype="float32")
            paddle.flatten(x_var, start_axis=2, stop_axis=10.0)

        self.assertRaises(ValueError, test_ValueError5)

        def test_InputError():
            out = paddle.flatten(x)

        self.assertRaises(ValueError, test_InputError)


class TestFlattenPython(unittest.TestCase):
    def test_python_api(self):
        paddle.set_device("gcu")
        image_shape = (2, 3, 4, 4)
        x = (
            np.arange(
                image_shape[0] * image_shape[1] * image_shape[2] * image_shape[3]
            ).reshape(image_shape)
            / 100.0
        )
        x = x.astype("float32")

        def test_InputError():
            out = paddle.flatten(x)

        self.assertRaises(ValueError, test_InputError)

        def test_Negative():
            paddle.disable_static()
            img = paddle.to_tensor(x)
            out = paddle.flatten(img, start_axis=-2, stop_axis=-1)
            return out.numpy().shape

        res_shape = test_Negative()
        self.assertTrue((2, 3, 16) == res_shape)


if __name__ == "__main__":
    unittest.main()
