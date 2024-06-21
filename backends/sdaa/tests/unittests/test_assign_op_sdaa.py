# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np

from op_test import OpTest
import paddle
import paddle.base as base
from paddle import static

SEED = 2021


class TestAssign(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "assign"
        self.python_api = paddle.assign
        self.init_dtype()
        self.init_shape()

        x = np.random.random(self.input_shape).astype(self.dtype)
        self.inputs = {"X": x}

        self.attrs = {}
        self.outputs = {"Out": x}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.input_shape = (3, 3)

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestAssignShape1(TestAssign):
    def init_shape(self):
        self.input_shape = (2, 768)


class TestAssignShape2(TestAssign):
    def init_shape(self):
        self.input_shape = 1024


class TestAssignShape3(TestAssign):
    def init_shape(self):
        self.input_shape = (2, 2, 255)


class TestAssignFp16(TestAssign):
    def init_dtype(self):
        self.dtype = np.float16


class TestAssignDouble(TestAssign):
    def init_dtype(self):
        self.dtype = np.double


class TestAssignUint8(TestAssign):
    def init_dtype(self):
        self.dtype = np.uint8


class TestAssignInt8(TestAssign):
    def init_dtype(self):
        self.dtype = np.int8


class TestAssignInt16(TestAssign):
    def init_dtype(self):
        self.dtype = np.int16


class TestAssignInt32(TestAssign):
    def init_dtype(self):
        self.dtype = np.int32


class TestAssignInt64(TestAssign):
    def init_dtype(self):
        self.dtype = np.int64


class TestAssignBool(TestAssign):
    def init_dtype(self):
        self.dtype = np.bool_


class TestAssignApi(unittest.TestCase):
    def test_assign_NumpyArray(self):
        paddle.set_device("sdaa")
        for dtype in [np.bool_, np.float32, np.int32, np.int64]:
            with base.dygraph.guard():
                array = np.random.random(size=(100, 10)).astype(dtype)
                result1 = paddle.zeros(shape=[3, 3], dtype="float32")
                paddle.assign(array, result1)
            np.testing.assert_allclose(result1.numpy(), array, rtol=1e-05)

    def test_assign_List(self):
        paddle.set_device("sdaa")
        l = [1, 2, 3]
        result = paddle.assign(l)
        np.testing.assert_allclose(result.numpy(), np.array(l), rtol=1e-05)

    def test_assign_BasicTypes(self):
        paddle.set_device("sdaa")
        result1 = paddle.assign(2)
        result2 = paddle.assign(3.0)
        result3 = paddle.assign(True)
        np.testing.assert_allclose(result1.numpy(), np.array([2]), rtol=1e-05)
        np.testing.assert_allclose(result2.numpy(), np.array([3.0]), rtol=1e-05)
        np.testing.assert_allclose(result3.numpy(), np.array([1]), rtol=1e-05)

    def test_clone(self):
        paddle.set_device("sdaa")
        self.python_api = paddle.clone

        x = paddle.ones([2])
        x.stop_gradient = False
        x.retain_grads()
        clone_x = paddle.clone(x)
        clone_x.retain_grads()

        y = clone_x**3
        y.backward()

        np.testing.assert_array_equal(x, [1, 1])
        np.testing.assert_array_equal(clone_x.grad.numpy(), [3, 3])
        np.testing.assert_array_equal(x.grad.numpy(), [3, 3])
        paddle.enable_static()

        with static.program_guard(static.Program(), static.Program()):
            x_np = np.random.randn(2, 3).astype("float32")
            x = paddle.static.data("X", shape=[2, 3])
            clone_x = paddle.clone(x)
            exe = paddle.static.Executor()
            y_np = exe.run(
                paddle.static.default_main_program(),
                feed={"X": x_np},
                fetch_list=[clone_x],
            )[0]

        np.testing.assert_array_equal(y_np, x_np)
        paddle.disable_static()


class TestAssignError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        paddle.set_device("sdaa")
        with static.program_guard(static.Program(), static.Program()):
            # The type of input must be Variable or numpy.ndarray.
            x1 = base.create_lod_tensor(
                np.array([[-1]]), [[1]], paddle.CustomPlace("sdaa", 0)
            )
            self.assertRaises(TypeError, paddle.assign, x1)
            # When the type of input is numpy.ndarray, the dtype of input must be float32, int32.
            x2 = np.array([[2.5, 2.5]], dtype="uint8")
            self.assertRaises(TypeError, paddle.assign, x2)
        paddle.disable_static()

    def test_type_error(self):
        paddle.enable_static()
        paddle.set_device("sdaa")
        with static.program_guard(static.Program(), static.Program()):
            x = [paddle.randn([3, 3]), paddle.randn([3, 3])]
            # not support to assign list(var)
            self.assertRaises(TypeError, paddle.assign, x)
        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
