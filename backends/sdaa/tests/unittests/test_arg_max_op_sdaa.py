# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.base.core as core

paddle.enable_static()


class BaseTestCase(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (3, 4, 5)
        self.dtype = "float32"
        self.axis = 0

    def setUp(self):
        self.set_sdaa()
        self.python_api = paddle.argmax
        self.initTestCase()
        self.x = (1000 * np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {"X": self.x}
        self.attrs = {"axis": self.axis}
        self.outputs = {"Out": np.argmax(self.x, axis=self.axis)}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestArgMaxSameValue1(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dtype = "float32"
        self.axis = 0

    def setUp(self):
        self.set_sdaa()
        self.python_api = paddle.argmax
        self.initTestCase()
        self.x = np.array([1, 2, 3, 5, 4, 5]).astype(self.dtype)
        self.inputs = {"X": self.x}
        self.attrs = {"axis": self.axis}
        self.outputs = {"Out": np.argmax(self.x, axis=self.axis)}


class TestArgMaxSameValue2(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dtype = "float16"
        self.axis = 0

    def setUp(self):
        self.set_sdaa()
        self.python_api = paddle.argmax
        self.initTestCase()
        self.x = np.array([[2, 3, 5, 5], [3, 2, 5, 5]]).astype(self.dtype)
        self.inputs = {"X": self.x}
        self.attrs = {"axis": self.axis}
        self.outputs = {"Out": np.argmax(self.x, axis=self.axis)}


# test argmax, dtype: float16
class TestArgMaxFloat16Case1(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (3, 4, 5)
        self.dtype = "float16"
        self.axis = -1


class TestArgMaxFloat16Case2(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (3, 4, 5)
        self.dtype = "float16"
        self.axis = 0


class TestArgMaxFloat16Case3(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (3, 4, 5)
        self.dtype = "float16"
        self.axis = 1


class TestArgMaxFloat16Case4(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (3, 4, 5)
        self.dtype = "float16"
        self.axis = 2


class TestArgMaxFloat16Case5(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (3, 4)
        self.dtype = "float16"
        self.axis = -1


class TestArgMaxFloat16Case6(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (3, 4)
        self.dtype = "float16"
        self.axis = 0


class TestArgMaxFloat16Case7(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (3, 4)
        self.dtype = "float16"
        self.axis = 1


class TestArgMaxFloat16Case8(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (1,)
        self.dtype = "float16"
        self.axis = 0


class TestArgMaxFloat16Case9(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (2,)
        self.dtype = "float16"
        self.axis = 0


class TestArgMaxFloat16Case10(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (3,)
        self.dtype = "float16"
        self.axis = 0


# test argmax, dtype: float32
class TestArgMaxFloat32Case1(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (3, 4, 5)
        self.dtype = "float32"
        self.axis = -1


class TestArgMaxFloat32Case2(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (3, 4, 5)
        self.dtype = "float32"
        self.axis = 0


class TestArgMaxFloat32Case3(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (3, 4, 5)
        self.dtype = "float32"
        self.axis = 1


class TestArgMaxFloat32Case4(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (3, 4, 5)
        self.dtype = "float32"
        self.axis = 2


class TestArgMaxFloat32Case5(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (3, 4)
        self.dtype = "float32"
        self.axis = -1


class TestArgMaxFloat32Case6(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (3, 4)
        self.dtype = "float32"
        self.axis = 0


class TestArgMaxFloat32Case7(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (3, 4)
        self.dtype = "float32"
        self.axis = 1


class TestArgMaxFloat32Case8(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (1,)
        self.dtype = "float32"
        self.axis = 0


class TestArgMaxFloat32Case9(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (2,)
        self.dtype = "float32"
        self.axis = 0


class TestArgMaxFloat32Case10(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (3,)
        self.dtype = "float32"
        self.axis = 0


class BaseTestComplex1_1(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (4, 5, 6)
        self.dtype = "float32"
        self.axis = 2

    def setUp(self):
        self.set_sdaa()
        self.python_api = paddle.argmax
        self.initTestCase()
        self.x = (np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {"X": self.x}
        self.attrs = {"axis": self.axis, "dtype": int(core.VarDesc.VarType.INT64)}
        self.outputs = {"Out": np.argmax(self.x, axis=self.axis).astype("int64")}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class BaseTestComplex1_2(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def initTestCase(self):
        self.op_type = "arg_max"
        self.dims = (4, 5, 6)
        self.dtype = "float16"
        self.axis = 2

    def setUp(self):
        self.set_sdaa()
        self.python_api = paddle.argmax
        self.initTestCase()
        self.x = (np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {"X": self.x}
        self.attrs = {"axis": self.axis, "dtype": int(core.VarDesc.VarType.INT64)}
        self.outputs = {"Out": np.argmax(self.x, axis=self.axis).astype("int64")}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestArgMaxAPI(unittest.TestCase):
    def initTestCase(self):
        self.dims = (3, 4, 5)
        self.dtype = "float32"
        self.axis = 0

    def setUp(self):
        self.python_api = paddle.argmax
        self.initTestCase()
        self.__class__.use_custom_device = True
        self.place = [paddle.CustomPlace("sdaa", 0)]

    def test_dygraph_api(self):
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2022)
            numpy_input = (np.random.random(self.dims)).astype(self.dtype)
            tensor_input = paddle.to_tensor(numpy_input)
            numpy_output = np.argmax(numpy_input, axis=self.axis)
            paddle_output = paddle.argmax(tensor_input, axis=self.axis)
            np.testing.assert_allclose(numpy_output, paddle_output.numpy(), rtol=1e-05)
            paddle.enable_static()

        for place in self.place:
            run(place)


class TestArgMaxAPI_2(unittest.TestCase):
    def initTestCase(self):
        self.dims = (3, 4, 5)
        self.dtype = "float32"
        self.axis = 0
        self.keep_dims = True

    def setUp(self):
        self.python_api = paddle.argmax
        self.initTestCase()
        self.__class__.use_custom_device = True
        self.place = [paddle.CustomPlace("sdaa", 0)]

    def test_dygraph_api(self):
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2022)
            numpy_input = (np.random.random(self.dims)).astype(self.dtype)
            tensor_input = paddle.to_tensor(numpy_input)
            numpy_output = np.argmax(numpy_input, axis=self.axis).reshape(1, 4, 5)
            paddle_output = paddle.argmax(
                tensor_input, axis=self.axis, keepdim=self.keep_dims
            )
            np.testing.assert_allclose(numpy_output, paddle_output.numpy(), rtol=1e-05)
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)


class TestArgMaxAPI_3(unittest.TestCase):
    def initTestCase(self):
        self.dims = (1, 9)
        self.dtype = "float32"

    def setUp(self):
        self.python_api = paddle.argmax
        self.initTestCase()
        self.__class__.use_custom_device = True
        self.place = [paddle.CustomPlace("sdaa", 0)]

    def test_dygraph_api(self):
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2022)
            numpy_input = (np.random.random(self.dims)).astype(self.dtype)
            tensor_input = paddle.to_tensor(numpy_input)
            numpy_output = np.argmax(numpy_input)
            paddle_output = paddle.argmax(tensor_input)
            np.testing.assert_allclose(numpy_output, paddle_output.numpy(), rtol=1e-05)
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)


if __name__ == "__main__":
    unittest.main()
