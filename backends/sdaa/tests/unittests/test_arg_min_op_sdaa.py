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

paddle.enable_static()


class BaseTestCase(OpTest):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (3, 4)
        self.dtype = "float32"
        self.axis = 1

    def setUp(self):
        self.initTestCase()
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)
        np.random.seed(2021)
        self.x = (np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {"X": self.x}
        self.attrs = {"axis": self.axis}
        self.outputs = {"Out": np.argmin(self.x, axis=self.axis)}

    def test_check_output(self):
        self.check_output_with_place(self.place)


@unittest.skip("sdaa not support the dtype is int32 for output")
# test param dtype is 2(int32)
class TestArgMinParamDtypeInt32(BaseTestCase):
    def setUp(self):
        self.initTestCase()
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)
        np.random.seed(2021)
        self.x = (np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {"X": self.x}
        self.attrs = {"axis": self.axis, "dtype": 2}
        self.outputs = {"Out": np.argmin(self.x, axis=self.axis)}


# test argmin, dtype: float16
class TestArgMinFloat16Case1(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (3, 4, 5)
        self.dtype = "float16"
        self.axis = -1


class TestArgMinFloat16Case2(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (3, 4, 5)
        self.dtype = "float16"
        self.axis = 0


class TestArgMinFloat16Case3(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (3, 4, 5)
        self.dtype = "float16"
        self.axis = 1


class TestArgMinFloat16Case4(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (3, 4, 5)
        self.dtype = "float16"
        self.axis = 2


class TestArgMinFloat16Case5(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (3, 4)
        self.dtype = "float16"
        self.axis = -1


class TestArgMinFloat16Case6(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (3, 4)
        self.dtype = "float16"
        self.axis = 0


class TestArgMinFloat16Case7(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (3, 4)
        self.dtype = "float16"
        self.axis = 1


class TestArgMinFloat16Case8(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (1,)
        self.dtype = "float16"
        self.axis = 0


class TestArgMinFloat16Case9(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (2,)
        self.dtype = "float16"
        self.axis = 0


class TestArgMinFloat16Case10(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (3,)
        self.dtype = "float16"
        self.axis = 0


# test argmin, dtype: float32
class TestArgMinFloat32Case1(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (3, 4, 5)
        self.dtype = "float32"
        self.axis = -1


class TestArgMinFloat32Case2(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (3, 4, 5)
        self.dtype = "float32"
        self.axis = 0


class TestArgMinFloat32Case3(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (3, 4, 5)
        self.dtype = "float32"
        self.axis = 1


class TestArgMinFloat32Case4(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (3, 4, 5)
        self.dtype = "float32"
        self.axis = 2


class TestArgMinFloat32Case5(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (3, 4)
        self.dtype = "float32"
        self.axis = -1


class TestArgMinFloat32Case6(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (3, 4)
        self.dtype = "float32"
        self.axis = 0


class TestArgMinFloat32Case7(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (3, 4)
        self.dtype = "float32"
        self.axis = 1


class TestArgMinFloat32Case8(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (1,)
        self.dtype = "float32"
        self.axis = 0


class TestArgMinFloat32Case9(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (2,)
        self.dtype = "float32"
        self.axis = 0


class TestArgMinFloat32Case10(BaseTestCase):
    def initTestCase(self):
        self.op_type = "arg_min"
        self.python_api = paddle.tensor.argmin
        self.dims = (3,)
        self.dtype = "float32"
        self.axis = 0


class TestArgMinAPI(unittest.TestCase):
    def initTestCase(self):
        self.dims = (3, 4, 5)
        self.dtype = "float32"
        self.axis = 0

    def setUp(self):
        self.initTestCase()
        self.__class__.use_custom_device = True
        self.place = [paddle.CustomPlace("sdaa", 0)]

    def test_dygraph_api(self):
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            numpy_input = (np.random.random(self.dims)).astype(self.dtype)
            tensor_input = paddle.to_tensor(numpy_input)
            numpy_output = np.argmin(numpy_input, axis=self.axis)
            paddle_output = paddle.argmin(tensor_input, axis=self.axis)
            self.assertEqual(np.allclose(numpy_output, paddle_output.numpy()), True)
            assert paddle_output.dtype == paddle.int64
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
        self.initTestCase()
        self.__class__.use_custom_device = True
        self.place = [paddle.CustomPlace("sdaa", 0)]

    def test_dygraph_api(self):
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            numpy_input = (np.random.random(self.dims)).astype(self.dtype)
            tensor_input = paddle.to_tensor(numpy_input)
            numpy_output = np.argmin(numpy_input, axis=self.axis).reshape(1, 4, 5)
            paddle_output = paddle.argmin(
                tensor_input, axis=self.axis, keepdim=self.keep_dims
            )
            self.assertEqual(np.allclose(numpy_output, paddle_output.numpy()), True)
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)


if __name__ == "__main__":
    unittest.main()
