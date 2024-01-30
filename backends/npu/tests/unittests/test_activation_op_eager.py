# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest

import numpy as np

import paddle
from tests.op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float

from npu_utils import check_soc_version

SEED = 2021


class TestActivationBf16(OpTest):
    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def setUp(self):
        self.set_npu()
        self.op_type = "exp"
        self.init_dtype()
        self.init_shape()
        self.init_kernel_type()
        self.check_dygraph = True
        self.python_api = paddle.exp

        np.random.seed(2049)
        x = convert_float_to_uint16(
            np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        )
        out = np.exp(convert_uint16_to_float(x))

        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {"Out": out}

    @check_soc_version
    def test_check_output(self):
        check_dygraph = False
        if hasattr(self, "check_dygraph"):
            check_dygraph = self.check_dygraph
        self.check_output_with_place(self.place, check_dygraph=check_dygraph)

    @check_soc_version
    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        check_dygraph = False
        if hasattr(self, "check_dygraph"):
            check_dygraph = self.check_dygraph
        self.check_grad_with_place(
            self.place, ["X"], "Out", check_dygraph=check_dygraph
        )

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [11, 17]

    def init_kernel_type(self):
        pass


class TestSquareBf16(TestActivationBf16):
    def setUp(self):
        self.set_npu()
        self.op_type = "square"
        self.python_api = paddle.square
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = convert_float_to_uint16(
            np.random.uniform(0.1, 1, self.shape).astype(np.float32)
        )
        out = np.square(convert_uint16_to_float(x))

        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {"Out": out}

    @check_soc_version
    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X"], "Out", max_relative_error=0.007, check_dygraph=True
        )

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=True)


class TestSqrt(TestActivationBf16):
    def setUp(self):
        self.set_npu()
        self.op_type = "sqrt"
        self.python_api = paddle.sqrt
        self.init_dtype()
        self.init_shape()

        np.random.seed(1023)
        x = convert_float_to_uint16(
            np.random.uniform(0.1, 1, self.shape).astype(np.float32)
        )
        out = np.sqrt(convert_uint16_to_float(x))

        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {"Out": out}

    @check_soc_version
    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out", check_dygraph=True)

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=True)


class TestSinBF16(OpTest):
    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def setUp(self):
        self.set_npu()
        self.op_type = "sin"
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = np.sin(convert_uint16_to_float(convert_float_to_uint16(x)))

        self.inputs = {"X": convert_float_to_uint16(OpTest.np_dtype_to_base_dtype(x))}
        self.outputs = {"Out": out}

    def init_shape(self):
        self.shape = [10, 12]

    def init_dtype(self):
        self.dtype = np.float32

    def init_kernel_type(self):
        pass

    @unittest.skip("sin_grad not implemented on NPU yet")
    def test_check_grad(self):
        pass

    @check_soc_version
    def test_check_output(self):
        check_dygraph = False
        if hasattr(self, "check_dygraph"):
            check_dygraph = self.check_dygraph
        self.check_output_with_place(self.place, check_dygraph=check_dygraph)


class TestSiluBF16(OpTest):
    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def setUp(self):
        self.set_npu()
        self.op_type = "silu"
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = x / (np.exp(-convert_uint16_to_float(convert_float_to_uint16(x))) + 1)
        self.inputs = {"X": convert_float_to_uint16(OpTest.np_dtype_to_base_dtype(x))}
        self.outputs = {"Out": out}

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [11, 17]

    def init_kernel_type(self):
        pass

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place)

    @check_soc_version
    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
        )


class TestCos(OpTest):
    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def setUp(self):
        self.set_npu()
        self.op_type = "cos"
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = np.cos(convert_uint16_to_float(convert_float_to_uint16(x)))

        self.inputs = {"X": convert_float_to_uint16(OpTest.np_dtype_to_base_dtype(x))}
        self.outputs = {"Out": out}

    def init_shape(self):
        self.shape = [10, 12]

    def init_dtype(self):
        self.dtype = np.float32

    @check_soc_version
    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(self.place, ["X"], "Out", max_relative_error=0.01)


class TestPow(OpTest):
    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def setUp(self):
        self.set_npu()
        self.op_type = "pow"
        self.python_api = paddle.pow
        self.check_dygraph = True
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(1, 2, self.shape).astype(self.dtype)
        out = np.power(convert_uint16_to_float(convert_float_to_uint16(x)), 3)

        self.inputs = {"X": convert_float_to_uint16(OpTest.np_dtype_to_base_dtype(x))}
        self.attrs = {"factor": 3.0}
        self.outputs = {"Out": out}

    def init_shape(self):
        self.shape = []

    def init_dtype(self):
        self.dtype = np.float32

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=self.check_dygraph)

    @check_soc_version
    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(
            self.place, ["X"], "Out", check_dygraph=self.check_dygraph
        )


if __name__ == "__main__":
    unittest.main()
