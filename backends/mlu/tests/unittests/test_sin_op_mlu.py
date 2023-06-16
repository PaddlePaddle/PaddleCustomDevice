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

from __future__ import print_function

import unittest
import numpy as np

from tests.op_test import OpTest
import paddle

paddle.enable_static()


class TestSin(OpTest):
    def set_mlu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("mlu", 0)

    def setUp(self):
        self.set_mlu()
        self.op_type = "sin"
        self.init_dtype()
        self.init_shape()
        self.check_dygraph = True

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = np.sin(x)

        self.inputs = {"X": OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {"Out": out}

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_output(self):
        check_dygraph = False
        if hasattr(self, "check_dygraph"):
            check_dygraph = self.check_dygraph
        self.check_output_with_place(
            self.place, check_dygraph=check_dygraph, rtol=1e-05, atol=1e-5
        )

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")

    def init_dtype(self):
        self.dtype = np.float32


class TestSin_ZeroDim(TestSin):
    def init_shape(self):
        self.shape = []


class TestSinHalf(OpTest):
    def setUp(self):
        self.op_type = "sin"
        self.dtype = "float16"
        self.set_mlu()
        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = np.sin(x)

        self.inputs = {"X": OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {"Out": out}

    def set_mlu(self):
        self.place = paddle.CustomPlace("mlu", 0)
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


if __name__ == "__main__":
    unittest.main()
