# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle

from tests.op_test import OpTest

paddle.enable_static()


class TestTransposeOp(OpTest):
    def setUp(self):
        self.set_gcu()
        self.op_type = "transpose2"
        self.place = paddle.CustomPlace("gcu", 0)
        self.init_dtype()
        self.init_shape_axis()

        self.inputs = {"X": np.random.random(self.shape).astype(self.dtype)}
        self.attrs = {"axis": self.axis, "data_format": "AnyLayout"}
        self.outputs = {"Out": self.inputs["X"].transpose(self.axis)}

    def set_gcu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape_axis(self):
        self.shape = (3, 40)
        self.axis = (1, 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestCase2D(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (100, 200)
        self.axis = (1, 0)


class TestCase3D1(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (3, 4, 10)
        self.axis = (0, 2, 1)


class TestCase3D2(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 16, 96)
        self.axis = (0, 2, 1)


class TestCase4D1(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 4, 5)
        self.axis = (0, 2, 3, 1)


class TestCase4D2(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 10, 12, 16)
        self.axis = (3, 1, 2, 0)


class TestCase4D3(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 10, 2, 16)
        self.axis = (0, 1, 3, 2)


class TestCase5D(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 4, 5, 6)
        self.axis = (4, 2, 3, 1, 0)


class TestTransposeOpFP16(TestTransposeOp):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad(self):
        pass


class TestTransposeOpInt64(TestTransposeOp):
    def init_dtype(self):
        self.dtype = np.int64

    def test_check_grad(self):
        pass


if __name__ == "__main__":
    unittest.main()
