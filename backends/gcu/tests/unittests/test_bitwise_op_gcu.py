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


# ---------------  TEST OP: BitwiseNot ----------------- #
class TestBitwiseNot(OpTest):
    def setUp(self):
        self.op_type = "bitwise_not"
        self.init_dtype()
        self.init_shape()
        self.init_bound()
        self.init_place()

        x = np.random.randint(self.low, self.high, self.x_shape, dtype=self.dtype)
        out = np.bitwise_not(x)

        self.inputs = {"X": x}
        self.outputs = {"Out": out}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass

    def init_place(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def init_dtype(self):
        self.dtype = np.int32

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]

    def init_bound(self):
        self.low = -100
        self.high = 100


class TestBitwiseNot_ZeroDim(TestBitwiseNot):
    def init_shape(self):
        self.x_shape = []


class TestBitwiseNotUInt8(TestBitwiseNot):
    def init_dtype(self):
        self.dtype = np.uint8

    def init_bound(self):
        self.low = 0
        self.high = 100


class TestBitwiseNotInt8(TestBitwiseNot):
    def init_dtype(self):
        self.dtype = np.int8

    def init_shape(self):
        self.x_shape = [4, 5]


class TestBitwiseNotInt16(TestBitwiseNot):
    def init_dtype(self):
        self.dtype = np.int16

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]


class TestBitwiseNotInt64(TestBitwiseNot):
    def init_dtype(self):
        self.dtype = np.int64

    def init_shape(self):
        self.x_shape = [1, 4, 1]


class TestBitwiseNotBool(TestBitwiseNot):
    def setUp(self):
        self.op_type = "bitwise_not"
        self.init_shape()
        self.init_place()

        x = np.random.choice([True, False], self.x_shape)
        out = np.bitwise_not(x)

        self.inputs = {"X": x}
        self.outputs = {"Out": out}


if __name__ == "__main__":
    unittest.main()
