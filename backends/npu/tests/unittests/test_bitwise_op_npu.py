#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from npu_utils import check_run_big_shape_test

paddle.enable_static()


# ----------------- TEST OP: BitwiseAnd ----------------- #
class TestBitwiseAnd(OpTest):
    def setUp(self):
        self.op_type = "bitwise_and"
        self.init_dtype()
        self.init_shape()
        self.init_bound()
        self.init_place()

        x = np.random.randint(self.low, self.high, self.x_shape, dtype=self.dtype)
        y = np.random.randint(self.low, self.high, self.y_shape, dtype=self.dtype)
        out = np.bitwise_and(x, y)

        self.inputs = {"X": x, "Y": y}
        self.outputs = {"Out": out}

    def init_place(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass

    def init_dtype(self):
        self.dtype = np.int32

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [2, 3, 4, 5]

    def init_bound(self):
        self.low = -100
        self.high = 100


class TestBitwiseAnd_ZeroDim1(TestBitwiseAnd):
    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = []


class TestBitwiseAnd_ZeroDim2(TestBitwiseAnd):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseAndUInt8(TestBitwiseAnd):
    def init_dtype(self):
        self.dtype = np.uint8

    def init_bound(self):
        self.low = 0
        self.high = 100


class TestBitwiseAndInt8(TestBitwiseAnd):
    def init_dtype(self):
        self.dtype = np.int8

    def init_shape(self):
        self.x_shape = [4, 5]
        self.y_shape = [2, 3, 4, 5]


@check_run_big_shape_test()
class TestBitwiseAndInt8_1(TestBitwiseAndInt8):
    def init_shape(self):
        self.x_shape = [2, 1, 4096, 4096]
        self.y_shape = [2, 1, 4096, 4096]


class TestBitwiseAndInt16(TestBitwiseAnd):
    def init_dtype(self):
        self.dtype = np.int16

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [4, 1]


@check_run_big_shape_test()
class TestBitwiseAndInt16_1(TestBitwiseAndInt16):
    def init_shape(self):
        self.x_shape = [2, 1, 4096, 4096]
        self.y_shape = [2, 1, 4096, 4096]


class TestBitwiseAndInt64(TestBitwiseAnd):
    def init_dtype(self):
        self.dtype = np.int64

    def init_shape(self):
        self.x_shape = [1, 4, 1]
        self.y_shape = [2, 3, 4, 5]


@check_run_big_shape_test()
class TestBitwiseAndInt64_1(TestBitwiseAndInt64):
    def init_shape(self):
        self.x_shape = [2, 1, 4096, 4096]
        self.y_shape = [2, 1, 4096, 4096]


class TestBitwiseAndBool(TestBitwiseAnd):
    def setUp(self):
        self.op_type = "bitwise_and"
        self.init_shape()
        self.init_place()

        x = np.random.choice([True, False], self.x_shape)
        y = np.random.choice([True, False], self.y_shape)
        out = np.bitwise_and(x, y)

        self.inputs = {"X": x, "Y": y}
        self.outputs = {"Out": out}


@check_run_big_shape_test()
class TestBitwiseAndBool_1(TestBitwiseAndBool):
    def init_shape(self):
        self.x_shape = [2, 1, 4096, 4096]
        self.y_shape = [2, 1, 4096, 4096]


# ----------------- TEST OP: BitwiseOr ------------------ #
class TestBitwiseOr(OpTest):
    def setUp(self):
        self.op_type = "bitwise_or"
        self.init_dtype()
        self.init_shape()
        self.init_bound()
        self.init_place()

        x = np.random.randint(self.low, self.high, self.x_shape, dtype=self.dtype)
        y = np.random.randint(self.low, self.high, self.y_shape, dtype=self.dtype)
        out = np.bitwise_or(x, y)

        self.inputs = {"X": x, "Y": y}
        self.outputs = {"Out": out}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def init_place(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def test_check_grad(self):
        pass

    def init_dtype(self):
        self.dtype = np.int32

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [2, 3, 4, 5]

    def init_bound(self):
        self.low = -100
        self.high = 100


class TestBitwiseOr_ZeroDim1(TestBitwiseOr):
    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = []


class TestBitwiseOr_ZeroDim2(TestBitwiseOr):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseOrUInt8(TestBitwiseOr):
    def init_dtype(self):
        self.dtype = np.uint8

    def init_bound(self):
        self.low = 0
        self.high = 100


class TestBitwiseOrInt8(TestBitwiseOr):
    def init_dtype(self):
        self.dtype = np.int8

    def init_shape(self):
        self.x_shape = [4, 5]
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseOrInt16(TestBitwiseOr):
    def init_dtype(self):
        self.dtype = np.int16

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [4, 1]


class TestBitwiseOrInt64(TestBitwiseOr):
    def init_dtype(self):
        self.dtype = np.int64

    def init_shape(self):
        self.x_shape = [1, 4, 1]
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseOrBool(TestBitwiseOr):
    def setUp(self):
        self.op_type = "bitwise_or"
        self.init_shape()
        self.init_place()

        x = np.random.choice([True, False], self.x_shape)
        y = np.random.choice([True, False], self.y_shape)
        out = np.bitwise_or(x, y)

        self.inputs = {"X": x, "Y": y}
        self.outputs = {"Out": out}


# ----------------- TEST OP: BitwiseXor ---------------- #
class TestBitwiseXor(OpTest):
    def setUp(self):
        self.op_type = "bitwise_xor"
        self.init_dtype()
        self.init_shape()
        self.init_bound()
        self.init_place()

        x = np.random.randint(self.low, self.high, self.x_shape, dtype=self.dtype)
        y = np.random.randint(self.low, self.high, self.y_shape, dtype=self.dtype)
        out = np.bitwise_xor(x, y)

        self.inputs = {"X": x, "Y": y}
        self.outputs = {"Out": out}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass

    def init_place(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def init_dtype(self):
        self.dtype = np.int32

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [2, 3, 4, 5]

    def init_bound(self):
        self.low = -100
        self.high = 100


class TestBitwiseXor_ZeroDim1(TestBitwiseXor):
    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = []


class TestBitwiseXor_ZeroDim2(TestBitwiseXor):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseXorUInt8(TestBitwiseXor):
    def init_dtype(self):
        self.dtype = np.uint8

    def init_bound(self):
        self.low = 0
        self.high = 100


class TestBitwiseXorInt8(TestBitwiseXor):
    def init_dtype(self):
        self.dtype = np.int8

    def init_shape(self):
        self.x_shape = [4, 5]
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseXorInt16(TestBitwiseXor):
    def init_dtype(self):
        self.dtype = np.int16

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [4, 1]


class TestBitwiseXorInt64(TestBitwiseXor):
    def init_dtype(self):
        self.dtype = np.int64

    def init_shape(self):
        self.x_shape = [1, 4, 1]
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseXorBool(TestBitwiseXor):
    def setUp(self):
        self.op_type = "bitwise_xor"
        self.init_shape()
        self.init_place()

        x = np.random.choice([True, False], self.x_shape)
        y = np.random.choice([True, False], self.y_shape)
        out = np.bitwise_xor(x, y)

        self.inputs = {"X": x, "Y": y}
        self.outputs = {"Out": out}


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
        self.place = paddle.CustomPlace("npu", 0)

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


@check_run_big_shape_test()
class TestBitwiseNotUInt8_1(TestBitwiseNotUInt8):
    def init_shape(self):
        self.x_shape = [4096, 4096]


class TestBitwiseNotInt8(TestBitwiseNot):
    def init_dtype(self):
        self.dtype = np.int8

    def init_shape(self):
        self.x_shape = [4, 5]


@check_run_big_shape_test()
class TestBitwiseNotInt8_1(TestBitwiseNotInt8):
    def init_shape(self):
        self.x_shape = [4096, 4096]


class TestBitwiseNotInt16(TestBitwiseNot):
    def init_dtype(self):
        self.dtype = np.int16

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]


@check_run_big_shape_test()
class TestBitwiseNotInt16_1(TestBitwiseNotInt16):
    def init_shape(self):
        self.x_shape = [4096, 4096]


class TestBitwiseNotInt64(TestBitwiseNot):
    def init_dtype(self):
        self.dtype = np.int64

    def init_shape(self):
        self.x_shape = [1, 4, 1]


@check_run_big_shape_test()
class TestBitwiseNotInt64_1(TestBitwiseNotInt64):
    def init_shape(self):
        self.x_shape = [4096, 4096]


class TestBitwiseNotBool(TestBitwiseNot):
    def setUp(self):
        self.op_type = "bitwise_not"
        self.init_shape()
        self.init_place()

        x = np.random.choice([True, False], self.x_shape)
        out = np.bitwise_not(x)

        self.inputs = {"X": x}
        self.outputs = {"Out": out}


@check_run_big_shape_test()
class TestBitwiseNotBool_1(TestBitwiseNotBool):
    def init_shape(self):
        self.x_shape = [4096, 4096]


if __name__ == "__main__":
    unittest.main()
