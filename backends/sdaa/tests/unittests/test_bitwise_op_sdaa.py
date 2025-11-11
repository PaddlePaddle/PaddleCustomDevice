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

import unittest

import numpy as np
from op_test import OpTest

import paddle

paddle.enable_static()


# ----------------- TEST OP: BitwiseOr ------------------ #
class TestBitwiseOr(OpTest):
    def setUp(self):
        self.op_type = "bitwise_or"
        self.python_api = paddle.tensor.logic.bitwise_or
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
        self.place = paddle.CustomPlace("sdaa", 0)

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


class TestBitwiseOr_ZeroDim3(TestBitwiseOr):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = []


class TestBitwiseOrUInt8(TestBitwiseOr):
    def init_dtype(self):
        self.dtype = np.uint8

    def init_bound(self):
        self.low = 0
        self.high = 100


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
        self.python_api = paddle.tensor.logic.bitwise_or
        self.init_shape()
        self.init_place()

        x = np.random.choice([True, False], self.x_shape)
        y = np.random.choice([True, False], self.y_shape)
        out = np.bitwise_or(x, y)

        self.inputs = {"X": x, "Y": y}
        self.outputs = {"Out": out}


class TestBitwiseOrBoolCase1(TestBitwiseOrBool):
    def init_shape(self):
        self.x_shape = [1]
        self.y_shape = [1]


# ---------------  TEST OP: BitwiseNot ----------------- #
class TestBitwiseNot(OpTest):
    def setUp(self):
        self.op_type = "bitwise_not"
        self.python_api = paddle.tensor.logic.bitwise_not
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
        self.place = paddle.CustomPlace("sdaa", 0)

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


@unittest.skip("tecodnnBitwiseNotTensor calculation error when dtype is uint8")
class TestBitwiseNotUInt8(TestBitwiseNot):
    def init_dtype(self):
        self.dtype = np.uint8

    def init_bound(self):
        self.low = 0
        self.high = 100


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
        self.python_api = paddle.tensor.logic.bitwise_not
        self.init_shape()
        self.init_place()

        x = np.random.choice([True, False], self.x_shape)
        out = np.bitwise_not(x)

        self.inputs = {"X": x}
        self.outputs = {"Out": out}


class TestBitwiseNotBoolCase1(TestBitwiseNotBool):
    def init_shape(self):
        self.x_shape = [1]


# ----------------- TEST OP: BitwiseAnd ----------------- #
class TestBitwiseAnd(OpTest):
    def setUp(self):
        self.op_type = "bitwise_and"
        self.python_api = paddle.tensor.logic.bitwise_and
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
        self.place = paddle.CustomPlace("sdaa", 0)

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


class TestBitwiseAnd_ZeroDim3(TestBitwiseAnd):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = []


class TestBitwiseAndUInt8(TestBitwiseAnd):
    def init_dtype(self):
        self.dtype = np.uint8

    def init_bound(self):
        self.low = 0
        self.high = 100


class TestBitwiseAndInt16(TestBitwiseAnd):
    def init_dtype(self):
        self.dtype = np.int16

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [4, 1]


class TestBitwiseAndInt64(TestBitwiseAnd):
    def init_dtype(self):
        self.dtype = np.int64

    def init_shape(self):
        self.x_shape = [1, 4, 1]
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseAndBool(TestBitwiseAnd):
    def setUp(self):
        self.op_type = "bitwise_and"
        self.python_api = paddle.tensor.logic.bitwise_and
        self.init_shape()
        self.init_place()

        x = np.random.choice([True, False], self.x_shape)
        y = np.random.choice([True, False], self.y_shape)
        out = np.bitwise_and(x, y)

        self.inputs = {"X": x, "Y": y}
        self.outputs = {"Out": out}


# ----------------- TEST OP: BitwiseXor ---------------- #
class TestBitwiseXor(OpTest):
    def setUp(self):
        self.op_type = "bitwise_xor"
        self.python_api = paddle.tensor.logic.bitwise_xor
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
        self.place = paddle.CustomPlace("sdaa", 0)

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


class TestBitwiseXor_ZeroDim3(TestBitwiseXor):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = []


class TestBitwiseXorUInt8(TestBitwiseXor):
    def init_dtype(self):
        self.dtype = np.uint8

    def init_bound(self):
        self.low = 0
        self.high = 100


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
        self.python_api = paddle.tensor.logic.bitwise_xor
        self.init_shape()
        self.init_place()

        x = np.random.choice([True, False], self.x_shape)
        y = np.random.choice([True, False], self.y_shape)
        out = np.bitwise_xor(x, y)

        self.inputs = {"X": x, "Y": y}
        self.outputs = {"Out": out}


if __name__ == "__main__":
    unittest.main()
