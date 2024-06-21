# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest

from op_test import OpTest
import paddle
import paddle.framework.dtype as dtypes

paddle.enable_static()


def fill_any_like_wrapper(x, value, out_dtype=None, name=None):
    if isinstance(out_dtype, int):
        tmp_dtype = dtypes.dtype(out_dtype)
    else:
        tmp_dtype = out_dtype
    return paddle.full_like(x, value, tmp_dtype, name)


class TestFillAnyOp(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "fill_any_like"
        self.python_api = fill_any_like_wrapper
        self.dtype = np.float32
        self.shape = [2, 3, 4, 5]
        self.value = 0.0

        self.init()

        self.inputs = {"X": np.random.random(self.shape).astype(self.dtype)}
        self.attrs = {"value": self.value}
        self.outputs = {"Out": np.full(self.shape, self.value, self.dtype)}

    def init(self):
        pass

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestFillAnyOpInt8(TestFillAnyOp):
    def init(self):
        self.dtype = np.int8
        self.value = 8


class TestFillAnyOpUint8(TestFillAnyOp):
    def init(self):
        self.dtype = np.uint8
        self.value = 8


class TestFillAnyOpInt16(TestFillAnyOp):
    def init(self):
        self.dtype = np.int16
        self.value = 16


class TestFillAnyOpInt32(TestFillAnyOp):
    def init(self):
        self.dtype = np.int32
        self.value = 32


class TestFillAnyOpInt64(TestFillAnyOp):
    def init(self):
        self.dtype = np.int64
        self.value = 64


class TestFillAnyOpFloat16(TestFillAnyOp):
    def init(self):
        self.dtype = np.float16
        self.value = 0.12

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestFillAnyOpFloat32(TestFillAnyOp):
    def init(self):
        self.dtype = np.float32
        self.value = 0.32


class TestFillAnyOpFloat64(TestFillAnyOp):
    def init(self):
        self.dtype = np.float64
        self.value = 0.32


class TestFillAnyOpBool(TestFillAnyOp):
    def init(self):
        self.dtype = bool
        self.value = True


class TestFillAnyOpValue1(TestFillAnyOp):
    def init(self):
        self.value = 6.66


class TestFillAnyOpValue2(TestFillAnyOp):
    def init(self):
        self.value = 1e-9


class TestFillAnyOpShape(TestFillAnyOp):
    def init(self):
        self.shape = [12, 10]


if __name__ == "__main__":
    unittest.main()
