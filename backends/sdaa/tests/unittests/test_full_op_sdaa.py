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
from paddle.base.framework import convert_np_dtype_to_dtype_

paddle.enable_static()
SEED = 2021


def fill_wrapper(shape, value=0.0):
    out = paddle.full(shape=shape, fill_value=value)
    return out


class TestFillConstant(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "fill_constant"
        self.python_api = fill_wrapper
        self.dtype = np.float32
        self.input = 9.0
        self.init_case()

        self.inputs = {}
        self.attrs = {
            "shape": [123, 92],
            "value": self.input,
            "dtype": convert_np_dtype_to_dtype_(self.dtype),
        }
        self.outputs = {"Out": np.full((123, 92), self.input)}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_case(self):
        self.dtype = np.float32
        self.input = 9.0

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestFillConstantInt8(TestFillConstant):
    def init_case(self):
        self.dtype = np.int8
        self.input = 4


class TestFillConstantUint8(TestFillConstant):
    def init_case(self):
        self.dtype = np.uint8
        self.input = 4


class TestFillConstantInt16(TestFillConstant):
    def init_case(self):
        self.dtype = np.int16
        self.input = 8


class TestFillConstantInt32(TestFillConstant):
    def init_case(self):
        self.dtype = np.int32
        self.input = 16


class TestFillConstantInt64(TestFillConstant):
    def init_case(self):
        self.dtype = np.int64
        self.input = 32


class TestFillConstantFP16(TestFillConstant):
    def init_case(self):
        self.dtype = np.float16
        self.input = 0.1


class TestFillConstantFP32(TestFillConstant):
    def init_case(self):
        self.dtype = np.float32
        self.input = 0.99


class TestFillConstantFP64(TestFillConstant):
    def init_case(self):
        self.dtype = np.float64
        self.input = 0.99


class TestFillConstantBool(TestFillConstant):
    def init_case(self):
        self.dtype = bool
        self.input = True


class TestFillConstantInf(TestFillConstant):
    def init_case(self):
        self.dtype = np.float64
        self.input = np.inf


class TestFillConstantWithPlaceType(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "fill_constant"
        self.python_api = fill_wrapper

        self.init_dtype()

        self.inputs = {}
        self.attrs = {"shape": [123, 92], "value": 1.8, "place_type": 0}
        self.outputs = {"Out": np.full((123, 92), 1.8)}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestFillConstantWithPlaceType(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "fill_constant"
        self.python_api = fill_wrapper

        self.init_dtype()

        self.inputs = {}
        self.attrs = {"shape": [123, 92], "value": 1.8, "place_type": 0}
        self.outputs = {"Out": np.full((123, 92), 1.8)}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == "__main__":
    unittest.main()
