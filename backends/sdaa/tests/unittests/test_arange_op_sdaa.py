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


def arange_wrapper(start, end, step, dtype="float32"):
    return paddle.arange(start, end, step, dtype)


class TestRangeOp(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "range"
        self.python_api = arange_wrapper
        self.init_config()
        self.inputs = {
            "Start": np.array([self.case[0]]).astype(self.dtype),
            "End": np.array([self.case[1]]).astype(self.dtype),
            "Step": np.array([self.case[2]]).astype(self.dtype),
        }

        self.outputs = {
            "Out": np.arange(self.case[0], self.case[1], self.case[2]).astype(
                self.dtype
            )
        }

    def init_config(self):
        self.dtype = np.float32
        self.case = (0, 1, 0.2)

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestFloatRangeOpCase0(TestRangeOp):
    def init_config(self):
        self.dtype = np.float32
        self.case = (0, 5, 1)


class TestInt32RangeOpCase0(TestRangeOp):
    def init_config(self):
        self.dtype = np.int32
        self.case = (0, 5, 2)


class TestInt32RangeOpCase1(TestRangeOp):
    def init_config(self):
        self.dtype = np.int32
        self.case = (10, 1, -2)


class TestInt32RangeOpCase2(TestRangeOp):
    def init_config(self):
        self.dtype = np.int32
        self.case = (-1, -10, -2)


class TestInt64RangeOpCase0(TestRangeOp):
    def init_config(self):
        self.dtype = np.int64
        self.case = (0, 5, 2)


class TestInt64RangeOpCase1(TestRangeOp):
    def init_config(self):
        self.dtype = np.int64
        self.case = (10, 1, -2)


class TestInt64RangeOpCase2(TestRangeOp):
    def init_config(self):
        self.dtype = np.int64
        self.case = (-1, -10, -2)


if __name__ == "__main__":
    unittest.main()
