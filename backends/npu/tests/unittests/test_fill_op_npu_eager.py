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

import numpy as np
import unittest
import os

select_npu = os.environ.get("FLAGS_selected_npus", 0)

from tests.op_test import convert_float_to_uint16, convert_uint16_to_float
import paddle
import paddle.base as base

from npu_utils import check_soc_version


class TestFill_API(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.shape = [9, 9]
        self.value = 6.0

    @check_soc_version
    def test_dygraph(self):
        paddle.disable_static(paddle.CustomPlace("npu", select_npu))
        with base.dygraph.guard():
            input_np = np.random.random(self.shape).astype(self.dtype)
            input = paddle.to_tensor(input_np)

            input_np.fill(self.value)
            input.fill_(self.value)
            print(input)

            self.assertTrue(
                (input == input_np).all(),
                msg="flii_ output is wrong, out =" + str(input_np),
            )


class TestFill_BF16API(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.shape = [9, 9]
        self.value = 6.0

    @check_soc_version
    def test_dygraph(self):
        paddle.disable_static(paddle.CustomPlace("npu", select_npu))
        with base.dygraph.guard():
            input_np = convert_float_to_uint16(
                np.random.random(self.shape).astype(self.dtype)
            )
            input = paddle.to_tensor(input_np)

            input_np = convert_uint16_to_float(input_np)
            input_np.fill(self.value)
            input.fill_(self.value)

            self.assertTrue(
                (convert_uint16_to_float(input) == input_np).all(),
                msg="flii_ output is wrong, out =" + str(input_np),
            )


class TestFill_FP16API(TestFill_API):
    def setUp(self):
        self.dtype = "float16"
        self.shape = [9, 9]
        self.value = 6.0


class TestFill_FLOAT64API(TestFill_API):
    def setUp(self):
        self.dtype = "float64"
        self.shape = [9, 9]
        self.value = 6.0


class TestFill_IntAPI(TestFill_API):
    def setUp(self):
        self.dtype = "int"
        self.shape = [9, 9]
        self.value = 6


class TestFill_Int32API(TestFill_API):
    def setUp(self):
        self.dtype = "int32"
        self.shape = [9, 9]
        self.value = 6


class TestFill_BoolAPI(TestFill_API):
    def setUp(self):
        self.dtype = "bool"
        self.shape = [9, 9]
        self.value = False


if __name__ == "__main__":
    unittest.main()
