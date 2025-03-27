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

from tests.op_test import convert_float_to_uint16, convert_uint16_to_float
import paddle
import paddle.base as base

import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)


class TestIndexPut(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.shape = [9, 9]
        self.value = [6.0, 6.0]

    def test_dygraph(self):
        index = [paddle.to_tensor([1, 1]), paddle.to_tensor([3, 4])]
        index_np = np.array([[1, 1], [3, 4]])

        with base.dygraph.guard():
            input_np = np.random.random(self.shape).astype(self.dtype)
            input = paddle.to_tensor(input_np)

            input_np[index_np[0], index_np[1]] = self.value
            output = paddle.index_put(
                input, index, paddle.to_tensor(self.value, dtype=self.dtype)
            )

            self.assertTrue(
                (output == input_np).all(),
                msg="flii_ output is wrong, out =" + str(input_np),
            )


class TestIndexPut_BF16API(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.shape = [9, 9]
        self.value = [6.0, 6.0]

    def test_dygraph(self):
        index = [paddle.to_tensor([1, 1]), paddle.to_tensor([3, 4])]
        index_np = np.array([[1, 1], [3, 4]])

        paddle.disable_static(
            paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))
        )
        with base.dygraph.guard():
            input_np = convert_float_to_uint16(
                np.random.random(self.shape).astype(self.dtype)
            )
            input = paddle.to_tensor(input_np)

            input_np = convert_uint16_to_float(input_np)
            input_np[index_np[0], index_np[1]] = self.value
            output = paddle.index_put(
                input, index, paddle.to_tensor(self.value, dtype="bfloat16")
            )

            # print(output)
            # print(input_np)
            self.assertTrue(
                (output == input_np).all(),
                msg="flii_ output is wrong, out =" + str(input_np),
            )


class TestIndexPut_FP16API(TestIndexPut):
    def setUp(self):
        self.dtype = "float16"
        self.shape = [9, 9]
        self.value = [6.0, 6.0]


# class TestIndexPut_FLOAT64API(TestIndexPut):
#     def setUp(self):
#         self.dtype = "float64"
#         self.shape = [9, 9]
#         self.value = 6.0


class TestIndexPut_IntAPI(TestIndexPut):
    def setUp(self):
        self.dtype = "int16"
        self.shape = [9, 9]
        self.value = [6, 6]


class TestIndexPut_Int32API(TestIndexPut):
    def setUp(self):
        self.dtype = "int32"
        self.shape = [9, 9]
        self.value = [6, 6]


class TestIndexPut_BoolAPI(TestIndexPut):
    def setUp(self):
        self.dtype = "bool"
        self.shape = [9, 9]
        self.value = [False, False]


if __name__ == "__main__":
    unittest.main()
