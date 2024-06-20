# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from op_test import OpTest
import paddle
import paddle.base as base
import paddle.base.framework as framework

paddle.enable_static()
SEED = 2021


class TestAssignValueOp(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)

        self.op_type = "assign_value"
        self.inputs = {}
        self.attrs = {}
        self.init_data()

        self.attrs["shape"] = self.value.shape
        self.attrs["dtype"] = framework.convert_np_dtype_to_dtype_(self.value.dtype)
        self.outputs = {"Out": self.value}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_data(self):
        self.value = np.random.random(size=(2, 5)).astype(np.float32)
        self.attrs["fp32_values"] = [float(v) for v in self.value.flat]

    def test_forward(self):
        self.check_output_with_place(self.place)


class TestAssignValueOp2(TestAssignValueOp):
    def init_data(self):
        self.value = np.random.random(size=(2, 5)).astype(np.int32)
        self.attrs["int32_values"] = [int(v) for v in self.value.flat]


class TestAssignValueOp3(TestAssignValueOp):
    def init_data(self):
        self.value = np.random.random(size=(2, 5)).astype(np.int64)
        self.attrs["int64_values"] = [int(v) for v in self.value.flat]


class TestAssignValueOp4(TestAssignValueOp):
    def init_data(self):
        self.value = np.random.choice(a=[False, True], size=(2, 5)).astype(np.bool_)
        self.attrs["bool_values"] = [int(v) for v in self.value.flat]


class TestAssignApi(unittest.TestCase):
    def setUp(self):
        self.init_dtype()
        self.value = (-100 + 200 * np.random.random(size=(2, 5))).astype(self.dtype)
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_dtype(self):
        self.dtype = "float32"

    def test_assign(self):
        main_program = base.Program()
        with base.program_guard(main_program):
            x = paddle.tensor.create_tensor(dtype=self.dtype)
            paddle.assign(self.value, output=x)

        exe = base.Executor(self.place)
        [fetched_x] = exe.run(main_program, feed={}, fetch_list=[x])
        self.assertTrue(
            np.array_equal(fetched_x, self.value),
            "fetch_x=%s val=%s" % (fetched_x, self.value),
        )
        self.assertEqual(fetched_x.dtype, self.value.dtype)


class TestAssignApi2(TestAssignApi):
    def init_dtype(self):
        self.dtype = "int32"


class TestAssignApi3(TestAssignApi):
    def init_dtype(self):
        self.dtype = "int64"


class TestAssignApi4(TestAssignApi):
    def setUp(self):
        self.init_dtype()
        self.value = np.random.choice(a=[False, True], size=(2, 5)).astype(np.bool_)
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_dtype(self):
        self.dtype = "bool"


if __name__ == "__main__":
    unittest.main()
