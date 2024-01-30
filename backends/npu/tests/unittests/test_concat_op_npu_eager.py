#  Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np
import paddle

from npu_utils import check_soc_version
from tests.op_test import (
    OpTest,
    convert_float_to_uint16,
    convert_uint16_to_float,
    skip_check_grad_ci,
)

SEED = 2021


# ----------------Concat bf16----------------
class TestConcatOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "concat"
        self.place = paddle.CustomPlace("npu", 0)
        self.init_test_data()

        self.inputs = {"X": [("x0", self.x0), ("x1", self.x1), ("x2", self.x2)]}
        self.attrs = {"axis": self.axis}
        if self.axis < 0:
            self.actual_axis = self.axis + len(self.x0.shape)
            self.actual_axis = self.actual_axis if self.actual_axis > 0 else 0
        else:
            self.actual_axis = self.axis

        cpu_x0 = convert_uint16_to_float(self.x0)
        cpu_x1 = convert_uint16_to_float(self.x1)
        cpu_x2 = convert_uint16_to_float(self.x2)
        self.outputs = {
            "Out": np.concatenate((cpu_x0, cpu_x1, cpu_x2), axis=self.actual_axis)
        }

    def set_npu(self):
        self.__class__.use_custom_device = True

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place)

    @check_soc_version
    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["x0", "x2"], "Out", numeric_place=paddle.CPUPlace()
        )
        self.check_grad_with_place(
            self.place, ["x1"], "Out", numeric_place=paddle.CPUPlace()
        )
        self.check_grad_with_place(
            self.place, ["x2"], "Out", numeric_place=paddle.CPUPlace()
        )

    def init_test_data(self):
        self.x0 = convert_float_to_uint16(
            np.random.random((1, 4, 50)).astype(np.float32)
        )
        self.x1 = convert_float_to_uint16(
            np.random.random((2, 4, 50)).astype(np.float32)
        )
        self.x2 = convert_float_to_uint16(
            np.random.random((3, 4, 50)).astype(np.float32)
        )
        self.axis = 0


class TestConcatOp2(TestConcatOp):
    def init_test_data(self):
        self.x0 = convert_float_to_uint16(
            np.random.random((2, 3, 4, 5)).astype(np.float32)
        )
        self.x1 = convert_float_to_uint16(
            np.random.random((2, 3, 4, 5)).astype(np.float32)
        )
        self.x2 = convert_float_to_uint16(
            np.random.random((2, 3, 4, 5)).astype(np.float32)
        )
        self.axis = 1


@skip_check_grad_ci(reason="The function 'check_grad' for large inputs is too slow.")
class TestConcatOp3(TestConcatOp):
    def init_test_data(self):
        self.x0 = convert_float_to_uint16(
            np.random.random((1, 256, 170, 256)).astype(np.float32)
        )
        self.x1 = convert_float_to_uint16(
            np.random.random((1, 256, 170, 256)).astype(np.float32)
        )
        self.x2 = convert_float_to_uint16(
            np.random.random((1, 256, 170, 256)).astype(np.float32)
        )
        self.axis = 1

    def test_check_grad(self):
        pass


class TestConcatOp4(TestConcatOp):
    def init_test_data(self):
        self.x0 = convert_float_to_uint16(
            np.random.random((5, 1, 4, 5)).astype(np.float32)
        )
        self.x1 = convert_float_to_uint16(
            np.random.random((5, 2, 4, 5)).astype(np.float32)
        )
        self.x2 = convert_float_to_uint16(
            np.random.random((5, 3, 4, 5)).astype(np.float32)
        )
        self.axis = -3


if __name__ == "__main__":
    unittest.main()
