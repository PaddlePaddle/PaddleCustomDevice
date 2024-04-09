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
from tests.op_test import (
    OpTest,
    convert_float_to_uint16,
    convert_uint16_to_float,
)

SEED = 2021


class TestElementwiseSubOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_sub"
        self.place = paddle.CustomPlace("npu", 0)
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(self.x),
            "Y": OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.attrs = {"axis": self.axis, "use_mkldnn": self.use_mkldnn}
        self.outputs = {"Out": self.out}

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_kernel_type(self):
        self.use_mkldnn = False

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.subtract(self.x, self.y)

    def init_dtype(self):
        self.dtype = np.int32

    def init_axis(self):
        self.axis = 0

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestElementwiseSubOpInt64(TestElementwiseSubOp):
    def init_dtype(self):
        self.dtype = np.int64


class TestElementwiseSubOpFloat32(TestElementwiseSubOp):
    def init_dtype(self):
        self.dtype = np.float32

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place,
            ["X", "Y"],
            "Out",
        )

    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(
            self.place,
            ["Y"],
            "Out",
            no_grad_set=set("X"),
        )

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            no_grad_set=set("Y"),
        )


class TestElementwiseSubOpFloat16(TestElementwiseSubOpFloat32):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseSubOpBfloat16(TestElementwiseSubOpFloat32):
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_sub"
        self.place = paddle.CustomPlace("npu", 0)
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()

        x = convert_float_to_uint16(self.x)
        y = convert_float_to_uint16(self.y)
        self.inputs = {
            "X": x,
            "Y": y,
        }
        self.attrs = {"axis": self.axis, "use_mkldnn": self.use_mkldnn}
        self.outputs = {"Out": convert_uint16_to_float(self.out)}


if __name__ == "__main__":
    unittest.main()
