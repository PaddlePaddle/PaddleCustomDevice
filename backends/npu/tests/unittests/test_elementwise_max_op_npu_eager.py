#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from tests.op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float
from npu_utils import check_soc_version

SEED = 2021


class TestElementwiseMasOpBF16(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_max"
        self.place = paddle.CustomPlace("npu", 0)

        self.init_dtype()
        self.init_input_output()
        self.init_axis()

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(self.x),
            "Y": OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.attrs = {"axis": self.axis}
        self.outputs = {"Out": self.out}

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.uint16
        self.np_dtype = np.float32

    def init_input_output(self):
        np_x = np.random.uniform(0.1, 1, [13, 17]).astype(self.np_dtype)
        sgn = np.random.choice([-1, 1], [13, 17]).astype(self.np_dtype)
        np_y = np_x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype(self.np_dtype)
        self.x = convert_float_to_uint16(np_x)
        self.y = convert_float_to_uint16(np_y)
        self.out = np.maximum(
            convert_uint16_to_float(np_x), convert_uint16_to_float(np_y)
        )

    def init_axis(self):
        self.axis = -1

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place, atol=0.004)

    @check_soc_version
    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ["X", "Y"], "Out")

    @check_soc_version
    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(self.place, ["Y"], "Out", no_grad_set=set("X"))

    @check_soc_version
    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(self.place, ["X"], "Out", no_grad_set=set("Y"))


if __name__ == "__main__":
    unittest.main()
