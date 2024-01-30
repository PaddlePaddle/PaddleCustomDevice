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
)


class ElementwiseMulOpBF16(OpTest):
    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_mul"
        self.dtype = np.float32
        self.axis = -1
        self.init_input_output()

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(self.x),
            "Y": OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.outputs = {"Out": self.out}
        self.attrs = {"axis": self.axis}

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place)

    @check_soc_version
    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ["X", "Y"], "Out")

    @check_soc_version
    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(self.place, ["Y"], "Out", no_grad_set=set("X"))

    @check_soc_version
    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(self.place, ["X"], "Out", no_grad_set=set("Y"))

    def init_input_output(self):
        self.x = convert_float_to_uint16(
            np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        )
        self.y = convert_float_to_uint16(
            np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        )
        self.out = np.multiply(
            convert_uint16_to_float(self.x), convert_uint16_to_float(self.y)
        )


if __name__ == "__main__":
    unittest.main()
