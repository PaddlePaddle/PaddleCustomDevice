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
from tests.op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float

from npu_utils import check_soc_version


class TestStackOpBf16(OpTest):
    def initDefaultParameters(self):
        self.num_inputs = 4
        self.input_dim = (5, 6, 7)
        self.axis = 0

    def initParameters(self):
        pass

    def get_x_names(self):
        x_names = []
        for i in range(self.num_inputs):
            x_names.append("x{}".format(i))
        return x_names

    def setUp(self):
        self.initDefaultParameters()
        self.initParameters()
        self.op_type = "stack"
        self.set_npu()
        self.init_dtype()
        self.x = []
        self.y = []
        for i in range(self.num_inputs):
            self.x.append(
                convert_float_to_uint16(
                    np.random.random(size=self.input_dim).astype(np.float32)
                )
            )

        tmp = []
        x_names = self.get_x_names()
        for i in range(self.num_inputs):
            tmp.append((x_names[i], self.x[i]))

        self.inputs = {"X": tmp}
        for i in self.x:
            self.y.append(convert_uint16_to_float(i))
        self.outputs = {"Y": np.stack(self.y, axis=self.axis)}
        self.attrs = {"axis": self.axis}

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def init_dtype(self):
        self.dtype = np.float32

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place)

    @check_soc_version
    def test_check_grad(self):
        self.check_grad_with_place(self.place, self.get_x_names(), "Y")


if __name__ == "__main__":
    unittest.main()
