# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division, print_function

import unittest

import numpy as np
import paddle
from tests.op_test import OpTest, convert_uint16_to_float, convert_float_to_uint16
from npu_utils import check_soc_version


class TestNPUWhereOpBF16(OpTest):
    def setUp(self):
        self.op_type = "where"
        self.set_npu()
        self.init_config()
        self.inputs = {"Condition": self.cond, "X": self.x, "Y": self.y}
        self.outputs = {
            "Out": np.where(
                self.cond,
                convert_uint16_to_float(self.x),
                convert_uint16_to_float(self.y),
            )
        }

    def init_config(self):
        self.x = convert_float_to_uint16(
            np.random.uniform(-3, 5, (100)).astype(np.float32)
        )
        self.y = convert_float_to_uint16(
            np.random.uniform(-3, 5, (100)).astype(np.float32)
        )
        self.cond = np.zeros((100)).astype("bool")

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place)

    @check_soc_version
    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ["X", "Y"], "Out")


if __name__ == "__main__":
    unittest.main()
