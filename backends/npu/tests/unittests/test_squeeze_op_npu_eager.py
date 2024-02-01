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

import numpy as np
import unittest
import os

select_npu = os.environ.get("FLAGS_selected_npus", 0)
from tests.op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float
import paddle

from npu_utils import check_soc_version


# input x is bfloat16
class TestSqueeze2Op(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "squeeze2"
        self.init_test_case()
        middle_inputs = np.random.random(self.ori_shape).astype("float32")
        middle_inputs = convert_float_to_uint16(middle_inputs)
        self.inputs = {"X": middle_inputs}
        self.init_attrs()
        self.outputs = {
            "Out": convert_uint16_to_float(middle_inputs).reshape(self.new_shape),
            "XShape": np.random.random(self.ori_shape).astype("float32"),
        }

    def set_npu(self):
        self.__class__.use_custom_device = True

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(
            paddle.CustomPlace("npu", select_npu), no_check_set=["XShape"]
        )

    @check_soc_version
    def test_check_grad(self):
        self.check_grad_with_place(paddle.CustomPlace("npu", select_npu), ["X"], "Out")

    def init_test_case(self):
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0, 2)
        self.new_shape = (3, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


if __name__ == "__main__":
    unittest.main()
