# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at #
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
from numpy import linalg as LA
from tests.op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float
import paddle

from npu_utils import check_soc_version


class TestL2LossOpBf16(OpTest):
    """Test npu squared_l2_norm"""

    def setUp(self):
        self.set_npu()
        self.place = paddle.CustomPlace("npu", select_npu)
        self.op_type = "squared_l2_norm"
        self.max_relative_error = 0.05

        X = convert_float_to_uint16(
            np.random.uniform(-1, 1, (13, 19)).astype("float32")
        )
        X[np.abs(X) < self.max_relative_error] = 0.1
        self.inputs = {"X": X}
        self.outputs = {
            "Out": np.array([np.square(LA.norm(convert_uint16_to_float(X)))])
        }

    def set_npu(self):
        self.__class__.use_custom_device = True

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(place=self.place)

    @check_soc_version
    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X"], "Out", max_relative_error=self.max_relative_error
        )


if __name__ == "__main__":
    unittest.main()
