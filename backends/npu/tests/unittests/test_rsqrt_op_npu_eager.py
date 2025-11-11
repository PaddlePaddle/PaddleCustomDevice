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

SEED = 2021


class TestRsqrtBF16(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "rsqrt"
        self.place = paddle.CustomPlace("npu", 0)

        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(np.float32)
        out = 1.0 / np.sqrt(convert_uint16_to_float(convert_float_to_uint16(x)))

        self.inputs = {"X": convert_float_to_uint16(OpTest.np_dtype_to_base_dtype(x))}
        self.attrs = {}
        self.outputs = {"Out": out}

    def set_npu(self):
        self.__class__.use_custom_device = True

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place)

    @check_soc_version
    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            max_relative_error=0.009,
            numeric_place=paddle.CPUPlace(),
        )


if __name__ == "__main__":
    unittest.main()
