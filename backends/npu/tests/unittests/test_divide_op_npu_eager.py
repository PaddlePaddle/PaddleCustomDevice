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


class TestElementwiseDivBf16(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_div"
        self.place = paddle.CustomPlace("npu", 0)

        np.random.seed(SEED)
        x = convert_float_to_uint16(np.random.uniform(1, 2, [3, 4]).astype(np.float32))
        y = convert_float_to_uint16(np.random.uniform(1, 2, [3, 4]).astype(np.float32))
        out = convert_uint16_to_float(np.divide(x, y))

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(x),
            "Y": OpTest.np_dtype_to_base_dtype(y),
        }
        self.attrs = {}
        self.outputs = {"Out": out}

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)

    @check_soc_version
    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place,
            ["X", "Y"],
            "Out",
            max_relative_error=0.007,
        )


if __name__ == "__main__":
    unittest.main()
