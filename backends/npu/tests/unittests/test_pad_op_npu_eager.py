#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
import paddle
from tests.op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float
from npu_utils import check_soc_version

paddle.enable_static()


class TestPadOp(OpTest):
    def setUp(self):
        self.op_type = "pad"
        self.set_npu()
        self.init_dtype()
        self.initTestCase()

        np_input = np.random.random(self.shape).astype(self.dtype)
        self.input = convert_float_to_uint16(np_input)
        self.inputs = {
            "X": self.input,
        }
        self.attrs = {}
        self.attrs["paddings"] = np.array(self.paddings).flatten()
        self.attrs["pad_value"] = self.pad_value
        self.outputs = {
            "Out": np.pad(
                convert_uint16_to_float(self.inputs["X"]),
                self.paddings,
                mode="constant",
                constant_values=self.pad_value,
            )
        }

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place)

    @check_soc_version
    def test_check_grad_normal(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(self.place, ["X"], "Out", max_relative_error=0.6)
        else:
            self.check_grad_with_place(self.place, ["X"], "Out")

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def initTestCase(self):
        self.shape = (16, 16)
        self.paddings = [(1, 1), (2, 3)]
        self.pad_value = 0.0


if __name__ == "__main__":
    unittest.main()
