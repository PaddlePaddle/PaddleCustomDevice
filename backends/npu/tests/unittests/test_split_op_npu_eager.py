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

from __future__ import print_function, division

import numpy as np
import unittest
from tests.op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float
import paddle
from npu_utils import check_soc_version, check_run_big_shape_test


class TestNPUSplitOpBF16(OpTest):
    def set_plugin(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("npu", 0)

    def setUp(self):
        self.set_plugin()
        self._set_op_type()
        self.init_dtype()
        self.init_data()
        self.inputs = {"X": self.x}
        self.attrs = {"axis": self.axis, "sections": self.sections, "num": self.num}

        out = np.split(
            convert_uint16_to_float(self.x), self.indices_or_sections, self.axis
        )
        self.outputs = {
            "Out": [
                ("out%d" % i, convert_uint16_to_float(out[i])) for i in range(len(out))
            ]
        }

    def init_data(self):
        self.np_x = np.random.random((4, 5, 6)).astype("float32")
        self.x = convert_float_to_uint16(self.np_x)
        self.axis = 2
        self.sections = []
        self.num = 3
        self.indices_or_sections = 3

    def init_dtype(self):
        self.dtype = np.uint16

    def _set_op_type(self):
        self.op_type = "split"

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place, atol=0.004)

    def test_check_grad(self):
        pass


@check_run_big_shape_test()
class TestNPUSplitOpRank1(TestNPUSplitOpBF16):
    def init_data(self):
        self.x = np.random.random((2, 4096, 1, 1280)).astype(self.dtype)
        self.axis = 1
        self.sections = []
        self.num = 1
        self.indices_or_sections = 1


@check_run_big_shape_test()
class TestNPUSplitOpRank2(TestNPUSplitOpBF16):
    def init_data(self):
        self.x = np.random.random((8192, 7168)).astype(self.dtype)
        self.axis = 1
        self.sections = []
        self.num = 1
        self.indices_or_sections = 1


if __name__ == "__main__":
    unittest.main()
