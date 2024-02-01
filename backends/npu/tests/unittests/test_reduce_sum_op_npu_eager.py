# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import os

select_npu = os.environ.get("FLAGS_selected_npus", 0)
import numpy as np
import paddle

from tests.op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float

from npu_utils import check_soc_version


class TestReduceSumBF16AllReduce(OpTest):
    def setUp(self):
        self.set_npu()
        self.init_attr()
        self.place = paddle.CustomPlace("npu", select_npu)
        self.op_type = "reduce_sum"
        self.shape_x = [6, 2, 10]
        self.x = np.random.uniform(-1, 1, self.shape_x).astype(np.float32)
        np_uint16_x = convert_float_to_uint16(self.x)
        self.np_uint16_to_fp32_x = convert_uint16_to_float(np_uint16_x)
        self.attrs = {
            "dim": self.axis,
            "keep_dim": self.keep_dim,
        }
        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(np_uint16_x),
        }
        if self.reduce_all:
            self.outputs = {"Out": np.sum(self.np_uint16_to_fp32_x)}
        else:
            self.outputs = {
                "Out": np.sum(
                    self.np_uint16_to_fp32_x,
                    axis=self.attrs["dim"],
                    keepdims=self.attrs["keep_dim"],
                )
            }

    def init_attr(self):
        self.keep_dim = False
        self.axis = None
        self.reduce_all = True

    def set_npu(self):
        self.__class__.use_custom_device = True

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place)

    @check_soc_version
    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestReduceSumBF16Axis(TestReduceSumBF16AllReduce):
    def init_attr(self):
        self.keep_dim = False
        self.axis = (1,)
        self.reduce_all = False


class TestReduceSumBF16Keepdim(TestReduceSumBF16AllReduce):
    def init_attr(self):
        self.keep_dim = True
        self.axis = (2,)
        self.reduce_all = False


if __name__ == "__main__":
    unittest.main()
