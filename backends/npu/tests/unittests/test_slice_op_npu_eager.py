# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
EPOCH = 100


class TestSliceOpBF16Tensor(OpTest):
    def setUp(self):
        self.op_type = "slice"
        self.set_npu()
        self.init_dtype()
        self.config()
        self.inputs = {
            "Input": self.input,
            "StartsTensor": self.starts,
            "EndsTensor": self.ends,
        }
        self.outputs = {"Out": self.out}
        self.attrs = {
            "axes": self.axes,
            "starts": [0],
            "ends": [1],
            "infer_flags": self.infer_flags,
        }

    def config(self):
        np_input = np.random.random([10, 5, 6]).astype("float32")
        self.input = convert_float_to_uint16(np_input)
        self.starts = np.array([0]).astype("int32")
        self.ends = np.array([1]).astype("int32")
        self.axes = [1]
        self.infer_flags = [-1]
        self.out = convert_uint16_to_float(self.input)[:, 0:1, :]

    def init_dtype(self):
        self.dtype = np.uint16

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("npu", 0)

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place, atol=0.004)

    @check_soc_version
    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place, ["Input"], "Out", max_relative_error=0.004
        )


if __name__ == "__main__":
    unittest.main()
