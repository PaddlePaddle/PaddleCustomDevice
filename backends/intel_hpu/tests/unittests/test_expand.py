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
from tests.op_test import OpTest, skip_check_grad_ci

paddle.enable_static()
SEED = 2021


class TestExpandV2HPUOp(OpTest):
    def setUp(self):
        self.set_hpu()
        self.place = paddle.CustomPlace("intel_hpu", 0)
        self.op_type = "expand_v2"
        self.init_dtype()
        self.init_data()
        self.attrs = {"shape": self.shape}
        self.input = np.random.random(self.ori_shape).astype(self.dtype)
        self.output = np.tile(self.input, self.expand_times)
        self.inputs = {"X": self.input}
        self.outputs = {"Out": self.output}

    def init_data(self):
        self.ori_shape = (4, 1, 15)
        self.shape = (2, -1, 4, -1)
        self.expand_times = (2, 1, 4, 1)

    def set_hpu(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == "__main__":
    unittest.main()
