#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
from tests.op_test import OpTest
import paddle
from scipy.special import erf, expit

SEED = 2021

def swish(x):
    out = x.astype('float32') * expit(x.astype('float32'))
    return out.astype(x.dtype)

class TestSwiGluFP32_1(OpTest):
    def setUp(self):
        self.set_npu()
        self.init_dtype()
        self.op_type = "swiglu"
        self.place = paddle.CustomPlace("npu", 0)

        rows = 20
        cols = 512
        x = (np.random.rand(rows, cols) * 16).astype(self.dtype)
        bias = np.random.rand(cols).astype(self.dtype)
        res_tmp = (x + bias).astype(self.dtype)
        res_tmp_head = res_tmp[:, : cols // 2]
        res_tmp_tail = res_tmp[:, cols // 2 :]
        res_tmp_head_act = swish(res_tmp_head)
        y = res_tmp_head_act * res_tmp_tail

        self.inputs = {"x": res_tmp_head, "y": res_tmp_tail}
        self.outputs = {"out": y}

    def init_dtype(self):
        self.dtype = np.float32

    def set_npu(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

class TestSwiGluFP32_2(OpTest):
    def setUp(self):
        self.set_npu()
        self.init_dtype()
        self.op_type = "swiglu"
        self.place = paddle.CustomPlace("npu", 0)

        rows = 20
        cols = 512
        x = (np.random.rand(rows, cols) * 16).astype(self.dtype)
        bias = np.random.rand(cols).astype(self.dtype)
        res_tmp = (x + bias).astype(self.dtype)
        res_tmp_head = res_tmp[:, : cols // 2]
        res_tmp_tail = res_tmp[:, cols // 2 :]
        res_tmp_head_act = swish(res_tmp_head)
        y = res_tmp_head_act * res_tmp_tail

        inputx = np.concatenate((res_tmp_head, res_tmp_tail), axis=-1)

        self.inputs = {"x": inputx}
        self.outputs = {"out": y}

    def init_dtype(self):
        self.dtype = np.float32

    def set_npu(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == "__main__":
    unittest.main()
