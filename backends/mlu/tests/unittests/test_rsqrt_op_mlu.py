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

import unittest

import numpy as np
import paddle
from tests.op_test import OpTest

paddle.enable_static()
SEED = 2023


class TestRsqrt(OpTest):
    def setUp(self):
        self.set_mlu()
        self.op_type = "rsqrt"

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = 1.0 / np.sqrt(x)

        self.inputs = {"X": OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {}
        self.outputs = {"Out": out}

    def set_mlu(self):
        self.place = paddle.CustomPlace("mlu", 0)
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out", max_relative_error=0.02)


class TestRsqrtFp16(TestRsqrt):
    def init_dtype(self):
        self.dtype = np.float16


if __name__ == "__main__":
    unittest.main()
