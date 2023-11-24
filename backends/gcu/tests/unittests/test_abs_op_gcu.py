# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from tests.op_test import OpTest
import paddle

paddle.enable_static()


class TestGCUAbs(OpTest):
    def setUp(self):
        self.op_type = "abs"
        self.set_device()
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [4, 25]).astype(self.dtype)
        # Because we set delta = 0.005 in calculating numeric gradient,
        # if x is too small, such as 0.002, x_neg will be -0.003
        # x_pos will be 0.007, so the numeric gradient is inaccurate.
        # we should avoid this
        x[np.abs(x) < 0.005] = 0.02
        out = np.abs(x)

        self.inputs = {"X": x}
        self.outputs = {"Out": out}

    def set_device(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("gcu", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestGCUAbsInt64(TestGCUAbs):
    def init_dtype(self):
        self.dtype = np.int64

    # get_numeric_gradient not support int64
    def test_check_grad(self):
        pass


if __name__ == "__main__":
    unittest.main()
