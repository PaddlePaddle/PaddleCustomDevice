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

from op_test import OpTest
import paddle

paddle.enable_static()


def python_core_api(x):
    return paddle._C_ops.mean_all(x)


class TestMeanAll(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "mean"
        self.python_api = python_core_api
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()

        x = np.random.random([1, 2, 10, 10]).astype(self.dtype)
        self.inputs = {"X": x}
        out = np.mean(x)
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_no_input(self):
        self.check_grad_with_place(
            paddle.CustomPlace("sdaa", 0),
            ["X"],
            "Out",
            max_relative_error=0.006,
        )


class TestMeanAllFP16(TestMeanAll):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=0.1)


if __name__ == "__main__":
    unittest.main()
