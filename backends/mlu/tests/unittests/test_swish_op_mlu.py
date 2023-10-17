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

from __future__ import print_function
import paddle

from tests.op_test import OpTest

import numpy as np
import unittest

paddle.enable_static()
SEED = 2021
np.random.seed(SEED)


def ref_swish(x):
    from scipy.special import expit

    out = x * expit(x)
    return out


class TestSwish(OpTest):
    def setUp(self):
        self.set_mlu()
        self.op_type = "swish"
        self.place = paddle.set_device("mlu")

        self.init_dtype()
        self.init_config()
        out = ref_swish(self.x)

        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(self.x)}
        self.attrs = {}
        self.outputs = {"Out": out}

    def init_config(self):
        self.x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.__class__.use_custom_device = True
        self.use_dynamic_create_class = False

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(
            self.place,
        )

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out", max_relative_error=0.009)


class TestCase1(TestSwish):
    def init_config(self):
        self.x = np.random.uniform(-2, 2, []).astype(self.dtype)


class TestCase2(TestSwish):
    def init_config(self):
        self.x = np.random.uniform(-2, 2, [1024, 8]).astype(self.dtype)


class TestCase3(TestSwish):
    def init_config(self):
        self.x = np.random.uniform(-2, 2, [4, 1, 15, 15]).astype(self.dtype)


class TestCase4(TestSwish):
    def init_config(self):
        self.x = np.random.uniform(-2, 2, [4, 1, 22, 22]).astype(self.dtype)


if __name__ == "__main__":
    unittest.main()
