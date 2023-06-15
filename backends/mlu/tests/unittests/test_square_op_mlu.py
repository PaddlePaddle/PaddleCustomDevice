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

import unittest

import numpy as np
import paddle
from tests.op_test import OpTest

paddle.enable_static()
SEED = 2021


class TestSquare(OpTest):
    def set_mlu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("mlu", 0)

    def setUp(self):
        self.set_mlu()
        self.op_type = "square"
        self.python_api = paddle.square
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.square(x)

        self.inputs = {"X": OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {"Out": out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(
            self.place, ["X"], "Out", max_relative_error=0.007, check_dygraph=True
        )

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=True)

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [11, 17]


class TestSquare_ZeroDim(TestSquare):
    def init_shape(self):
        self.shape = []


# ------------------ Test Fp16 ----------------------
def create_test_act_fp16_class(parent, atol=1e-3, grad_check=True, grad_atol=0.80):
    class TestActFp16(parent):
        def init_dtype(self):
            self.dtype = np.float16

        def test_check_output(self):
            self.check_output_with_place(self.place, atol=atol)

        def test_check_grad(self):
            if grad_check:
                self.check_grad_with_place(
                    self.place, ["X"], "Out", max_relative_error=grad_atol
                )

    cls_name = "{0}_{1}".format(parent.__name__, "fp16")
    TestActFp16.__name__ = cls_name
    globals()[cls_name] = TestActFp16


create_test_act_fp16_class(TestSquare)


if __name__ == "__main__":
    unittest.main()
