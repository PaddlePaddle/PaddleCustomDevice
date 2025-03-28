#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from tests.op_test import OpTest
import paddle
import os

paddle.enable_static()
intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)


class TestAllOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_all"
        self.set_hpu()
        self.initTestCase()
        self.attrs = {
            "dim": self.axis,
            "keep_dim": self.keep_dim,
        }
        self.inputs = {"X": np.random.random(self.shape).astype(self.dtype)}
        self.outputs = {
            "Out": self.inputs["X"].all(axis=self.axis, keepdims=self.keep_dim)
        }

    def set_hpu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))

    def initTestCase(self):
        self.shape = (5, 6, 10)
        self.axis = (0,)
        self.keep_dim = True
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


class TestAllF16(TestAllOp):
    def initTestCase(self):
        self.shape = (2, 5, 3, 2, 2)
        self.axis = (0,)
        self.keep_dim = True
        self.dtype = np.float16


class TestAllBool(TestAllOp):
    def initTestCase(self):
        self.shape = (2, 5, 10)
        self.axis = (0, 1)
        self.keep_dim = True
        self.dtype = np.int32


if __name__ == "__main__":
    unittest.main()
