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

import unittest
import os

select_npu = os.environ.get("FLAGS_selected_npus", 0)

import numpy as np
from tests.op_test import OpTest

import paddle


class TestFillDiagonalOpNPU(OpTest):
    def setUp(self):
        self.op_type = "fill_diagonal"
        self.set_npu()
        self.init_kernel_type()
        x1 = np.random.random((10, 10)).astype(self.dtype)
        x2 = x1.copy()
        value = 0
        offset = 0
        wrap = False
        np.fill_diagonal(x2, value, wrap)

        self.inputs = {"X": x1}
        self.outputs = {"Out": x2}
        self.attrs = {"value": value, "offset": offset, "wrap": wrap}

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", select_npu)

    def init_kernel_type(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestTensorFillDiagonalInt32(TestFillDiagonalOpNPU):
    def init_kernel_type(self):
        self.dtype = np.int32

    # int32 not supported for grad check
    def test_check_grad(self):
        pass


class TestTensorFillDiagonalInt64(TestFillDiagonalOpNPU):
    def init_kernel_type(self):
        self.dtype = np.int64

    # int64 not supported for grad check
    def test_check_grad(self):
        pass


class TestTensorFillDiagonalWrapTrue(TestFillDiagonalOpNPU):
    def setUp(self):
        self.op_type = "fill_diagonal"
        self.set_npu()
        self.init_kernel_type()
        x1 = np.random.random((10, 10)).astype(self.dtype)
        x2 = x1.copy()
        value = 0
        offset = 0
        wrap = True
        np.fill_diagonal(x2, value, wrap)

        self.inputs = {"X": x1}
        self.outputs = {"Out": x2}
        self.attrs = {"value": value, "offset": offset, "wrap": wrap}


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
