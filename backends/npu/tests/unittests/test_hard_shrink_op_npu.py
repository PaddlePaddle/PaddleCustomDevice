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
import os

select_npu = os.environ.get("FLAGS_selected_npus", 0)

import numpy as np
import paddle
from tests.op_test import OpTest


def ref_hard_shrink(x, threshold=0.5):
    mask_low = x < -threshold
    mask_up = x > threshold
    mask = np.logical_or(mask_low, mask_up)
    mask.astype(x.dtype)
    return x * mask


paddle.enable_static()


class TestHardShrink(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "hard_shrink"
        self.place = paddle.CustomPlace("npu", select_npu)
        self.init_dtype()

        x = np.random.uniform(-5, 5, [10, 12]).astype(self.dtype)
        threshold = 0.5

        out = ref_hard_shrink(x, threshold)
        self.inputs = {"X": x}
        self.attrs = {"threshold": threshold}
        self.outputs = {"Out": out}

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestHardShrinkFp16(TestHardShrink):
    def init_kernel_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


if __name__ == "__main__":
    unittest.main()
