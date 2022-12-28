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
from tests.op_test import OpTest


def ref_hard_shrink_grad(x, threshold=0.5):
    up = threshold
    lo = -threshold
    mask_low = x < lo
    mask_up = x > up
    mask = np.logical_or(mask_low, mask_up)
    mask.astype(x.dtype)
    return mask


def ref_hard_shrink(x, threshold=0.5):
    return x * ref_hard_shrink_grad(x, threshold)


paddle.enable_static()


class TestHardShrink(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "hard_shrink"
        self.place = paddle.CustomPlace("npu", 0)
        self.init_dtype()

        x = np.random.uniform(-5, 5, [10, 12]).astype(self.dtype)
        threshold = 0.5

        out = ref_hard_shrink(x, threshold)
        self.inputs = {"X": x}
        self.attrs = {"threshold": threshold}
        self.outputs = {"Out": out}
        self.x_grad = ref_hard_shrink_grad(x, threshold)

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X"], "Out", user_defined_grads=[self.x_grad]
        )


class TestHardShrinkFp16(TestHardShrink):
    def init_kernel_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X"], "Out", user_defined_grads=[self.x_grad]
        )


if __name__ == "__main__":
    unittest.main()
