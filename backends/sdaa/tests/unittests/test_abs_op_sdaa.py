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

import numpy as np
import unittest

from op_test import OpTest
import paddle

paddle.enable_static()


# @skip_check_grad_ci(reason="Haven not implement abs grad kernel.")
class TestAbsKernel(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "abs"
        self.init_dtype()
        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [4, 25]).astype(self.dtype)

        x[np.abs(x) < 0.005] = 0.02
        out = np.abs(x)

        self.inputs = {"X": x}
        self.outputs = {"Out": out}
        self.python_api = paddle.abs

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestAbsKernelFP16(TestAbsKernel):
    def init_dtype(self):
        self.dtype = np.float16

    # kernel 'mean' is not registerd and cpu 'mean' kernel do not support fp16
    def test_check_grad(self):
        pass


class TestAbsKernelINT64(TestAbsKernel):
    def init_dtype(self):
        self.dtype = np.int64

    # get_numeric_gradient not support int64
    def test_check_grad(self):
        pass


# This case comes from UIE-X-base model.
class TestAbsKernelINT64Case1(TestAbsKernel):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "abs"
        self.init_dtype()
        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [4, 561, 561]).astype(self.dtype)

        x[np.abs(x) < 0.005] = 0.02
        out = np.abs(x)

        self.inputs = {"X": x}
        self.outputs = {"Out": out}
        self.python_api = paddle.abs

    def init_dtype(self):
        self.dtype = np.int64

    # get_numeric_gradient not support int64
    def test_check_grad(self):
        pass


if __name__ == "__main__":
    unittest.main()
