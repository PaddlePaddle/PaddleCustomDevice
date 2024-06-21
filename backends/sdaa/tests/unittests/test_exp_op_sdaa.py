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
SEED = 2021


class TestExp(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "exp"
        self.python_api = paddle.exp
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = np.exp(x)
        self.inputs = {"X": x}
        self.outputs = {"Out": out}
        grad_out = np.ones(out.shape).astype(self.dtype)
        self.grad_out = grad_out
        grad_x = self.compute_gradient(grad_out, out)
        self.grad_x = grad_x

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def compute_gradient(self, grad_out, out):
        return grad_out * out

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            user_defined_grads=[self.grad_x],
            user_defined_grad_outputs=[self.grad_out],
        )


if __name__ == "__main__":
    unittest.main()
