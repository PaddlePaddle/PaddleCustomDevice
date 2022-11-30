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

import unittest

import numpy as np
import paddle
from tests.op_test import OpTest

paddle.enable_static()


class TestAdagradOp1(OpTest):
    """Test Adagrad operator with explicit attributes"""

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def setUp(self):
        self.set_npu()
        self.init_dtype()
        self.op_type = "adagrad"
        param = np.random.random((123, 321)).astype(self.dtype)
        grad = np.random.random((123, 321)).astype(self.dtype)
        moment = np.zeros((123, 321)).astype(self.dtype)
        lr = 0.01
        epsilon = 1e-8

        self.inputs = {
            "Param": param,
            "Grad": grad,
            "Moment": moment,
            "LearningRate": np.array([lr]).astype(self.dtype),
        }

        self.attrs = {"epsilon": epsilon}

        moment_out = moment + grad * grad
        param_out = param - lr * grad / (np.sqrt(moment_out) + epsilon)

        self.outputs = {"ParamOut": param_out, "MomentOut": moment_out}

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5, check_dygraph=False)


class TestAdagradOp2(OpTest):
    """Test Adagrad operator with default attributes"""

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def setUp(self):
        self.set_npu()
        self.init_dtype()
        self.op_type = "adagrad"

        param = np.random.random((5, 5)).astype(self.dtype)
        grad = np.random.random((5, 5)).astype(self.dtype)
        moment = np.zeros((5, 5)).astype(self.dtype)
        lr = 0.01
        epsilon = 1e-6

        self.inputs = {
            "Param": param,
            "Grad": grad,
            "Moment": moment,
            "LearningRate": np.array([lr]).astype("float32"),
        }

        self.attrs = {"epsilon": epsilon}

        moment_out = moment + grad * grad
        param_out = param - lr * grad / (np.sqrt(moment_out) + epsilon)

        self.outputs = {"ParamOut": param_out, "MomentOut": moment_out}

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
