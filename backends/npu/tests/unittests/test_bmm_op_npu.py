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

paddle.enable_static()
SEED = 2021


def reference_matmul(X, Y):
    """Reference forward implementation using np.matmul."""
    Out = np.matmul(X, Y)
    if not Out.shape:
        # We do not support 0-dimensional Tensors (scalars). So where
        # np.matmul outputs a scalar, we must convert to a Tensor of
        # shape (1, ) instead.
        # Everywhere else, we are compatible with np.matmul.
        Out = np.array([Out], dtype="float64")
    return Out


class TestBmmOp(OpTest):
    """
    case 0
    """

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def config(self):
        self.x_shape = (10, 2, 5)
        self.y_shape = (10, 5, 8)

    def init_kernel_type(self):
        self.dtype = "float32"

    def setUp(self):
        self.set_npu()
        self.init_kernel_type()
        self.config()
        self.op_type = "bmm"
        x = np.random.random(self.x_shape).astype(self.dtype)
        y = np.random.random(self.y_shape).astype(self.dtype)
        # -0.1 ~ 0.1
        x = -0.1 + 0.2 * x
        y = -0.1 + 0.2 * y
        result = reference_matmul(x, y)
        result = result.astype(self.dtype)
        self.inputs = {
            "X": x,
            "Y": y,
        }
        self.outputs = {"Out": result}

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X", "Y"], "Out", numeric_place=paddle.CPUPlace()
        )


class TestBmmOp1(TestBmmOp):
    """
    case 1
    """

    def config(self):
        self.x_shape = (40, 10, 10)
        self.y_shape = (40, 10, 10)

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X", "Y"], "Out", numeric_place=paddle.CPUPlace()
        )


class TestBmmOp2(TestBmmOp):
    """
    case 2
    """

    def config(self):
        self.x_shape = (4, 10, 80)
        self.y_shape = (4, 80, 1)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["X", "Y"],
            "Out",
            numeric_place=paddle.CPUPlace(),
            max_relative_error=1e-2,
        )

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
