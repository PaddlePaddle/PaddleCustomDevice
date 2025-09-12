#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

import paddle

paddle.enable_static()


def get_places(self):
    return [paddle.CustomPlace("custom_cpu", 0)]


OpTest._get_places = get_places


def reference_matmul(X, Y, transpose_X=False, transpose_Y=False):
    """Reference forward implementation using np.matmul."""
    # np.matmul does not support the transpose flags, so we manually
    # transpose X and Y appropriately.
    if transpose_X:
        if X.ndim == 1:
            X = X.reshape((X.size,))
        elif X.ndim == 2:
            X = X.T
        else:
            dim = [i for i in range(len(X.shape))]
            dim[-1], dim[len(X.shape) - 2] = dim[len(X.shape) - 2], dim[-1]
            X = np.transpose(X, tuple(dim))
    if transpose_Y:
        if Y.ndim == 1:
            Y = Y.reshape((Y.size,))
        else:
            dim = [i for i in range(len(Y.shape))]
            dim[-1], dim[len(Y.shape) - 2] = dim[len(Y.shape) - 2], dim[-1]
            Y = np.transpose(Y, tuple(dim))

    Out = np.matmul(X, Y)
    if not Out.shape:
        # We do not support 0-dimensional Tensors (scalars). So where
        # np.matmul outputs a scalar, we must convert to a Tensor of
        # shape (1, ) instead.
        # Everywhere else, we are compatible with np.matmul.
        Out = np.array([Out], dtype="float64")
    return Out


class TestMatMulOp(OpTest):
    """
    case 1
    """

    def config(self):
        self.x_shape = (100,)
        self.y_shape = (100,)
        self.trans_x = False
        self.trans_y = False

    def init_kernel_type(self):
        self.dtype = "float64"

    def setUp(self):
        self.init_kernel_type()
        self.config()
        self.op_type = "matmul_v2"
        x = np.random.random(self.x_shape).astype(self.dtype)
        y = np.random.random(self.y_shape).astype(self.dtype)
        x = np.ones_like(np.random.random(self.x_shape).astype(self.dtype))
        y = (
            np.array(range(np.product(self.y_shape)))
            .reshape(self.y_shape)
            .astype(self.dtype)
        )
        # -0.1 ~ 0.1
        x = -0.1 + 0.2 * x
        y = -0.1 + 0.2 * y
        result = reference_matmul(x, y, self.trans_x, self.trans_y)
        self.inputs = {
            "X": x,
            "Y": y,
        }
        self.attrs = {"trans_x": self.trans_x, "trans_y": self.trans_y}
        self.outputs = {"Out": result}

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad(self):
        self.check_grad(["X", "Y"], "Out", check_eager=False)


class TestMatMul2Dx1D(TestMatMulOp):
    def config(self):
        self.x_shape = (2, 100)
        self.y_shape = 100
        self.trans_x = False
        self.trans_y = False


class TestMatMul2Dx1D_transX(TestMatMulOp):
    def config(self):
        self.x_shape = (100, 2)
        self.y_shape = 100
        self.trans_x = True
        self.trans_y = False


class TestMatMul1Dx2D(TestMatMulOp):
    def config(self):
        self.x_shape = 100
        self.y_shape = (100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMul1Dx2D_transY(TestMatMulOp):
    def config(self):
        self.x_shape = 100
        self.y_shape = (2, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatMul3Dx1D(TestMatMulOp):
    def config(self):
        self.x_shape = (2, 3, 100)
        self.y_shape = (100,)
        self.trans_x = False
        self.trans_y = False


class TestMatMul3Dx1D_TransX(TestMatMulOp):
    def config(self):
        self.x_shape = (2, 100, 3)
        self.y_shape = (100,)
        self.trans_x = True
        self.trans_y = False


class TestMatMul1Dx3D(TestMatMulOp):
    def config(self):
        self.x_shape = (100,)
        self.y_shape = (2, 100, 3)
        self.trans_x = False
        self.trans_y = False


class TestMatMul1Dx3D_TransY(TestMatMulOp):
    def config(self):
        self.x_shape = (100,)
        self.y_shape = (2, 3, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatMul2Dx2D(TestMatMulOp):
    def config(self):
        self.x_shape = (11, 12)
        self.y_shape = (12, 13)
        self.trans_x = False
        self.trans_y = False


class TestMatMul2Dx2D_TransX(TestMatMul2Dx2D):
    def config(self):
        self.x_shape = (12, 11)
        self.y_shape = (12, 13)
        self.trans_x = True
        self.trans_y = False


class TestMatMul2Dx2D_TransY(TestMatMul2Dx2D):
    def config(self):
        self.x_shape = (11, 12)
        self.y_shape = (13, 12)
        self.trans_x = False
        self.trans_y = True


class TestMatMul2Dx2D_TransXY(TestMatMul2Dx2D):
    def config(self):
        self.x_shape = (12, 11)
        self.y_shape = (13, 12)
        self.trans_x = True
        self.trans_y = True


class TestMatMul3Dx2D(TestMatMulOp):
    def config(self):
        self.x_shape = (5, 11, 12)
        self.y_shape = (12, 13)
        self.trans_x = False
        self.trans_y = False


class TestMatMul3Dx2D_TransXY(TestMatMulOp):
    def config(self):
        self.x_shape = (5, 12, 11)
        self.y_shape = (13, 12)
        self.trans_x = True
        self.trans_y = True


class TestMatMul3Dx2D_TransX(TestMatMulOp):
    def config(self):
        self.x_shape = (5, 12, 11)
        self.y_shape = (12, 13)
        self.trans_x = True
        self.trans_y = False


class TestMatMul3Dx2D_TransY(TestMatMulOp):
    def config(self):
        self.x_shape = (5, 11, 12)
        self.y_shape = (13, 12)
        self.trans_x = False
        self.trans_y = True


class TestMatMul2Dx3D(TestMatMulOp):
    def config(self):
        self.x_shape = (13, 12)
        self.y_shape = (5, 12, 11)
        self.trans_x = False
        self.trans_y = False


class TestMatMul2Dx3D_TransX(TestMatMulOp):
    def config(self):
        self.x_shape = (12, 13)
        self.y_shape = (5, 12, 11)
        self.trans_x = True
        self.trans_y = False

    def test_check_grad(self):
        self.check_grad(["X", "Y"], "Out")


class TestMatMul2Dx3D_TransY(TestMatMulOp):
    def config(self):
        self.x_shape = (13, 12)
        self.y_shape = (5, 11, 12)
        self.trans_x = False
        self.trans_y = True


class TestMatMul2Dx3D_TransXY(TestMatMulOp):
    def config(self):
        self.x_shape = (12, 13)
        self.y_shape = (5, 11, 12)
        self.trans_x = True
        self.trans_y = True


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
