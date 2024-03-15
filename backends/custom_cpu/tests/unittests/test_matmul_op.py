#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.base as base

paddle.enable_static()


def get_places(self):
    return [paddle.CustomPlace("custom_cpu", 0)]


OpTest._get_places = get_places


def reference_matmul(X, Y, transpose_x=False, transpose_y=False):
    """Reference forward implementation using np.matmul."""
    # np.matmul does not support the transpose flags, so we manually
    # transpose X and Y appropriately.
    if transpose_x:
        if X.ndim == 1:
            X = X.reshape((X.size,))
        elif X.ndim == 2:
            X = X.T
        else:
            dim = [i for i in range(len(X.shape))]
            dim[-1], dim[len(X.shape) - 2] = dim[len(X.shape) - 2], dim[-1]
            X = np.transpose(X, tuple(dim))
    if transpose_y:
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


class API_TestMm(unittest.TestCase):
    def test_out(self):
        with base.program_guard(base.Program()):
            x = paddle.static.data(name="x", shape=[2], dtype="float64")
            y = paddle.static.data(name="y", shape=[2], dtype="float64")
            res = paddle.static.data(name="output", shape=[1], dtype="float64")
            result = paddle.mm(x, y)
            exe = base.Executor(base.CustomPlace("custom_cpu", 0))
            data1 = np.random.rand(2)
            data2 = np.random.rand(2)
            np_res = exe.run(feed={"x": data1, "y": data2}, fetch_list=[result])
            expected_result = np.matmul(data1.reshape(1, 2), data2.reshape(2, 1))

        self.assertTrue(
            np.allclose(np_res, expected_result, atol=1e-5),
            "two value is\
            {}\n{}, check diff!".format(
                np_res, expected_result
            ),
        )

    def test_out_fp16(self):
        with base.program_guard(base.Program()):
            x = paddle.static.data(name="x", shape=[2], dtype="float16")
            y = paddle.static.data(name="y", shape=[2], dtype="float16")
            res = paddle.static.data(name="output", shape=[1], dtype="float16")
            result = paddle.mm(x, y)
            exe = base.Executor(base.CustomPlace("custom_cpu", 0))
            data1 = np.random.rand(2).astype("float16")
            data2 = np.random.rand(2).astype("float16")
            np_res = exe.run(feed={"x": data1, "y": data2}, fetch_list=[result])
            expected_result = np.matmul(data1.reshape(1, 2), data2.reshape(2, 1))

        self.assertTrue(
            np.allclose(np_res, expected_result, atol=1e-3),
            "two value is\
            {}\n{}, check diff!".format(
                np_res, expected_result
            ),
        )

    def test_dygraph_without_out(self):
        device = base.CustomPlace("custom_cpu", 0)
        with base.dygraph.guard(device):
            input_array1 = np.random.rand(3, 4).astype("float64")
            input_array2 = np.random.rand(4, 3).astype("float64")
            data1 = paddle.to_tensor(input_array1)
            data2 = paddle.to_tensor(input_array2)
            out = paddle.mm(data1, data2)
            expected_result = np.matmul(input_array1, input_array2)
        self.assertTrue(np.allclose(expected_result, out.numpy()))


class Test_API_Matmul2DX2D(unittest.TestCase):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = (3, 4)
        self.shape_y = (4, 3)
        self.trans_x = False
        self.trans_y = False

    def test_dygraph_without_out(self):
        self.init_shape()

        device = base.CustomPlace("custom_cpu", 0)
        with base.dygraph.guard(device):
            input_array1 = np.random.random(self.shape_x).astype(self.dtype)
            input_array2 = np.random.random(self.shape_y).astype(self.dtype)
            data1 = paddle.to_tensor(input_array1)
            data2 = paddle.to_tensor(input_array2)
            out = paddle.matmul(
                data1, data2, transpose_x=self.trans_x, transpose_y=self.trans_y
            )
            expected_result = reference_matmul(
                input_array1,
                input_array2,
                transpose_x=self.trans_x,
                transpose_y=self.trans_y,
            )
        self.assertTrue(np.allclose(expected_result, out.numpy()))


class Test_API_Matmul2DX2D_transX(Test_API_Matmul2DX2D):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = (4, 3)
        self.shape_y = (4, 3)
        self.trans_x = True
        self.trans_y = False


class Test_API_Matmul2DX2D_transY(Test_API_Matmul2DX2D):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = (3, 4)
        self.shape_y = (3, 4)
        self.trans_x = False
        self.trans_y = True


class Test_API_Matmul2DX2D_transXY(Test_API_Matmul2DX2D):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = (4, 3)
        self.shape_y = (3, 4)
        self.trans_x = True
        self.trans_y = True


class Test_API_Matmul1DX2D(Test_API_Matmul2DX2D):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = 4
        self.shape_y = (4, 3)
        self.trans_x = False
        self.trans_y = False


class Test_API_Matmul1DX2D_transY(Test_API_Matmul2DX2D):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = 4
        self.shape_y = (3, 4)
        self.trans_x = False
        self.trans_y = True


class Test_API_Matmul2DX1D(Test_API_Matmul2DX2D):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = (3, 4)
        self.shape_y = 4
        self.trans_x = False
        self.trans_y = False


class Test_API_Matmul2DX1D_transX(Test_API_Matmul2DX2D):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = (4, 3)
        self.shape_y = 4
        self.trans_x = True
        self.trans_y = False


class Test_API_Matmul1DX3D(Test_API_Matmul2DX2D):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = 4
        self.shape_y = (5, 4, 3)
        self.trans_x = False
        self.trans_y = False


class Test_API_Matmul1DX3D_transY(Test_API_Matmul2DX2D):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = 4
        self.shape_y = (5, 3, 4)
        self.trans_x = False
        self.trans_y = True


class Test_API_Matmul3DX1D(Test_API_Matmul2DX2D):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = (5, 3, 4)
        self.shape_y = 4
        self.trans_x = False
        self.trans_y = False


class Test_API_Matmul3DX1D_transX(Test_API_Matmul2DX2D):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = (5, 4, 3)
        self.shape_y = 4
        self.trans_x = True
        self.trans_y = False


class Test_API_Matmul3DX2D(Test_API_Matmul2DX2D):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = (5, 3, 4)
        self.shape_y = (4, 3)
        self.trans_x = False
        self.trans_y = False


class Test_API_Matmul3DX2D_transX(Test_API_Matmul2DX2D):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = (5, 4, 3)
        self.shape_y = (4, 3)
        self.trans_x = True
        self.trans_y = False


class Test_API_Matmul3DX2D_transY(Test_API_Matmul2DX2D):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = (5, 3, 4)
        self.shape_y = (3, 4)
        self.trans_x = False
        self.trans_y = True


class Test_API_Matmul3DX2D_transXY(Test_API_Matmul2DX2D):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = (5, 4, 3)
        self.shape_y = (3, 4)
        self.trans_x = True
        self.trans_y = True


class Test_API_Matmul2DX3D(Test_API_Matmul2DX2D):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = (4, 3)
        self.shape_y = (5, 3, 4)
        self.trans_x = False
        self.trans_y = False


class Test_API_Matmul2DX3D_transX(Test_API_Matmul2DX2D):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = (3, 4)
        self.shape_y = (5, 3, 4)
        self.trans_x = True
        self.trans_y = False


class Test_API_Matmul2DX3D_transY(Test_API_Matmul2DX2D):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = (4, 3)
        self.shape_y = (5, 4, 3)
        self.trans_x = False
        self.trans_y = True


class Test_API_Matmul2DX3D_transXY(Test_API_Matmul2DX2D):
    def init_shape(self):
        self.dtype = "float64"
        self.shape_x = (3, 4)
        self.shape_y = (5, 4, 3)
        self.trans_x = True
        self.trans_y = True


if __name__ == "__main__":
    unittest.main()
