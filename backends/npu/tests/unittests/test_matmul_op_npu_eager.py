# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import os

select_npu = os.environ.get("FLAGS_selected_npus", 0)

import paddle

from tests.op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float

from npu_utils import check_soc_version

SEED = 2021


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


class TestMatMulOpBf16Case(OpTest):
    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", select_npu)

    def config(self):
        self.x_shape = (100,)
        self.y_shape = (100,)
        self.trans_x = False
        self.trans_y = False

    def init_kernel_type(self):
        self.dtype = np.uint16

    def setUp(self):
        self.set_npu()
        self.init_kernel_type()
        self.config()
        self.op_type = "matmul_v2"
        x = np.random.random(self.x_shape).astype(np.float32)
        y = np.random.random(self.y_shape).astype(np.float32)
        # -0.1 ~ 0.1
        x = -0.1 + 0.2 * x
        y = -0.1 + 0.2 * y
        np_uint16_x = convert_float_to_uint16(x)
        np_uint16_y = convert_float_to_uint16(y)
        np_uint16_to_fp32_x = convert_uint16_to_float(np_uint16_x)
        np_uint16_to_fp32_y = convert_uint16_to_float(np_uint16_y)
        result = reference_matmul(
            np_uint16_to_fp32_x, np_uint16_to_fp32_y, self.trans_x, self.trans_y
        )
        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(np_uint16_x),
            "Y": OpTest.np_dtype_to_base_dtype(np_uint16_y),
        }
        self.attrs = {"trans_x": self.trans_x, "trans_y": self.trans_y}
        self.outputs = {"Out": result}

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place, atol=0.004)

    @check_soc_version
    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], ["Out"])


if __name__ == "__main__":
    unittest.main()
