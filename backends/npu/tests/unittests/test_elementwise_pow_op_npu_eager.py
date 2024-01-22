# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from tests.op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float

from npu_utils import check_soc_version

SEED = 2021


def ComputeGrad(x, y, out, axis):
    grad = 1 / out.size
    shape_x = x.shape
    shape_y = y.shape
    shape_out = out.shape
    reduce_axes_x = []
    reduce_axes_y = []

    if shape_x != shape_out:
        if len(shape_x) < len(shape_out):
            src_axis = axis
        else:
            src_axis = 0

        for ax in range(len(shape_out)):
            if (ax < src_axis or ax >= src_axis + len(shape_x)) or (
                shape_out[ax] > 1 and shape_x[ax - src_axis] == 1
            ):
                reduce_axes_x.append(ax)

    if shape_y != shape_out:
        if len(shape_y) < len(shape_out):
            src_axis = axis
        else:
            src_axis = 0

        for ax in range(len(shape_out)):
            if (ax < src_axis or ax >= src_axis + len(shape_y)) or (
                shape_out[ax] > 1 and shape_y[ax - src_axis] == 1
            ):
                reduce_axes_y.append(ax)

    if len(reduce_axes_x) > 0:
        for i in reduce_axes_x:
            x = np.expand_dims(x, axis=i)

    if len(reduce_axes_y) > 0:
        for i in reduce_axes_y:
            y = np.expand_dims(y, axis=i)

    dx = y * np.power(x, y - 1) * grad
    dy = np.log(x) * np.power(x, y) * grad

    if len(reduce_axes_x) > 0:
        for i, element in enumerate(reduce_axes_x):
            dx = np.add.reduce(dx, element - i)

    if len(reduce_axes_y) > 0:
        for i, element in enumerate(reduce_axes_y):
            dy = np.add.reduce(dy, element - i)

    return dx, dy


class TestElementwisePowBf16(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_pow"
        self.place = paddle.CustomPlace("npu", 0)

        self.init_dtype()
        self.init_input_output()
        self.init_axis()

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(convert_float_to_uint16(self.x)),
            "Y": OpTest.np_dtype_to_base_dtype(convert_float_to_uint16(self.y)),
        }
        self.attrs = {"axis": self.axis}
        self.outputs = {"Out": convert_uint16_to_float(self.out)}

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)

    def init_axis(self):
        self.axis = -1

    def init_input_output(self):
        np.random.seed(SEED)
        self.x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        self.out = np.power(self.x, self.y)

    @check_soc_version
    def test_check_grad_normal(self):
        dx, dy = ComputeGrad(self.x, self.y, self.out, self.axis)
        self.check_grad_with_place(
            self.place, ["X", "Y"], "Out", user_defined_grads=[dx, dy]
        )


if __name__ == "__main__":
    unittest.main()
