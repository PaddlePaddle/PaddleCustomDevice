#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
from tests.op_test import OpTest
import paddle

paddle.enable_static()


class TestElementwiseMaxOp(OpTest):
    def setUp(self):
        self.set_device()
        self.op_type = "elementwise_max"

        self.init_dtype()
        self.init_input_output()
        self.init_axis()

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(self.x),
            "Y": OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.attrs = {"axis": self.axis}
        self.outputs = {"Out": self.out}

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def init_input_output(self):
        # If x and y have the same value, the max() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        sgn = np.random.choice([-1, 1], [13, 17]).astype(self.dtype)
        self.y = self.x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.maximum(self.x, self.y)

    def init_axis(self):
        self.axis = -1

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place, ["X", "Y"], "Out", max_relative_error=0.5
            )
        else:
            self.check_grad_with_place(
                self.place,
                ["X", "Y"],
                "Out",
            )

    def test_check_grad_ingore_x(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place, ["Y"], "Out", no_grad_set=set("X"), max_relative_error=0.9
            )
        else:
            self.check_grad_with_place(
                self.place,
                ["Y"],
                "Out",
                no_grad_set=set("X"),
            )

    def test_check_grad_ingore_y(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place, ["X"], "Out", no_grad_set=set("Y"), max_relative_error=0.1
            )
        else:
            self.check_grad_with_place(
                self.place,
                ["X"],
                "Out",
                no_grad_set=set("Y"),
            )


class TestElementwiseMaxOp_FP16(TestElementwiseMaxOp):
    def init_dtype(self):
        self.dtype = np.float16


class TestElementwiseMaxOp_broadcast_1(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (2, 3, 4, 5)).astype(self.dtype)
        sgn = np.random.choice([-1, 1], (2, 3, 1, 5)).astype(self.dtype)
        self.y = self.x + sgn * np.random.uniform(1, 2, (2, 3, 1, 5)).astype(self.dtype)
        self.out = np.maximum(self.x, self.y)


class TestElementwiseMaxOp_broadcast_2(TestElementwiseMaxOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.5, 1, (2, 3, 4, 5)).astype(self.dtype)
        sgn = np.random.choice([-1, 1], (2, 3, 1, 1)).astype(self.dtype)
        self.y = self.x + sgn * np.random.uniform(1, 2, (2, 3, 1, 1)).astype(self.dtype)
        self.out = np.maximum(self.x, self.y)


if __name__ == "__main__":
    unittest.main()
