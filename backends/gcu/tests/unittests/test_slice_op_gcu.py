#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from tests.op_test import OpTest
import paddle.base as base
import paddle

paddle.enable_static()


class TestSliceOp(OpTest):
    def setUp(self):
        self.op_type = "slice"
        self.set_device()
        self.config()
        self.inputs = {"Input": self.input}
        self.outputs = {"Out": self.out}
        self.attrs = {
            "axes": self.axes,
            "starts": self.starts,
            "ends": self.ends,
            "infer_flags": self.infer_flags,
        }

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1:3, 0:3, 2:4, :]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place, ["Input"], "Out", max_relative_error=0.006
        )


class TestCase1(TestSliceOp):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [-3, 0, 2]
        self.ends = [3, 100, -1]
        self.axes = [0, 1, 2]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[-3:3, 0:100, 2:-1, :]


class TestCase2(TestSliceOp):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [-3, 0, 2]
        self.ends = [3, 100, -1]
        self.axes = [0, 1, 3]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[-3:3, 0:100, :, 2:-1]


class TestSliceOp_decs_dim(OpTest):
    def setUp(self):
        self.op_type = "slice"
        self.set_device()
        self.config()
        self.inputs = {"Input": self.input}
        self.outputs = {"Out": self.out}
        self.attrs = {
            "axes": self.axes,
            "starts": self.starts,
            "ends": self.ends,
            "infer_flags": self.infer_flags,
            "decrease_axis": self.decrease_axis,
        }

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1, 0:3, 2:4, :]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place, ["Input"], "Out", max_relative_error=0.006
        )


class TestSliceApiEager(unittest.TestCase):
    def test_slice_api(self):
        paddle.set_device("gcu")
        with paddle.base.dygraph.guard():
            a = paddle.rand(shape=[4, 5, 6], dtype="float32")
            a.stop_gradient = False
            axes = [0, 1, 2]
            starts = [-3, 0, 2]
            ends = [3, 2, 4]
            a_1 = paddle.slice(a, axes=axes, starts=starts, ends=ends)
            a_1.backward()
            grad_truth = paddle.zeros_like(a)
            grad_truth[-3:3, 0:2, 2:4] = 1
            np.testing.assert_array_equal(grad_truth, a.gradient())

            np.testing.assert_allclose(a_1.numpy(), a[-3:3, 0:2, 2:4], rtol=1e-05)


class TestImperativeVarBaseGetItem(unittest.TestCase):
    def test_getitem_with_long(self):
        paddle.set_device("gcu")
        with base.dygraph.guard():
            data = np.random.random((2, 80, 16128)).astype("float32")
            var = base.dygraph.to_variable(data)
            sliced = var[:, 10:, : var.shape[1]]  # var.shape[1] is 80L here
            self.assertEqual(sliced.shape, [2, 70, 80])

            sliced = var[:, var.shape[0] :, var.shape[0] : var.shape[1]]
            self.assertEqual(sliced.shape, [2, 78, 78])

    def test_getitem_with_float(self):
        def test_float_in_slice_item():
            paddle.set_device("gcu")
            with base.dygraph.guard():
                data = np.random.random((2, 80, 16128)).astype("float32")
                var = base.dygraph.to_variable(data)
                sliced = var[:, 1.1:, : var.shape[1]]

        self.assertRaises(Exception, test_float_in_slice_item)

        def test_float_in_index():
            paddle.set_device("gcu")
            with base.dygraph.guard():
                data = np.random.random((2, 80, 16128)).astype("float32")
                var = base.dygraph.to_variable(data)
                sliced = var[1.1]

        self.assertRaises(Exception, test_float_in_index)


if __name__ == "__main__":
    unittest.main()
