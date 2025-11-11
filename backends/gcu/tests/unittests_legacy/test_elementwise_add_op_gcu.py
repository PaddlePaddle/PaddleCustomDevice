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

import unittest

import numpy as np
from tests.op_test import OpTest, skip_check_grad_ci

import paddle
from paddle import base
from paddle.base import core


class TestElementwiseAddOp(OpTest):
    def init_kernel_type(self):
        self.use_mkldnn = False

    def setUp(self):
        self.op_type = "elementwise_add"
        self.python_api = paddle.add
        self.public_python_api = paddle.add
        self.init_dtype()
        self.set_device()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(self.x),
            "Y": OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.attrs = {"axis": self.axis, "use_mkldnn": self.use_mkldnn}
        self.outputs = {"Out": self.out}

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def check_dygraph(self):
        return not self.use_mkldnn and self.axis == -1

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(
            self.place, ["X", "Y"], "Out", max_relative_error=0.01
        )

    def test_check_grad_ingore_x(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(
            self.place, ["Y"], "Out", no_grad_set=set("X"), max_relative_error=0.01
        )

    def test_check_grad_ingore_y(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(
            self.place, ["X"], "Out", no_grad_set=set("Y"), max_relative_error=0.01
        )

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.add(self.x, self.y)

    def init_dtype(self):
        self.dtype = np.float32

    def init_axis(self):
        self.axis = -1


class TestElementwiseAddOp_ZeroDim1(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, []).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, []).astype(self.dtype)
        self.out = np.add(self.x, self.y)


class TestElementwiseAddOp_ZeroDim2(TestElementwiseAddOp_ZeroDim1):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, []).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.add(self.x, self.y)


class TestElementwiseAddOp_ZeroDim3(TestElementwiseAddOp_ZeroDim1):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, []).astype(self.dtype)
        self.out = np.add(self.x, self.y)


class TestFP16ElementwiseAddOp(TestElementwiseAddOp):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


@skip_check_grad_ci(reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestElementwiseAddOp_scalar(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)
        self.out = self.x + self.y


@skip_check_grad_ci(reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestFP16ElementwiseAddOp_scalar(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)
        self.out = self.x + self.y


@skip_check_grad_ci(reason="[skip shape check] Use y_shape(1,1) to test broadcast.")
class TestElementwiseAddOp_scalar2(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(1, 1).astype(self.dtype)
        self.out = self.x + self.y


@skip_check_grad_ci(reason="[skip shape check] Use y_shape(1,1) to test broadcast.")
class TestFP16ElementwiseAddOp_scalar2(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(1, 1).astype(self.dtype)
        self.out = self.x + self.y


class TestElementwiseAddOp_Vector(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.random((100,)).astype(self.dtype)
        self.y = np.random.random((100,)).astype(self.dtype)
        self.out = np.add(self.x, self.y)


class TestFP16ElementwiseAddOp_Vector(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.random((100,)).astype(self.dtype)
        self.y = np.random.random((100,)).astype(self.dtype)
        self.out = np.add(self.x, self.y)


class TestElementwiseAddOp_broadcast_0(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(100, 2, 3).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x + self.y.reshape(100, 1, 1)
        self.python_api = paddle.add

    def init_axis(self):
        self.axis = 0

    def if_check_prim(self):
        self.check_prim = False


@skip_check_grad_ci(reason="the numerical method is not accurate enough on fp16")
class TestFP16ElementwiseAddOp_broadcast_0(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 100).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 1, 100)
        self.python_api = paddle.add

    def init_axis(self):
        self.axis = -1


class TestElementwiseAddOp_rowwise_add_0(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 10, 12).astype(self.dtype)
        self.y = np.random.rand(10, 12).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 10, 12)

    def init_axis(self):
        self.axis = 1


@skip_check_grad_ci(reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestFP16ElementwiseAddOp_rowwise_add_0(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(100, 1).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)
        self.out = self.x + self.y


class TestElementwiseAddOp_channelwise_add(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(100, 2, 3).astype(self.dtype)
        self.y = np.random.rand(100, 1, 1).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = -1


class TestFP16ElementwiseAddOp_channelwise_add(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(100, 2, 3).astype(self.dtype)
        self.y = np.random.rand(100, 1, 1).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = -1


class TestElementwiseAddOp_commonuse_add1(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 100).astype(self.dtype)
        self.y = np.random.rand(1, 1, 100).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = -1


class TestElementwiseFP16AddOp_commonuse_add1(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 100).astype(self.dtype)
        self.y = np.random.rand(1, 1, 100).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = -1


class TestElementwiseAddOp_commonuse_add2(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(10, 3, 1, 4).astype(self.dtype)
        self.y = np.random.rand(10, 1, 12, 1).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = -1


class TestElementwiseAddOp_xsize_lessthan_ysize_add(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(10, 12).astype(self.dtype)
        self.y = np.random.rand(2, 2, 10, 12).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = 2


class TestElementwiseAddOp_same_shape_ysize_large(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(10, 1, 12).astype(self.dtype)
        self.y = np.random.rand(10, 2, 12).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = 0


class TestAddApi(unittest.TestCase):
    def _executed_api(self, x, y, name=None):
        return paddle.add(x, y, name)

    def test_name(self):
        with base.program_guard(base.Program()):
            x = paddle.static.data(name="x", shape=[2, 3], dtype="float32")
            y = paddle.static.data(name="y", shape=[2, 3], dtype="float32")

            y_1 = self._executed_api(x, y, name="add_res")
            self.assertEqual(("add_res" in y_1.name), True)

    def test_declarative(self):
        paddle.enable_static()
        with base.program_guard(base.Program()):

            def gen_data():
                return {
                    "x": np.array([2, 3, 4]).astype("float32"),
                    "y": np.array([1, 5, 2]).astype("float32"),
                }

            x = paddle.static.data(name="x", shape=[3], dtype="float32")
            y = paddle.static.data(name="y", shape=[3], dtype="float32")
            z = self._executed_api(x, y)

            place = paddle.CustomPlace("gcu", 0)
            exe = base.Executor(place)
            z_value = exe.run(feed=gen_data(), fetch_list=[z.name])
            z_expected = np.array([3.0, 8.0, 6.0])
            self.assertEqual((z_value == z_expected).all(), True)
        paddle.disable_static()

    def test_dygraph(self):
        paddle.disable_static()
        paddle.set_device("gcu")
        with base.dygraph.guard():
            np_x = np.array([2, 3, 4]).astype("float32")
            np_y = np.array([1, 5, 2]).astype("float32")
            x = paddle.to_tensor(np_x)
            y = paddle.to_tensor(np_y)
            z = self._executed_api(x, y)
            np_z = z.numpy()
            z_expected = np.array([3.0, 8.0, 6.0])
            self.assertEqual((np_z == z_expected).all(), True)
        paddle.enable_static()


class TestAddInplaceApi(TestAddApi):
    def _executed_api(self, x, y, name=None):
        return x.add_(y, name)


class TestAddInplaceBroadcastSuccess(unittest.TestCase):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 3, 4).astype("float")
        self.y_numpy = np.random.rand(3, 4).astype("float")

    def test_broadcast_success(self):
        paddle.disable_static()
        paddle.set_device("gcu")
        self.init_data()
        x = paddle.to_tensor(self.x_numpy)
        y = paddle.to_tensor(self.y_numpy)
        inplace_result = x.add_(y)
        numpy_result = self.x_numpy + self.y_numpy
        self.assertEqual((inplace_result.numpy() == numpy_result).all(), True)
        paddle.enable_static()


class TestAddInplaceBroadcastSuccess2(TestAddInplaceBroadcastSuccess):
    def init_data(self):
        self.x_numpy = np.random.rand(1, 2, 3, 1).astype("float")
        self.y_numpy = np.random.rand(3, 1).astype("float")


class TestAddInplaceBroadcastSuccess3(TestAddInplaceBroadcastSuccess):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 3, 1, 5).astype("float")
        self.y_numpy = np.random.rand(1, 3, 1, 5).astype("float")


class TestAddInplaceBroadcastError(unittest.TestCase):
    def init_data(self):
        self.x_numpy = np.random.rand(3, 4).astype("float")
        self.y_numpy = np.random.rand(2, 3, 4).astype("float")

    def test_broadcast_errors(self):
        paddle.disable_static()
        paddle.set_device("gcu")
        self.init_data()
        x = paddle.to_tensor(self.x_numpy)
        y = paddle.to_tensor(self.y_numpy)

        def broadcast_shape_error():
            x.add_(y)

        self.assertRaises(ValueError, broadcast_shape_error)
        paddle.enable_static()


class TestAddInplaceBroadcastError2(TestAddInplaceBroadcastError):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 1, 4).astype("float")
        self.y_numpy = np.random.rand(2, 3, 4).astype("float")


class TestAddInplaceBroadcastError3(TestAddInplaceBroadcastError):
    def init_data(self):
        self.x_numpy = np.random.rand(5, 2, 1, 4).astype("float")
        self.y_numpy = np.random.rand(2, 3, 4).astype("float")


class TestBoolAddFloatElementwiseAddop(unittest.TestCase):
    def test_static_add(self):
        paddle.enable_static()
        a = 1.5
        b = paddle.full([4, 5, 6], True, dtype="bool")
        c = a + b
        self.assertTrue(c.dtype == core.VarDesc.VarType.FP32)
        paddle.enable_static()

    def test_dygraph_add(self):
        paddle.disable_static()
        paddle.set_device("gcu")
        a = 1.5
        b = paddle.full([2], True, dtype="bool")
        # special case: scalar + tensor(bool)
        c = a + b
        self.assertTrue(c.dtype == core.VarDesc.VarType.FP32)

        np_a = np.random.random((2, 3, 4)).astype(np.float64)
        np_b = np.random.random((2, 3, 4)).astype(np.float64)

        tensor_a = paddle.to_tensor(np_a, dtype="float32")
        tensor_b = paddle.to_tensor(np_b, dtype="float32")

        # normal case: tensor + tensor
        expect_out = np_a + np_b
        actual_out = tensor_a + tensor_b
        np.testing.assert_allclose(actual_out, expect_out)

        # normal case: tensor + scalar
        expect_out = np_a + 1
        actual_out = tensor_a + 1
        np.testing.assert_allclose(actual_out, expect_out)

        # normal case: scalar + tenor
        expect_out = 1 + np_a
        actual_out = 1 + tensor_a
        np.testing.assert_allclose(actual_out, expect_out)

        paddle.enable_static()


class TestElementwiseAddop1(unittest.TestCase):
    def test_dygraph_add(self):
        paddle.disable_static()
        paddle.set_device("gcu")

        np_a = np.random.random((2, 3, 4)).astype(np.float32)
        np_b = np.random.random((2, 3, 4)).astype(np.float32)

        tensor_a = paddle.to_tensor(np_a, dtype="float32")
        tensor_b = paddle.to_tensor(np_b, dtype="float32")

        # normal case: nparray + tenor
        expect_out = np_a + np_b
        actual_out = np_a + tensor_b
        np.testing.assert_allclose(actual_out, expect_out)

        # normal case: tensor + nparray
        actual_out = tensor_a + np_b
        np.testing.assert_allclose(actual_out, expect_out)

        paddle.enable_static()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
