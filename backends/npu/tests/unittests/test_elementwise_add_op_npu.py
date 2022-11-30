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

import os
import numpy as np
import unittest

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from tests.op_test import OpTest, skip_check_grad_ci

paddle.enable_static()


class TestElementwiseAddOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_add"
        self.place = paddle.CustomPlace("npu", 0)
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()

        self.inputs = {
            "X": OpTest.np_dtype_to_fluid_dtype(self.x),
            "Y": OpTest.np_dtype_to_fluid_dtype(self.y),
        }
        self.attrs = {"axis": self.axis, "use_mkldnn": self.use_mkldnn}
        self.outputs = {"Out": self.out}

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_kernel_type(self):
        self.use_mkldnn = False

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.add(self.x, self.y)

    def init_dtype(self):
        self.dtype = np.float32

    def init_axis(self):
        self.axis = -1

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        if self.dtype == np.int64:
            return

        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place,
                ["X", "Y"],
                "Out",
                max_relative_error=0.15,
            )
        else:
            self.check_grad_with_place(
                self.place,
                ["X", "Y"],
                "Out",
                max_relative_error=0.006,
            )

    def test_check_grad_ingore_x(self):
        if self.dtype == np.int64:
            return

        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place,
                ["Y"],
                "Out",
                no_grad_set=set("X"),
                max_relative_error=0.92,
            )
        else:
            self.check_grad_with_place(
                self.place,
                ["Y"],
                "Out",
                no_grad_set=set("X"),
                max_relative_error=0.006,
            )

    def test_check_grad_ingore_y(self):
        if self.dtype == np.int64:
            return

        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place,
                ["X"],
                "Out",
                no_grad_set=set("Y"),
                max_relative_error=0.8,
            )
        else:
            self.check_grad_with_place(
                self.place,
                ["X"],
                "Out",
                no_grad_set=set("Y"),
                max_relative_error=0.006,
            )


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestFP16ElementwiseAddOp(TestElementwiseAddOp):
    def init_dtype(self):
        self.dtype = np.float16


class TestINT64ElementwiseAddOp(TestElementwiseAddOp):
    def init_dtype(self):
        self.dtype = np.int64


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestElementwiseAddOp_scalar(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)
        self.out = self.x + self.y


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestFP16ElementwiseAddOp_scalar(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)
        self.out = self.x + self.y


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1,1) to test broadcast.")
class TestElementwiseAddOp_scalar2(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(1, 1).astype(self.dtype)
        self.out = self.x + self.y


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1,1) to test broadcast.")
class TestFP16ElementwiseAddOp_scalar2(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(1, 1).astype(self.dtype)
        self.out = self.x + self.y


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestAddAPI(unittest.TestCase):
    def test_name(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x", shape=[2, 3], dtype="float32")
            y = paddle.static.data(name="y", shape=[2, 3], dtype="float32")

            y_1 = paddle.add(x, y, name="add_res")
            self.assertEqual(("add_res" in y_1.name), True)

    def test_static(self):
        with paddle.static.program_guard(paddle.static.Program()):

            x_np = np.array([2, 3, 4]).astype("float32")
            y_np = np.array([1, 5, 2]).astype("float32")

            x = paddle.static.data(name="x", shape=[3], dtype="float32")
            y = paddle.static.data(name="y", shape=[3], dtype="float32")

            x_reshape = paddle.reshape(x, [3, 1])
            y_reshape = paddle.reshape(y, [3, 1])
            z = paddle.add(x_reshape, y_reshape)
            z = paddle.reshape(z, shape=[3])

            place = paddle.CustomPlace("npu", 0)
            exe = paddle.static.Executor(place)
            x_value, y_value, z_value = exe.run(
                feed={"x": x_np, "y": y_np}, fetch_list=[x, y, z]
            )

            z_expected = np.array([3.0, 8.0, 6.0])
            self.assertEqual(
                (x_value == x_np).all(),
                True,
                msg="x_value = {}, but expected {}".format(x_value, x_np),
            )
            self.assertEqual(
                (y_value == y_np).all(),
                True,
                msg="y_value = {}, but expected {}".format(y_value, y_np),
            )
            self.assertEqual(
                (z_value == z_expected).all(),
                True,
                msg="z_value = {}, but expected {}".format(z_value, z_expected),
            )


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestAddError(unittest.TestCase):
    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            # the input of elementwise_add must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], paddle.CustomPlace("npu", 0)
            )
            y1 = fluid.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], paddle.CustomPlace("npu", 0)
            )
            self.assertRaises(TypeError, paddle.add, x1, y1)

            # the input dtype must be float16 or float32 or float64 or int32 or int64
            x2 = paddle.static.data(name="x2", shape=[3, 4, 5, 6], dtype="uint8")
            y2 = paddle.static.data(name="y2", shape=[3, 4, 5, 6], dtype="uint8")
            self.assertRaises(TypeError, paddle.add, x2, y2)


class TestElementwiseAddOp_Vector(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.random((100,)).astype(self.dtype)
        self.y = np.random.random((100,)).astype(self.dtype)
        self.out = np.add(self.x, self.y)


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
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

    def init_axis(self):
        self.axis = 0


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestFP16ElementwiseAddOp_broadcast_0(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(100, 2, 3).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x + self.y.reshape(100, 1, 1)

    def init_axis(self):
        self.axis = 0


class TestElementwiseAddOp_broadcast_1(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 100, 3).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 100, 1)

    def init_axis(self):
        self.axis = 1


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestFP16ElementwiseAddOp_broadcast_1(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 100, 3).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 100, 1)

    def init_axis(self):
        self.axis = 1


class TestElementwiseAddOp_broadcast_2(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 100).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 1, 100)


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestFP16ElementwiseAddOp_broadcast_2(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 100).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 1, 100)


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestElementwiseAddOp_broadcast_3(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 10, 12, 1).astype(self.dtype)
        self.y = np.random.rand(10, 12).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 10, 12, 1)

    def init_axis(self):
        self.axis = 1


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestFP16ElementwiseAddOp_broadcast_3(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 10, 12, 3).astype(self.dtype)
        self.y = np.random.rand(10, 12).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 10, 12, 1)

    def init_axis(self):
        self.axis = 1


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestElementwiseAddOp_broadcast_4(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(100, 2, 1, 2).astype(self.dtype)
        self.y = np.random.rand(100, 1).astype(self.dtype)
        self.out = self.x + self.y.reshape(100, 1, 1, 1)

    def init_axis(self):
        self.axis = 0


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestFP16ElementwiseAddOp_broadcast_4(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(100, 2, 1, 2).astype(self.dtype)
        self.y = np.random.rand(100, 1).astype(self.dtype)
        self.out = self.x + self.y.reshape(100, 1, 1, 1)

    def init_axis(self):
        self.axis = 0


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestElementwiseAddOp_broadcast_5(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(10, 3, 12).astype(self.dtype)
        self.y = np.random.rand(10, 1, 12).astype(self.dtype)
        self.out = self.x + self.y


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestFP16ElementwiseAddOp_broadcast_5(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(10, 3, 12).astype(self.dtype)
        self.y = np.random.rand(10, 1, 12).astype(self.dtype)
        self.out = self.x + self.y


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestElementwiseAddOp_broadcast_6(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 12, 3, 5).astype(self.dtype)
        self.y = np.random.rand(2, 12, 1, 5).astype(self.dtype)
        self.out = self.x + self.y


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestElementwiseAddOp_broadcast_7(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(1, 1, 20, 5).astype(self.dtype)
        self.y = np.random.rand(20, 5, 1, 1).astype(self.dtype)
        self.out = self.x + self.y


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestFP16ElementwiseAddOp_broadcast_6(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 12, 3, 5).astype(self.dtype)
        self.y = np.random.rand(2, 12, 1, 5).astype(self.dtype)
        self.out = self.x + self.y


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestElementwiseAddOp_rowwise_add_0(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 10, 12).astype(self.dtype)
        self.y = np.random.rand(10, 12).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 10, 12)

    def init_axis(self):
        self.axis = 1


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestFP16ElementwiseAddOp_rowwise_add_0(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 10, 12).astype(self.dtype)
        self.y = np.random.rand(10, 12).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 10, 12)

    def init_axis(self):
        self.axis = 1


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestElementwiseAddOp_rowwise_add_1(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(100, 1).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 1)

    def init_axis(self):
        self.axis = 1


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestFP16ElementwiseAddOp_rowwise_add_1(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(100, 1).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 1)

    def init_axis(self):
        self.axis = 1


class TestElementwiseAddOp_channelwise_add(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(100, 2, 3).astype(self.dtype)
        self.y = np.random.rand(100, 1, 1).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = -1


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestFP16ElementwiseAddOp_channelwise_add(TestFP16ElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(100, 2, 3).astype(self.dtype)
        self.y = np.random.rand(100, 1, 1).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = -1


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestElementwiseAddOp_commonuse_add1(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 100).astype(self.dtype)
        self.y = np.random.rand(1, 1, 100).astype(self.dtype)
        self.out = self.x + self.y

    def init_axis(self):
        self.axis = -1


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
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


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestElementwiseAddOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # the input of elementwise_add must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], paddle.CustomPlace("npu", 0)
            )
            y1 = fluid.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], paddle.CustomPlace("npu", 0)
            )
            self.assertRaises(TypeError, fluid.layers.elementwise_add, x1, y1)

            # the input dtype of elementwise_add must be float16 or float32 or float64 or int32 or int64
            # float16 only can be set on GPU place
            x2 = fluid.layers.data(name="x2", shape=[3, 4, 5, 6], dtype="uint8")
            y2 = fluid.layers.data(name="y2", shape=[3, 4, 5, 6], dtype="uint8")
            self.assertRaises(TypeError, fluid.layers.elementwise_add, x2, y2)


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestAddApi(unittest.TestCase):
    def _executed_api(self, x, y, name=None):
        return paddle.add(x, y, name)

    def test_name(self):
        with fluid.program_guard(fluid.Program()):
            x = fluid.data(name="x", shape=[2, 3], dtype="float32")
            y = fluid.data(name="y", shape=[2, 3], dtype="float32")

            y_1 = self._executed_api(x, y, name="add_res")
            self.assertEqual(("add_res" in y_1.name), True)

    def test_declarative(self):
        with fluid.program_guard(fluid.Program()):

            def gen_data():
                return {
                    "x": np.array([2, 3, 4]).astype("float32"),
                    "y": np.array([1, 5, 2]).astype("float32"),
                }

            x = fluid.data(name="x", shape=[3], dtype="float32")
            y = fluid.data(name="y", shape=[3], dtype="float32")
            z = self._executed_api(x, y)

            place = paddle.CustomPlace("npu", 0)
            exe = fluid.Executor(place)
            z_value = exe.run(feed=gen_data(), fetch_list=[z.name])
            z_expected = np.array([3.0, 8.0, 6.0])
            self.assertEqual((z_value == z_expected).all(), True)

    def test_dygraph(self):
        with fluid.dygraph.guard(paddle.CustomPlace("npu", 0)):
            np_x = np.array([2, 3, 4]).astype("float32")
            np_y = np.array([1, 5, 2]).astype("float32")
            x = fluid.dygraph.to_variable(np_x)
            y = fluid.dygraph.to_variable(np_y)
            z = self._executed_api(x, y)
            np_z = z.numpy()
            z_expected = np.array([3.0, 8.0, 6.0])
            self.assertEqual((np_z == z_expected).all(), True)


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestAddInplaceApi(TestAddApi):
    def _executed_api(self, x, y, name=None):
        return x.add_(y, name)


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestAddInplaceBroadcastSuccess(unittest.TestCase):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 3, 4).astype("float")
        self.y_numpy = np.random.rand(3, 4).astype("float")

    def test_broadcast_success(self):
        paddle.disable_static(place=paddle.CustomPlace("npu", 0))
        self.init_data()
        x = paddle.to_tensor(self.x_numpy)
        y = paddle.to_tensor(self.y_numpy)
        inplace_result = x.add_(y)
        numpy_result = self.x_numpy + self.y_numpy
        self.assertEqual((inplace_result.numpy() == numpy_result).all(), True)
        paddle.enable_static()


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestAddInplaceBroadcastSuccess2(TestAddInplaceBroadcastSuccess):
    def init_data(self):
        self.x_numpy = np.random.rand(1, 2, 3, 1).astype("float")
        self.y_numpy = np.random.rand(3, 1).astype("float")


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestAddInplaceBroadcastSuccess3(TestAddInplaceBroadcastSuccess):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 3, 1, 5).astype("float")
        self.y_numpy = np.random.rand(1, 3, 1, 5).astype("float")


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestAddInplaceBroadcastError(unittest.TestCase):
    def init_data(self):
        self.x_numpy = np.random.rand(3, 4).astype("float")
        self.y_numpy = np.random.rand(2, 3, 4).astype("float")

    def test_broadcast_errors(self):
        paddle.disable_static(place=paddle.CustomPlace("npu", 0))
        self.init_data()
        x = paddle.to_tensor(self.x_numpy)
        y = paddle.to_tensor(self.y_numpy)

        def broadcast_shape_error():
            x.add_(y)

        self.assertRaises(ValueError, broadcast_shape_error)
        paddle.enable_static()


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestAddInplaceBroadcastError2(TestAddInplaceBroadcastError):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 1, 4).astype("float")
        self.y_numpy = np.random.rand(2, 3, 4).astype("float")


@unittest.skipIf(os.getenv('FLAGS_use_graph_engine', None) == '1', "cann error")
class TestAddInplaceBroadcastError3(TestAddInplaceBroadcastError):
    def init_data(self):
        self.x_numpy = np.random.rand(5, 2, 1, 4).astype("float")
        self.y_numpy = np.random.rand(2, 3, 4).astype("float")


class TestAddAPIWithNPUStroageFormat(unittest.TestCase):
    def setUp(self):
        self.shape_x = [4, 6, 24, 24]
        self.shape_y = [1, 6, 1, 1]
        self.x = np.random.random(self.shape_x).astype(np.float32)
        self.y = np.random.random(self.shape_y).astype(np.float32)
        self.format = 3  # ACL_FORMAT_NC1HWC0 = 3
        self.place = paddle.CustomPlace("npu", 0)

    def test_api_static(self):
        paddle.enable_static()

        main_program = paddle.static.default_main_program()
        startup_program = paddle.static.default_startup_program()
        with paddle.static.program_guard(main_program, startup_program):
            x_data = paddle.static.data(
                shape=self.shape_x, name="data_x", dtype="float32"
            )
            y_data = paddle.static.data(
                shape=self.shape_y, name="data_y", dtype="float32"
            )
            out_expect = paddle.add(x=x_data, y=y_data)

            x_format = paddle.incubate._npu_identity(x=x_data, format=self.format)
            y_format = paddle.incubate._npu_identity(x=y_data, format=self.format)
            out_format = paddle.add(x=x_format, y=y_format)
            out_actual = paddle.incubate._npu_identity(x=out_format, format=-1)
        exe = paddle.static.Executor()
        exe.run(startup_program)
        result = exe.run(
            main_program,
            feed={x_data.name: self.x, y_data.name: self.y},
            fetch_list=[out_expect, out_actual],
        )

        np.testing.assert_allclose(result[0], result[1], rtol=1e-08)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)

        # fwd and bwd with normal format
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        x.stop_gradient = False
        y.stop_gradient = False
        out_expect = paddle.add(x, y)
        loss = out_expect.sum()
        loss.backward()
        x_grad_expect = x.grad
        y_grad_expect = y.grad

        # fwd and bwd with storage format
        x_format = paddle.incubate._npu_identity(x, self.format)
        y_format = paddle.incubate._npu_identity(y, self.format)
        x_format.stop_gradient = False
        y_format.stop_gradient = False
        out_format = paddle.add(x_format, y_format)
        loss_format = out_format.sum()
        loss_format.backward()
        out_actual = paddle.incubate._npu_identity(out_format, -1)
        x_grad_actual = paddle.incubate._npu_identity(x_format.grad, -1)
        y_grad_actual = paddle.incubate._npu_identity(y_format.grad, -1)

        # compare results
        np.testing.assert_allclose(out_expect.numpy(), out_actual.numpy(), rtol=1e-08)
        np.testing.assert_allclose(
            x_grad_expect.numpy(), x_grad_actual.numpy(), rtol=1e-08
        )
        np.testing.assert_allclose(
            y_grad_expect.numpy(), y_grad_actual.numpy(), rtol=1e-08
        )
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
