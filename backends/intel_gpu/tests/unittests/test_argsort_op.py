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
from op_test import OpTest
import paddle
import paddle.base as base
import numpy as np
import six
import paddle.base.core as core

from paddle.base.framework import Program, grad_var_name
from paddle.base.executor import Executor

paddle.enable_static()

np.random.seed(123)


def get_places(self):
    return [paddle.CustomPlace("intel_gpu", 0)]


OpTest._get_places = get_places


class PyArgsort(object):
    def __init__(self, input_shape, axis, descending, dtype):
        self.x = np.random.random(input_shape).astype(dtype)
        self.label = np.random.random(input_shape).astype(dtype)
        if axis < 0:
            self.axis = axis + len(self.x.shape)
        else:
            self.axis = axis
        self.descending = descending

    def forward(self):
        if self.descending:
            self.indices = np.flip(
                np.argsort(self.x, kind="quicksort", axis=self.axis), self.axis
            )
            self.sorted_x = np.flip(
                np.sort(self.x, kind="quicksort", axis=self.axis), self.axis
            )
        else:
            self.indices = np.argsort(self.x, kind="quicksort", axis=self.axis)
            self.sorted_x = np.sort(self.x, kind="quicksort", axis=self.axis)
        self.loss = self.sorted_x * self.label
        self.loss = np.sum(self.loss)
        out = (
            np.array(self.indices, dtype=self.indices.dtype),
            np.array(self.sorted_x, dtype=self.sorted_x.dtype),
            np.array([self.loss], dtype=self.loss.dtype),
        )
        return out


def create_tensor(np_data, place):
    tensor = core.LoDTensor()
    tensor.set(np_data, place)
    return tensor


class TestArgsortOpCPU(unittest.TestCase):
    def setup_program(self):
        self.main_program = Program()
        self.startup_program = Program()
        self.init_place()

    def setUp(self):
        self.init_axis()
        self.init_datatype()
        self.init_direction()
        self.init_inputshape()

        self.setup_program()
        self.feed_data_field = {"x", "label"}
        self.grad_data_field = {"x"}

        self.py_argsort = PyArgsort(
            self.input_shape, self.axis, self.descending, self.dtype
        )

        with base.program_guard(self.main_program, self.startup_program):
            x = base.layers.data(name="x", shape=self.input_shape, dtype=self.dtype)
            x.stop_gradient = False
            label = base.layers.data(
                name="label", shape=self.input_shape, dtype=self.dtype
            )
            self.sorted_x, self.index = base.layers.argsort(
                input=x, axis=self.axis, descending=self.descending
            )
            self.sorted_x.stop_gradient = False
            loss = base.layers.elementwise_mul(self.sorted_x, label)
            self.loss = base.layers.reduce_sum(loss)

    def forward(self):
        self.feed_map = {
            x: create_tensor(getattr(self.py_argsort, x), self.place)
            for x in self.feed_data_field
        }
        exe = Executor(self.place)
        out = exe.run(
            self.main_program,
            feed=self.feed_map,
            fetch_list=[self.index, self.sorted_x, self.loss],
        )
        return out

    def backward(self):
        self.feed_map = {
            x: create_tensor(getattr(self.py_argsort, x), self.place)
            for x in self.feed_data_field
        }
        fetch_list = [
            self.main_program.global_block().var(grad_var_name(x))
            for x in self.grad_data_field
        ]
        exe = Executor(self.place)
        out = exe.run(
            self.main_program,
            feed=self.feed_map,
            fetch_list=fetch_list,
            return_numpy=False,
        )
        return out

    # def test_backward(self, numeric_grad_delta=1e-5, max_relative_error=1e-7):
    #     self.check_forward()

    #     with base.program_guard(self.main_program, self.startup_program):
    #         append_backward(self.loss)

    #     ana_grad = [np.array(x) for x in self.backward()]

    #     num_grad = self.get_numerical_gradient(delta=numeric_grad_delta)
    #     self.assert_is_close(
    #         num_grad,
    #         ana_grad,
    #         'x',
    #         max_relative_error=max_relative_error,
    #         msg_prefix="Gradient Check On %s" % str(self.place))

    def check_forward(self):
        pd_outputs = self.forward()
        py_outputs = self.py_argsort.forward()
        for pd_output, py_output in zip(pd_outputs, py_outputs):
            self.assertEqual(pd_output.shape, py_output.shape)
            self.assertTrue(np.allclose(pd_output, py_output, atol=0, equal_nan=False))

    def get_numerical_gradient(self, delta=1e-7):
        if self.dtype == "float16":
            delta = np.array(delta).astype(np.float16)
        feed_list = [getattr(self.py_argsort, x) for x in self.grad_data_field]
        grad_list = [np.zeros_like(x) for x in feed_list]
        for feed, grad in zip(feed_list, grad_list):
            for f, g in np.nditer([feed, grad], op_flags=["readwrite"]):
                o = float(f)
                f[...] = o + delta
                y_pos = self.forward()[2]

                f[...] = o - delta
                y_neg = self.forward()[2]

                f[...] = o
                dout_dfeed = (y_pos - y_neg) / (delta * 2)
                g[...] = dout_dfeed[0]

        return grad_list

    def assert_is_close(
        self, numeric_grads, analytic_grads, names, max_relative_error, msg_prefix
    ):
        for a, b, name in six.moves.zip(numeric_grads, analytic_grads, names):
            abs_a = np.abs(a)
            abs_a[abs_a < 1e-3] = 1

            diff_mat = np.abs(a - b) / abs_a
            max_diff = np.max(diff_mat)

            def err_msg():
                offset = np.argmax(diff_mat > max_relative_error)
                return (
                    "%s error, %s variable %s max gradient diff %f over limit %f, "
                    "the first error element is %d, expected %f, but got %f."
                ) % (
                    "argsort",
                    msg_prefix,
                    name,
                    max_diff,
                    max_relative_error,
                    offset,
                    a.flatten()[offset],
                    b.flatten()[offset],
                )

            self.assertLessEqual(max_diff, max_relative_error, err_msg())

    def init_axis(self):
        self.axis = -1

    def init_datatype(self):
        self.dtype = "float32"

    def init_direction(self):
        self.descending = False

    def init_inputshape(self):
        self.input_shape = (2, 2, 2, 2, 3)

    def init_place(self):
        self.place = core.CustomPlace("intel_gpu", 0)


class TestArgsortOpAxis0CPU(TestArgsortOpCPU):
    def init_axis(self):
        self.axis = 0


class TestArgsortOpAxis1CPU(TestArgsortOpCPU):
    def init_axis(self):
        self.axis = 1


class TestArgsortOpAxis2CPU(TestArgsortOpCPU):
    def init_axis(self):
        self.axis = 2


class TestArgsortOpAxisNeg1CPU(TestArgsortOpCPU):
    def init_axis(self):
        self.axis = -1


class TestArgsortOpAxisNeg2CPU(TestArgsortOpCPU):
    def init_axis(self):
        self.axis = -2


class TestArgsortOpDescendingAxisCPU(TestArgsortOpCPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis0CPU(TestArgsortOpAxis0CPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis1CPU(TestArgsortOpAxis1CPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxis2CPU(TestArgsortOpAxis2CPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxisNeg1CPU(TestArgsortOpAxisNeg1CPU):
    def init_direction(self):
        self.descending = True


class TestArgsortOpDescendingAxisNeg2CPU(TestArgsortOpAxisNeg2CPU):
    def init_direction(self):
        self.descending = True


class TestArgsortErrorOnCPU(unittest.TestCase):
    def setUp(self):
        self.place = core.CustomPlace("intel_gpu", 0)

    def test_error(self):
        def test_base_var_type():
            with base.program_guard(base.Program()):
                x = [1]
                output = base.layers.argsort(input=x)
            self.assertRaises(TypeError, test_base_var_type)

        def test_paddle_var_type():
            with base.program_guard(base.Program()):
                x = [1]
                output = paddle.argsort(input=x)
            self.assertRaises(TypeError, test_paddle_var_type)


class TestArgsort(unittest.TestCase):
    def init(self):
        self.input_shape = [
            10,
        ]
        self.axis = 0

    def setUp(self):
        self.init()
        self.place = core.CustomPlace("intel_gpu", 0)
        self.data = np.random.rand(*self.input_shape).astype("float32")

    def test_api(self):
        paddle.enable_static()
        with base.program_guard(base.Program()):
            input = base.data(name="input", shape=self.input_shape, dtype="float32")

            output = paddle.argsort(input, axis=self.axis)
            output2 = paddle.argsort(input, axis=self.axis, descending=True)

            exe = base.Executor(self.place)
            result, result2 = exe.run(
                feed={"input": self.data}, fetch_list=[output, output2]
            )  # 始终有一半会被擦掉

            np_result = np.argsort(self.data, axis=self.axis)
            self.assertEqual((result == np_result).all(), True)

            np_result2 = np.argsort(-self.data, axis=self.axis).astype("int64")
            self.assertEqual((result2 == np_result2).all(), True)


class TestArgsort2(TestArgsort):
    def init(self):
        self.input_shape = [10, 1]
        self.axis = 0


class TestArgsort3(TestArgsort):
    def init(self):
        self.input_shape = [1, 10]
        self.axis = 1


# TODO(Zhiwei35): support argsort of dims > 3
# class TestArgsort4(TestArgsort):
#     def init(self):
#         self.input_shape = [2, 3, 4]
#         self.axis = 1


class TestStableArgsort(unittest.TestCase):
    def init(self):
        self.input_shape = [
            30,
        ]
        self.axis = 0
        self.data = np.array([100.0, 50.0, 10.0] * 10)

    def setUp(self):
        self.init()
        self.place = core.CustomPlace("intel_gpu", 0)

    def test_api(self):
        with base.program_guard(base.Program()):
            input = paddle.static.data(
                name="input", shape=self.input_shape, dtype="float64"
            )

            output = paddle.argsort(input, axis=self.axis, stable=True)
            output2 = paddle.argsort(
                input, axis=self.axis, descending=True, stable=True
            )

            exe = base.Executor(self.place)
            result, result2 = exe.run(
                feed={"input": self.data}, fetch_list=[output, output2]
            )

            np_result = np.argsort(self.data, axis=self.axis, kind="stable")
            self.assertEqual((result == np_result).all(), True)

            np_result2 = np.argsort(-self.data, axis=self.axis, kind="stable")
            self.assertEqual((result2 == np_result2).all(), True)


class TestStableArgsort2(TestStableArgsort):
    def init(self):
        self.input_shape = [30, 1]
        self.data = np.array([100.0, 50.0, 10.0] * 10).reshape(self.input_shape)
        self.axis = 0


class TestStableArgsort3(TestStableArgsort):
    def init(self):
        self.input_shape = [1, 30]
        self.data = np.array([100.0, 50.0, 10.0] * 10).reshape(self.input_shape)
        self.axis = 1


class TestArgsortImperative(unittest.TestCase):
    def init(self):
        self.input_shape = [
            100,
        ]
        self.axis = 0

    def setUp(self):
        self.init()
        self.input_data = np.random.rand(*self.input_shape)
        self.place = core.CustomPlace("intel_gpu", 0)

    def test_api(self):
        paddle.disable_static(self.place)
        var_x = paddle.to_tensor(self.input_data)
        out = paddle.argsort(var_x, axis=self.axis)
        expect = np.argsort(self.input_data, axis=self.axis)
        self.assertEqual((expect == out.numpy()).all(), True)
        out2 = paddle.argsort(var_x, axis=self.axis, descending=True)
        expect2 = np.argsort(-self.input_data, axis=self.axis)
        self.assertEqual((expect2 == out2.numpy()).all(), True)
        paddle.enable_static()


class TestArgsortImperative2(TestArgsortImperative):
    def init(self):
        self.input_shape = [100, 1]
        self.axis = 0


class TestArgsortImperative3(TestArgsortImperative):
    def init(self):
        self.input_shape = [1, 10000]
        self.axis = 1


class TestArgsortImperative4(TestArgsortImperative):
    def init(self):
        self.input_shape = [17, 60]
        self.axis = 1


class TestArgsortImperative5(TestArgsortImperative):
    def init(self):
        self.input_shape = [60, 70]
        self.axis = 1


class TestArgsortImperative6(TestArgsortImperative):
    def init(self):
        self.input_shape = [10, 70]
        self.axis = 1


class TestArgsortLeNetCase(TestArgsortImperative):
    def init(self):
        self.input_shape = [64, 10]
        self.axis = 1


# TODO(Zhiwei35): support argsort of dims > 3
# class TestArgsortImperative4(TestArgsortImperative):
#     def init(self):
#         self.input_shape = [2, 3, 4]
#         self.axis = 1

# TODO(Zhiwei35): support argsort with NaN
# class TestArgsortWithInputNaN(unittest.TestCase):
#     def init(self):
#         self.axis = 0

#     def setUp(self):
#         self.init()
#         self.input_data = np.array([1.0, np.nan, 3.0, 2.0])
#         self.place = core.CustomPlace("intel_gpu", 0)

#     def test_api(self):
#         paddle.disable_static(self.place)
#         var_x = paddle.to_tensor(self.input_data)
#         out = paddle.argsort(var_x, axis=self.axis)
#         self.assertEqual((out.numpy() == np.array([0, 3, 2, 1])).all(), True)
#         out = paddle.argsort(var_x, axis=self.axis, descending=True)
#         self.assertEqual((out.numpy() == np.array([1, 2, 3, 0])).all(), True)
#         paddle.enable_static()


class TestStableArgsortImperative(unittest.TestCase):
    def init(self):
        self.input_shape = [
            30,
        ]
        self.axis = 0
        self.input_data = np.array([100.0, 50.0, 10.0] * 10)

    def setUp(self):
        self.init()
        self.place = core.CustomPlace("intel_gpu", 0)

    def test_api(self):
        paddle.disable_static(self.place)
        var_x = paddle.to_tensor(self.input_data)
        out = paddle.argsort(var_x, axis=self.axis, stable=True)
        expect = np.argsort(self.input_data, axis=self.axis, kind="stable")
        self.assertEqual((expect == out.numpy()).all(), True)

        out2 = paddle.argsort(var_x, axis=self.axis, descending=True, stable=True)
        expect2 = np.argsort(-self.input_data, axis=self.axis, kind="stable")
        self.assertEqual((expect2 == out2.numpy()).all(), True)

        paddle.enable_static()


class TestStableArgsortImperative2(TestStableArgsortImperative):
    def init(self):
        self.input_shape = [30, 1]
        self.input_data = np.array([100.0, 50.0, 10.0] * 10).reshape(self.input_shape)
        self.axis = 0


class TestStableArgsortImperative3(TestStableArgsortImperative):
    def init(self):
        self.input_shape = [1, 30]
        self.input_data = np.array([100.0, 50.0, 10.0] * 10).reshape(self.input_shape)
        self.axis = 1


if __name__ == "__main__":
    unittest.main()
