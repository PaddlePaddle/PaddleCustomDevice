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

import numpy as np
import unittest
from op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci
import paddle
import paddle.base as base

paddle.enable_static()
SEED = 2021


def broadcast_wrapper(shape=[1, 10, 12, 1]):
    def div_wrapper(x, y, axis=-1):
        return paddle.divide(x, y.reshape(shape))

    return div_wrapper


class TestElementwiseDiv(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "elementwise_div"
        self.python_api = paddle.divide
        self.public_python_api = paddle.divide
        self.init_args()
        self.init_dtype()
        self.init_shape()

        x = self.gen_data(self.x_shape).astype(self.val_dtype)
        y = self.gen_data(self.y_shape).astype(self.val_dtype)
        out = self.compute_output(x, y).astype(self.val_dtype)
        grad_out = np.ones(out.shape).astype(self.val_dtype)
        grad_x = self.compute_gradient_x(grad_out, y).astype(self.val_dtype)
        grad_y = self.compute_gradient_y(grad_out, out, y).astype(self.val_dtype)

        # Convert np.float32 data to np.uint16 for bfloat16 Paddle OP
        if self.dtype == np.uint16:
            x = convert_float_to_uint16(x)
            y = convert_float_to_uint16(y)
            out = convert_float_to_uint16(out)
            grad_out = convert_float_to_uint16(grad_out)
            grad_x = convert_float_to_uint16(grad_x)
            grad_y = convert_float_to_uint16(grad_y)

        self.inputs = {"X": x, "Y": y}
        self.outputs = {"Out": out}
        self.grad_out = grad_out
        self.grad_x = grad_x
        self.grad_y = grad_y

    def init_args(self):
        self.check_dygraph = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_dtype(self):
        self.dtype = np.float32
        self.val_dtype = np.float32

    def init_shape(self):
        self.x_shape = [13, 17]
        self.y_shape = [13, 17]

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def gen_data(self, shape):
        return np.random.uniform(0.1, 1, shape)

    def compute_output(self, x, y):
        return x / y

    def compute_gradient_x(self, grad_out, y):
        return grad_out / y

    def compute_gradient_y(self, grad_out, out, y):
        return -1 * grad_out * out / y

    def test_check_output(self):
        if self.place is None:
            self.check_output()
        else:
            self.check_output_with_place(self.place)

    def test_check_gradient(self):
        check_list = []
        check_list.append(
            {
                "grad": ["X", "Y"],
                "no_grad": None,
                "val_grad": [self.grad_x, self.grad_y],
            }
        )
        check_list.append(
            {"grad": ["Y"], "no_grad": set("X"), "val_grad": [self.grad_y]}
        )
        check_list.append(
            {"grad": ["X"], "no_grad": set("Y"), "val_grad": [self.grad_x]}
        )
        for check_option in check_list:
            check_args = [check_option["grad"], "Out"]
            check_kwargs = {
                "no_grad_set": check_option["no_grad"],
                "user_defined_grads": check_option["val_grad"],
                "user_defined_grad_outputs": [self.grad_out],
                "check_dygraph": self.check_dygraph,
                "check_prim": self.check_prim,
            }
            if self.place is None:
                self.check_grad(*check_args, **check_kwargs)
            else:
                check_args.insert(0, self.place)
                self.check_grad_with_place(*check_args, **check_kwargs)


@skip_check_grad_ci(reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestElementwiseDivOpScalar(TestElementwiseDiv):
    def init_shape(self):
        self.x_shape = [20, 3, 4]
        self.y_shape = [1]

    def compute_gradient_y(self, grad_out, out, y):
        return np.array([np.sum(-1 * grad_out * out / y)])


class TestElementwiseDivOpVector(TestElementwiseDiv):
    def init_shape(self):
        self.x_shape = [100]
        self.y_shape = [100]


class TestElementwiseDivNd1(TestElementwiseDiv):
    def init_shape(self):
        self.x_shape = [2, 3, 3, 2, 8]
        self.y_shape = [2, 3, 3, 2, 8]


class TestElementwiseDivNd2(TestElementwiseDiv):
    def init_shape(self):
        self.x_shape = [2, 3, 3, 2, 3, 8]
        self.y_shape = [2, 3, 3, 2, 3, 8]


class TestElementwiseDivOpBroadcast1(TestElementwiseDiv):
    def init_shape(self):
        self.x_shape = [100, 3, 4]
        self.y_shape = [100]
        self.attrs = {"axis": 0}
        self.python_api = broadcast_wrapper(shape=[100, 1, 1])

    def compute_output(self, x, y):
        return x / y.reshape(100, 1, 1)

    def compute_gradient_x(self, grad_out, y):
        return grad_out / y.reshape(100, 1, 1)

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(-1 * grad_out * out / y.reshape(100, 1, 1), axis=(1, 2))


class TestElementwiseDivOpBroadcast2(TestElementwiseDiv):
    def init_shape(self):
        self.x_shape = [2, 100, 4]
        self.y_shape = [100]
        self.attrs = {"axis": 1}
        self.python_api = broadcast_wrapper(shape=[1, 100, 1])

    def compute_output(self, x, y):
        return x / y.reshape(1, 100, 1)

    def compute_gradient_x(self, grad_out, y):
        return grad_out / y.reshape(1, 100, 1)

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(-1 * grad_out * out / y.reshape(1, 100, 1), axis=(0, 2))


class TestElementwiseDivOpBroadcast3(TestElementwiseDiv):
    def init_shape(self):
        self.x_shape = [2, 3, 100]
        self.y_shape = [100]
        self.python_api = broadcast_wrapper(shape=[1, 1, 100])

    def compute_output(self, x, y):
        return x / y.reshape(1, 1, 100)

    def compute_gradient_x(self, grad_out, y):
        return grad_out / y.reshape(1, 1, 100)

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(-1 * grad_out * out / y.reshape(1, 1, 100), axis=(0, 1))


class TestElementwiseDivOpBroadcast4(TestElementwiseDiv):
    def init_shape(self):
        self.x_shape = [2, 10, 12, 5]
        self.y_shape = [10, 12]
        self.attrs = {"axis": 1}
        self.python_api = broadcast_wrapper(shape=[1, 10, 12, 1])

    def compute_output(self, x, y):
        return x / y.reshape(1, 10, 12, 1)

    def compute_gradient_x(self, grad_out, y):
        return grad_out / y.reshape(1, 10, 12, 1)

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(-1 * grad_out * out / y.reshape(1, 10, 12, 1), axis=(0, 3))


class TestElementwiseDivOpBroadcast5(TestElementwiseDiv):
    def init_shape(self):
        self.x_shape = [2, 3, 50]
        self.y_shape = [2, 1, 50]

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(-1 * grad_out * out / y, axis=(1)).reshape(2, 1, 50)


class TestElementwiseDivOpBroadcast6(TestElementwiseDiv):
    def init_shape(self):
        self.x_shape = [2, 3, 4, 20]
        self.y_shape = [2, 3, 1, 20]

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(-1 * grad_out * out / y, axis=(2)).reshape(2, 3, 1, 20)


class TestElementwiseDivOpCommonuse1(TestElementwiseDiv):
    def init_shape(self):
        self.x_shape = [2, 3, 100]
        self.y_shape = [1, 1, 100]

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(-1 * grad_out * out / y, axis=(0, 1)).reshape(1, 1, 100)


class TestElementwiseDivOpCommonuse2(TestElementwiseDiv):
    def init_shape(self):
        self.x_shape = [2, 1, 100]
        self.y_shape = [1, 4, 100]

    def compute_gradient_x(self, grad_out, y):
        return np.sum(grad_out / y, axis=(1)).reshape(2, 1, 100)

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(-1 * grad_out * out / y, axis=(0)).reshape(1, 4, 100)


class TestElementwiseDivOpCommonuse3(TestElementwiseDiv):
    def init_shape(self):
        self.x_shape = [30, 3, 1, 2]
        self.y_shape = [30, 1, 4, 2]

    def compute_gradient_x(self, grad_out, y):
        return np.sum(grad_out / y, axis=(2)).reshape(30, 3, 1, 2)

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(-1 * grad_out * out / y, axis=(1)).reshape(30, 1, 4, 2)


class TestElementwiseDivOpCommonuse4(TestElementwiseDiv):
    def init_shape(self):
        self.x_shape = [30, 3, 1, 5]
        self.y_shape = [30, 1, 4, 1]

    def compute_gradient_x(self, grad_out, y):
        return np.sum(grad_out / y, axis=(2)).reshape(30, 3, 1, 5)

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(-1 * grad_out * out / y, axis=(1, 3)).reshape(30, 1, 4, 1)


class TestElementwiseDivOpCommonuse5(TestElementwiseDiv):
    def init_shape(self):
        self.x_shape = [30, 3, 1, 5, 2]
        self.y_shape = [30, 1, 4, 5, 2]

    def compute_gradient_x(self, grad_out, y):
        return np.sum(grad_out / y, axis=(2)).reshape(30, 3, 1, 5, 2)

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(-1 * grad_out * out / y, axis=(1)).reshape(30, 1, 4, 5, 2)


class TestElementwiseDivOpXsizeLessThanYsize(TestElementwiseDiv):
    def init_shape(self):
        self.x_shape = [10, 12]
        self.y_shape = [2, 3, 10, 12]
        self.attrs = {"axis": 2}

    def compute_gradient_x(self, grad_out, y):
        return np.sum(grad_out / y, axis=(0, 1))


def create_test_fp16_class(parent, max_relative_error=2e-3):
    class TestElementwiseDivFP16Op(parent):
        def init_dtype(self):
            self.dtype = np.float16
            self.val_dtype = np.float16

        def test_check_gradient(self):
            check_list = []
            check_list.append(
                {
                    "grad": ["X", "Y"],
                    "no_grad": None,
                    "val_grad": [self.grad_x, self.grad_y],
                }
            )
            check_list.append(
                {"grad": ["Y"], "no_grad": set("X"), "val_grad": [self.grad_y]}
            )
            check_list.append(
                {"grad": ["X"], "no_grad": set("Y"), "val_grad": [self.grad_x]}
            )
            for check_option in check_list:
                check_args = [check_option["grad"], "Out"]
                check_kwargs = {
                    "no_grad_set": check_option["no_grad"],
                    "user_defined_grads": check_option["val_grad"],
                    "user_defined_grad_outputs": [self.grad_out],
                    "check_dygraph": self.check_dygraph,
                    "max_relative_error": max_relative_error,
                }
                if self.place is None:
                    self.check_grad(*check_args, **check_kwargs)
                else:
                    check_args.insert(0, self.place)
                    self.check_grad_with_place(*check_args, **check_kwargs)

    cls_name = "{}_{}".format(parent.__name__, "Fp16")
    TestElementwiseDivFP16Op.__name__ = cls_name
    globals()[cls_name] = TestElementwiseDivFP16Op


create_test_fp16_class(TestElementwiseDiv)
create_test_fp16_class(TestElementwiseDivOpScalar)
create_test_fp16_class(TestElementwiseDivOpVector)
create_test_fp16_class(TestElementwiseDivNd1)
create_test_fp16_class(TestElementwiseDivNd2)
create_test_fp16_class(TestElementwiseDivOpBroadcast1)
create_test_fp16_class(TestElementwiseDivOpBroadcast2)
create_test_fp16_class(TestElementwiseDivOpBroadcast3)
create_test_fp16_class(TestElementwiseDivOpBroadcast4)
create_test_fp16_class(TestElementwiseDivOpBroadcast5)
create_test_fp16_class(TestElementwiseDivOpBroadcast6)
create_test_fp16_class(TestElementwiseDivOpCommonuse1)
create_test_fp16_class(TestElementwiseDivOpCommonuse2)
create_test_fp16_class(TestElementwiseDivOpCommonuse3)
create_test_fp16_class(TestElementwiseDivOpCommonuse4)
create_test_fp16_class(TestElementwiseDivOpCommonuse5)
create_test_fp16_class(TestElementwiseDivOpXsizeLessThanYsize)


def create_test_double_class(parent):
    class TestElementwiseDivDoubleOp(parent):
        def init_dtype(self):
            self.dtype = np.double
            self.val_dtype = np.double

    cls_name = "{}_{}".format(parent.__name__, "Double")
    TestElementwiseDivDoubleOp.__name__ = cls_name
    globals()[cls_name] = TestElementwiseDivDoubleOp


create_test_double_class(TestElementwiseDiv)
create_test_double_class(TestElementwiseDivOpScalar)
create_test_double_class(TestElementwiseDivOpVector)
create_test_double_class(TestElementwiseDivNd1)
create_test_double_class(TestElementwiseDivNd2)
create_test_double_class(TestElementwiseDivOpBroadcast1)
create_test_double_class(TestElementwiseDivOpBroadcast2)
create_test_double_class(TestElementwiseDivOpBroadcast3)
create_test_double_class(TestElementwiseDivOpBroadcast4)
create_test_double_class(TestElementwiseDivOpBroadcast5)
create_test_double_class(TestElementwiseDivOpBroadcast6)
create_test_double_class(TestElementwiseDivOpCommonuse1)
create_test_double_class(TestElementwiseDivOpCommonuse2)
create_test_double_class(TestElementwiseDivOpCommonuse3)
create_test_double_class(TestElementwiseDivOpCommonuse4)
create_test_double_class(TestElementwiseDivOpCommonuse5)
create_test_double_class(TestElementwiseDivOpXsizeLessThanYsize)


def create_test_uint8_class(parent):
    class TestElementwiseDivUint8Op(parent):
        def init_dtype(self):
            self.dtype = np.uint8
            self.val_dtype = np.uint8

        def gen_data(self, shape):
            return np.random.randint(2, 10, shape)

        def test_check_gradient(self):
            pass

    cls_name = "{}_{}".format(parent.__name__, "Uint8")
    TestElementwiseDivUint8Op.__name__ = cls_name
    globals()[cls_name] = TestElementwiseDivUint8Op


create_test_uint8_class(TestElementwiseDiv)
create_test_uint8_class(TestElementwiseDivOpScalar)
create_test_uint8_class(TestElementwiseDivOpVector)
create_test_uint8_class(TestElementwiseDivNd1)
create_test_uint8_class(TestElementwiseDivNd2)
create_test_uint8_class(TestElementwiseDivOpBroadcast1)
create_test_uint8_class(TestElementwiseDivOpBroadcast2)
create_test_uint8_class(TestElementwiseDivOpBroadcast3)
create_test_uint8_class(TestElementwiseDivOpBroadcast4)
create_test_uint8_class(TestElementwiseDivOpBroadcast5)
create_test_uint8_class(TestElementwiseDivOpBroadcast6)
create_test_uint8_class(TestElementwiseDivOpCommonuse1)
create_test_uint8_class(TestElementwiseDivOpCommonuse2)
create_test_uint8_class(TestElementwiseDivOpCommonuse3)
create_test_uint8_class(TestElementwiseDivOpCommonuse4)
create_test_uint8_class(TestElementwiseDivOpCommonuse5)
create_test_uint8_class(TestElementwiseDivOpXsizeLessThanYsize)


def create_test_int8_class(parent):
    class TestElementwiseDivInt8Op(parent):
        def init_dtype(self):
            self.dtype = np.int8
            self.val_dtype = np.int8

        def gen_data(self, shape):
            return np.random.randint(2, 10, shape)

        def test_check_gradient(self):
            pass

    cls_name = "{}_{}".format(parent.__name__, "Int8")
    TestElementwiseDivInt8Op.__name__ = cls_name
    globals()[cls_name] = TestElementwiseDivInt8Op


create_test_int8_class(TestElementwiseDiv)
create_test_int8_class(TestElementwiseDivOpScalar)
create_test_int8_class(TestElementwiseDivOpVector)
create_test_int8_class(TestElementwiseDivNd1)
create_test_int8_class(TestElementwiseDivNd2)
create_test_int8_class(TestElementwiseDivOpBroadcast1)
create_test_int8_class(TestElementwiseDivOpBroadcast2)
create_test_int8_class(TestElementwiseDivOpBroadcast3)
create_test_int8_class(TestElementwiseDivOpBroadcast4)
create_test_int8_class(TestElementwiseDivOpBroadcast5)
create_test_int8_class(TestElementwiseDivOpBroadcast6)
create_test_int8_class(TestElementwiseDivOpCommonuse1)
create_test_int8_class(TestElementwiseDivOpCommonuse2)
create_test_int8_class(TestElementwiseDivOpCommonuse3)
create_test_int8_class(TestElementwiseDivOpCommonuse4)
create_test_int8_class(TestElementwiseDivOpCommonuse5)
create_test_int8_class(TestElementwiseDivOpXsizeLessThanYsize)


def create_test_int16_class(parent):
    class TestElementwiseDivInt16Op(parent):
        def init_dtype(self):
            self.dtype = np.int16
            self.val_dtype = np.int16

        def gen_data(self, shape):
            return np.random.randint(2, 10, shape)

        def test_check_gradient(self):
            pass

    cls_name = "{}_{}".format(parent.__name__, "Int16")
    TestElementwiseDivInt16Op.__name__ = cls_name
    globals()[cls_name] = TestElementwiseDivInt16Op


create_test_int16_class(TestElementwiseDiv)
create_test_int16_class(TestElementwiseDivOpScalar)
create_test_int16_class(TestElementwiseDivOpVector)
create_test_int16_class(TestElementwiseDivNd1)
create_test_int16_class(TestElementwiseDivNd2)
create_test_int16_class(TestElementwiseDivOpBroadcast1)
create_test_int16_class(TestElementwiseDivOpBroadcast2)
create_test_int16_class(TestElementwiseDivOpBroadcast3)
create_test_int16_class(TestElementwiseDivOpBroadcast4)
create_test_int16_class(TestElementwiseDivOpBroadcast5)
create_test_int16_class(TestElementwiseDivOpBroadcast6)
create_test_int16_class(TestElementwiseDivOpCommonuse1)
create_test_int16_class(TestElementwiseDivOpCommonuse2)
create_test_int16_class(TestElementwiseDivOpCommonuse3)
create_test_int16_class(TestElementwiseDivOpCommonuse4)
create_test_int16_class(TestElementwiseDivOpCommonuse5)
create_test_int16_class(TestElementwiseDivOpXsizeLessThanYsize)


def create_test_int32_class(parent):
    class TestElementwiseDivInt32Op(parent):
        def init_dtype(self):
            self.dtype = np.int32
            self.val_dtype = np.int32

        def gen_data(self, shape):
            return np.random.randint(2, 10, shape)

        def test_check_gradient(self):
            pass

    cls_name = "{}_{}".format(parent.__name__, "Int32")
    TestElementwiseDivInt32Op.__name__ = cls_name
    globals()[cls_name] = TestElementwiseDivInt32Op


create_test_int32_class(TestElementwiseDiv)
create_test_int32_class(TestElementwiseDivOpScalar)
create_test_int32_class(TestElementwiseDivOpVector)
create_test_int32_class(TestElementwiseDivNd1)
create_test_int32_class(TestElementwiseDivNd2)
create_test_int32_class(TestElementwiseDivOpBroadcast1)
create_test_int32_class(TestElementwiseDivOpBroadcast2)
create_test_int32_class(TestElementwiseDivOpBroadcast3)
create_test_int32_class(TestElementwiseDivOpBroadcast4)
create_test_int32_class(TestElementwiseDivOpBroadcast5)
create_test_int32_class(TestElementwiseDivOpBroadcast6)
create_test_int32_class(TestElementwiseDivOpCommonuse1)
create_test_int32_class(TestElementwiseDivOpCommonuse2)
create_test_int32_class(TestElementwiseDivOpCommonuse3)
create_test_int32_class(TestElementwiseDivOpCommonuse4)
create_test_int32_class(TestElementwiseDivOpCommonuse5)
create_test_int32_class(TestElementwiseDivOpXsizeLessThanYsize)


def create_test_int64_class(parent):
    class TestElementwiseDivInt64Op(parent):
        def init_dtype(self):
            self.dtype = np.int64
            self.val_dtype = np.int64

        def gen_data(self, shape):
            return np.random.randint(2, 10, shape)

        def test_check_gradient(self):
            pass

    cls_name = "{}_{}".format(parent.__name__, "Int64")
    TestElementwiseDivInt64Op.__name__ = cls_name
    globals()[cls_name] = TestElementwiseDivInt64Op


create_test_int64_class(TestElementwiseDiv)
create_test_int64_class(TestElementwiseDivOpScalar)
create_test_int64_class(TestElementwiseDivOpVector)
create_test_int64_class(TestElementwiseDivNd1)
create_test_int64_class(TestElementwiseDivNd2)
create_test_int64_class(TestElementwiseDivOpBroadcast1)
create_test_int64_class(TestElementwiseDivOpBroadcast2)
create_test_int64_class(TestElementwiseDivOpBroadcast3)
create_test_int64_class(TestElementwiseDivOpBroadcast4)
create_test_int64_class(TestElementwiseDivOpBroadcast5)
create_test_int64_class(TestElementwiseDivOpBroadcast6)
create_test_int64_class(TestElementwiseDivOpCommonuse1)
create_test_int64_class(TestElementwiseDivOpCommonuse2)
create_test_int64_class(TestElementwiseDivOpCommonuse3)
create_test_int64_class(TestElementwiseDivOpCommonuse4)
create_test_int64_class(TestElementwiseDivOpCommonuse5)
create_test_int64_class(TestElementwiseDivOpXsizeLessThanYsize)


class TestElementwiseDivBroadcast(unittest.TestCase):
    def test_shape_with_batch_sizes(self):
        with base.program_guard(base.Program()):
            x_var = paddle.static.data(
                name="x", dtype="float32", shape=[None, 3, None, None]
            )
            one = 2.0
            out = one / x_var
            exe = base.Executor(base.CPUPlace())
            x = np.random.uniform(0.1, 0.6, (1, 3, 32, 32)).astype("float32")
            (out_result,) = exe.run(feed={"x": x}, fetch_list=[out])
            self.assertEqual((out_result == (2 / x)).all(), True)


class TestDivideOp(unittest.TestCase):
    def test_name(self):
        with base.program_guard(base.Program()):
            x = paddle.static.data(name="x", shape=[2, 3], dtype="float32")
            y = paddle.static.data(name="y", shape=[2, 3], dtype="float32")

            y_1 = paddle.divide(x, y, name="div_res")
            self.assertEqual(("div_res" in y_1.name), True)

    def test_dygraph(self):
        with base.dygraph.guard():
            np_x = np.array([2, 3, 4]).astype("float64")
            np_y = np.array([1, 5, 2]).astype("float64")
            x = paddle.to_tensor(np_x)
            y = paddle.to_tensor(np_y)
            z = paddle.divide(x, y)
            np_z = z.numpy()
            z_expected = np.array([2.0, 0.6, 2.0])
            self.assertEqual((np_z == z_expected).all(), True)


class TestElementwiseDivop(unittest.TestCase):
    def test_dygraph_div(self):
        paddle.disable_static()
        paddle.device.set_device("sdaa")

        np_a = np.random.random((2, 3, 4)).astype(np.float32)
        np_b = np.random.random((2, 3, 4)).astype(np.float32)
        np_a[np.abs(np_a) < 0.0005] = 0.002
        np_b[np.abs(np_b) < 0.0005] = 0.002

        tensor_a = paddle.to_tensor(np_a, dtype="float32")
        tensor_b = paddle.to_tensor(np_b, dtype="float32")

        # normal case: nparray / tenor
        expect_out = np_a / np_b
        actual_out = np_a / tensor_b
        np.testing.assert_allclose(actual_out, expect_out, atol=1e-6)

        # normal case: tensor / nparray
        actual_out = tensor_a / np_b
        np.testing.assert_allclose(actual_out, expect_out, atol=1e-6)

        paddle.enable_static()


class TestElementwiseDivNet(unittest.TestCase):
    def _test(self, run_sdaa=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.uniform(1, 2, [32, 32]).astype("float32")
        b_np = np.random.uniform(1, 2, [32, 32]).astype("float32")
        c_np = np.random.uniform(1, 2, [32, 32]).astype("float32")
        d_np = np.random.uniform(1, 2, [32, 32]).astype("float32")
        label_np = np.random.randint(2, size=(32, 1)).astype("int64")

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[32, 32], dtype="float32")
            b = paddle.static.data(name="b", shape=[32, 32], dtype="float32")
            c = paddle.static.data(name="c", shape=[32, 32], dtype="float32")
            d = paddle.static.data(name="d", shape=[32, 32], dtype="float32")
            label = paddle.static.data(name="label", shape=[32, 1], dtype="int64")

            e = paddle.multiply(a, b)
            f = paddle.multiply(c, d)
            f.stop_gradient = True
            g = paddle.divide(e, f)

            fc_1 = paddle.static.nn.fc(x=g, size=128)
            prediction = paddle.static.nn.fc(x=fc_1, size=2, activation="softmax")

            cost = paddle.nn.functional.cross_entropy(input=prediction, label=label)
            loss = paddle.mean(cost)
            sgd = paddle.optimizer.Momentum(learning_rate=0.01)
            sgd.minimize(loss)

        if run_sdaa:
            place = paddle.CustomPlace("sdaa", 0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        print("Start run on {}".format(place))
        for epoch in range(100):

            pred_res, loss_res = exe.run(
                main_prog,
                feed={"a": a_np, "b": b_np, "c": c_np, "d": d_np, "label": label_np},
                fetch_list=[prediction, loss],
            )
            if epoch % 10 == 0:
                print(
                    "Epoch {} | Prediction[0]: {}, Loss: {}".format(
                        epoch, pred_res[0], loss_res
                    )
                )

        return pred_res, loss_res

    def test_sdaa(self):
        cpu_pred, cpu_loss = self._test(False)
        sdaa_pred, sdaa_loss = self._test(True)

        self.assertTrue(np.allclose(sdaa_pred, cpu_pred))
        self.assertTrue(np.allclose(sdaa_loss, cpu_loss))


if __name__ == "__main__":
    unittest.main()
