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

from op_test import OpTest
import paddle
import paddle.base as base

paddle.enable_static()
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


class TestMatMulV2Op(OpTest):
    """
    case 1
    """

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def config(self):
        self.x_shape = (100,)
        self.y_shape = (100,)
        self.trans_x = False
        self.trans_y = False

    def init_kernel_type(self):
        self.dtype = np.float32

    def setUp(self):
        self.set_sdaa()
        self.init_kernel_type()
        self.config()
        self.op_type = "matmul_v2"
        self.python_api = paddle.tensor.matmul
        x = np.random.random(self.x_shape).astype(self.dtype)
        y = np.random.random(self.y_shape).astype(self.dtype)
        # -0.1 ~ 0.1
        x = -0.1 + 0.2 * x
        y = -0.1 + 0.2 * y
        result = reference_matmul(x, y, self.trans_x, self.trans_y)
        result = result.astype(self.dtype)
        self.inputs = {
            "X": x,
            "Y": y,
        }
        self.attrs = {"trans_x": self.trans_x, "trans_y": self.trans_y}
        self.outputs = {"Out": result}

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-7)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["X", "Y"],
            "Out",
        )

    def check_grad_with_place(
        self,
        place,
        inputs_to_check,
        output_names,
        no_grad_set=None,
        numeric_grad_delta=0.005,
        in_place=False,
        max_relative_error=0.005,
        user_defined_grads=None,
        user_defined_grad_outputs=None,
        check_dygraph=True,
        numeric_place=None,
    ):
        if self.dtype == np.float32:
            numeric_place = paddle.CPUPlace()
        super().check_grad_with_place(
            place,
            inputs_to_check,
            output_names,
            no_grad_set,
            numeric_grad_delta,
            in_place,
            max_relative_error,
            user_defined_grads,
            user_defined_grad_outputs,
            check_dygraph,
            numeric_place=numeric_place,
        )


class TestMatMulOp2(TestMatMulV2Op):
    """
    case 2
    """

    def config(self):
        self.x_shape = (100,)
        self.y_shape = (1, 3, 2, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatMulOp3(TestMatMulV2Op):
    """
    case 3
    """

    def config(self):
        self.x_shape = (100,)
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp4(TestMatMulV2Op):
    """
    case 4
    """

    def config(self):
        self.x_shape = (100,)
        self.y_shape = (1, 2, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp5(TestMatMulV2Op):
    """
    case 5
    """

    def config(self):
        self.x_shape = (1, 1, 100, 1)
        self.y_shape = (100,)
        self.trans_x = True
        self.trans_y = False


class TestMatMulOp6(TestMatMulV2Op):
    """
    case 6
    """

    def config(self):
        self.x_shape = (1, 2, 102, 1)
        self.y_shape = (102,)
        self.trans_x = True
        self.trans_y = False


class TestMatMulOp7(TestMatMulV2Op):
    """
    case 7
    """

    def config(self):
        self.x_shape = (1, 2, 1, 100)
        self.y_shape = (100,)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp8(TestMatMulV2Op):
    """
    case 8
    """

    def config(self):
        self.x_shape = (1, 1, 2, 100)
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp9(TestMatMulV2Op):
    """
    case 9
    """

    def config(self):
        self.x_shape = (1, 1, 1, 100)
        self.y_shape = (2, 1, 2, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatMulOp10(TestMatMulV2Op):
    """
    case 10
    """

    def config(self):
        self.x_shape = (1, 1, 25, 4)
        self.y_shape = (1, 2, 4, 25)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp11(TestMatMulV2Op):
    """
    case 11
    """

    def config(self):
        self.x_shape = (2, 1, 2, 100)
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp12(TestMatMulV2Op):
    """
    case 12
    """

    def config(self):
        self.x_shape = (2, 1, 4, 25)
        self.y_shape = (1, 1, 4, 25)
        self.trans_x = True
        self.trans_y = False


class TestMatMulOp13(TestMatMulV2Op):
    """
    case 13
    """

    def config(self):
        self.x_shape = (2, 2, 10, 10)
        self.y_shape = (2, 2, 10, 10)
        self.trans_x = True
        self.trans_y = False


class TestMatMulOp14(TestMatMulV2Op):
    """
    case 14_1
    """

    def config(self):
        self.x_shape = (3, 1, 6, 6)
        self.y_shape = (1, 2, 6, 9)
        self.trans_x = True
        self.trans_y = False


class TestMatMulOp15(TestMatMulV2Op):
    """
    case 14_2
    """

    def config(self):
        self.x_shape = (3, 1, 6, 6)
        self.y_shape = (1, 2, 6, 9)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp16(TestMatMulV2Op):
    """
    case 16 : to check the gradient for special case
    """

    def config(self):
        self.x_shape = 100
        self.y_shape = (1, 2, 2, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp17(TestMatMulV2Op):
    """
    case 17 : to check the gradient for special case
    """

    def config(self):
        self.x_shape = (2, 1, 100)
        self.y_shape = 100
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp18(TestMatMulV2Op):
    """
    case 18
    """

    def config(self):
        self.x_shape = (2, 2, 10, 10)
        self.y_shape = (2, 2, 10, 10)
        self.trans_x = True
        self.trans_y = True


class TestMatMulOp19(TestMatMulV2Op):
    """
    case 19
    """

    def config(self):
        self.x_shape = (2, 100)
        self.y_shape = (100,)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp20(TestMatMulV2Op):
    """
    case 20
    """

    def config(self):
        self.x_shape = (100, 2)
        self.y_shape = (100,)
        self.trans_x = True
        self.trans_y = False


class TestMatMulOp21(TestMatMulV2Op):
    """
    case 21
    """

    def config(self):
        self.x_shape = (32, 32)
        self.y_shape = (32, 32)
        self.trans_x = True
        self.trans_y = True


class TestMatMulOpBroadcast1(TestMatMulV2Op):
    """
    case 14_3
    """

    def config(self):
        self.x_shape = (3, 1, 10, 10)
        self.y_shape = (1, 2, 10, 10)
        self.trans_x = True
        self.trans_y = True


class TestMatMulOpBroadcast2(TestMatMulV2Op):
    """
    case 14_4
    """

    def config(self):
        self.x_shape = (3, 1, 10, 10)
        self.y_shape = (1, 2, 10, 10)
        self.trans_x = False
        self.trans_y = True


class TestMatMulOpBroadcast3(TestMatMulV2Op):
    """
    case 14_5
    test unilaterial broadcast
    """

    def config(self):
        self.x_shape = (3, 1, 10, 10)
        self.y_shape = (10, 10)
        self.trans_x = False
        self.trans_y = True


class TestMatMulOpBroadcast4(TestMatMulV2Op):
    """
    case 14_6
    test unilaterial broadcast
    """

    def config(self):
        self.x_shape = (3, 1, 10, 10)
        self.y_shape = (10, 10)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOpBroadcast5(TestMatMulV2Op):
    """
    case 14_7
    test unilaterial broadcast
    """

    def config(self):
        self.x_shape = (3, 1, 10, 10)
        self.y_shape = (10, 10)
        self.trans_x = True
        self.trans_y = False


class TestMatMulOpBroadcast6(TestMatMulV2Op):
    """
    case 14_6
    test unilaterial broadcast
    """

    def config(self):
        self.x_shape = (10, 10)
        self.y_shape = (3, 1, 10, 10)
        self.trans_x = False
        self.trans_y = True


class TestMatMulOpBroadcast7(TestMatMulV2Op):
    """
    case 14_7
    test unilaterial broadcast
    """

    def config(self):
        self.x_shape = (10, 10)
        self.y_shape = (3, 1, 10, 10)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOpBroadcast8(TestMatMulV2Op):
    """
    case 14_8
    test unilaterial broadcast
    """

    def config(self):
        self.x_shape = (10, 10)
        self.y_shape = (3, 1, 10, 10)
        self.trans_x = True
        self.trans_y = False


class TestMatMulOpBroadcast9(TestMatMulV2Op):
    """
    case 14_9
    test unilaterial broadcast
    """

    def config(self):
        self.x_shape = (3, 10, 10, 1, 5)
        self.y_shape = (10, 10, 5, 4)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOpBroadcast10(TestMatMulV2Op):
    """
    case 14_10
    test unilaterial broadcast
    """

    def config(self):
        self.x_shape = (10, 10, 1, 5)
        self.y_shape = (3, 10, 10, 5, 4)
        self.trans_x = False
        self.trans_y = False


# --------------------test matmul fp16--------------------


def create_test_fp16_class(parent, atol=0.001, max_relative_error=1.5):
    class TestMatMulOpFp16Case(parent):
        def init_kernel_type(self):
            self.dtype = np.float16

        def test_check_output(self):
            self.check_output_with_place(self.place, atol=atol)

        def test_check_grad(self):
            self.check_grad_with_place(
                self.place, ["X", "Y"], "Out", max_relative_error=max_relative_error
            )

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestMatMulOpFp16Case.__name__ = cls_name
    globals()[cls_name] = TestMatMulOpFp16Case


create_test_fp16_class(TestMatMulV2Op)
create_test_fp16_class(TestMatMulOp2)
create_test_fp16_class(TestMatMulOp3)
create_test_fp16_class(TestMatMulOp4)
create_test_fp16_class(TestMatMulOp5)
create_test_fp16_class(TestMatMulOp6)
create_test_fp16_class(TestMatMulOp7)
create_test_fp16_class(TestMatMulOp8)
create_test_fp16_class(TestMatMulOp9)
create_test_fp16_class(TestMatMulOp10)
create_test_fp16_class(TestMatMulOp11)
create_test_fp16_class(TestMatMulOp12)
create_test_fp16_class(TestMatMulOp13)
create_test_fp16_class(TestMatMulOp14)
create_test_fp16_class(TestMatMulOp15)
create_test_fp16_class(TestMatMulOp16)
create_test_fp16_class(TestMatMulOp17)
create_test_fp16_class(TestMatMulOp18)
create_test_fp16_class(TestMatMulOp19)
create_test_fp16_class(TestMatMulOp20)
create_test_fp16_class(TestMatMulOp21)
create_test_fp16_class(TestMatMulOpBroadcast1)
create_test_fp16_class(TestMatMulOpBroadcast2)
create_test_fp16_class(TestMatMulOpBroadcast3)
create_test_fp16_class(TestMatMulOpBroadcast4)
create_test_fp16_class(TestMatMulOpBroadcast5)
create_test_fp16_class(TestMatMulOpBroadcast6)
create_test_fp16_class(TestMatMulOpBroadcast7)
create_test_fp16_class(TestMatMulOpBroadcast8)
create_test_fp16_class(TestMatMulOpBroadcast9)
create_test_fp16_class(TestMatMulOpBroadcast10)


class TestMatMulV2API(unittest.TestCase):
    def setUp(self):
        self.places = [paddle.CPUPlace(), paddle.CustomPlace("sdaa", 0)]

    def check_static_result(self, place):
        with base.program_guard(base.Program(), base.Program()):
            input_x = paddle.static.data(name="input_x", shape=[4, 3], dtype=np.float32)
            input_y = paddle.static.data(name="input_y", shape=[3, 4], dtype=np.float32)

            result = paddle.matmul(input_x, input_y)

            x_np = np.random.random([4, 3]).astype("float32")
            y_np = np.random.random([3, 4]).astype("float32")

            exe = base.Executor(place)
            fetches = exe.run(
                base.default_main_program(),
                feed={"input_x": x_np, "input_y": y_np},
                fetch_list=[result],
            )

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                input_x = np.random.random([4, 3]).astype(np.float32)
                input_y = np.random.random([3, 4]).astype(np.float32)
                x = paddle.to_tensor(input_x)
                y = paddle.to_tensor(input_y)
                result = paddle.matmul(x, y)


# This case is used to test that the stride value passed into the
# tecoblasGetWorkspaceSize API is set correctly when using
# tecoblasHgemmStridedBatched.
# Meanwhile, the test case will be invalidated after a change in
# blas's workspace computation policy or a change in test shape.
# The test shape is:
# x_shape = (1, 12, 561, 561) and y_shape = (1, 12, 561, 64).
class TestMatmulStride(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        paddle.device.set_device("sdaa")

        self.x_shape = (1, 12, 561, 561)
        self.y_shape = (1, 12, 561, 64)
        self.trans_x = True
        self.trans_y = False
        self.dtype = np.float16

    def test_matmul_stride(self):
        def get_temp_var():
            out_temp = np.random.uniform(low=0, high=1, size=(1, 12, 561, 64)).astype(
                self.dtype
            )
            workspace = np.random.uniform(low=0, high=1, size=(17694720)).astype(
                np.uint8
            )

            out_temp = paddle.to_tensor(out_temp)
            workspace = paddle.to_tensor(workspace)

            global check_tensor
            check_tensor = paddle.full(shape=[3, 2], fill_value=1.0)

        x_np = np.random.uniform(low=0, high=1, size=self.x_shape).astype(self.dtype)
        y_np = np.random.uniform(low=0, high=1, size=self.y_shape).astype(self.dtype)

        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = paddle.to_tensor(y_np, stop_gradient=False)

        get_temp_var()

        out = paddle.matmul(x, y, self.trans_x, self.trans_y)

        base_check_data = np.full(shape=[3, 2], fill_value=1.0)

        np.testing.assert_array_equal(
            check_tensor.numpy(),
            base_check_data,
        )


if __name__ == "__main__":
    unittest.main()
