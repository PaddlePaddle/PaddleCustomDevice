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


class TestMatMulV2Op_HIGHPERFORMANCE(OpTest):
    """
    case 1
    """

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)
        import os

        os.environ["HIGH_PERFORMANCE_GEMM"] = "1"

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
        self.check_output_with_place(self.place, atol=0.001)

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
        max_relative_error=1.5,
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


class TestMatMulOp2_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 2
    """

    def config(self):
        self.x_shape = (100,)
        self.y_shape = (1, 3, 2, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatMulOp3_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 3
    """

    def config(self):
        self.x_shape = (100,)
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp4_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 4
    """

    def config(self):
        self.x_shape = (100,)
        self.y_shape = (1, 2, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp5_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 5
    """

    def config(self):
        self.x_shape = (1, 1, 100, 1)
        self.y_shape = (100,)
        self.trans_x = True
        self.trans_y = False


class TestMatMulOp6_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 6
    """

    def config(self):
        self.x_shape = (1, 2, 102, 1)
        self.y_shape = (102,)
        self.trans_x = True
        self.trans_y = False


class TestMatMulOp7_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 7
    """

    def config(self):
        self.x_shape = (1, 2, 1, 100)
        self.y_shape = (100,)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp8_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 8
    """

    def config(self):
        self.x_shape = (1, 1, 2, 100)
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


@unittest.skip("high performance gemm not support bidirectional broadcast")
class TestMatMulOp9_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 9
    """

    def config(self):
        self.x_shape = (1, 1, 1, 100)
        self.y_shape = (2, 1, 2, 100)
        self.trans_x = False
        self.trans_y = True


@unittest.skip("high performance gemm not support bidirectional broadcast")
class TestMatMulOp10_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 10
    """

    def config(self):
        self.x_shape = (1, 1, 25, 4)
        self.y_shape = (1, 2, 4, 25)
        self.trans_x = False
        self.trans_y = False


@unittest.skip("high performance gemm not support bidirectional broadcast")
class TestMatMulOp11_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 11
    """

    def config(self):
        self.x_shape = (2, 1, 2, 100)
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


@unittest.skip("high performance gemm not support bidirectional broadcast")
class TestMatMulOp12_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 12
    """

    def config(self):
        self.x_shape = (2, 1, 4, 25)
        self.y_shape = (1, 1, 4, 25)
        self.trans_x = True
        self.trans_y = False


class TestMatMulOp13_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 13
    """

    def config(self):
        self.x_shape = (2, 2, 10, 10)
        self.y_shape = (2, 2, 10, 10)
        self.trans_x = True
        self.trans_y = False


@unittest.skip("high performance gemm not support bidirectional broadcast")
class TestMatMulOp14_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 14_1
    """

    def config(self):
        self.x_shape = (3, 1, 6, 6)
        self.y_shape = (1, 2, 6, 9)
        self.trans_x = True
        self.trans_y = False


@unittest.skip("high performance gemm not support bidirectional broadcast")
class TestMatMulOp15_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 14_2
    """

    def config(self):
        self.x_shape = (3, 1, 6, 6)
        self.y_shape = (1, 2, 6, 9)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp16_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 16 : to check the gradient for special case
    """

    def config(self):
        self.x_shape = 100
        self.y_shape = (1, 2, 2, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp17_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 17 : to check the gradient for special case
    """

    def config(self):
        self.x_shape = (2, 1, 100)
        self.y_shape = 100
        self.trans_x = False
        self.trans_y = False


@unittest.skip("high performance gemm not support bidirectional broadcast")
class TestMatMulOpBroadcast1_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 14_3
    """

    def config(self):
        self.x_shape = (3, 1, 10, 10)
        self.y_shape = (1, 2, 10, 10)
        self.trans_x = True
        self.trans_y = True


@unittest.skip("high performance gemm not support bidirectional broadcast")
class TestMatMulOpBroadcast2_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 14_4
    """

    def config(self):
        self.x_shape = (3, 1, 10, 10)
        self.y_shape = (1, 2, 10, 10)
        self.trans_x = False
        self.trans_y = True


class TestMatMulOpBroadcast3_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 14_5
    test unilaterial broadcast
    """

    def config(self):
        self.x_shape = (3, 1, 10, 10)
        self.y_shape = (10, 10)
        self.trans_x = False
        self.trans_y = True


class TestMatMulOpBroadcast4_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 14_6
    test unilaterial broadcast
    """

    def config(self):
        self.x_shape = (3, 1, 10, 10)
        self.y_shape = (10, 10)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOpBroadcast5_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 14_7
    test unilaterial broadcast
    """

    def config(self):
        self.x_shape = (3, 1, 10, 10)
        self.y_shape = (10, 10)
        self.trans_x = True
        self.trans_y = False


class TestMatMulOpBroadcast6_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 14_6
    test unilaterial broadcast
    """

    def config(self):
        self.x_shape = (10, 10)
        self.y_shape = (3, 1, 10, 10)
        self.trans_x = False
        self.trans_y = True


class TestMatMulOpBroadcast7_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 14_7
    test unilaterial broadcast
    """

    def config(self):
        self.x_shape = (10, 10)
        self.y_shape = (3, 1, 10, 10)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOpBroadcast8_HIGHPERFORMANCE(TestMatMulV2Op_HIGHPERFORMANCE):
    """
    case 14_8
    test unilaterial broadcast
    """

    def config(self):
        self.x_shape = (10, 10)
        self.y_shape = (3, 1, 10, 10)
        self.trans_x = True
        self.trans_y = False


class TestMatMulV2API_HIGHPERFORMANCE(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
