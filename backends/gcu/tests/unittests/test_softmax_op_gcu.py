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

import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
from tests.op_test import OpTest
from tests.utils import static_guard


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = (x - np.max(x)).clip(-64.0)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def ref_softmax(x, axis=None, dtype=None):
    x_t = x.copy()
    if dtype is not None:
        x_t = x_t.astype(dtype)
    if axis is None:
        axis = -1
    return np.apply_along_axis(stable_softmax, axis, x_t)


class TestSoftmaxOp(OpTest):
    def get_x_shape(self):
        return [10, 10]

    def get_axis(self):
        return -1

    def setUp(self):
        self.op_type = "softmax"
        self.set_device()
        self.use_cudnn = False
        self.use_mkldnn = False
        self.dtype = np.float32
        self.init_kernel_type()
        self.shape = self.get_x_shape()
        self.axis = self.get_axis()

        np.random.seed(0)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.apply_along_axis(stable_softmax, self.axis, x)

        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {"Out": out}
        self.attrs = {
            "axis": self.axis,
            "use_cudnn": self.use_cudnn,
            "use_mkldnn": self.use_mkldnn,
        }
        self.enable_cinn = True

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def init_kernel_type(self):
        pass

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out", max_relative_error=0.01)


class TestSoftmaxOpFp64(TestSoftmaxOp):
    def init_dtype(self):
        self.dtype = np.float64


class TestSoftmaxOp2(TestSoftmaxOp):
    def get_x_shape(self):
        return [2, 3, 4, 5]


class TestSoftmaxOp3(TestSoftmaxOp):
    def get_x_shape(self):
        return [2, 3, 4, 5]

    def get_axis(self):
        return 0


class TestSoftmaxOp4(TestSoftmaxOp):
    def get_x_shape(self):
        return [2, 3, 4, 5]

    def get_axis(self):
        return 1


class TestSoftmaxOp5(TestSoftmaxOp):
    def get_x_shape(self):
        return [2, 3, 4, 5]

    def get_axis(self):
        return 2


class TestSoftmaxOp6(TestSoftmaxOp):
    def get_x_shape(self):
        return [2, 3, 4, 5]

    def get_axis(self):
        return 3


class TestSoftmaxAPI(unittest.TestCase):
    def setUp(self):
        self.place = (
            paddle.CustomPlace("gcu", 0)
            if ("gcu" in paddle.base.core.get_all_custom_device_type())
            else paddle.CPUPlace()
        )
        self.x_np = np.random.uniform(-1.0, 1.0, [2, 3, 4, 5]).astype("float32")
        self.out_ref = np.apply_along_axis(stable_softmax, -1, self.x_np)
        self.executed_api()

    def executed_api(self):
        self.softmax = F.softmax

    def test_dygraph_check(self):
        paddle.disable_static(self.place)

        x = paddle.to_tensor(self.x_np)
        out1 = self.softmax(x)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.Softmax()
        out2 = m(x)
        out_ref = ref_softmax(self.x_np, axis=-1, dtype=None)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

        out1 = self.softmax(x, axis=0)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.Softmax(axis=0)
        out2 = m(x)
        out_ref = ref_softmax(self.x_np, axis=0, dtype=None)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

        out = self.softmax(x, dtype=np.float32)
        out_ref = ref_softmax(self.x_np, axis=-1, dtype=np.float32)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)

        paddle.enable_static()

    def test_error(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                # The input type must be Variable.
                self.assertRaises(TypeError, self.softmax, 1)
                # The input dtype must be float16, float32, float64.
                x_int32 = paddle.static.data(
                    name="x_int32", shape=[2, 3], dtype="int32"
                )
                self.assertRaises(TypeError, self.softmax, x_int32)
                # support the input dtype is float16
                x_fp16 = paddle.static.data(
                    name="x_fp16", shape=[2, 3], dtype="float16"
                )
                self.softmax(x_fp16)


if __name__ == "__main__":
    unittest.main()
