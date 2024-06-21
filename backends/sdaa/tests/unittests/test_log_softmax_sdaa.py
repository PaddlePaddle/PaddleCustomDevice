# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

from __future__ import print_function
import numpy as np
import unittest

from op_test import OpTest
import paddle
import paddle.nn.functional as F

paddle.enable_static()


def ref_log_softmax(x):
    shiftx = x - np.max(x)
    out = shiftx - np.log(np.exp(shiftx).sum())
    return out


def ref_log_softmax_grad(x, axis):
    if axis < 0:
        axis += len(x.shape)
    out = np.apply_along_axis(ref_log_softmax, axis, x)
    axis_dim = x.shape[axis]
    dout = np.full_like(x, fill_value=1.0 / x.size)
    dx = dout - np.exp(out) * dout.copy().sum(axis=axis, keepdims=True).repeat(
        axis_dim, axis=axis
    )
    return dx


class TestLogSoftmaxOp(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.python_api = paddle.nn.functional.log_softmax
        self.op_type = "log_softmax"
        self.dtype = np.float32
        self.shape = [64, 1000]
        self.axis = -1
        self.set_attrs()
        self.set_dtype()
        x = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
        if self.axis < 0:
            axis = self.axis + len(self.shape)
            out = np.apply_along_axis(ref_log_softmax, axis, x)
            self.x_grad = ref_log_softmax_grad(x, axis)
        else:
            out = np.apply_along_axis(ref_log_softmax, self.axis, x)
            self.x_grad = ref_log_softmax_grad(x, self.axis)
        self.inputs = {"X": x}
        self.outputs = {"Out": out}
        self.attrs = {"axis": self.axis}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def set_attrs(self):
        pass

    def set_dtype(self):
        pass

    def test_check_output(self):
        if self.dtype == np.float16:
            self.check_output_with_place(self.place, atol=1e-2)
        else:
            self.check_output_with_place(self.place)

    def test_check_grad(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place,
                ["X"],
                ["Out"],
                user_defined_grads=[self.x_grad],
                max_relative_error=0.02,
            )
        else:
            self.check_grad_with_place(
                self.place, ["X"], ["Out"], user_defined_grads=[self.x_grad]
            )


class TestLogSoftmaxZero(TestLogSoftmaxOp):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.python_api = paddle.nn.functional.log_softmax
        self.op_type = "log_softmax"
        self.dtype = np.float32
        self.shape = []
        self.axis = -1
        self.set_attrs()
        self.set_dtype()
        x = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
        out = np.full(shape=[], fill_value=0.0)
        self.x_grad = np.full(shape=[], fill_value=0.0)
        self.inputs = {"X": x}
        self.outputs = {"Out": out}
        self.attrs = {"axis": self.axis}


def test_class(op_type, typename):
    class TestLogSoftmaxShape(TestLogSoftmaxOp):
        def set_attrs(self):
            self.shape = [12, 10]

        def set_dtype(self):
            self.dtype = typename

    cls_name = "{0}_{1}_1".format(op_type, typename)
    TestLogSoftmaxShape.__name__ = cls_name
    globals()[cls_name] = TestLogSoftmaxShape


def test_class2(op_type, typename):
    class TestLogSoftmaxAxis(TestLogSoftmaxOp):
        def set_attrs(self):
            self.shape = [3, 4, 5]

        def set_dtype(self):
            self.dtype = typename

    cls_name = "{0}_{1}_2".format(op_type, typename)

    TestLogSoftmaxAxis.__name__ = cls_name
    globals()[cls_name] = TestLogSoftmaxAxis


def test_class3(op_type, typename, axis):
    class TestLogSoftmaxAxis(TestLogSoftmaxOp):
        def set_attrs(self):
            self.shape = [2, 3, 4, 5]
            self.axis = axis

        def set_dtype(self):
            self.dtype = typename

    cls_name = "{0}_{1}_3".format(op_type, typename)

    TestLogSoftmaxAxis.__name__ = cls_name
    globals()[cls_name] = TestLogSoftmaxAxis


for _typename in {np.float32, np.float16}:
    test_class("logsoftmax", _typename)
    test_class2("logsoftmax", _typename)
    for i in range(4):
        test_class3("logsoftmax", _typename, i)
        test_class3("logsoftmax", _typename, -i)


class TestNNLogSoftmaxAPI(unittest.TestCase):
    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1.0, 1.0, self.x_shape).astype(np.float32)
        self.place = (
            paddle.CustomPlace("sdaa", 0)
            if ("sdaa" in paddle.base.core.get_all_custom_device_type())
            else paddle.CPUPlace()
        )

    def check_api(self, axis=-1):
        ref_out = np.apply_along_axis(ref_log_softmax, axis, self.x)

        logsoftmax = paddle.nn.LogSoftmax(axis)
        # test static api
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x", shape=self.x_shape)
            y = logsoftmax(x)
            exe = paddle.static.Executor(self.place)
            out = exe.run(feed={"x": self.x}, fetch_list=[y])
        self.assertTrue(np.allclose(out[0], ref_out))

        # test dygrapg api
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        y = logsoftmax(x)
        self.assertTrue(np.allclose(y.numpy(), ref_out))
        paddle.enable_static()

    def test_check_api(self):
        self.check_api(-1)


class TestNNFunctionalLogSoftmaxAPI(unittest.TestCase):
    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.place = (
            paddle.CustomPlace("sdaa", 0)
            if ("sdaa" in paddle.base.core.get_all_custom_device_type())
            else paddle.CPUPlace()
        )

    def check_api(self, axis=-1, dtype=None):
        x = self.x.copy()
        if dtype is not None:
            x = x.astype(dtype)
        ref_out = np.apply_along_axis(ref_log_softmax, axis, x)
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x", shape=self.x_shape)
            y = F.log_softmax(x, axis, dtype)
            exe = paddle.static.Executor(self.place)
            out = exe.run(feed={"x": self.x}, fetch_list=[y])
        self.assertTrue(np.allclose(out[0], ref_out))

        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        y = F.log_softmax(x, axis, dtype)
        self.assertTrue(np.allclose(y.numpy(), ref_out), True)
        paddle.enable_static()

    def test_check_api(self):
        self.check_api(-1)


if __name__ == "__main__":
    unittest.main()
