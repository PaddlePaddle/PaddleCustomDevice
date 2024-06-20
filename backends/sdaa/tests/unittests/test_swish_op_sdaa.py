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
from scipy.special import expit
import paddle.nn.functional as F
from paddle import base, static

paddle.enable_static()
SEED = 1024


def ref_swish(x):
    out = x * expit(x)
    return out


class TestSwishOp(OpTest):
    def setUp(self):
        self.op_type = "swish"
        self.python_api = paddle.nn.functional.swish
        self.set_sdaa()
        self.init_dtype()
        self.init_shape()
        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = ref_swish(x)

        self.inputs = {"X": x}
        self.attrs = {"beta": 1.0}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestSwishOpFp16(TestSwishOp):
    def init_dtype(self):
        self.dtype = np.float16


class TestSwish_ZeroDim(TestSwishOp):
    def init_shape(self):
        self.shape = []


class TestSwishAPI(unittest.TestCase):
    # test paddle.nn.Swish, paddle.nn.functional.swish
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(np.float32)
        self.place = paddle.CustomPlace("sdaa", 0)

    def test_static_api(self):
        paddle.set_device("sdaa")
        paddle.enable_static()
        with paddle.static.program_guard(static.Program(), static.Program()):
            x = paddle.static.data("X", self.x_np.shape, self.x_np.dtype)
            out1 = F.swish(x)
            swish = paddle.nn.Swish()
            out2 = swish(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={"X": self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_swish(self.x_np)
        for r in res:
            np.testing.assert_allclose(out_ref, r, rtol=1e-05)
        paddle.disable_static()

    def test_dygraph_api(self):
        paddle.disable_static()
        paddle.set_device("sdaa")
        x = paddle.to_tensor(self.x_np)
        out1 = F.swish(x)
        swish = paddle.nn.Swish()
        out2 = swish(x)
        out_ref = ref_swish(self.x_np)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_base_api(self):
        paddle.set_device("sdaa")
        paddle.enable_static()
        with paddle.static.program_guard(static.Program(), static.Program()):
            x = paddle.static.data("X", self.x_np.shape, self.x_np.dtype)
            out = paddle.nn.functional.swish(x)
            exe = base.Executor(self.place)
            res = exe.run(feed={"X": self.x_np}, fetch_list=[out])
        out_ref = ref_swish(self.x_np)
        np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)
        paddle.disable_static()

    def test_errors(self):
        paddle.set_device("sdaa")
        with paddle.static.program_guard(static.Program(), static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.swish, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.static.data(name="x_int32", shape=[12, 10], dtype="int32")
            self.assertRaises(TypeError, F.swish, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.static.data(name="x_fp16", shape=[12, 10], dtype="float16")
            F.swish(x_fp16)


if __name__ == "__main__":
    unittest.main()
