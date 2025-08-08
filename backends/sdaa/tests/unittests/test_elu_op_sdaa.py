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

paddle.enable_static()
# SEED = 1024
SEED = 2021


def elu(x, alpha):
    out_ref = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    return out_ref.astype(x.dtype)


class TestELU(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "elu"
        self.python_api = paddle.nn.functional.elu
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        self.init_shape()
        np.random.seed(SEED)

        alpha = self.get_alpha()
        x = np.random.uniform(-3, 3, self.shape).astype(self.dtype)
        out = elu(x, alpha)
        self.inputs = {"X": x}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")

    def get_alpha(self):
        return 1.0

    def test_check_inplace(self):
        self.check_inplace_output_with_place(self.place)


class TestELUAlpha(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "elu"
        self.python_api = paddle.nn.functional.elu
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        self.init_shape()
        np.random.seed(SEED)

        alpha = self.get_alpha()
        self.attrs = {"alpha": alpha}
        x = np.random.uniform(-3, 3, self.shape).astype(self.dtype)
        out = elu(x, alpha)

        self.inputs = {"X": x}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [10, 12]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out", max_relative_error=0.0065)
        # self.check_grad_with_place(self.place, ['X'], 'Out')

    def get_alpha(self):
        return 0.2

    def test_check_inplace(self):
        self.check_inplace_output_with_place(self.place)


class TestELU_ZeroDim(TestELU):
    def init_shape(self):
        self.shape = []


class TestEluFp16(TestELU):
    def init_dtype(self):
        self.dtype = np.float16


class TestEluFp16Alpha(TestELUAlpha):
    def get_alpha(self):
        return 0.2

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestELUAPI(unittest.TestCase):
    # test paddle.nn.ELU, paddle.nn.functional.elu
    def setUp(self):
        np.random.seed(SEED)
        self.x_np = np.random.uniform(-3, 3, [10, 12]).astype("float32")
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)
        self.executed_api()

    def executed_api(self):
        self.elu = paddle.nn.functional.elu

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data("X", [10, 12], dtype="float32")
            out1 = self.elu(x, 1.0)
            m = paddle.nn.ELU(1.0)
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={"X": self.x_np}, fetch_list=[out1, out2])
        out_ref = elu(self.x_np, 1.0)
        for r in res:
            np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = self.elu(x)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.ELU()
        out2 = m(x)
        out_ref = elu(self.x_np, 1.0)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)

        out1 = self.elu(x, 0.2)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.ELU(0.2)
        out2 = m(x)
        out_ref = elu(self.x_np, 0.2)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, self.elu, 1)
            # The input dtype must be float16, float32.
            x_int32 = paddle.static.data(name="x_int32", shape=[10, 12], dtype="int32")
            self.assertRaises(TypeError, self.elu, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.static.data(name="x_fp16", shape=[10, 12], dtype="float16")
            self.elu(x_fp16)


if __name__ == "__main__":
    unittest.main()
