# BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
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

import unittest

import numpy as np
from op_test import OpTest, skip_check_grad_ci
from scipy.special import erf

import paddle
import paddle.base.dygraph as dg
from paddle import static

paddle.enable_static()
SEED = 2021


class TestErfOp(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "erf"
        self.public_python_api = paddle.erf
        self.python_api = paddle.erf

        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_shape()
        self.init_dtype()

        np.random.seed(SEED)
        x = np.random.uniform(-1, 1, size=self.x_shape).astype(self.dtype)
        y_ref = erf(x).astype(self.dtype)
        self.inputs = {"X": x}
        self.outputs = {"Out": y_ref}

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.x_shape = [11, 17]

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

    # def test_check_grad(self):
    #     self.check_grad_with_place(self.place, ['X'], 'Out')


class TestErfLayer(unittest.TestCase):
    def setUp(self):
        self.x = np.random.uniform(-3, 3, size=(11, 17)).astype(np.float32)
        self.y = erf(self.x)

    def _test_dygraph(self, place):
        with dg.guard(place) as g:
            x_var = paddle.to_tensor(self.x)
            y_var = paddle.erf(x_var)
            y_test = y_var.numpy()
        np.testing.assert_allclose(self.y, y_test, rtol=1e-05)

    def test_dygraph(self):
        self._test_dygraph(paddle.CustomPlace("sdaa", 0))
        self._test_dygraph(paddle.CPUPlace())

    def _test_static(self, place):
        mp, sp = static.Program(), static.Program()
        with static.program_guard(mp, sp):
            x = static.data("x", shape=[11, 17], dtype="float32")
            y = paddle.erf(x)

        exe = static.Executor(place)
        exe.run(sp)
        [y_np] = exe.run(mp, feed={"x": self.x}, fetch_list=[y])
        np.testing.assert_allclose(self.y, y_np, rtol=1e-05)

    def test_static(self):
        self._test_static(paddle.CustomPlace("sdaa", 0))
        self._test_static(paddle.CPUPlace())


@skip_check_grad_ci("erf_grad not support on sdaa")
class TestErfFP16OP(TestErfOp):
    def init_dtype(self):
        self.dtype = np.float16

    # def test_check_grad(self):
    #     self.check_grad_with_place(self.place, ['X'], 'Out')


if __name__ == "__main__":
    unittest.main()
