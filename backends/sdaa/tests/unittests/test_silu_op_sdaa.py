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
SEED = 2021


class TestSilu(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "silu"
        self.python_api = paddle.nn.functional.silu
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = x / (np.exp(-x) + 1)

        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(x)}
        self.attrs = {}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestSiluFp16(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "silu"
        self.python_api = paddle.nn.functional.silu
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [13, 14]).astype(self.dtype)
        out = x / (1 + np.exp(-x))

        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(x)}
        self.attrs = {}
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestSiluAPI(unittest.TestCase):
    # test paddle.nn.Silu, paddle.nn.functional.silu
    def setUp(self):
        self.x_np = np.random.uniform(-1, 1, [11, 17]).astype("float32")
        self.place = paddle.CustomPlace("sdaa", 0)

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data("X", [11, 17])
            out1 = paddle.nn.functional.silu(x)
            m = paddle.nn.Silu()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={"X": self.x_np}, fetch_list=[out1, out2])
        out_ref = self.x_np / (1 + np.exp(-self.x_np))
        for r in res:
            np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = paddle.nn.functional.silu(x)
        m = paddle.nn.Silu()
        out2 = m(x)
        out_ref = self.x_np / (1 + np.exp(-self.x_np))
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, paddle.nn.functional.silu, 1)
            # The input dtype must be float16, float32.
            x_int32 = paddle.static.data(name="x_int32", shape=[11, 17], dtype="int32")
            self.assertRaises(TypeError, paddle.nn.functional.silu, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.static.data(name="x_fp16", shape=[11, 17], dtype="float16")
            paddle.nn.functional.silu(x_fp16)


if __name__ == "__main__":
    unittest.main()
