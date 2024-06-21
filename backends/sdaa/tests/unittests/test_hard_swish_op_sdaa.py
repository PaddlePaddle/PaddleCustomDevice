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
import paddle.nn.functional as F
import paddle

from op_test import OpTest

import numpy as np
import unittest

paddle.enable_static()
SEED = 2021
np.random.seed(SEED)


def ref_hardswish(x, threshold=6.0, scale=6.0, offset=3.0):
    x_dtype = x.dtype
    if x_dtype == "float16":
        x_dtype = "float16"
        x = x.astype("float32")
    return (x * np.minimum(np.maximum(x + offset, 0.0), threshold) / scale).astype(
        x_dtype
    )


def ref_hard_swish_grad(x, threshold=6.0, scale=6.0, offset=3.0):
    dout = np.full_like(x, fill_value=1.0 / x.size)
    tmp = ((x + offset) < threshold).astype(x.dtype)
    dx = dout * (
        ((x + offset) > 0).astype(x.dtype) * (2 * x + offset) * tmp / scale + 1.0 - tmp
    )
    return dx


class TestHardSwish(OpTest):
    def setUp(self):
        self.op_type = "hard_swish"
        self.init_dtype()
        self.init_shape()
        self.python_api = paddle.nn.functional.hardswish

        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True

        np.random.seed(1024)
        x = np.random.uniform(-6, 6, self.shape).astype(self.dtype)
        threshold = 6.0
        scale = 6.0
        offset = 3.0
        x[np.abs(x + offset) < 0.005] = 0.02
        x[np.abs(x - threshold + offset) < 0.005] = threshold - offset + 0.02
        out = ref_hardswish(x, threshold, scale, offset)

        self.x_grad = ref_hard_swish_grad(x, threshold, scale, offset)
        self.inputs = {"X": x}
        self.attrs = {"threshold": threshold, "scale": scale, "offset": offset}
        self.outputs = {"Out": out}

    def init_shape(self):
        self.shape = [10, 12]

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            user_defined_grads=[self.x_grad],
        )


class TestHardSwishFP16(TestHardSwish):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")


class TestHardswishAPI(unittest.TestCase):
    # test paddle.nn.Hardswish, paddle.nn.functional.hardswish
    def setUp(self):
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(np.float32)
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True

    def test_static_api(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data("X", self.x_np.shape, self.x_np.dtype)
            out1 = F.hardswish(x)
            m = paddle.nn.Hardswish()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={"X": self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_hardswish(self.x_np)
        for r in res:
            np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor([11648.0, 11448.0])
        out1 = F.hardswish(x)
        m = paddle.nn.Hardswish()
        out2 = m(x)
        out_ref = [11648.0, 11448.0]
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.hardswish, 1)
            # The input dtype must be float16, float32.
            x_int32 = paddle.static.data(name="x_int32", shape=[12, 10], dtype="int32")
            self.assertRaises(TypeError, F.hardswish, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.static.data(name="x_fp16", shape=[12, 10], dtype="float16")
            F.hardswish(x_fp16)


if __name__ == "__main__":
    unittest.main()
