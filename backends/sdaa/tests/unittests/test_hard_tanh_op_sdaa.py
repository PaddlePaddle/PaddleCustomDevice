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

import unittest
import numpy as np
import paddle
import paddle.nn.functional as F
from op_test import OpTest, skip_check_grad_ci, convert_float_to_uint16

paddle.enable_static()
SEED = 2021


def ref_hardtanh(x, min=-1.0, max=1.0):
    out = np.copy(x)
    out[np.abs(x - min) < 0.005] = min + 0.02
    out[np.abs(x - max) < 0.005] = max + 0.02
    out = np.minimum(np.maximum(x, min), max)
    return out


@skip_check_grad_ci("hard_tanh_grad not support on sdaa")
class TestHardTanh(OpTest):
    def setUp(self):
        self.op_type = "brelu"
        self.python_api = paddle.nn.functional.hardtanh
        self.set_sdaa()
        self.init_dtype()
        self.shape = [10, 12]

        np.random.seed(1024)
        x = np.random.uniform(-5, 5, self.shape).astype(self.dtype)
        t_min = -1.0
        t_max = 1.0

        x[np.abs(x - t_min) < 0.005] = t_min + 0.02
        x[np.abs(x - t_max) < 0.005] = t_max + 0.02
        t = np.copy(x)
        t[t < t_min] = t_min
        t[t > t_max] = t_max

        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(x)}
        self.attrs = {"t_min": t_min, "t_max": t_max}
        self.outputs = {"Out": t}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_dtype(self):
        self.dtype = "float32"

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


@skip_check_grad_ci("hard_tanh_grad not support on sdaa")
class TestHardTanhBF16(OpTest):
    def setUp(self):
        self.op_type = "brelu"
        self.python_api = paddle.nn.functional.hardtanh
        self.set_sdaa()
        self.init_dtype()
        self.shape = [10, 12]

        np.random.seed(1024)
        x = np.random.uniform(-5, 5, self.shape).astype(np.float32)
        t_min = -1.0
        t_max = 1.0

        x[np.abs(x - t_min) < 0.005] = t_min + 0.02
        x[np.abs(x - t_max) < 0.005] = t_max + 0.02
        t = np.copy(x)
        t[t < t_min] = t_min
        t[t > t_max] = t_max

        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(convert_float_to_uint16(x))}
        self.attrs = {"t_min": t_min, "t_max": t_max}
        self.outputs = {"Out": convert_float_to_uint16(t)}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_dtype(self):
        self.dtype = np.uint16

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-2)

    def test_check_grad(self):
        pass


class TestHardTanhFP16(TestHardTanh):
    def set_attrs(self):
        self.dtype = "float16"


class TestHardTanhTminTmax(TestHardTanh):
    def set_attrs(self):
        self.t_min = -1.2
        self.t_max = 2.4


class TestHardTanh_ZeroDim(TestHardTanh):
    def init_shape(self):
        self.shape = []


class TestHardTanhAPI(unittest.TestCase):
    # test paddle.nn.Hardtanh, paddle.nn.functional.hardtanh
    def setUp(self):
        self.x_np = np.random.uniform(-5, 5, [10, 12]).astype(np.float32)
        self.place = paddle.CustomPlace("sdaa", 0)

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data("X", self.x_np.shape, self.x_np.dtype)
            out1 = F.hardtanh(x)
            m = paddle.nn.Hardtanh()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={"X": self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_hardtanh(self.x_np)
        for r in res:
            np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.hardtanh(x)
        m = paddle.nn.Hardtanh()
        out2 = m(x)
        out_ref = ref_hardtanh(self.x_np)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
