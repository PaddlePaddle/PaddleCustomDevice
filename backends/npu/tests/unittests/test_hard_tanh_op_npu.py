# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
from tests.op_test import OpTest

paddle.enable_static()
SEED = 2021


def ref_hardtanh(x, min=-1.0, max=1.0):
    out = np.copy(x)
    out[np.abs(x - min) < 0.005] = min + 0.02
    out[np.abs(x - max) < 0.005] = max + 0.02
    out = np.minimum(np.maximum(x, min), max)
    return out


class TestHardTanh(OpTest):
    def setUp(self):
        self.op_type = "brelu"
        self.set_npu()
        self.init_dtype()
        self.shape = [10, 12]

        np.random.seed(1024)
        x = np.random.uniform(-5, 5, self.shape).astype(self.dtype)
        t_min = -1.0
        t_max = 1.0
        # The same with TestAbs
        x[np.abs(x - t_min) < 0.005] = t_min + 0.02
        x[np.abs(x - t_max) < 0.005] = t_max + 0.02
        t = np.copy(x)
        t[t < t_min] = t_min
        t[t > t_max] = t_max

        self.inputs = {"X": OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {"t_min": t_min, "t_max": t_max}
        self.outputs = {"Out": t}

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def init_dtype(self):
        self.dtype = "float32"

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            numeric_place=paddle.CPUPlace(),
        )


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
        self.place = paddle.CustomPlace("npu", 0)

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
