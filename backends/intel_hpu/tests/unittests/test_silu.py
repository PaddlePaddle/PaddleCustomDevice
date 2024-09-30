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
from scipy.special import erf, expit

import paddle
import paddle.base as base
import paddle.nn.functional as F
from tests.op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float
from tests.utils import static_guard


paddle.enable_static()
SEED = 2021


class TestActivation(OpTest):
    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("intel_hpu", 0)

    def setUp(self):
        self.set_npu()
        self.op_type = "exp"
        self.init_dtype()
        self.init_shape()
        self.init_kernel_type()
        self.check_dygraph = True
        self.python_api = paddle.exp

        np.random.seed(2049)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.exp(x)

        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {"Out": out}

    def test_check_output(self):
        check_dygraph = False
        if hasattr(self, "check_dygraph"):
            check_dygraph = self.check_dygraph
        self.check_output_with_place(self.place, check_dygraph=check_dygraph)

    def test_check_grad(self):
        pass

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [11, 17]

    def init_kernel_type(self):
        pass


class TestSilu(TestActivation):
    def setUp(self):
        self.set_npu()
        self.op_type = "silu"
        self.init_dtype()
        self.init_shape()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = x / (np.exp(-x) + 1)
        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {"Out": out}

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


class TestSiluAPI(unittest.TestCase):
    # test paddle.nn.Silu, paddle.nn.functional.silu
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [11, 17]).astype("float32")
        self.place = paddle.CustomPlace("intel_hpu", 0)

    def test_static_api(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data("X", [11, 17])
                out1 = F.silu(x)
                m = paddle.nn.Silu()
                out2 = m(x)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={"X": self.x_np}, fetch_list=[out1, out2])
            out_ref = self.x_np / (1 + np.exp(-self.x_np))
            for r in res:
                np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np)
        out1 = F.silu(x)
        m = paddle.nn.Silu()
        out2 = m(x)
        out_ref = self.x_np / (1 + np.exp(-self.x_np))
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()

if __name__ == "__main__":
    unittest.main()
