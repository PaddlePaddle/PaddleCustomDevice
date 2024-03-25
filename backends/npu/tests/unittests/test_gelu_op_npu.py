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

import unittest

import numpy as np
import paddle
import paddle.base as base
import paddle.nn.functional as F
from tests.op_test import OpTest

paddle.enable_static()
SEED = 2024


def ref_gelu(x):
    # approximate gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

class TestGelu(OpTest):
    def setUp(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)
        self.op_type = "gelu"
        self.dtype = "float32"
        self.set_attrs()
        self.init_shape()

        x = np.random.uniform(-0.1, 0.1, self.shape)

        out = ref_gelu(x)

        self.inputs = {"X": x}
        self.outputs = {"Out": out}

    def test_check_output(self):
        check_dygraph = False
        if hasattr(self, "check_dygraph"):
            check_dygraph = self.check_dygraph
        self.check_output_with_place(self.place, check_dygraph=check_dygraph)

    def test_check_grad(self):
        check_dygraph = False
        if hasattr(self, "check_dygraph"):
            check_dygraph = self.check_dygraph
        self.check_grad_with_place(
            self.place, ["X"], "Out", check_dygraph=check_dygraph
        )

    def init_shape(self):
        self.shape = [10, 12]

    def set_attrs(self):
        pass


class TestGeluFP16(TestGelu):
    def set_attrs(self):
        self.dtype = "float16"

class TestGeluBF16(TestGelu):
    def set_attrs(self):
        self.dtype = "bfloat16"

class TestGeluAPI(unittest.TestCase):
    # test paddle.nn.gelu, paddle.nn.functional.gelu
    def setUp(self):
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(np.float32)
        self.place = paddle.CustomPlace("npu", 0)

    def test_static_api(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data("X", self.x_np.shape, self.x_np.dtype)
            out1 = F.gelu(x)
            m = paddle.nn.GELU()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={"X": self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_gelu(self.x_np)
        for r in res:
            np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.gelu(x)
        m = paddle.nn.GELU()
        out2 = m(x)
        out_ref = ref_gelu(self.x_np)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_base_api(self):
        with base.program_guard(base.Program()):
            x = paddle.static.data("X", self.x_np.shape, self.x_np.dtype)
            out = paddle.nn.functional.gelu(x)
            exe = base.Executor(self.place)
            res = exe.run(feed={"X": self.x_np}, fetch_list=[out])
        out_ref = ref_gelu(self.x_np)
        np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)

        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out = paddle.nn.functional.gelu(x)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)
        paddle.enable_static()

    # def test_errors(self):
    #     with paddle.static.program_guard(paddle.static.Program()):
    #         # The input type must be Variable.
    #         self.assertRaises(TypeError, F.hardsigmoid, 1)
    #         # The input dtype must be float16, float32, bfloat16.
    #         x_int32 = paddle.static.data(name="x", shape=[12, 10], dtype="int32")
    #         self.assertRaises(TypeError, F.hardsigmoid, x_int32)
    #         # support the input dtype is float16
    #         x_fp16 = paddle.static.data(name="x_fp16", shape=[12, 10], dtype="float16")
    #         F.GELU(x_fp16)


if __name__ == "__main__":
    unittest.main()
