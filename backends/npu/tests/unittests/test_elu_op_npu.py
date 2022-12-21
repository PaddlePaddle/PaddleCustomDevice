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
import paddle.fluid as fluid
import paddle.nn.functional as F
from tests.op_test import OpTest

SEED = 2021
paddle.enable_static()


def ref_elu(
    x,
    alpha=1.0,
):
    """Reference forward implementation"""
    out = np.copy(x)
    out_flat = out.flatten()
    for i in range(out_flat.size):
        if out_flat[i] < 0:
            out_flat[i] = alpha * (np.exp(x) - 1)
    out = out.reshape(x.shape)
    return out


class EluTest(OpTest):
    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def init_kernel_type(self):
        self.dtype = np.float32

    def setUp(self):
        self.set_npu()
        self.init_kernel_type()
        self.op_type = "elu"
        self.python_api = paddle.nn.functional.elu

        alpha = 0.2

        np.random.seed(SEED)
        x = np.random.uniform(-5, 7, [11, 17]).astype(self.dtype)

        result = ref_elu(x, alpha)

        self.inputs = {"X": x}
        self.attrs = {"alpha": alpha}
        self.outputs = {"Out": result}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X"], "Out", max_relative_error=0.5, numeric_grad_delta=0.001
        )


class EluTestFp16(EluTest):
    def init_kernel_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X"], "Out", max_relative_error=0.5, numeric_grad_delta=0.001
        )


class TestEluAPI(unittest.TestCase):
    # test paddle.nn.ELU, paddle.nn.functional.elu
    def setUp(self):
        self.alpha = 0.2
        np.random.seed(SEED)
        self.x_np = np.random.normal(size=[3, 5, 5]).astype(np.float32)
        self.place = paddle.CustomPlace("npu", 0)

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data("X", self.x_np.shape, self.x_np.dtype)
            out1 = F.elu(x, self.alpha)
            elu = paddle.nn.ELU(self.alpha)
            out2 = elu(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={"X": self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_elu(self.x_np, self.alpha)
        for r in res:
            np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.elu(x, self.alpha)
        elu = paddle.nn.ELU(self.alpha)
        out2 = elu(x)
        out_ref = ref_elu(self.x_np, self.alpha)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_fluid_api(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program()):
            x = fluid.data("X", self.x_np.shape, self.x_np.dtype)
            out = F.elu(x, self.alpha)
            exe = fluid.Executor(self.place)
            res = exe.run(feed={"X": self.x_np}, fetch_list=[out])
        out_ref = ref_elu(self.x_np, self.alpha)
        np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
