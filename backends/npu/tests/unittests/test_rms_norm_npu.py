#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import unittest

import numpy as np
import paddle
from paddle.base import core
from tests.op_test import convert_float_to_uint16

from npu_utils import check_soc_version


for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
    if lib.endswith(".so"):
        paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
            lib
        )
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")


class TestNPURMSNormFP32(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CustomPlace("npu", 0)
        self.x_shape = (10, 128)
        self.gamma_shape = (128,)
        self.eps = 1e-6
        self.init_dtype()

    def init_dtype(self):
        self.dtype = "float32"

    def check_result(self, golden_res, fused_res):
        if self.dtype == "float32":
            rtol = 1e-04
            atol = 1e-04
        elif self.dtype == "float16":
            rtol = 1e-03
            atol = 1e-03
        elif self.dtype == "bfloat16":
            rtol = 8e-3
            atol = 8e-3
        else:
            self.assertTrue(
                False,
                msg="NPURMSNorm input dtype only supports bfloat16, \
                     float16 and float32, but got "
                + self.dtype,
            )

        golden_y, golden_rstd, golden_dx, golden_dgamma = golden_res
        fused_y, fused_rstd, fused_dx, fused_dgamma = fused_res

        np.testing.assert_allclose(golden_y, fused_y, rtol=rtol, atol=atol)
        np.testing.assert_allclose(golden_rstd, fused_rstd, rtol=rtol, atol=atol)
        np.testing.assert_allclose(golden_dx, fused_dx, rtol=rtol, atol=atol)
        np.testing.assert_allclose(golden_dgamma, fused_dgamma, rtol=rtol, atol=atol)

    def golden_rms_norm(self, x_, gamma, eps):
        x = x_.cast("float32")
        var = paddle.mean(paddle.pow(x, 2), axis=-1, keepdim=True)
        std = paddle.sqrt(var + eps)
        rstd = 1 / std
        y = x * rstd
        y = y.cast(self.dtype)
        y = y * gamma
        y.backward()
        dx = x_.grad
        dgamma = gamma.grad
        if self.dtype == "bfloat16":
            y = y.cast("float32")
            dx = dx.cast("float32")
            dgamma = dgamma.cast("float32")
        return y.numpy(), rstd.numpy(), dx.numpy(), dgamma.numpy()

    def fused_rms_norm(self, x, gamma, eps):
        y, rstd = core.eager._run_custom_op("rms_norm_npu", x, gamma, eps)
        y.backward()
        dx = x.grad
        dgamma = gamma.grad
        if self.dtype == "bfloat16":
            y = y.cast("float32")
            dx = dx.cast("float32")
        return y.numpy(), rstd.numpy(), dx.numpy(), dgamma.numpy()

    def gen_input(self):
        np_x = np.random.randn(self.x_shape[0], self.x_shape[1]).astype(np.float32)
        np_gamma = np.random.randn(self.gamma_shape[0]).astype(np.float32)
        return np_x, np_gamma

    @check_soc_version
    def test_rms_norm(self):
        np_x, np_gamma = self.gen_input()
        golden_x = paddle.to_tensor(
            np_x, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        golden_gamma = paddle.to_tensor(
            np_gamma, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        fused_x = paddle.to_tensor(
            np_x, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        fused_gamma = paddle.to_tensor(
            np_gamma, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        golden_res = self.golden_rms_norm(golden_x, golden_gamma, self.eps)
        fused_res = self.fused_rms_norm(fused_x, fused_gamma, self.eps)

        self.check_result(golden_res, fused_res)


class TestNPURMSNormFP16(TestNPURMSNormFP32):
    def init_dtype(self):
        self.dtype = "float16"


class TestNPURMSNormBF16(TestNPURMSNormFP32):
    def init_dtype(self):
        self.dtype = "bfloat16"

    def gen_input(self):
        np_x = np.random.randn(self.x_shape[0], self.x_shape[1]).astype(np.float32)
        np_gamma = np.random.randn(self.gamma_shape[0]).astype(np.float32)
        np_uint16_x = convert_float_to_uint16(np_x)
        np_uint16_gamma = convert_float_to_uint16(np_gamma)
        return np_uint16_x, np_uint16_gamma


if __name__ == "__main__":
    np.random.seed(123)
    unittest.main()
