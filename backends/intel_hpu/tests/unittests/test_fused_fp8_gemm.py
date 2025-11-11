#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
import paddle
import paddlenlp_ops
from tests.op_test import skip_check_grad_ci


def reference_gemm(X, Y, transpose_x=False, transpose_y=False, scaler=1.0, bias=0.0):
    if transpose_x:
        if X.ndim == 1:
            X = X.reshape((X.size,))
        elif X.ndim == 2:
            X = X.T
        else:
            dim = [i for i in range(len(X.shape))]
            dim[-1], dim[len(X.shape) - 2] = dim[len(X.shape) - 2], dim[-1]
            X = np.transpose(X, tuple(dim))
    if transpose_y:
        if Y.ndim == 1:
            Y = Y.reshape((Y.size,))
        else:
            dim = [i for i in range(len(Y.shape))]
            dim[-1], dim[len(Y.shape) - 2] = dim[len(Y.shape) - 2], dim[-1]
            Y = np.transpose(Y, tuple(dim))

    out = scaler * (X @ Y) + bias

    return out


@skip_check_grad_ci(reason="fused_fp8_gemm ops not support gradient calculation.")
class TestFusedFp8Gemm(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CustomPlace("intel_hpu", 0)
        self.init_params()

    def init_params(self):
        self.shape = (4, 4)
        self.scaler = 1.0
        self.bias = 0.0
        self.tranposeA = False
        self.tranposeB = False

    def check_result(self, np_res, ops_res):
        np.testing.assert_allclose(np_res, ops_res, rtol=0.1, atol=0.1)

    def fused_fp8_gemm_custom(self, x, y, bias):
        x_f16 = paddle.to_tensor(x, dtype="float16")
        y_f16 = paddle.to_tensor(y, dtype="float16")
        bias_f16 = paddle.to_tensor(bias, dtype="float16")

        x_fp8 = x_f16.to(dtype="float8_e4m3fn")
        y_fp8 = y_f16.to(dtype="float8_e4m3fn")
        bias_fp8 = bias_f16.to(dtype="float8_e4m3fn")

        output = paddlenlp_ops.fused_fp8_gemm(
            x=x_fp8,
            y=y_fp8,
            bias=bias_fp8,
            transpose_x=self.tranposeA,
            transpose_y=self.tranposeB,
            scale=self.scaler,
            output_dtype="bfloat16",
            act="identity",
        )

        return output.to(dtype="float16")

    def prepare_input(self):
        np.random.seed(1024)
        a = np.random.rand(*self.shape).astype(np.float16)

        np.random.seed(2048)
        b = np.random.rand(*self.shape).astype(np.float16)

        if self.tranposeA:
            M = a.shape[1]
        else:
            M = a.shape[0]
        if self.tranposeB:
            N = b.shape[0]
        else:
            N = b.shape[1]

        shape_bais = (M, N)

        bias = np.full(shape_bais, self.bias, dtype=np.float16)

        return a, b, bias

    def test_fp8_gemm(self):
        input_a, input_b, bias = self.prepare_input()
        custom_res = self.fused_fp8_gemm_custom(input_a, input_b, bias)
        ref_res = reference_gemm(
            input_a, input_b, self.tranposeA, self.tranposeB, self.scaler, self.bias
        )
        self.check_result(ref_res, custom_res)


@skip_check_grad_ci(reason="fused_fp8_gemm ops not support gradient calculation.")
class TestGemm(TestFusedFp8Gemm):
    def init_params(self):
        self.shape = (4, 4)
        self.scaler = 1.0
        self.bias = 0.0
        self.tranposeA = False
        self.tranposeB = False


class TestGemmTranpose(TestFusedFp8Gemm):
    def init_params(self):
        self.shape = (4, 4)
        self.scaler = 1.0
        self.bias = 0.0
        self.tranposeA = True
        self.tranposeB = True


class TestGemmTranposeScale(TestFusedFp8Gemm):
    def init_params(self):
        self.shape = (4, 4)
        self.scaler = 100.0
        self.bias = 0.0
        self.tranposeA = True
        self.tranposeB = True


class TestGemmTranposeScaleBias(TestFusedFp8Gemm):
    def init_params(self):
        self.shape = (4, 4)
        self.scaler = 100.0
        self.bias = 25.0
        self.tranposeA = True
        self.tranposeB = True


if __name__ == "__main__":
    unittest.main()
