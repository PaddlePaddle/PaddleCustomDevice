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
import paddle
import paddle_sdaa

import os
import numpy as np

SEED = 2023

np.random.seed(SEED)
paddle.seed(SEED)


class TestCustomMatmul(unittest.TestCase):
    def __test_gemm(self, a, b):
        # matmul input should be float32 to excuted sgemmex
        a_gt = paddle.clone(a).cast("float32")
        b_gt = paddle.clone(b).cast("float32")

        a_custom = paddle.clone(a)
        b_custom = paddle.clone(b)

        a_gt.stop_gradient = False
        b_gt.stop_gradient = False
        a_custom.stop_gradient = False
        b_custom.stop_gradient = False

        c_gt = paddle.matmul(a_gt, b_gt)
        c_gt.backward()

        c_custom = paddle_sdaa.ops.matmul(a_custom, b_custom)
        c_custom.backward()

        np.testing.assert_array_equal(c_custom.numpy(), c_gt.numpy())
        np.testing.assert_array_equal(a_custom.grad.numpy(), a_gt.grad.numpy())
        np.testing.assert_array_equal(b_custom.grad.numpy(), b_gt.grad.numpy())

    def __test_gemm_shape(self, M, N, K, dtype="float32"):
        a = paddle.randn([1, M, K], dtype=dtype)
        b = paddle.randn([K, N], dtype=dtype)
        self.__test_gemm(a, b)

        a = paddle.randn([M, K], dtype=dtype)
        b = paddle.randn([K, N], dtype=dtype)
        self.__test_gemm(a, b)

        if dtype != "float16":
            # when input dtype is fp16, executed function of blas is not same. so the result is not equal.
            a = paddle.randn([1, 3, M, K], dtype=dtype)
            b = paddle.randn([K, N], dtype=dtype)
            self.__test_gemm(a, b)

    def test_llama_gemm(self):
        paddle.set_device("sdaa")
        os.environ["HIGH_PERFORMANCE_GEMM"] = "1"
        os.environ["SDAA_LAUNCH_BLOCKING"] = "1"
        os.environ["TECO_LAUNCH_BLOCKING"] = "1"
        self.__test_gemm_shape(M=4096, N=3840, K=5120)
        self.__test_gemm_shape(M=4096, N=6912, K=5120)
        self.__test_gemm_shape(M=4096, N=5120, K=3456)
        self.__test_gemm_shape(M=4096, N=3840, K=5120, dtype="float16")
        self.__test_gemm_shape(M=4096, N=6912, K=5120, dtype="float16")
        self.__test_gemm_shape(M=4096, N=5120, K=3456, dtype="float16")


if __name__ == "__main__":
    unittest.main()
