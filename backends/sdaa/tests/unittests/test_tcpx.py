# BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
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

import numpy as np

SEED = 2023

np.random.seed(SEED)
paddle.seed(SEED)


class TestTcpx(unittest.TestCase):
    def test_net(self):
        paddle.set_device("sdaa")
        input_sdaa_1 = paddle.arange(100, dtype="float32").reshape([10, 10])
        input_sdaa_2 = paddle.ones(shape=[10, 5], dtype="float32")
        input_sdaa_1.stop_gradient = False
        input_sdaa_2.stop_gradient = False
        output_sdaa = paddle.matmul(2 * input_sdaa_1, input_sdaa_2)
        output_sdaa.backward()
        grad_sdaa_1 = input_sdaa_1.grad.numpy()
        grad_sdaa_2 = input_sdaa_2.grad.numpy()
        assert "sdaa" in str(output_sdaa.place)
        paddle.set_device("cpu")
        input_cpu_1 = paddle.arange(100, dtype="float32").reshape([10, 10])
        input_cpu_2 = paddle.ones(shape=[10, 5], dtype="float32")
        input_cpu_1.stop_gradient = False
        input_cpu_2.stop_gradient = False
        output_cpu = paddle.matmul(2 * input_cpu_1, input_cpu_2)
        output_cpu.backward()
        grad_cpu_1 = input_cpu_1.grad.numpy()
        grad_cpu_2 = input_cpu_2.grad.numpy()
        np.testing.assert_allclose(output_cpu.numpy(), output_sdaa.numpy(), atol=1e-10)
        np.testing.assert_allclose(grad_cpu_1, grad_sdaa_1, atol=1e-10)
        np.testing.assert_allclose(grad_cpu_2, grad_sdaa_2, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
