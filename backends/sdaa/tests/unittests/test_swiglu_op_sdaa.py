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

import paddle
import paddle.nn.functional as F
import numpy as np
import unittest
import paddle_sdaa

seed = 1234
paddle.seed(seed)
np.random.seed(seed)

seq_len = 1024
bs = 2
hidden_size = 13824


class TestFusedSwiglu(unittest.TestCase):
    def test_fused_swiglu(self):
        rand_input = np.random.randn(bs, seq_len, hidden_size)

        base_rand_input = paddle.to_tensor(rand_input, dtype="float32")
        base_rand_input.stop_gradient = False
        gate_out, up_out = paddle.chunk(base_rand_input, chunks=2, axis=-1)
        base_result = F.silu(gate_out) * up_out
        base_result.backward(base_result)

        rand_input_fused = paddle.to_tensor(rand_input, dtype="float32")
        rand_input_fused.stop_gradient = False
        fused_result = paddle_sdaa.ops.fused_swiglu(rand_input_fused)
        fused_result.backward(fused_result)

        np.testing.assert_allclose(
            fused_result.numpy(), base_result.numpy(), rtol=1e-5, atol=1e-8
        )

        np.testing.assert_allclose(
            rand_input_fused.grad.numpy(),
            base_rand_input.grad.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
