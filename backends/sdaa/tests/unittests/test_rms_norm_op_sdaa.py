#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle_sdaa


class TestRMSNormOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(20)
        batch = 32
        cols = 256
        self.x_np = np.random.random([batch, cols])
        self.scale_np = np.random.random([cols])
        self.epsilon = 1e-6

    def naive_rms_norm(self, x_np, scale_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype), stop_gradient=False)
        scale = paddle.to_tensor(scale_np.astype(dtype), stop_gradient=False)

        variance = x.pow(2).mean(-1, keepdim=True)
        out = paddle.rsqrt(variance + self.epsilon) * x
        out = out * scale

        out.backward()

        x_grad = x.grad.detach()
        scale_grad = scale.grad.detach()

        paddle.enable_static()

        return out.numpy(), x_grad.numpy(), scale_grad.numpy()

    def fused_rms_norm(self, x_np, scale_np, dtype):
        paddle.disable_static()
        x = paddle.to_tensor(x_np.astype(dtype), stop_gradient=False)
        scale = paddle.to_tensor(scale_np.astype(dtype), stop_gradient=False)

        out = paddle_sdaa.ops.fused_rms_norm(x, scale, self.epsilon)

        out.backward()

        x_grad = x.grad.detach()
        scale_grad = scale.grad.detach()

        paddle.enable_static()

        return out.numpy(), x_grad.numpy(), scale_grad.numpy()

    def test_rmsnorm_fp32(self):
        naive_out, naive_x_grad, naive_scale_grad = self.naive_rms_norm(
            self.x_np, self.scale_np, "float32"
        )

        out, x_grad, scale_grad = self.fused_rms_norm(
            self.x_np, self.scale_np, "float32"
        )

        np.testing.assert_allclose(
            naive_out,
            out,
            rtol=1e-6,
            atol=1e-6,
        )

        np.testing.assert_allclose(
            naive_x_grad,
            x_grad,
            rtol=1e-6,
            atol=1e-6,
        )

        np.testing.assert_allclose(
            naive_scale_grad,
            scale_grad,
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
