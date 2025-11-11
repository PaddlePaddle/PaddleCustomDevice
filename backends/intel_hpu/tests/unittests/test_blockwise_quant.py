# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


@skip_check_grad_ci(
    reason="fused_blockwise_quant ops not support gradient calculation."
)
class TestFusedQuant(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CustomPlace("intel_hpu", 0)
        self.init_params()

    def init_params(self):
        self.shape = (256, 256)

    def prepare_input(self):
        np.random.seed(1024)
        x = np.random.rand(*self.shape).astype(np.float16)
        x_bf16 = paddle.to_tensor(x, dtype="bfloat16")

        return x_bf16

    def test_quant(self):
        input = self.prepare_input()
        quant, scale = paddlenlp_ops.fused_blockwise_quant(input)
        ref_quant, ref_scale = paddlenlp_ops.blockwise_quant_to_fp8(input)

        assert paddle.allclose(
            quant.to(paddle.float32).cpu(),
            ref_quant.to(paddle.float32).cpu(),
            rtol=1e-5,
        ), "quant is not match to ref quant"

        assert paddle.allclose(
            scale.to(paddle.float32).cpu(),
            ref_scale.to(paddle.float32).cpu(),
            rtol=1e-5,
        ), "scale is not match to ref scale"


if __name__ == "__main__":
    unittest.main()
