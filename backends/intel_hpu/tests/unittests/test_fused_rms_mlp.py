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

import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)


def fused_rms_mlp(
    x,
    ln_scales,
    gate_weight,
    up_weight,
    down_weight,
    epsilon,
):
    def swiglu_naive(x, up=None):
        if up is not None:
            gate = x
        else:
            gate, up = paddle.chunk(x, chunks=2, axis=-1)
        silu = gate / (paddle.exp(-gate) + 1)
        return silu * up

    hidden_states = paddle.incubate.nn.functional.fused_rms_norm(
        x, ln_scales, None, epsilon, 2
    )[0]

    gate = paddle.matmul(hidden_states, gate_weight)
    up = paddle.matmul(hidden_states, up_weight)
    swiglu = swiglu_naive(x=gate, up=up)
    res = paddle.matmul(swiglu, down_weight)

    return res.numpy()


class Test_Fused_MLP_OP(unittest.TestCase):
    def setUp(self):
        self.init_dtype()
        self.batch_size = 2
        self.seq_length = 16

    def set_hpu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))
        paddle.seed(20241213)

    def init_dtype(self):
        self.dtype = "float32"

    def prepare_input(
        self,
        batch_size=2,
        seqence_len=16,
        hidden_size=256,
        intermediate_size=1024,
        dtype="bfloat16",
    ):
        with paddle.no_grad():
            x = paddle.rand(
                [batch_size, seqence_len, hidden_size], dtype=paddle.float32
            ).to(paddle.bfloat16)

            ln_scales = paddle.rand([hidden_size], dtype=paddle.bfloat16)
            gate_weight = paddle.normal(
                mean=0.0, std=0.02, shape=[hidden_size, intermediate_size]
            ).astype(dtype)
            up_weight = paddle.normal(
                mean=1.0, std=0.05, shape=[hidden_size, intermediate_size]
            ).astype(dtype)
            down_weight = paddle.normal(
                mean=0.5, std=0.12, shape=[intermediate_size, hidden_size]
            ).astype(dtype)
            proj_weight = paddle.concat([gate_weight, up_weight], axis=1)

            epsilon = 1e-06

        return x, ln_scales, proj_weight, gate_weight, up_weight, down_weight, epsilon

    def HPU_Fused_RMS_MLP_OP(self):
        (
            x,
            ln_scales,
            proj_weight,
            _,
            _,
            down_weight,
            epsilon,
        ) = self.prepare_input()

        fused_mlp_out = paddlenlp_ops.fused_rms_mlp(
            x, ln_scales, proj_weight, down_weight, epsilon
        )
        return fused_mlp_out

    def NP_Fused_RMS_MLP_OP(self):
        (
            x,
            ln_scales,
            _,
            gate_weight,
            up_weight,
            down_weight,
            epsilon,
        ) = self.prepare_input()

        np_mlp_out_ref = fused_rms_mlp(
            x, ln_scales, gate_weight, up_weight, down_weight, epsilon
        )
        return np_mlp_out_ref

    def check_result(self, np_result, fused_result):
        np.testing.assert_allclose(np_result, fused_result)

    def test_fused_mlp(self):
        result_fused_mlp = self.HPU_Fused_RMS_MLP_OP()
        result_np_result = self.NP_Fused_RMS_MLP_OP()

        self.check_result(result_np_result, result_fused_mlp)


if __name__ == "__main__":
    unittest.main()
