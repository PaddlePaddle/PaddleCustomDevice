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

paddle.device.set_device("intel_hpu")


def fused_mlp(
    x,
    gate_weight,
    up_weight,
    down_weight,
):
    def swiglu_naive(x, up=None):
        if up is not None:
            gate = x
        else:
            gate, up = paddle.chunk(x, chunks=2, axis=-1)
        silu = gate / (paddle.exp(-gate) + 1)
        return silu * up

    gate = paddle.matmul(x, gate_weight)
    up = paddle.matmul(x, up_weight)
    swiglu = swiglu_naive(x=gate, up=up)
    res = paddle.matmul(swiglu, down_weight)

    return res.numpy()


class Test_Fused_MLP_OP(unittest.TestCase):
    def setUp(self):
        self.init_dtype()
        self.batch_size = 8
        self.seq_length = 128

    def set_hpu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))
        paddle.seed(20241214)

    def init_dtype(self):
        self.dtype = "float32"

    def prepare_input(
        self,
        batch_size=8,
        seqence_len=128,
        hidden_size=256,
        intermediate_size=512,
        dtype="bfloat16",
    ):
        with paddle.no_grad():
            x = paddle.rand(
                [batch_size, seqence_len, hidden_size], dtype=paddle.float32
            ).to(paddle.bfloat16)

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

        return x, gate_weight, up_weight, down_weight, proj_weight

    def HPU_Fused_MLP_OP(self, x, gate_weight, up_weight, down_weight):
        fused_mlp_out = paddlenlp_ops.fused_mlp(
            x,
            gate_weight,
            up_weight,
            down_weight,
        )
        return fused_mlp_out

    def HPU_Fused_GateUp_MLP_OP(self, x, down_weight, proj_weight):
        fused_gateup_mlp_out = paddlenlp_ops.fused_mlp(
            x,
            proj_weight,
            None,
            down_weight,
        )
        return fused_gateup_mlp_out

    def NP_Fused_MLP_OP(self, x, gate_weight, up_weight, down_weight):
        np_mlp_out_ref = fused_mlp(
            x,
            gate_weight,
            up_weight,
            down_weight,
        )
        return np_mlp_out_ref

    def check_result(self, np_result, fused_result, fused_gate_up_result):
        np.testing.assert_allclose(np_result, fused_result)
        np.testing.assert_allclose(np_result, fused_gate_up_result)

    def test_fused_mlp(self):
        x, gate_weight, up_weight, down_weight, proj_weight = self.prepare_input()

        result_fused_mlp = self.HPU_Fused_MLP_OP(x, gate_weight, up_weight, down_weight)
        result_fused_gate_up_mlp = self.HPU_Fused_GateUp_MLP_OP(
            x, down_weight, proj_weight
        )
        result_np_result = self.NP_Fused_MLP_OP(x, gate_weight, up_weight, down_weight)

        self.check_result(result_np_result, result_fused_mlp, result_fused_gate_up_mlp)


if __name__ == "__main__":
    unittest.main()
