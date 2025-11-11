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


def get_similarity(x, y):
    x = x.cpu().to("float32")
    y = y.cpu().to("float32")
    return paddle.nn.functional.cosine_similarity(
        x.flatten(), y.flatten(), axis=0
    ).item()


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

    return res.cast("float32").numpy()


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
                mean=0.0, std=0.05, shape=[hidden_size, intermediate_size]
            ).astype(dtype)
            down_weight = paddle.normal(
                mean=0.0, std=0.12, shape=[intermediate_size, hidden_size]
            ).astype(dtype)
            proj_weight = paddle.concat([gate_weight, up_weight], axis=1)

            epsilon = 1e-06

        return x, ln_scales, proj_weight, gate_weight, up_weight, down_weight, epsilon

    def HPU_Fused_RMS_MLP_OP(self, x, ln_scales, proj_weight, down_weight, epsilon):
        fused_mlp_out = paddlenlp_ops.fused_rms_mlp(
            x, ln_scales, proj_weight, down_weight, epsilon
        )
        return fused_mlp_out.cast("float32")

    def NP_Fused_RMS_MLP_OP(
        self, x, ln_scales, gate_weight, up_weight, down_weight, epsilon
    ):
        np_mlp_out_ref = fused_rms_mlp(
            x, ln_scales, gate_weight, up_weight, down_weight, epsilon
        )
        return np_mlp_out_ref

    def check_result(self, np_result, fused_result, atol=1e-2):
        np.testing.assert_allclose(np_result, fused_result, atol=atol)

    def test_fused_mlp(self):
        (
            x,
            ln_scales,
            proj_weight,
            gate_weight,
            up_weight,
            down_weight,
            epsilon,
        ) = self.prepare_input()
        result_fused_mlp = self.HPU_Fused_RMS_MLP_OP(
            x, ln_scales, proj_weight, down_weight, epsilon
        )
        result_np_result = self.NP_Fused_RMS_MLP_OP(
            x, ln_scales, gate_weight, up_weight, down_weight, epsilon
        )

        self.check_result(result_np_result, result_fused_mlp)


class Test_FP8_Fused_MLP_OP(Test_Fused_MLP_OP):
    def HPU_Fused_RMS_MLP_OP(self, x, ln_scales, proj_weight, down_weight, epsilon):
        proj_weight = proj_weight.transpose([1, 0])
        proj_weight0, proj_weight1 = paddle.split(
            proj_weight, num_or_sections=2, axis=0
        )
        proj_weight0 = proj_weight0.astype(paddle.float8_e4m3fn)
        proj_weight1 = proj_weight1.astype(paddle.float8_e4m3fn)

        down_weight = down_weight.transpose([1, 0])
        down_weight = down_weight.astype(paddle.float8_e4m3fn)

        one = paddle.to_tensor([1.0])
        fused_mlp_out = paddlenlp_ops.fused_fp8_rms_mlp(
            x,
            ln_scales,
            proj_weight0,
            proj_weight1,
            down_weight,
            one,
            one,
            one,
            one,
            one,
            epsilon,
        )

        return fused_mlp_out

    def test_fused_mlp(self):
        (
            x,
            ln_scales,
            proj_weight,
            gate_weight,
            up_weight,
            down_weight,
            epsilon,
        ) = self.prepare_input()
        result_np_result = self.NP_Fused_RMS_MLP_OP(
            x, ln_scales, gate_weight, up_weight, down_weight, epsilon
        )

        result_fused_mlp = self.HPU_Fused_RMS_MLP_OP(
            x, ln_scales, proj_weight, down_weight, epsilon
        )

        similarity = get_similarity(
            paddle.to_tensor(result_np_result), result_fused_mlp
        )
        print("similarity = ", similarity)
        assert (
            abs(1 - similarity) < 2e-2
        ), "similarity check fails between fp8 and bf16 outputs"


if __name__ == "__main__":
    unittest.main()
