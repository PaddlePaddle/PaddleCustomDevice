# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import argparse

import paddle
import paddlenlp_ops
import paddle.profiler as profiler
import numpy as np

paddle.device.set_device("intel_hpu")

paddle.seed(20241213)


def init_data(
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
        residual = paddle.rand(
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

    return (
        x,
        ln_scales,
        proj_weight,
        gate_weight,
        up_weight,
        down_weight,
        residual,
        epsilon,
    )


def ref_rms_mlp(
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


class refRmsMlpOP(paddle.nn.Layer):
    def __init__(
        self, x, ln_scales, gate_weight, up_weight, down_weight, residual, epsilon
    ):
        super().__init__()

        self.x = x + residual
        self.residual = x
        self.ln_scales = ln_scales
        self.gate_weight = gate_weight
        self.up_weight = up_weight
        self.down_weight = down_weight
        self.epsilon = epsilon

    def forward(self):
        mlp_out_ref = ref_rms_mlp(
            self.x,
            self.ln_scales,
            self.gate_weight,
            self.up_weight,
            self.down_weight,
            self.epsilon,
        )
        return mlp_out_ref


class fusedRmsMlpOP(paddle.nn.Layer):
    def __init__(self, x, ln_scales, proj_weight, down_weight, residual, epsilon):
        super().__init__()

        self.x = x + residual
        self.residual = x
        self.ln_scales = ln_scales
        self.proj_weight = proj_weight
        self.down_weight = down_weight
        self.epsilon = epsilon

    def forward(self):
        fused_rms_mlp_out = paddlenlp_ops.fused_rms_mlp(
            self.x,
            self.ln_scales,
            self.proj_weight,
            self.down_weight,
            self.epsilon,
        )
        return fused_rms_mlp_out


class fusedRmsMlpResOP(paddle.nn.Layer):
    def __init__(self, x, ln_scales, proj_weight, down_weight, residual, epsilon):
        super().__init__()

        self.x = x
        self.ln_scales = ln_scales
        self.proj_weight = proj_weight
        self.down_weight = down_weight
        self.residual = residual
        self.epsilon = epsilon

    def forward(self):
        fused_rms_mlp_out = paddlenlp_ops.fused_rms_mlp_res(
            self.x,
            self.ln_scales,
            self.proj_weight,
            self.down_weight,
            self.residual,
            self.epsilon,
        )
        return fused_rms_mlp_out


class fusedFP8RmsMlpResOP(paddle.nn.Layer):
    def __init__(self, x, ln_scales, proj_weight, down_weight, residual, epsilon):
        super().__init__()

        self.x = x
        self.ln_scales = ln_scales
        self.residual = residual
        self.epsilon = epsilon

        proj_weight = proj_weight.transpose([1, 0])
        proj_weight0, proj_weight1 = paddle.split(
            proj_weight, num_or_sections=2, axis=0
        )
        self.proj_weight0 = proj_weight0.astype(paddle.float8_e4m3fn)
        self.proj_weight1 = proj_weight1.astype(paddle.float8_e4m3fn)
        down_weight = down_weight.transpose([1, 0])
        self.down_weight = down_weight.astype(paddle.float8_e4m3fn)

    def forward(self):
        scale_one = paddle.to_tensor([1.0], dtype=paddle.float32)
        fused_rms_mlp_out = paddlenlp_ops.fused_fp8_rms_mlp_res(
            self.x,
            self.ln_scales,
            self.proj_weight0,
            self.proj_weight1,
            self.down_weight,
            self.residual,
            scale_one,
            scale_one,
            scale_one,
            scale_one,
            scale_one,
            self.epsilon,
        )
        return fused_rms_mlp_out


def run_profile(my_profile_func):
    prof = profiler.Profiler(
        targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.CUSTOM_DEVICE],
        on_trace_ready=profiler.export_chrome_tracing("./profile"),
    )
    prof.start()
    for iter in range(20):
        with paddle.no_grad():
            rms_mlp_out = my_profile_func()
    prof.stop()


def run_accuracy_check(
    x,
    ln_scales,
    proj_weight,
    gate_weight,
    up_weight,
    down_weight,
    residual,
    epsilon,
):
    ref_rms_mlp = refRmsMlpOP(
        x, ln_scales, gate_weight, up_weight, down_weight, residual, epsilon
    )
    fused_rms_mlp = fusedRmsMlpOP(
        x, ln_scales, proj_weight, down_weight, residual, epsilon
    )
    fused_rms_mlp_residual = fusedRmsMlpResOP(
        x, ln_scales, proj_weight, down_weight, residual, epsilon
    )
    fused_fp8_rms_mlp_residual = fusedFP8RmsMlpResOP(
        x, ln_scales, proj_weight, down_weight, residual, epsilon
    )

    golden_res = ref_rms_mlp()
    fused_rms_res = fused_rms_mlp()
    fused_rms_mlp_residual_res = fused_rms_mlp_residual()
    fused_fp8_rms_mlp_residual_res = fused_fp8_rms_mlp_residual()

    # Check FP8 accuracy
    close_mask = np.isclose(
        fused_fp8_rms_mlp_residual_res.numpy(), golden_res, rtol=5e-02
    )
    mismatch_count = np.sum(~close_mask)
    mismatch_percentage = mismatch_count / np.size(golden_res) * 100.0
    assert (
        mismatch_percentage <= 0.05
    ), f"Mismatched elements percentage: {mismatch_percentage:}% > {0.03}% threshold\n"


def main():
    parser = argparse.ArgumentParser(description="Run profile or accuracy check")
    parser.add_argument("--profile", action="store_true", help="Run profile")
    parser.add_argument("--accuracy", action="store_false", help="Run accuracy check")

    (
        x,
        ln_scales,
        proj_weight,
        gate_weight,
        up_weight,
        down_weight,
        residual,
        epsilon,
    ) = init_data()

    args = parser.parse_args()
    if args.profile:
        run_profile(
            fusedRmsMlpOP(x, ln_scales, proj_weight, down_weight, residual, epsilon)
        )
        run_profile(
            refRmsMlpOP(
                x, ln_scales, gate_weight, up_weight, down_weight, residual, epsilon
            )
        )
    else:
        run_accuracy_check(
            x,
            ln_scales,
            proj_weight,
            gate_weight,
            up_weight,
            down_weight,
            residual,
            epsilon,
        )


if __name__ == "__main__":
    main()
