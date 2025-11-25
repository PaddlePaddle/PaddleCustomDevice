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
import numpy as np

import paddle
import paddlenlp_ops
import paddle.profiler as profiler

paddle.device.set_device("intel_hpu")

paddle.seed(20241214)


def check_using_cosine_similarity(test_result, ref_result, required_similarity, logger):
    vec1 = test_result.to("float32").cpu().numpy().reshape(-1)
    vec2 = ref_result.to("float32").cpu().numpy().reshape(-1)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        cos_sim = 1.0 if np.array_equal(vec1, vec2) else 0.0
    else:
        cos_sim = np.dot(vec1, vec2) / (norm1 * norm2)

    print(f"Cosine similarity: {cos_sim}")
    return cos_sim >= required_similarity


def tensorwise_quant_to_fp8(tensor):
    return paddlenlp_ops.fused_quant(tensor)
    """
    x_abs = paddle.abs(tensor).astype(paddle.float32)
    x_amax = paddle.amax(x_abs)
    x_amax = paddle.clip(x_amax, min=1e-4)
    scale = paddle.to_tensor(x_amax / 240.0, dtype=paddle.bfloat16)
    x_scaled = (tensor / scale).astype(paddle.float8_e4m3fn)
    return x_scaled, scale
    """


def init_data(
    batch_size=8,
    seqence_len=1,
    hidden_size=2560,
    intermediate_size=3072,
    dtype="bfloat16",
    is_3D_hidden_states=False,
    fused_ffn1=True,
    permute_weights=False,
):
    with paddle.no_grad():
        if is_3D_hidden_states:
            hidden_states = (
                paddle.rand([batch_size * seqence_len, hidden_size], dtype="bfloat16")
                * 10
            ) - 5
        else:
            hidden_states = (
                paddle.rand([batch_size, seqence_len, hidden_size], dtype="bfloat16")
                * 10
            ) - 5

        gate_weight = (
            paddle.rand([hidden_size, intermediate_size], dtype="bfloat16")
        ) * 2.0 - 1.0
        up_weight = (
            paddle.rand([hidden_size, intermediate_size], dtype="bfloat16")
        ) * 2.0 - 1.0
        down_weight = (
            paddle.rand([intermediate_size, hidden_size], dtype="bfloat16")
        ) * 2.0 - 1.0
        if permute_weights:
            gate_weight = gate_weight.transpose([1, 0])
            up_weight = up_weight.transpose([1, 0])
            down_weight = down_weight.transpose([1, 0])
            up_gate_weight = paddle.concat([gate_weight, up_weight], axis=0)
        else:
            up_gate_weight = paddle.concat([gate_weight, up_weight], axis=1)

    if dtype == "bfloat16":
        if fused_ffn1:
            return hidden_states, up_gate_weight, None, down_weight
        else:
            return hidden_states, gate_weight, up_weight, down_weight
    elif dtype == "fp8":
        hidden_states_scaled, d_hidden_states_scales = tensorwise_quant_to_fp8(
            hidden_states
        )
        hidden_states_scale = 1.0 / d_hidden_states_scales
        d_intermediate_hidden_states_scales = paddle.to_tensor(
            [976], dtype=paddle.bfloat16
        )
        intermediate_hidden_states_scales = paddle.to_tensor(
            [1.0 / 976], dtype=paddle.bfloat16
        )
        gate_weight, d_gate_scale = tensorwise_quant_to_fp8(gate_weight)
        up_weight, d_up_scale = tensorwise_quant_to_fp8(up_weight)
        down_weight, d_down_scale = tensorwise_quant_to_fp8(down_weight)
        up_gate_weight, d_up_gate_scale = tensorwise_quant_to_fp8(up_gate_weight)
        if fused_ffn1:
            return (
                hidden_states,
                up_gate_weight,
                None,
                down_weight,
                hidden_states_scale,
                d_hidden_states_scales,
                d_up_gate_scale,
                None,
                intermediate_hidden_states_scales,
                d_intermediate_hidden_states_scales,
                d_down_scale,
            )
        else:
            return (
                hidden_states,
                gate_weight,
                up_weight,
                down_weight,
                hidden_states_scale,
                d_hidden_states_scales,
                d_gate_scale,
                d_up_scale,
                intermediate_hidden_states_scales,
                d_intermediate_hidden_states_scales,
                d_down_scale,
            )
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def ref_mlp(
    hidden_states,
    proj_weight,
    up_weight,
    down_weight,
    permuted_weights,
):
    def swiglu_naive(hidden_states, up=None):
        if up is not None:
            gate = hidden_states
        else:
            gate, up = paddle.chunk(hidden_states, chunks=2, axis=-1)
        silu = gate / (paddle.exp(-gate) + 1)
        return silu * up

    gate = paddle.matmul(hidden_states, proj_weight, transpose_y=permuted_weights)
    up = (
        paddle.matmul(hidden_states, up_weight, transpose_y=permuted_weights)
        if up_weight is not None
        else None
    )
    swiglu = swiglu_naive(hidden_states=gate, up=up)
    # _, d_scales_swiglu = tensorwise_quant_to_fp8(swiglu)
    # print(f"Reference intermediate_hidden_states_scales: {d_scales_swiglu.item()}")
    res = paddle.matmul(swiglu, down_weight, transpose_y=permuted_weights)

    return res


class refMlpOP(paddle.nn.Layer):
    def __init__(
        self,
        hidden_states,
        up_gate_weight,
        up_weight=None,
        down_weight=None,
        up_gate_scale=None,
        d_up_scale=None,
        d_down_scale=None,
        permuted_weights=False,
    ):
        super().__init__()
        self.hidden_states = hidden_states
        self.permuted_weights = permuted_weights
        if up_gate_weight.dtype != paddle.bfloat16:
            self.up_gate_weight = up_gate_weight.cast("bfloat16") * up_gate_scale
            self.up_weight = (
                (up_weight.cast("bfloat16") * d_up_scale)
                if up_weight is not None
                else None
            )
            self.down_weight = down_weight.cast("bfloat16") * d_down_scale
        else:
            self.up_gate_weight = up_gate_weight
            self.up_weight = up_weight
            self.down_weight = down_weight

    def forward(self):
        mlp_out_ref = ref_mlp(
            self.hidden_states,
            self.up_gate_weight,
            self.up_weight,
            self.down_weight,
            self.permuted_weights,
        )
        return mlp_out_ref


class fusedMlpOP(paddle.nn.Layer):
    def __init__(
        self, hidden_states, proj_weight=None, up_weight=None, down_weight=None
    ):
        super().__init__()
        self.hidden_states = hidden_states
        self.proj_weight = proj_weight
        self.up_weight = up_weight
        self.down_weight = down_weight

    def forward(self):
        """
        fused_mlp_out = paddlenlp_ops.fused_mlp_new(
            self.hidden_states,
            self.proj_weight,
            self.up_weight,
            self.down_weight,
        )
        """
        """
        fused_mlp_out = paddlenlp_ops.fused_mlp_bf16(
            self.hidden_states,
            self.proj_weight,
            self.up_weight,
            self.down_weight,
        )
        """
        fused_mlp_out = paddlenlp_ops.fused_mlp(
            self.hidden_states,
            self.proj_weight,
            self.up_weight,
            self.down_weight,
            None,
            None,
            None,
            None,
            None,
            False,
        )
        return fused_mlp_out

    def forward_profile(self):
        fused_mlp_out = paddlenlp_ops.fused_mlp(
            self.hidden_states,
            self.proj_weight,
            self.up_weight,
            self.down_weight,
            None,
            None,
            None,
            None,
            None,
            False,
        )
        for _ in range(9):
            fused_mlp_out = paddlenlp_ops.fused_mlp(
                fused_mlp_out,
                self.proj_weight,
                self.up_weight,
                self.down_weight,
                None,
                None,
                None,
                None,
                None,
                False,
            )
        return fused_mlp_out


class fusedFp8MlpOP(paddle.nn.Layer):
    def __init__(
        self,
        hidden_states,
        proj_weight,
        up_weight=None,
        down_weight=None,
        hidden_states_scale=None,
        d_hidden_states_scale=None,
        d_proj_scale=None,
        d_up_scale=None,
        intermediate_hidden_states_scales=None,
        d_intermediaete_hidden_states_scales=None,
        d_down_scale=None,
        permuted_weights=False,
    ):
        super().__init__()
        self.hidden_states = hidden_states
        self.proj_weight = proj_weight
        self.up_weight = up_weight
        self.down_weight = down_weight
        self.hidden_states_scale = hidden_states_scale
        self.d_proj_scale = d_proj_scale
        self.d_up_scale = d_up_scale
        self.intermediate_hidden_states_scales = intermediate_hidden_states_scales
        self.d_down_scale = d_down_scale
        self.permuted_weights = permuted_weights
        self.d_hidden_states_scale = d_hidden_states_scale
        self.d_intermediaete_hidden_states_scales = d_intermediaete_hidden_states_scales

    def forward(self):
        fused_fp8_mlp_out = paddlenlp_ops.fused_mlp(
            self.hidden_states,
            self.proj_weight,
            self.up_weight,
            self.down_weight,
            self.hidden_states_scale,  # 240/max
            self.d_proj_scale,
            self.d_up_scale,
            self.intermediate_hidden_states_scales,  # 240/max
            self.d_down_scale,
            self.permuted_weights,
        )
        """
        fused_fp8_mlp_out = paddlenlp_ops.fused_fp8_mlp_new(
            self.hidden_states,
            self.proj_weight,
            self.up_weight,
            self.down_weight,
            self.d_hidden_states_scale,  # max/240
            self.d_proj_scale,
            self.d_up_scale,
            self.d_intermediaete_hidden_states_scales,  # max/240
            self.d_down_scale,
            self.permuted_weights,
        )
        """
        return fused_fp8_mlp_out

    def forward_profile(self):
        fused_fp8_mlp_out = paddlenlp_ops.fused_mlp(
            self.hidden_states,
            self.proj_weight,
            self.up_weight,
            self.down_weight,
            self.hidden_states_scale,
            self.d_proj_scale,
            self.d_up_scale,
            self.intermediate_hidden_states_scales,
            self.d_down_scale,
            self.permuted_weights,
        )
        for _ in range(9):
            fused_fp8_mlp_out = paddlenlp_ops.fused_mlp(
                fused_fp8_mlp_out,
                self.proj_weight,
                self.up_weight,
                self.down_weight,
                self.hidden_states_scale,
                self.d_proj_scale,
                self.d_up_scale,
                self.intermediate_hidden_states_scales,
                self.d_down_scale,
                self.permuted_weights,
            )
        return fused_fp8_mlp_out

    def forward_profile_new(self):
        fused_fp8_mlp_out = paddlenlp_ops.fused_fp8_mlp_new(
            self.hidden_states,
            self.proj_weight,
            self.up_weight,
            self.down_weight,
            self.d_hidden_states_scale,
            self.d_proj_scale,
            self.d_up_scale,
            self.d_intermediaete_hidden_states_scales,
            self.d_down_scale,
            self.permuted_weights,
        )
        for _ in range(9):
            fused_fp8_mlp_out = paddlenlp_ops.fused_fp8_mlp_new(
                fused_fp8_mlp_out,
                self.proj_weight,
                self.up_weight,
                self.down_weight,
                self.d_hidden_states_scale,
                self.d_proj_scale,
                self.d_up_scale,
                self.d_intermediaete_hidden_states_scales,
                self.d_down_scale,
                self.permuted_weights,
            )
        return fused_fp8_mlp_out


def run_profile(profile_model):
    prof = profiler.Profiler(
        targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.CUSTOM_DEVICE],
        scheduler=(0, 40),
        on_trace_ready=profiler.export_chrome_tracing("./profile"),
    )
    prof.start()
    for iter in range(40):
        with paddle.no_grad():
            # mlp_out = profile_model.forward_profile()
            mlp_out = profile_model.forward_profile_new()
        paddle.device.synchronize()
        prof.step()
    prof.stop()


def run_accuracy_check(
    testcase,
    hidden_states,
    gate_weight,
    up_weight,
    down_weight,
    d_proj_scale=None,
    d_up_scale=None,
    d_down_scale=None,
    fused_res=None,
    permuted_weights=False,
):
    ref_mlp = refMlpOP(
        hidden_states,
        gate_weight,
        up_weight,
        down_weight,
        d_proj_scale,
        d_up_scale,
        d_down_scale,
        permuted_weights,
    )
    golden_res = ref_mlp()

    required_similarity = 0.99
    passed = check_using_cosine_similarity(
        fused_res, golden_res, required_similarity, None
    )
    if passed:
        print(
            f"------- {testcase} accuracy check passed (cosine similarity >= {required_similarity}). -------\n"
        )
    else:
        print(
            f"******* {testcase} accuracy check failed! (cosine similarity < {required_similarity}). *******\n"
        )
        print("fused_res: ", fused_res)
        print("golden_res: ", golden_res)


def main():
    parser = argparse.ArgumentParser(description="Run profile or accuracy check")
    parser.add_argument(
        "--profile", action="store_true", help="Run profile [default False]"
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        default=True,
        help="Run accuracy check [default True]",
    )
    parser.add_argument(
        "--testcase",
        type=str,
        default="all",
        choices=[
            "fuse_3D_bf16",
            "fuse_2D_bf16",
            "split_3D_bf16",
            "split_2D_bf16",
            "fuse_3D_fp8",
            "fuse_2D_fp8",
            "split_3D_fp8",
            "split_2D_fp8",
            "fuse_3D_permute_fp8",
            "fuse_2D_permute_fp8",
            "split_3D_permute_fp8",
            "split_2D_permute_fp8",
            "all",
        ],
        help="Test case to run.",
    )
    args = parser.parse_args()
    parser.print_help()

    if args.testcase == "all" or args.testcase == "fuse_3D_bf16":
        hidden_states, ffn1_weight, up_weight, down_weight = init_data(
            is_3D_hidden_states=True
        )
        fused_mlp = fusedMlpOP(hidden_states, ffn1_weight, None, down_weight)
        if args.accuracy:
            fused_res = fused_mlp()
            run_accuracy_check(
                "fuse_3D_bf16",
                hidden_states,
                ffn1_weight,
                up_weight,
                down_weight,
                fused_res=fused_res,
            )

    if args.testcase == "all" or args.testcase == "fuse_2D_bf16":
        hidden_states, ffn1_weight, up_weight, down_weight = init_data()
        fused_mlp = fusedMlpOP(hidden_states, ffn1_weight, None, down_weight)
        if args.accuracy:
            fused_res = fused_mlp()
            run_accuracy_check(
                "fuse_2D_bf16",
                hidden_states,
                ffn1_weight,
                up_weight,
                down_weight,
                fused_res=fused_res,
            )

    if args.testcase == "all" or args.testcase == "split_3D_bf16":
        hidden_states, ffn1_weight, up_weight, down_weight = init_data(
            is_3D_hidden_states=True, fused_ffn1=False
        )
        fused_mlp = fusedMlpOP(hidden_states, ffn1_weight, up_weight, down_weight)
        if args.accuracy:
            fused_res = fused_mlp()
            run_accuracy_check(
                "split_3D_bf16",
                hidden_states,
                ffn1_weight,
                up_weight,
                down_weight,
                fused_res=fused_res,
            )

    if args.testcase == "all" or args.testcase == "split_2D_bf16":
        hidden_states, gate_weight, up_weight, down_weight = init_data(fused_ffn1=False)
        fused_mlp = fusedMlpOP(hidden_states, gate_weight, up_weight, down_weight)
        if args.accuracy:
            fused_res = fused_mlp()
            run_accuracy_check(
                "split_2D_bf16",
                hidden_states,
                gate_weight,
                up_weight,
                down_weight,
                fused_res=fused_res,
            )

    if args.testcase == "all" or args.testcase == "fuse_3D_fp8":
        testcase = "fuse_3D_fp8"
        fused_ffn1 = True if "fuse" in testcase else False
        is_3D_hidden_states = True if "3D" in testcase else False
        dtype = "fp8" if "fp8" in testcase else "bfloat16"
        (
            hidden_states,
            proj_weight,
            up_weight,
            down_weight,
            hidden_states_scale,
            d_hidden_states_scales,
            d_proj_scale,
            d_up_scale,
            intermediate_hidden_states_scales,
            d_intermediate_hidden_states_scales,
            d_down_scale,
        ) = init_data(
            is_3D_hidden_states=is_3D_hidden_states, fused_ffn1=fused_ffn1, dtype=dtype
        )
        fused_mlp = fusedFp8MlpOP(
            hidden_states,
            proj_weight,
            up_weight,
            down_weight,
            hidden_states_scale,
            d_hidden_states_scales,
            d_proj_scale,
            d_up_scale,
            intermediate_hidden_states_scales,
            d_intermediate_hidden_states_scales,
            d_down_scale,
        )
        if args.accuracy:
            fused_res = fused_mlp()
            run_accuracy_check(
                testcase,
                hidden_states,
                proj_weight,
                up_weight,
                down_weight,
                d_proj_scale,
                d_up_scale,
                d_down_scale,
                fused_res,
            )

    if args.testcase == "all" or args.testcase == "fuse_2D_fp8":
        testcase = "fuse_2D_fp8"
        fused_ffn1 = True if "fuse" in testcase else False
        is_3D_hidden_states = True if "3D" in testcase else False
        dtype = "fp8" if "fp8" in testcase else "bfloat16"
        (
            hidden_states,
            proj_weight,
            up_weight,
            down_weight,
            hidden_states_scale,
            d_hidden_states_scales,
            d_proj_scale,
            d_up_scale,
            intermediate_hidden_states_scales,
            d_intermediate_hidden_states_scales,
            d_down_scale,
        ) = init_data(
            is_3D_hidden_states=is_3D_hidden_states, fused_ffn1=fused_ffn1, dtype=dtype
        )
        fused_mlp = fusedFp8MlpOP(
            hidden_states,
            proj_weight,
            up_weight,
            down_weight,
            hidden_states_scale,
            d_hidden_states_scales,
            d_proj_scale,
            d_up_scale,
            intermediate_hidden_states_scales,
            d_intermediate_hidden_states_scales,
            d_down_scale,
        )
        if args.accuracy:
            fused_res = fused_mlp()
            run_accuracy_check(
                testcase,
                hidden_states,
                proj_weight,
                up_weight,
                down_weight,
                d_proj_scale,
                d_up_scale,
                d_down_scale,
                fused_res,
            )

    if args.testcase == "all" or args.testcase == "split_3D_fp8":
        testcase = "split_3D_fp8"
        fused_ffn1 = True if "fuse" in testcase else False
        is_3D_hidden_states = True if "3D" in testcase else False
        dtype = "fp8" if "fp8" in testcase else "bfloat16"
        (
            hidden_states,
            proj_weight,
            up_weight,
            down_weight,
            hidden_states_scale,
            d_hidden_states_scales,
            d_proj_scale,
            d_up_scale,
            intermediate_hidden_states_scales,
            d_intermediate_hidden_states_scales,
            d_down_scale,
        ) = init_data(
            is_3D_hidden_states=is_3D_hidden_states, fused_ffn1=fused_ffn1, dtype=dtype
        )
        fused_mlp = fusedFp8MlpOP(
            hidden_states,
            proj_weight,
            up_weight,
            down_weight,
            hidden_states_scale,
            d_hidden_states_scales,
            d_proj_scale,
            d_up_scale,
            intermediate_hidden_states_scales,
            d_intermediate_hidden_states_scales,
            d_down_scale,
        )
        if args.accuracy:
            fused_res = fused_mlp()
            run_accuracy_check(
                testcase,
                hidden_states,
                proj_weight,
                up_weight,
                down_weight,
                d_proj_scale,
                d_up_scale,
                d_down_scale,
                fused_res,
            )

    if args.testcase == "all" or args.testcase == "split_2D_fp8":
        testcase = "split_2D_fp8"
        fused_ffn1 = True if "fuse" in testcase else False
        is_3D_hidden_states = True if "3D" in testcase else False
        dtype = "fp8" if "fp8" in testcase else "bfloat16"
        (
            hidden_states,
            proj_weight,
            up_weight,
            down_weight,
            hidden_states_scale,
            d_hidden_states_scales,
            d_proj_scale,
            d_up_scale,
            intermediate_hidden_states_scales,
            d_intermediate_hidden_states_scales,
            d_down_scale,
        ) = init_data(
            is_3D_hidden_states=is_3D_hidden_states, fused_ffn1=fused_ffn1, dtype=dtype
        )
        fused_mlp = fusedFp8MlpOP(
            hidden_states,
            proj_weight,
            up_weight,
            down_weight,
            hidden_states_scale,
            d_hidden_states_scales,
            d_proj_scale,
            d_up_scale,
            intermediate_hidden_states_scales,
            d_intermediate_hidden_states_scales,
            d_down_scale,
        )
        if args.accuracy:
            fused_res = fused_mlp()
            run_accuracy_check(
                testcase,
                hidden_states,
                proj_weight,
                up_weight,
                down_weight,
                d_proj_scale,
                d_up_scale,
                d_down_scale,
                fused_res,
            )

    if args.testcase == "all" or args.testcase == "fuse_3D_permute_fp8":
        testcase = "fuse_3D_permute_fp8"
        fused_ffn1 = True if "fuse" in testcase else False
        is_3D_hidden_states = True if "3D" in testcase else False
        permuted_weights = True if "permute" in testcase else False
        dtype = "fp8" if "fp8" in testcase else "bfloat16"
        (
            hidden_states,
            proj_weight,
            up_weight,
            down_weight,
            hidden_states_scale,
            d_hidden_states_scales,
            d_proj_scale,
            d_up_scale,
            intermediate_hidden_states_scales,
            d_intermediate_hidden_states_scales,
            d_down_scale,
        ) = init_data(
            is_3D_hidden_states=is_3D_hidden_states,
            fused_ffn1=fused_ffn1,
            dtype=dtype,
            permute_weights=permuted_weights,
        )
        fused_mlp = fusedFp8MlpOP(
            hidden_states,
            proj_weight,
            up_weight,
            down_weight,
            hidden_states_scale,
            d_hidden_states_scales,
            d_proj_scale,
            d_up_scale,
            intermediate_hidden_states_scales,
            d_intermediate_hidden_states_scales,
            d_down_scale,
            permuted_weights,
        )
        if args.accuracy:
            fused_res = fused_mlp()
            run_accuracy_check(
                testcase,
                hidden_states,
                proj_weight,
                up_weight,
                down_weight,
                d_proj_scale,
                d_up_scale,
                d_down_scale,
                fused_res,
                permuted_weights,
            )

    if args.testcase == "all" or args.testcase == "fuse_2D_permute_fp8":
        testcase = "fuse_2D_permute_fp8"
        fused_ffn1 = True if "fuse" in testcase else False
        is_3D_hidden_states = True if "3D" in testcase else False
        permuted_weights = True if "permute" in testcase else False
        dtype = "fp8" if "fp8" in testcase else "bfloat16"
        (
            hidden_states,
            proj_weight,
            up_weight,
            down_weight,
            hidden_states_scale,
            d_hidden_states_scales,
            d_proj_scale,
            d_up_scale,
            intermediate_hidden_states_scales,
            d_intermediate_hidden_states_scales,
            d_down_scale,
        ) = init_data(
            is_3D_hidden_states=is_3D_hidden_states,
            fused_ffn1=fused_ffn1,
            dtype=dtype,
            permute_weights=permuted_weights,
        )
        fused_mlp = fusedFp8MlpOP(
            hidden_states,
            proj_weight,
            up_weight,
            down_weight,
            hidden_states_scale,
            d_hidden_states_scales,
            d_proj_scale,
            d_up_scale,
            intermediate_hidden_states_scales,
            d_intermediate_hidden_states_scales,
            d_down_scale,
            permuted_weights,
        )
        if args.accuracy:
            fused_res = fused_mlp()
            run_accuracy_check(
                testcase,
                hidden_states,
                proj_weight,
                up_weight,
                down_weight,
                d_proj_scale,
                d_up_scale,
                d_down_scale,
                fused_res,
                permuted_weights,
            )

    if args.testcase == "all" or args.testcase == "split_3D_permute_fp8":
        testcase = "split_3D_permute_fp8"
        fused_ffn1 = True if "fuse" in testcase else False
        is_3D_hidden_states = True if "3D" in testcase else False
        permuted_weights = True if "permute" in testcase else False
        dtype = "fp8" if "fp8" in testcase else "bfloat16"
        (
            hidden_states,
            proj_weight,
            up_weight,
            down_weight,
            hidden_states_scale,
            d_hidden_states_scales,
            d_proj_scale,
            d_up_scale,
            intermediate_hidden_states_scales,
            d_intermediate_hidden_states_scales,
            d_down_scale,
        ) = init_data(
            is_3D_hidden_states=is_3D_hidden_states,
            fused_ffn1=fused_ffn1,
            dtype=dtype,
            permute_weights=permuted_weights,
        )
        fused_mlp = fusedFp8MlpOP(
            hidden_states,
            proj_weight,
            up_weight,
            down_weight,
            hidden_states_scale,
            d_hidden_states_scales,
            d_proj_scale,
            d_up_scale,
            intermediate_hidden_states_scales,
            d_intermediate_hidden_states_scales,
            d_down_scale,
            permuted_weights,
        )
        if args.accuracy:
            fused_res = fused_mlp()
            run_accuracy_check(
                testcase,
                hidden_states,
                proj_weight,
                up_weight,
                down_weight,
                d_proj_scale,
                d_up_scale,
                d_down_scale,
                fused_res,
                permuted_weights,
            )

    if args.testcase == "all" or args.testcase == "split_2D_permute_fp8":
        testcase = "split_2D_permute_fp8"
        fused_ffn1 = True if "fuse" in testcase else False
        is_3D_hidden_states = True if "3D" in testcase else False
        permuted_weights = True if "permute" in testcase else False
        dtype = "fp8" if "fp8" in testcase else "bfloat16"
        (
            hidden_states,
            proj_weight,
            up_weight,
            down_weight,
            hidden_states_scale,
            d_hidden_states_scales,
            d_proj_scale,
            d_up_scale,
            intermediate_hidden_states_scales,
            d_intermediate_hidden_states_scales,
            d_down_scale,
        ) = init_data(
            is_3D_hidden_states=is_3D_hidden_states,
            fused_ffn1=fused_ffn1,
            dtype=dtype,
            permute_weights=permuted_weights,
        )
        fused_mlp = fusedFp8MlpOP(
            hidden_states,
            proj_weight,
            up_weight,
            down_weight,
            hidden_states_scale,
            d_hidden_states_scales,
            d_proj_scale,
            d_up_scale,
            intermediate_hidden_states_scales,
            d_intermediate_hidden_states_scales,
            d_down_scale,
            permuted_weights,
        )
        if args.accuracy:
            fused_res = fused_mlp()
            run_accuracy_check(
                testcase,
                hidden_states,
                proj_weight,
                up_weight,
                down_weight,
                d_proj_scale,
                d_up_scale,
                d_down_scale,
                fused_res,
                permuted_weights,
            )

    if args.profile:
        run_profile(fused_mlp)


if __name__ == "__main__":
    main()
