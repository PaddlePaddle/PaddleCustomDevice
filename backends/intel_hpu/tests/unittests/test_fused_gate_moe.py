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

import os
import sys
import unittest
from parameterized import parameterized

import logging
import numpy as np

import paddle
import paddle.distributed as dist
import paddlenlp_ops

local_rank = dist.get_rank()
world_size = dist.get_world_size()

print(
    f"**************************************\n"
    f"      World size: {world_size}, Local rank: {local_rank}\n"
    f"**************************************"
)

if world_size == 1:
    intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 1)
    paddle.device.set_device(f"intel_hpu:{intel_hpus_module_id}")
else:
    paddle.set_device("intel_hpu")
    dist.init_parallel_env()

np.random.seed(2049)
paddle.seed(102)


class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


_first_run = True


def setup_logging(ep_rank, tp_rank, enable_logging=False):
    global _first_run

    logger = logging.getLogger(f"moe_ep_rank_{ep_rank}_tp_rank{tp_rank}")
    if enable_logging or os.getenv("ENABLE_LOGGING") == "1":
        log_file = f"test_logs_ep_rank_{ep_rank}_tp_rank_{tp_rank}.log"
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        mode = "w" if _first_run and os.path.exists(log_file) else "a"
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] ep_rank %(ep_rank)d tp_rank %(tp_rank)d: %(message)s"
            )
        )
        logger.addHandler(file_handler)

        stream_handler = FlushStreamHandler(sys.stdout)
        stream_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] ep_rank %(ep_rank)d tp_rank %(tp_rank)d: %(message)s"
            )
        )
        logger.addHandler(stream_handler)
        _first_run = False

    logger.info(
        "Logging initialized for ep_rank %d, tp_rank %d",
        ep_rank,
        tp_rank,
        extra={"ep_rank": ep_rank, "tp_rank": tp_rank},
    )
    return logger


def init_distributed(ep_size=1, tp_size=1):

    if not dist.is_initialized():
        try:
            dist.init_parallel_env()
        except Exception as e:
            raise RuntimeError("Failed to initialize distributed environment") from e

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size == 1:
        ep_size, tp_size = 1, 1
    elif ep_size == 1:
        tp_size = world_size
    elif tp_size == 1:
        ep_size = world_size

    if world_size != ep_size * tp_size:
        raise ValueError(
            f"Invalid configuration: ep_size ({ep_size}) * tp_size ({tp_size}) "
            f"= {ep_size * tp_size} != world_size ({world_size})"
        )

    ep_rank = global_rank // tp_size
    tp_rank = global_rank % tp_size

    # Create TP group
    if ep_size == 1:
        tp_ranks = list(range(world_size))
    else:
        tp_ranks = [ep_rank * tp_size + i for i in range(tp_size)]
    try:
        tp_group = dist.new_group(tp_ranks)
    except Exception as e:
        raise ValueError(f"Failed to create tp_group with ranks={tp_ranks}: {e}")

    # Create EP group
    if tp_size == 1:
        ep_ranks = list(range(world_size))
    else:
        ep_ranks = [i * tp_size + tp_rank for i in range(ep_size)]
    try:
        ep_group = dist.new_group(ep_ranks)
    except Exception as e:
        raise ValueError(f"Failed to create ep_group with ranks={ep_ranks}: {e}")

    return (ep_rank, ep_size, ep_group), (tp_rank, tp_size, tp_group)


def check_using_cosine_similarity(
    final_states, final_states_ref, required_similarity, ep_rank, tp_rank, logger
):
    vec1 = final_states.reshape(-1)
    vec2 = final_states_ref.reshape(-1)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        cos_sim = 1.0 if np.array_equal(vec1, vec2) else 0.0
    else:
        cos_sim = np.dot(vec1, vec2) / (norm1 * norm2)

    logger.info(
        f"Cosine similarity: {cos_sim}, \n"
        f"required_similarity: {required_similarity}, ",
        extra={"ep_rank": ep_rank, "tp_rank": tp_rank},
    )
    print(f"Cosine similarity: {cos_sim}")
    return cos_sim >= required_similarity


def tensorwise_cast_to_fp8(tensor, scale):
    scale = paddle.to_tensor(scale, dtype=tensor.dtype)
    x_scaled = (tensor * scale).cast(paddle.float8_e4m3fn)
    return x_scaled


def tensorwise_quant_to_fp8(tensor):
    """
    x_abs = paddle.abs(tensor).astype(paddle.float32)
    x_amax = paddle.amax(x_abs)
    x_amax = paddle.clip(x_amax, min=1e-4)
    scale = paddle.to_tensor(x_amax / 240.0, dtype=paddle.bfloat16)
    x_scaled = (tensor / scale).astype(paddle.float8_e4m3fn)
    return x_scaled, scale
    """
    return paddlenlp_ops.fused_quant(tensor)


def channelwise_quant_to_fp8(tensor):
    # Channel-wise quantization along the last dimension (N)
    x_abs = paddle.abs(tensor).astype(paddle.float32)
    x_amax = paddle.amax(x_abs, axis=0)  # shape: [N]
    x_amax = paddle.clip(x_amax, min=1e-4)
    scale = x_amax / 240.0  # shape: [N]
    scale = paddle.to_tensor(scale, dtype=paddle.bfloat16)
    x_scaled = (tensor / scale).astype(paddle.float8_e4m3fn)
    return x_scaled, scale


def blockwise_quant_to_fp8(tensorlist, block_size):
    q_tensor_list = []
    q_tensor_scales = []

    for x in tensorlist:
        assert x.dim() == 2
        m, n = x.shape
        x_padded = paddle.zeros(
            (
                (m + block_size - 1) // block_size * block_size,
                (n + block_size - 1) // block_size * block_size,
            ),
            dtype=x.dtype,
        )
        x_padded[:m, :n] = x
        x_view = paddle.view(
            x_padded, (-1, block_size, x_padded.shape[1] // block_size, block_size)
        )

        x_abs = paddle.abs(x_view).astype(paddle.float32)
        x_amax = paddle.amax(x_abs, axis=(1, 3), keepdim=True)
        x_amax = paddle.clip(x_amax, min=1e-4)
        x_scaled = (x_view * (240.0 / x_amax)).astype(paddle.float8_e4m3fn)

        q_tensor_list.append(x_scaled.view_as(x_padded)[:m, :n].contiguous())
        q_tensor_scales.append(
            paddle.view(x_amax / 240.0, (x_view.shape[0], x_view.shape[2]))
        )

    return (q_tensor_list, q_tensor_scales)


def generate_tensors(
    dtype,
    num_tokens,
    hidden_dim,
    ffn_dim,
    top_k,
    num_experts,
    permuted_weights,
    fused_weights,
    intermediate_dynamic_scale=None,
    hidden_states_dynamic_quant=False,
    weight_scale_type=None,
    block_size=None,
):
    if dtype == "bfloat16":
        paddle_dtype = paddle.bfloat16
    elif dtype == "fp8":
        paddle_dtype = paddle.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    hidden_states = (paddle.rand([num_tokens, hidden_dim], dtype=paddle_dtype) * 10) - 5
    route_gate_weight = (
        paddle.rand([hidden_dim, num_experts], dtype=paddle.float32) * 0.6
    ) - 0.3
    gate_correction_bias = (
        paddle.rand([1, num_experts], dtype=paddle.float32) * 128
    ) - 64
    up_weights = [
        (paddle.rand([hidden_dim, ffn_dim], dtype=paddle_dtype) * 0.6) - 0.3
        for _ in range(num_experts)
    ]
    gate_weights = [
        (paddle.rand([hidden_dim, ffn_dim], dtype=paddle_dtype) * 0.6) - 0.3
        for _ in range(num_experts)
    ]
    down_weights = [
        (paddle.rand([ffn_dim, hidden_dim], dtype=paddle_dtype) * 0.6) - 0.3
        for _ in range(num_experts)
    ]

    if permuted_weights:
        up_weights = [w.transpose([1, 0]) for w in up_weights]
        gate_weights = [w.transpose([1, 0]) for w in gate_weights]
        down_weights = [w.transpose([1, 0]) for w in down_weights]

    if fused_weights:
        up_gate_weights = [
            paddle.concat((w1, w2), axis=0)
            if permuted_weights
            else paddle.concat((w1, w2), axis=1)
            for w1, w2 in zip(up_weights, gate_weights)
        ]

    # fp8 scale weights handling
    if dtype == "bfloat16":
        d_scales_up_gate = None
        d_scales_down = None
        d_scales_hidden_states = None
        d_scales_intermediate_hidden_states = None
    elif dtype == "fp8":
        # weights cast to fp8, scales to tensor
        weight_quant_method = (
            channelwise_quant_to_fp8
            if weight_scale_type == "channelwise"
            else tensorwise_quant_to_fp8
        )
        if fused_weights:
            up_gate_weights, d_scales_up_gate = zip(
                *[weight_quant_method(w) for w in up_gate_weights]
            )
            up_gate_weights = list(up_gate_weights)
            d_scales_up_gate = list(d_scales_up_gate)
        else:
            up_weights, d_scales_up = zip(*[weight_quant_method(w) for w in up_weights])
            up_weights = list(up_weights)
            d_scales_up = list(d_scales_up)
            gate_weights, d_scales_gate = zip(
                *[weight_quant_method(w) for w in gate_weights]
            )
            gate_weights = list(gate_weights)
            d_scales_gate = list(d_scales_gate)
        down_weights, d_scales_down = zip(
            *[weight_quant_method(w) for w in down_weights]
        )
        down_weights = list(down_weights)
        d_scales_down = list(d_scales_down)

        if intermediate_dynamic_scale is False:
            d_scales_intermediate_hidden_states = [
                paddle.to_tensor([1.0], dtype=paddle_dtype)
                # paddle.ones(shape=[num_tokens, 1], dtype=paddle_dtype)
                for _ in range(num_experts)
            ]
        else:
            d_scales_intermediate_hidden_states = None

        if hidden_states_dynamic_quant is False:
            _, d_scales_hidden_states = tensorwise_quant_to_fp8(hidden_states)
            d_scales_hidden_states = paddle.to_tensor(
                d_scales_hidden_states, dtype=paddle_dtype
            )
            d_scales_hidden_states = 1.0 / d_scales_hidden_states
        else:
            d_scales_hidden_states = None
    elif dtype == "blockwise_fp8":
        up_weights, d_scales_up = blockwise_quant_to_fp8(up_weights, block_size)
        gate_weights, d_scales_gate = blockwise_quant_to_fp8(gate_weights, block_size)
        down_weights, d_scales_down = blockwise_quant_to_fp8(down_weights, block_size)
        # not done yet
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    paddle_data = (
        hidden_states,
        gate_correction_bias,
        route_gate_weight,
        up_gate_weights,
        down_weights,
        d_scales_hidden_states,
        d_scales_intermediate_hidden_states,
        d_scales_up_gate,
        d_scales_down,
    )
    return paddle_data


class MixtralSparseMoeRef:
    def __init__(self, dynamic_quant, dtype):
        super().__init__()
        self.dynamic_quant = dynamic_quant

        if dtype == "fp8":
            self.forward = self.forward_fp8
        else:
            self.forward = self.forward_bf16

    def forward_fp8(
        self,
        hidden_states,
        gate_weights,
        gate_correction_bias,
        up_gate_weights,
        down_weights,
        hidden_states_scales,
        intermediate_hidden_states_scales,
        gate_up_weights_scales,
        down_weights_scales,
        top_k,
        norm_topk_prob,
        permuted_weights,
        experts_min,
        experts_max,
        chunk_size,
    ):
        gate_out = paddle.matmul(hidden_states.cast("float32"), gate_weights)

        weights = paddle.nn.functional.softmax(gate_out, axis=-1)
        if gate_correction_bias is not None:
            scores = weights + gate_correction_bias
            _, selected_experts = paddle.topk(scores, top_k, axis=-1)
            routing_weights = paddle.index_sample(weights, selected_experts)
        else:
            routing_weights, selected_experts = paddle.topk(weights, top_k, axis=-1)
        if norm_topk_prob:
            routing_weights /= paddle.sum(routing_weights, axis=-1, keepdim=True)
        routing_weights = routing_weights.cast("bfloat16")

        if hidden_states_scales is None:
            hidden_states, hidden_states_scales = tensorwise_quant_to_fp8(hidden_states)
        else:
            hidden_states = tensorwise_cast_to_fp8(
                hidden_states, 1.0 / hidden_states_scales
            )

        common_inputs = (
            hidden_states,
            selected_experts,
            routing_weights.cast("bfloat16"),
        )
        weights = (up_gate_weights, down_weights)

        if self.dynamic_quant:
            intermediate_hidden_states_scales = None

        scales = (
            hidden_states_scales,
            intermediate_hidden_states_scales,
            gate_up_weights_scales,
            down_weights_scales,
        )

        common_params = (
            permuted_weights,
            "silu",  # activation,
            experts_min,
            experts_max,
            self.dynamic_quant,
            chunk_size,
        )

        fused_moe_out = paddlenlp_ops.mixture_of_experts_fp8(
            *common_inputs, *weights, *scales, *common_params
        )

        return fused_moe_out

    def forward_bf16(
        self,
        hidden_states,
        gate_weights,
        gate_correction_bias,
        up_gate_weights,
        down_weights,
        hidden_states_scales,
        intermediate_hidden_states_scales,
        gate_up_weights_scales,
        down_weights_scales,
        top_k,
        norm_topk_prob,
        permuted_weights,
        experts_min,
        experts_max,
        chunk_size,
    ):
        gate_out = paddle.matmul(hidden_states.cast("float32"), gate_weights)

        weights = paddle.nn.functional.softmax(gate_out, axis=-1)
        if gate_correction_bias is not None:
            scores = weights + gate_correction_bias
            _, selected_experts = paddle.topk(scores, top_k, axis=-1)
            routing_weights = paddle.index_sample(weights, selected_experts)
        else:
            routing_weights, selected_experts = paddle.topk(weights, top_k, axis=-1)
        if norm_topk_prob:
            routing_weights /= paddle.sum(routing_weights, axis=-1, keepdim=True)
        routing_weights = routing_weights.cast("bfloat16")

        common_inputs = (hidden_states, selected_experts, routing_weights)
        weights = (up_gate_weights, down_weights)

        common_params = (
            permuted_weights,
            "silu",  # activation,
            experts_min,
            experts_max,
            False,  # measurement_mode
            chunk_size,
        )

        fused_moe_out, _ = paddlenlp_ops.mixture_of_experts(
            *common_inputs, *weights, *common_params
        )

        return fused_moe_out


class FusedGateMoE:
    def __init__(
        self,
        num_experts,
        top_k,
        activation,
        permuted_weights,
        fused_weights,
        slice_max_expert,
        logger,
        ep_rank,
        ep_size,
        ep_group=None,
        tp_rank=0,
        tp_size=1,
        tp_group=None,
        dtype="fp8",
        intermediate_dynamic_scale=None,
        block_size=None,
        chunk_size=0,
    ):
        self.num_experts = num_experts
        self.permuted_weights = permuted_weights
        self.fused_weights = fused_weights
        self.intermediate_dynamic_scale = intermediate_dynamic_scale
        self.activation = activation
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.ep_group = ep_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group
        self.logger = logger
        self.dtype = dtype
        self.block_size = block_size
        self.top_k = top_k
        self.chunk_size = chunk_size

        if self.dtype == "bfloat16":
            self.fn = paddlenlp_ops.fused_gate_moe
        elif self.dtype == "fp8":
            self.fn = paddlenlp_ops.fused_gate_moe_fp8
        elif self.dtype == "blockwise_fp8":
            self.fn = paddlenlp_ops.fused_gate_moe_blockwise_fp8
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self.experts_per_rank = self.num_experts // self.ep_size
        self.experts_min = self.ep_rank * self.experts_per_rank
        self.experts_max = (self.ep_rank + 1) * self.experts_per_rank - 1
        if self.ep_rank == self.ep_size - 1:
            self.experts_max = self.num_experts - 1

        self.expert_slice = max(
            1, (self.experts_max - self.experts_min + 1) // slice_max_expert
        )
        self.expert_chunk = max(
            1, (self.experts_max - self.experts_min + 1) // self.expert_slice
        )

    def forward(
        self,
        hidden_states,
        gate_weights,
        gate_correction_bias,
        expert_weights,
        hidden_states_scale,
        intermediate_states_scales,
        weights_scales,
        compute_amax=False,
    ):
        common_inputs = (hidden_states, gate_weights, gate_correction_bias)
        # final_hidden_states = paddle.zeros_like(hidden_states)

        amax_per_expert = (
            paddle.zeros(self.num_experts, dtype="float32") if compute_amax else None
        )

        for idx in range(self.expert_slice):
            slice_experts_min = self.experts_min + (self.expert_chunk * idx)
            slice_experts_max = min(
                slice_experts_min + self.expert_chunk - 1, self.experts_max
            )
            common_params = (
                self.top_k,
                True,  # norm_topk_prob
                self.permuted_weights,
                self.activation,
                slice_experts_min,
                slice_experts_max,
            )
            slice_weights = (
                (
                    paddle.stack(
                        expert_weights[0][slice_experts_min : slice_experts_max + 1],
                        axis=0,
                    ),
                    paddle.stack(
                        expert_weights[1][slice_experts_min : slice_experts_max + 1],
                        axis=0,
                    ),
                )
                if self.fused_weights
                else (
                    paddle.stack(
                        expert_weights[0][slice_experts_min : slice_experts_max + 1]
                        + expert_weights[1][slice_experts_min : slice_experts_max + 1],
                        axis=0,
                    ),
                    paddle.stack(
                        expert_weights[2][slice_experts_min : slice_experts_max + 1],
                        axis=0,
                    ),
                )
            )
            if self.dtype == "fp8":
                slice_scales = (
                    (
                        hidden_states_scale,
                        None
                        if self.intermediate_dynamic_scale
                        else intermediate_states_scales[
                            slice_experts_min : slice_experts_max + 1
                        ],
                        paddle.stack(
                            weights_scales[0][
                                slice_experts_min : slice_experts_max + 1
                            ],
                            axis=0,
                        ),
                        paddle.stack(
                            weights_scales[1][
                                slice_experts_min : slice_experts_max + 1
                            ],
                            axis=0,
                        ),
                    )
                    if self.fused_weights
                    else (
                        hidden_states_scale,
                        None
                        if self.intermediate_dynamic_scale
                        else intermediate_states_scales[
                            slice_experts_min : slice_experts_max + 1
                        ],
                        paddle.stack(
                            weights_scales[0][slice_experts_min : slice_experts_max + 1]
                            + weights_scales[1][
                                slice_experts_min : slice_experts_max + 1
                            ],
                            axis=0,
                        ),
                        paddle.stack(
                            weights_scales[2][
                                slice_experts_min : slice_experts_max + 1
                            ],
                            axis=0,
                        ),
                    )
                )
            elif self.dtype == "blockwise_fp8":
                slice_scales = (
                    (
                        paddle.stack(
                            weights_scales[0][
                                slice_experts_min : slice_experts_max + 1
                            ],
                            axis=0,
                        ),
                        paddle.stack(
                            weights_scales[1][
                                slice_experts_min : slice_experts_max + 1
                            ],
                            axis=0,
                        ),
                    )
                    if self.fused_weights
                    else (
                        paddle.stack(
                            weights_scales[0][slice_experts_min : slice_experts_max + 1]
                            + weights_scales[1][
                                slice_experts_min : slice_experts_max + 1
                            ],
                            axis=0,
                        ),
                        paddle.stack(
                            weights_scales[2][
                                slice_experts_min : slice_experts_max + 1
                            ],
                            axis=0,
                        ),
                    )
                )

            if self.dtype == "fp8":
                slice_result = self.fn(
                    *common_inputs,
                    *slice_weights,
                    *slice_scales,
                    *common_params,
                    self.chunk_size,
                )
            elif self.dtype == "blockwise_fp8":
                slice_result = self.fn(
                    *common_inputs,
                    *slice_weights,
                    *slice_scales,
                    *common_params,
                    self.block_size,
                    self.chunk_size,
                )
            else:
                slice_result = self.fn(
                    *common_inputs,
                    *slice_weights,
                    *common_params,
                    self.chunk_size,
                )
                # paddlenlp_ops.fused_gate_moe no requirement to return amax
                slice_amax = None
            if compute_amax:
                amax_per_expert[slice_experts_min : slice_experts_max + 1] = slice_amax

            final_hidden_states = slice_result

        # EP: All-reduce for final output
        if self.tp_size > 1:
            try:
                dist.all_reduce(
                    final_hidden_states, op=dist.ReduceOp.SUM, group=self.tp_group
                )
                self.logger.info(
                    "TP All-reduce for MoE successfully.",
                    extra={"ep_rank": self.ep_rank, "tp_rank": self.tp_rank},
                )
                if compute_amax:
                    dist.all_reduce(
                        amax_per_expert, op=dist.ReduceOp.MAX, group=self.tp_group
                    )
                    self.logger.info(
                        "TP All-reduce for AMax successfully.",
                        extra={"ep_rank": self.ep_rank, "tp_rank": self.tp_rank},
                    )
            except Exception as e:
                self.logger.error(
                    f"Failed to perform TP All-reduce: {str(e)}",
                    extra={"ep_rank": self.ep_rank, "tp_rank": self.tp_rank},
                )
                raise

        if self.ep_size > 1:
            try:
                dist.all_reduce(
                    final_hidden_states, op=dist.ReduceOp.SUM, group=self.ep_group
                )
                self.logger.info(
                    "EP All-reduce for MoE successfully.",
                    extra={"ep_rank": self.ep_rank, "tp_rank": self.tp_rank},
                )
                if compute_amax:
                    dist.all_reduce(
                        amax_per_expert, op=dist.ReduceOp.MAX, group=self.ep_group
                    )
                    self.logger.info(
                        "EP All-reduce for AMax successfully.",
                        extra={"ep_rank": self.ep_rank, "tp_rank": self.tp_rank},
                    )
            except Exception as e:
                self.logger.error(
                    f"Failed to perform EP All-reduce: {str(e)}",
                    extra={"ep_rank": self.ep_rank, "tp_rank": self.tp_rank},
                )
                raise

        return final_hidden_states, amax_per_expert


DTYPES = ["bfloat16", "fp8"]  # ["bfloat16", "fp8"]
NUM_TOKENS = [32]
HIDDEN_DIMS = [4096]
FFN_DIMS = [2560]
TOP_K = [2]
NUM_EXPERTS = [8]
SLICE_MAX_EXPERT = [8]
FUSED_WEIGHTS = [True]  # [True, False]
ACTIVATIONS = ["silu"]  # ["gelu", "relu", "silu"]
PERMUTED_WEIGHTS = [False]  # [True, False]
EP_SIZE = [world_size]
TP_SIZE = [1]
# for bfloat16 only
COMPUTE_AMAX = [False]  # [True, False]
# for fp8 only
HIDDEN_STATES_DYNAMIC_SCALE = [True, False]
INTERMEDIATE_DYNAMIC_SCALE = [True, False]
# for blockwise_fp8 only
BLOCK_SIZES = [128]
WEIGHT_SCALE_TYPES = ["channelwise"]  # ["tensorwise", "channelwise"]


class MoETest(unittest.TestCase):
    @parameterized.expand(
        [
            (
                num_tokens,
                hidden_dim,
                ffn_dim,
                top_k,
                num_experts,
                slice_max_expert,
                fused_weights,
                activation,
                permuted_weights,
                ep_size,
                tp_size,
                intermediate_dynamic_scale if dtype == "fp8" else None,
                hidden_states_dynamic_quant if dtype == "fp8" else None,
                weight_scale_type if dtype == "fp8" else None,
                dtype,
            )
            for num_tokens in NUM_TOKENS
            for hidden_dim in HIDDEN_DIMS
            for ffn_dim in FFN_DIMS
            for top_k in TOP_K
            for num_experts in NUM_EXPERTS
            for slice_max_expert in SLICE_MAX_EXPERT
            for fused_weights in FUSED_WEIGHTS
            for activation in ACTIVATIONS
            for permuted_weights in PERMUTED_WEIGHTS
            for ep_size in EP_SIZE
            for tp_size in TP_SIZE
            for dtype in DTYPES
            for intermediate_dynamic_scale in (
                INTERMEDIATE_DYNAMIC_SCALE if dtype == "fp8" else [None]
            )
            for hidden_states_dynamic_quant in (
                HIDDEN_STATES_DYNAMIC_SCALE if dtype == "fp8" else [None]
            )
            for weight_scale_type in (WEIGHT_SCALE_TYPES if dtype == "fp8" else [None])
        ]
    )
    def test_fused_gate_moe(
        self,
        num_tokens,
        hidden_dim,
        ffn_dim,
        top_k,
        num_experts,
        slice_max_expert,
        fused_weights,
        activation,
        permuted_weights,
        ep_size,
        tp_size,
        intermediate_dynamic_scale,
        hidden_states_dynamic_quant,
        weight_scale_type,
        dtype="fp8",
    ):
        (ep_rank, ep_size, ep_group), (tp_rank, tp_size, tp_group) = init_distributed(
            ep_size, tp_size
        )
        logger = setup_logging(ep_rank=ep_rank, tp_rank=tp_rank)
        logger.debug(
            f"\n\n======================================="
            f"`test_mixture_of_experts_fp8`: \n"
            f" num_tokens={num_tokens}, hidden_dim={hidden_dim}, ffn_dim={ffn_dim}, \n"
            f" top_k={top_k}, num_experts={num_experts}, slice_max_expert={slice_max_expert}, \n"
            f" fused_weights={fused_weights}, permuted_weights={permuted_weights}, activation={activation}, \n"
            f" dtype={dtype}, intermediate_dynamic_scale={intermediate_dynamic_scale}, \n"
            f"  hidden_states_dynamic_quant={hidden_states_dynamic_quant}, ep_size={ep_size}, tp_size={tp_size}, \n",
            extra={"ep_rank": ep_rank, "tp_rank": tp_rank},
        )

        paddle.seed(ep_rank * 100 + tp_rank + 1024)
        device = "intel_hpu"
        out_tensors = generate_tensors(
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            top_k=top_k,
            num_experts=num_experts,
            permuted_weights=permuted_weights,
            fused_weights=fused_weights,
            intermediate_dynamic_scale=intermediate_dynamic_scale,
            hidden_states_dynamic_quant=hidden_states_dynamic_quant,
            weight_scale_type=weight_scale_type,
            dtype=dtype,
        )

        (
            hidden_states,
            gate_correction_bias,
            gate_weights,
            up_gate_weights,
            down_weights,
            d_scales_hidden_states,
            d_scales_intermediate_hidden_states,
            d_scales_up_gate,
            d_scales_down,
        ) = out_tensors

        # CPU Reference Implementation
        mixtral_ref = MixtralSparseMoeRef(intermediate_dynamic_scale, dtype)

        final_hidden_states_ref = mixtral_ref.forward(
            hidden_states,
            gate_weights,
            gate_correction_bias,
            up_gate_weights,
            down_weights,
            d_scales_hidden_states,
            d_scales_intermediate_hidden_states,
            d_scales_up_gate,
            d_scales_down,
            top_k,
            norm_topk_prob=True,
            permuted_weights=permuted_weights,
            experts_min=0,
            experts_max=num_experts - 1,
            chunk_size=0,
        )

        logger.debug(
            "\n===== Mixtral Moe numpy ref Output =====\n",
            extra={
                "ep_rank": ep_rank,
                "tp_rank": tp_rank,
                "final_hidden_states_ref_np": final_hidden_states_ref,
                "shape": final_hidden_states_ref.shape,
            },
        )

        # paddlenlp_ops.moe operator
        fused_gate_moe = FusedGateMoE(
            num_experts=num_experts,
            top_k=top_k,
            activation=activation,
            permuted_weights=permuted_weights,
            fused_weights=fused_weights,
            intermediate_dynamic_scale=intermediate_dynamic_scale,
            slice_max_expert=slice_max_expert,
            logger=logger,
            ep_rank=ep_rank,
            ep_size=ep_size,
            ep_group=ep_group,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            dtype=dtype,
            block_size=None,
            chunk_size=0,
        )

        final_hidden_states, amax_per_expert = fused_gate_moe.forward(
            hidden_states=hidden_states,
            gate_weights=gate_weights,
            gate_correction_bias=gate_correction_bias,
            expert_weights=(up_gate_weights, down_weights),
            hidden_states_scale=d_scales_hidden_states,
            intermediate_states_scales=d_scales_intermediate_hidden_states,
            weights_scales=(d_scales_up_gate, d_scales_down),
        )
        logger.debug(
            "\n===== paddlenlp_ops.mixture_of_experts Output =====\n",
            extra={
                "ep_rank": ep_rank,
                "tp_rank": tp_rank,
                "amax_per_expert": amax_per_expert,
                "final_hidden_states": final_hidden_states,
            },
        )

        required_similarity = 0.99
        similar = check_using_cosine_similarity(
            final_hidden_states.to("float32").cpu().numpy(),
            final_hidden_states_ref.to("float32").cpu().numpy(),
            required_similarity,
            ep_rank=ep_rank,
            tp_rank=tp_rank,
            logger=logger,
        )
        # print(f"--final_hidden_states_ref {final_hidden_states_ref}")
        # print(f"--final_hidden_states {final_hidden_states}")
        assert similar, f"Cosine similarity check failed: {similar}"


if __name__ == "__main__":
    # Set logging level to DEBUG to see debug messages
    logging.getLogger().setLevel(logging.WARNING)

    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(MoETest)

    # Create a test runner with the desired verbosity level
    runner = unittest.TextTestRunner(
        verbosity=2
    )  # Set verbosity to 2 for detailed output

    # Run the test suite
    runner.run(suite)
