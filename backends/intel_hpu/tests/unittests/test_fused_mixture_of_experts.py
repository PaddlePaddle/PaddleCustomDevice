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
import paddle.nn.functional as F
import paddle.distributed as dist
import paddlenlp_ops

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 1)
paddle.device.set_device(f"intel_hpu:{intel_hpus_module_id}")


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


def generate_moe_params(
    dtype,
    num_tokens,
    hidden_dim,
    ffn_dim,
    top_k,
    num_experts,
    permuted_weights,
    fused_weights,
    dynamic_scale=None,
    fp8_scales=None,
    block_size=None,
):
    if dtype == "float32":
        paddle_dtype = paddle.float32
    elif dtype == "float16":
        paddle_dtype = paddle.float16
    elif dtype == "bfloat16":
        paddle_dtype = paddle.bfloat16
    elif dtype == "fp8":
        paddle_dtype = paddle.bfloat16
    elif dtype == "blockwise_fp8":
        paddle_dtype = paddle.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    hidden_states_paddle = paddle.randn([num_tokens, hidden_dim], dtype=paddle_dtype)
    w1_paddle = [
        paddle.randn([hidden_dim, ffn_dim], dtype=paddle_dtype)
        for _ in range(num_experts)
    ]
    w2_paddle = [
        paddle.randn([hidden_dim, ffn_dim], dtype=paddle_dtype)
        for _ in range(num_experts)
    ]
    w3_paddle = [
        paddle.randn([ffn_dim, hidden_dim], dtype=paddle_dtype)
        for _ in range(num_experts)
    ]

    w1_numpy = [paddle.cast(w, "float32").numpy().astype(np.float32) for w in w1_paddle]
    w2_numpy = [paddle.cast(w, "float32").numpy().astype(np.float32) for w in w2_paddle]
    w3_numpy = [paddle.cast(w, "float32").numpy().astype(np.float32) for w in w3_paddle]
    expert_weights_numpy = (w1_numpy, w2_numpy, w3_numpy)

    if permuted_weights:
        w1_paddle = [w.transpose([1, 0]) for w in w1_paddle]
        w2_paddle = [w.transpose([1, 0]) for w in w2_paddle]
        w3_paddle = [w.transpose([1, 0]) for w in w3_paddle]

    if fused_weights:
        w12_paddle = [
            paddle.concat((w1, w2), axis=0)
            if permuted_weights
            else paddle.concat((w1, w2), axis=1)
            for w1, w2 in zip(w1_paddle, w2_paddle)
        ]

    # fp8 handling
    if dtype == "blockwise_fp8":
        if fused_weights:
            w12_paddle, d_scales_w12_pd = blockwise_quant_to_fp8(w12_paddle, block_size)
        else:
            w1_paddle, d_scales_w1_pd = blockwise_quant_to_fp8(w1_paddle, block_size)
            w2_paddle, d_scales_w2_pd = blockwise_quant_to_fp8(w2_paddle, block_size)
        w3_paddle, d_scales_w3_pd = blockwise_quant_to_fp8(w3_paddle, block_size)
    elif dtype == "fp8":
        d_scales_w1 = fp8_scales["d_scale_w1"]
        d_scales_w2 = fp8_scales["d_scale_w2"]
        d_scales_w3 = fp8_scales["d_scale_w3"]
        # weights cast to fp8, scales to tensor

        if fused_weights:
            d_scales_w12_pd = [
                paddle.to_tensor(scale, dtype=paddle_dtype) for scale in d_scales_w1
            ]
            w12_paddle = [
                (w / d_scale).cast(paddle.float8_e4m3fn)
                for w, d_scale in zip(w12_paddle, d_scales_w1, strict=False)
            ]
        else:
            d_scales_w1_pd = [
                paddle.to_tensor(scale, dtype=paddle_dtype) for scale in d_scales_w1
            ]
            w1_paddle = [
                (w / d_scale).cast(paddle.float8_e4m3fn)
                for w, d_scale in zip(w1_paddle, d_scales_w1, strict=False)
            ]
            d_scales_w2_pd = [
                paddle.to_tensor(scale, dtype=paddle_dtype) for scale in d_scales_w2
            ]
            w2_paddle = [
                (w / d_scale).cast(paddle.float8_e4m3fn)
                for w, d_scale in zip(w2_paddle, d_scales_w2, strict=False)
            ]
        d_scales_w3_pd = [
            paddle.to_tensor(scale, dtype=paddle_dtype) for scale in d_scales_w3
        ]
        w3_paddle = [
            (w / d_scale).cast(paddle.float8_e4m3fn)
            for w, d_scale in zip(w3_paddle, d_scales_w3, strict=False)
        ]

    weight_scales_pd = None
    if fused_weights:
        expert_weights_paddle = (w12_paddle, w3_paddle)
        if dtype == "fp8" or dtype == "blockwise_fp8":
            weight_scales_pd = (d_scales_w12_pd, d_scales_w3_pd)
    else:
        expert_weights_paddle = (w1_paddle, w2_paddle, w3_paddle)
        if dtype == "fp8" or dtype == "blockwise_fp8":
            weight_scales_pd = (d_scales_w1_pd, d_scales_w2_pd, d_scales_w3_pd)

    router_logits_paddle = paddle.randn([num_tokens, num_experts], dtype=paddle_dtype)
    router_probs_paddle = F.softmax(router_logits_paddle, axis=-1)
    router_weights_paddle, routing_table_paddle = paddle.topk(
        router_probs_paddle, k=top_k, axis=-1
    )
    router_weights_paddle = router_weights_paddle / (
        paddle.sum(router_weights_paddle, axis=-1, keepdim=True) + 1e-10
    )

    hidden_states_numpy = (
        (paddle.cast(hidden_states_paddle, dtype="float32")).numpy().astype(np.float32)
    )
    router_weights_numpy = (
        (paddle.cast(router_weights_paddle, dtype="float32")).numpy().astype(np.float32)
    )
    routing_table_numpy = routing_table_paddle.numpy().astype(np.int32)

    hidden_states_scale_pd, intermediate_states_scales_pd = None, None
    if dtype == "fp8":
        hidden_states_scale = fp8_scales["d_scale_hidden_states"]
        # hidden_states cast to fp8, scales to tensor
        hidden_states_scale_pd = paddle.to_tensor(
            hidden_states_scale, dtype=paddle_dtype
        )
        hidden_states_paddle = (hidden_states_paddle / hidden_states_scale).cast(
            paddle.float8_e4m3fn
        )
        if not dynamic_scale:
            intermediate_states_scales = fp8_scales[
                "d_scale_intermediate_hidden_states"
            ]
            intermediate_states_scales_pd = [
                paddle.to_tensor(scale, dtype=paddle_dtype)
                for scale in intermediate_states_scales
            ]

    numpy_data = (
        hidden_states_numpy,
        router_weights_numpy,
        routing_table_numpy,
        expert_weights_numpy,
    )
    paddle_data = (
        hidden_states_paddle,
        router_weights_paddle,
        routing_table_paddle,
        expert_weights_paddle,
        hidden_states_scale_pd,
        intermediate_states_scales_pd,
        weight_scales_pd,
    )
    return numpy_data, paddle_data


def generate_moe_params_static(
    dtype,
    num_tokens,
    hidden_dim,
    ffn_dim,
    top_k,
    num_experts,
    permuted_weights,
    tp_rank=0,
    tp_size=1,
):
    if dtype == "float32":
        paddle_dtype = paddle.float32
    elif dtype == "float16":
        paddle_dtype = paddle.float16
    elif dtype == "bfloat16":
        paddle_dtype = paddle.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Split weights for paddle_data (TP)
    ffn_dim_per_tp = ffn_dim // tp_size
    start_idx = tp_rank * ffn_dim_per_tp
    end_idx = (tp_rank + 1) * ffn_dim_per_tp

    row_values = paddle.arange(1, num_tokens + 1, dtype=paddle.float32) * 0.01
    hidden_states_paddle = (
        row_values.cast(paddle_dtype).reshape([-1, 1]).tile([1, hidden_dim])
    )
    w1_paddle = [
        paddle.full([hidden_dim, ffn_dim_per_tp], fill_value=0.01, dtype=paddle_dtype)
        for _ in range(num_experts)
    ]
    w2_paddle = [
        paddle.full([hidden_dim, ffn_dim_per_tp], fill_value=0.02, dtype=paddle_dtype)
        for _ in range(num_experts)
    ]
    w3_paddle = [
        paddle.full([ffn_dim_per_tp, hidden_dim], fill_value=0.03, dtype=paddle_dtype)
        for _ in range(num_experts)
    ]
    w1_numpy = [paddle.cast(w, "float32").numpy().astype(np.float32) for w in w1_paddle]
    w2_numpy = [paddle.cast(w, "float32").numpy().astype(np.float32) for w in w2_paddle]
    w3_numpy = [paddle.cast(w, "float32").numpy().astype(np.float32) for w in w3_paddle]
    expert_weights_numpy = (w1_numpy, w2_numpy, w3_numpy)

    if permuted_weights:
        w1_paddle = [w.transpose([1, 0]) for w in w1_paddle]
        w2_paddle = [w.transpose([1, 0]) for w in w2_paddle]
        w3_paddle = [w.transpose([1, 0]) for w in w3_paddle]
    expert_weights_paddle = (w1_paddle, w2_paddle, w3_paddle)

    router_logits_paddle = paddle.arange(
        num_tokens * num_experts, dtype=paddle_dtype
    ).reshape([num_tokens, num_experts])
    router_probs_paddle = F.softmax(router_logits_paddle, axis=-1)
    router_weights_paddle, routing_table_paddle = paddle.topk(
        router_probs_paddle, k=top_k, axis=-1
    )
    router_weights_paddle = router_weights_paddle / (
        paddle.sum(router_weights_paddle, axis=-1, keepdim=True) + 1e-10
    )

    hidden_states_numpy = (
        paddle.cast(hidden_states_paddle, dtype="float32").numpy().astype(np.float32)
    )
    router_weights_numpy = (
        paddle.cast(router_weights_paddle, dtype="float32").numpy().astype(np.float32)
    )
    routing_table_numpy = routing_table_paddle.numpy()

    numpy_data = (
        hidden_states_numpy,
        router_weights_numpy,
        routing_table_numpy,
        expert_weights_numpy,
    )
    paddle_data = (
        hidden_states_paddle,
        router_weights_paddle,
        routing_table_paddle,
        expert_weights_paddle,
    )
    return numpy_data, paddle_data


class MixtralBlockSparseMLP_Numpy:
    def __init__(self, w1, w2, w3, activation="silu"):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.activation_fn = self.__get_activation_fn(activation)

    def __get_activation_fn(self, activation):
        if activation == "gelu":

            def gelu(x):
                return (
                    x
                    * 0.5
                    * (
                        1.0
                        + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))
                    )
                )

            return gelu
        elif activation == "relu":
            return lambda x: np.maximum(0, x)
        elif activation == "silu":

            def silu(x):
                x_clipped = np.clip(x, -10.0, 10.0)
                sigmoid_x = 1 / (1 + np.exp(-x_clipped))
                return x * sigmoid_x

            return silu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, hidden_states, compute_amax=False):
        if hidden_states.size == 0:
            output = np.zeros_like(hidden_states)
            amax = 0.0 if compute_amax else None
            return output, amax

        hidden_states_w1 = self.activation_fn(np.matmul(hidden_states, self.w1))
        hidden_states_w2 = np.matmul(hidden_states, self.w2)
        intermediate = hidden_states_w1 * hidden_states_w2
        output = np.matmul(intermediate, self.w3)
        amax = np.max(np.abs(intermediate)) if compute_amax else None
        return output, amax


class MixtralSparseMoeRef_Numpy:
    def __init__(self, hidden_dim, num_experts, expert_weights, activation="silu"):
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        w1, w2, w3 = expert_weights
        self.experts = [
            MixtralBlockSparseMLP_Numpy(w1[i], w2[i], w3[i], activation)
            for i in range(num_experts)
        ]

    def forward(self, hidden_states, router_weights, routing_table):
        amax_per_expert = np.zeros(self.num_experts, dtype=np.float32)
        final_hidden_states = np.zeros_like(hidden_states)
        routing_table = routing_table.astype(np.int64)

        expert_mask = np.eye(self.num_experts, dtype=np.int64)[routing_table].transpose(
            2, 1, 0
        )

        for expert_idx in range(self.num_experts):
            idx, top_x = np.where(expert_mask[expert_idx])
            if idx.size == 0:
                continue
            current_state = hidden_states[top_x].reshape(-1, self.hidden_dim)
            current_hidden_states, current_amax = self.experts[expert_idx].forward(
                current_state, compute_amax=True
            )
            current_hidden_states *= router_weights[top_x, idx, None]

            for i, pos in enumerate(top_x):
                final_hidden_states[pos] += current_hidden_states[i]

            amax_per_expert[expert_idx] = (
                current_amax if current_amax is not None else 0.0
            )

        return final_hidden_states.reshape(hidden_states.shape), amax_per_expert


class FusedMoE:
    def __init__(
        self,
        num_experts,
        expert_weights,
        hidden_states_scale,
        intermediate_states_scales,
        weights_scales,
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
        dtype="bfloat16",
        dynamic_scale=None,
        block_size=None,
    ):
        self.num_experts = num_experts
        self.permuted_weights = permuted_weights
        self.fused_weights = fused_weights
        self.dynamic_scale = dynamic_scale
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

        if self.dtype == "fp8":
            self.fn = paddlenlp_ops.mixture_of_experts_fp8
        elif self.dtype == "blockwise_fp8":
            self.fn = paddlenlp_ops.mixture_of_experts_blockwise_fp8
        else:
            self.fn = paddlenlp_ops.mixture_of_experts

        self.expert_weights = expert_weights
        self.hidden_scale = hidden_states_scale
        self.intermediate_hidden_scales = intermediate_states_scales
        self.weights_scales = weights_scales

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

    def forward(self, hidden_states, router_weights, routing_table, compute_amax=False):
        common_inputs = (hidden_states, routing_table, router_weights)
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
                self.permuted_weights,
                self.activation,
                slice_experts_min,
                slice_experts_max,
            )
            slice_weights = (
                (
                    self.expert_weights[0][slice_experts_min : slice_experts_max + 1],
                    self.expert_weights[1][slice_experts_min : slice_experts_max + 1],
                )
                if self.fused_weights
                else (
                    self.expert_weights[0][slice_experts_min : slice_experts_max + 1]
                    + self.expert_weights[1][slice_experts_min : slice_experts_max + 1],
                    self.expert_weights[2][slice_experts_min : slice_experts_max + 1],
                )
            )
            if self.dtype == "fp8":
                slice_scales = (
                    (
                        self.hidden_scale,
                        None
                        if self.dynamic_scale
                        else self.intermediate_hidden_scales[
                            slice_experts_min : slice_experts_max + 1
                        ],
                        self.weights_scales[0][
                            slice_experts_min : slice_experts_max + 1
                        ],
                        self.weights_scales[1][
                            slice_experts_min : slice_experts_max + 1
                        ],
                    )
                    if self.fused_weights
                    else (
                        self.hidden_scale,
                        None
                        if self.dynamic_scale
                        else self.intermediate_hidden_scales[
                            slice_experts_min : slice_experts_max + 1
                        ],
                        self.weights_scales[0][
                            slice_experts_min : slice_experts_max + 1
                        ]
                        + self.weights_scales[1][
                            slice_experts_min : slice_experts_max + 1
                        ],
                        self.weights_scales[2][
                            slice_experts_min : slice_experts_max + 1
                        ],
                    )
                )
            elif self.dtype == "blockwise_fp8":
                slice_scales = (
                    (
                        self.weights_scales[0][
                            slice_experts_min : slice_experts_max + 1
                        ],
                        self.weights_scales[1][
                            slice_experts_min : slice_experts_max + 1
                        ],
                    )
                    if self.fused_weights
                    else (
                        self.weights_scales[0][
                            slice_experts_min : slice_experts_max + 1
                        ]
                        + self.weights_scales[1][
                            slice_experts_min : slice_experts_max + 1
                        ],
                        self.weights_scales[2][
                            slice_experts_min : slice_experts_max + 1
                        ],
                    )
                )

            chunk_size = 0
            if self.dtype == "fp8":
                slice_result = self.fn(
                    *common_inputs,
                    *slice_weights,
                    *slice_scales,
                    *common_params,
                    self.dynamic_scale,
                    chunk_size,
                )
            elif self.dtype == "blockwise_fp8":
                slice_result = self.fn(
                    *common_inputs,
                    *slice_weights,
                    *slice_scales,
                    *common_params,
                    self.block_size,
                    chunk_size,
                )
            else:
                slice_result, slice_amax = self.fn(
                    *common_inputs,
                    *slice_weights,
                    *common_params,
                    compute_amax,
                    chunk_size,
                )
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


DTYPES = ["bfloat16", "fp8", "blockwise_fp8"]
NUM_TOKENS = [32]
HIDDEN_DIMS = [4096]
FFN_DIMS = [2560]
TOP_K = [2]
NUM_EXPERTS = [8]
SLICE_MAX_EXPERT = [8]
FUSED_WEIGHTS = [True, False]
ACTIVATIONS = ["gelu", "relu", "silu"]
PERMUTED_WEIGHTS = [True, False]
EP_SIZE = [1]
TP_SIZE = [1]
# for bfloat16 only
COMPUTE_AMAX = [True, False]
# for fp8 only
DYNAMIC_SCALE = [True, False]
FP8_SCALES = [
    {
        "d_scale_hidden_states": 3.27,
        "d_scale_intermediate_hidden_states": [
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
        ],
        "d_scale_w1": [4.35, 1.49, 1.12, 2.22, 8.33, 1.28, 2.94, 1.79],
        "d_scale_w2": [1.10, 2.13, 2.78, 1.22, 3.45, 1.59, 1.35, 1.72],
        "d_scale_w3": [6.67, 1.09, 2.08, 2.70, 1.56, 1.23, 1.89, 3.85],
    },
    {
        "d_scale_hidden_states": 1.0,
        "d_scale_intermediate_hidden_states": [
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
        ],
        "d_scale_w1": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "d_scale_w2": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "d_scale_w3": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
]
# for blockwise_fp8 only
BLOCK_SIZES = [128]


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
                compute_amax,
                ep_size,
                tp_size,
                # dtype,
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
            for compute_amax in COMPUTE_AMAX
            for ep_size in EP_SIZE
            for tp_size in TP_SIZE
            # for dtype in DTYPES
        ]
    )
    def test_mixture_of_experts(
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
        compute_amax,
        ep_size,
        tp_size,
        dtype="bfloat16",
    ):
        (ep_rank, ep_size, ep_group), (tp_rank, tp_size, tp_group) = init_distributed(
            ep_size, tp_size
        )
        logger = setup_logging(ep_rank=ep_rank, tp_rank=tp_rank)
        logger.debug(
            f"\n\n======================================="
            f"`test_mixture_of_experts`: \n"
            f" num_tokens={num_tokens}, hidden_dim={hidden_dim}, ffn_dim={ffn_dim}, \n"
            f" top_k={top_k}, num_experts={num_experts}, slice_max_expert={slice_max_expert}, \n"
            f" fused_weights={fused_weights}, permuted_weights={permuted_weights}, activation={activation}, \n"
            f" dtype={dtype}, compute_amax={compute_amax}, \n"
            f" ep_size={ep_size}, tp_size={tp_size}, \n",
            extra={"ep_rank": ep_rank, "tp_rank": tp_rank},
        )

        paddle.seed(ep_rank * 100 + tp_rank + 1024)
        device = "intel_hpu"
        if ep_size == 1 and tp_size == 1:
            numpy_data, paddle_data = generate_moe_params(
                num_tokens=num_tokens,
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                top_k=top_k,
                num_experts=num_experts,
                permuted_weights=permuted_weights,
                fused_weights=fused_weights,
                dynamic_scale=None,
                fp8_scales=None,
                dtype=dtype,
                block_size=None,
            )
        else:
            numpy_data, paddle_data = generate_moe_params_static(
                num_tokens=num_tokens,
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                top_k=top_k,
                num_experts=num_experts,
                permuted_weights=permuted_weights,
                dtype=dtype,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )

        (
            hidden_states_np,
            router_weights_np,
            routing_table_np,
            expert_weights_np,
        ) = numpy_data
        (
            hidden_states_pd,
            router_weights_pd,
            routing_table_pd,
            expert_weights_pd,
            hidden_states_scale_pd,
            intermediate_states_scales_pd,
            weight_scales_pd,
        ) = paddle_data

        # CPU Reference Implementation
        mixtral_ref_np = MixtralSparseMoeRef_Numpy(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            expert_weights=expert_weights_np,
            activation=activation,
        )

        final_hidden_states_ref_np, amax_per_expert_ref_np = mixtral_ref_np.forward(
            hidden_states=hidden_states_np,
            router_weights=router_weights_np,
            routing_table=routing_table_np,
        )

        logger.debug(
            "\n===== Mixtral Moe numpy ref Output =====\n",
            extra={
                "ep_rank": ep_rank,
                "tp_rank": tp_rank,
                "amax_per_expert_ref_np": amax_per_expert_ref_np,
                "final_hidden_states_ref_np": final_hidden_states_ref_np,
                "shape": final_hidden_states_ref_np.shape,
            },
        )

        # paddlenlp_ops.moe operator
        fused_moe = FusedMoE(
            num_experts=num_experts,
            expert_weights=expert_weights_pd,
            hidden_states_scale=hidden_states_scale_pd,
            intermediate_states_scales=intermediate_states_scales_pd,
            weights_scales=weight_scales_pd,
            activation=activation,
            permuted_weights=permuted_weights,
            fused_weights=fused_weights,
            dynamic_scale=None,
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
        )

        final_hidden_states, amax_per_expert = fused_moe.forward(
            hidden_states=hidden_states_pd,
            router_weights=router_weights_pd,
            routing_table=routing_table_pd,
            compute_amax=compute_amax,
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
            final_hidden_states_ref_np,
            required_similarity,
            ep_rank=ep_rank,
            tp_rank=tp_rank,
            logger=logger,
        )
        assert similar, f"Cosine similarity check failed: {similar}"

        if compute_amax:
            assert device in str(amax_per_expert.place)
            mask = amax_per_expert_ref_np != 0
            fused_op_vals = amax_per_expert.to("cpu").numpy()[mask]
            ref_vals = amax_per_expert_ref_np[mask]
            logger.debug(f"amax_per_expert: {fused_op_vals}, ref: {ref_vals}")
            rtol = 0.01
            atol = 0.01
            if mask.any():
                logger.info(
                    f"Comparing amax: \n"
                    f"fused_moe={fused_op_vals.tolist()}, \n"
                    f"ref_mixtral_moe={ref_vals.tolist()} \n",
                    extra={"ep_rank": ep_rank, "tp_rank": tp_rank},
                )
                np.testing.assert_allclose(
                    fused_op_vals, ref_vals, rtol=rtol, atol=atol
                )

            if ep_size > 1 or tp_size > 1:
                logger.info(
                    "Destroying communication groups",
                    extra={"ep_rank": ep_rank, "tp_rank": tp_rank},
                )
                dist.destroy_process_group(ep_group)
                dist.destroy_process_group(tp_group)

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
                dynamic_scale,
                fp8_scales,
                # dtype,
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
            for dynamic_scale in DYNAMIC_SCALE
            for fp8_scales in FP8_SCALES
            # for dtype in DTYPES
        ]
    )
    def test_mixture_of_experts_fp8(
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
        dynamic_scale,
        fp8_scales,
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
            f" dtype={dtype}, dynamic_scale={dynamic_scale}, \n"
            f" ep_size={ep_size}, tp_size={tp_size}, \n",
            extra={"ep_rank": ep_rank, "tp_rank": tp_rank},
        )

        paddle.seed(ep_rank * 100 + tp_rank + 1024)
        device = "intel_hpu"
        if ep_size == 1 and tp_size == 1:
            numpy_data, paddle_data = generate_moe_params(
                num_tokens=num_tokens,
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                top_k=top_k,
                num_experts=num_experts,
                permuted_weights=permuted_weights,
                fused_weights=fused_weights,
                dynamic_scale=dynamic_scale,
                fp8_scales=fp8_scales,
                dtype=dtype,
            )
        else:
            numpy_data, paddle_data = generate_moe_params_static(
                num_tokens=num_tokens,
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                top_k=top_k,
                num_experts=num_experts,
                permuted_weights=permuted_weights,
                dtype=dtype,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )

        (
            hidden_states_np,
            router_weights_np,
            routing_table_np,
            expert_weights_np,
        ) = numpy_data
        (
            hidden_states_pd,
            router_weights_pd,
            routing_table_pd,
            expert_weights_pd,
            hidden_states_scale_pd,
            intermediate_states_scales_pd,
            weight_scales_pd,
        ) = paddle_data

        # CPU Reference Implementation
        mixtral_ref_np = MixtralSparseMoeRef_Numpy(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            expert_weights=expert_weights_np,
            activation=activation,
        )

        final_hidden_states_ref_np, amax_per_expert_ref_np = mixtral_ref_np.forward(
            hidden_states=hidden_states_np,
            router_weights=router_weights_np,
            routing_table=routing_table_np,
        )

        logger.debug(
            "\n===== Mixtral Moe numpy ref Output =====\n",
            extra={
                "ep_rank": ep_rank,
                "tp_rank": tp_rank,
                "amax_per_expert_ref_np": amax_per_expert_ref_np,
                "final_hidden_states_ref_np": final_hidden_states_ref_np,
                "shape": final_hidden_states_ref_np.shape,
            },
        )

        # paddlenlp_ops.moe operator
        fused_moe = FusedMoE(
            num_experts=num_experts,
            expert_weights=expert_weights_pd,
            hidden_states_scale=hidden_states_scale_pd,
            intermediate_states_scales=intermediate_states_scales_pd,
            weights_scales=weight_scales_pd,
            activation=activation,
            permuted_weights=permuted_weights,
            fused_weights=fused_weights,
            dynamic_scale=dynamic_scale,
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
        )

        final_hidden_states, amax_per_expert = fused_moe.forward(
            hidden_states=hidden_states_pd,
            router_weights=router_weights_pd,
            routing_table=routing_table_pd,
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
            final_hidden_states_ref_np,
            required_similarity,
            ep_rank=ep_rank,
            tp_rank=tp_rank,
            logger=logger,
        )
        assert similar, f"Cosine similarity check failed: {similar}"

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
                block_size,
                # dtype,
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
            for block_size in BLOCK_SIZES
            # for dtype in DTYPES
        ]
    )
    def test_mixture_of_experts_blockwise_fp8(
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
        block_size,
        dtype="blockwise_fp8",
    ):
        (ep_rank, ep_size, ep_group), (tp_rank, tp_size, tp_group) = init_distributed(
            ep_size, tp_size
        )
        logger = setup_logging(ep_rank=ep_rank, tp_rank=tp_rank)
        logger.debug(
            f"\n\n======================================="
            f"`test_mixture_of_experts_blockwise_fp8`: \n"
            f" num_tokens={num_tokens}, hidden_dim={hidden_dim}, ffn_dim={ffn_dim}, \n"
            f" top_k={top_k}, num_experts={num_experts}, slice_max_expert={slice_max_expert}, \n"
            f" fused_weights={fused_weights}, permuted_weights={permuted_weights}, activation={activation}, \n"
            f" dtype={dtype}, block_size={block_size}, \n"
            f" ep_size={ep_size}, tp_size={tp_size}, \n",
            extra={"ep_rank": ep_rank, "tp_rank": tp_rank},
        )

        paddle.seed(ep_rank * 100 + tp_rank + 1024)
        device = "intel_hpu"
        if ep_size == 1 and tp_size == 1:
            numpy_data, paddle_data = generate_moe_params(
                num_tokens=num_tokens,
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                top_k=top_k,
                num_experts=num_experts,
                permuted_weights=permuted_weights,
                fused_weights=fused_weights,
                dtype=dtype,
                block_size=block_size,
            )
        else:
            numpy_data, paddle_data = generate_moe_params_static(
                num_tokens=num_tokens,
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                top_k=top_k,
                num_experts=num_experts,
                permuted_weights=permuted_weights,
                dtype=dtype,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )

        (
            hidden_states_np,
            router_weights_np,
            routing_table_np,
            expert_weights_np,
        ) = numpy_data
        (
            hidden_states_pd,
            router_weights_pd,
            routing_table_pd,
            expert_weights_pd,
            hidden_states_scale_pd,
            intermediate_states_scales_pd,
            weight_scales_pd,
        ) = paddle_data

        # CPU Reference Implementation
        mixtral_ref_np = MixtralSparseMoeRef_Numpy(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            expert_weights=expert_weights_np,
            activation=activation,
        )

        final_hidden_states_ref_np, amax_per_expert_ref_np = mixtral_ref_np.forward(
            hidden_states=hidden_states_np,
            router_weights=router_weights_np,
            routing_table=routing_table_np,
        )

        logger.debug(
            "\n===== Mixtral Moe numpy ref Output =====\n",
            extra={
                "ep_rank": ep_rank,
                "tp_rank": tp_rank,
                "amax_per_expert_ref_np": amax_per_expert_ref_np,
                "final_hidden_states_ref_np": final_hidden_states_ref_np,
                "shape": final_hidden_states_ref_np.shape,
            },
        )

        # paddlenlp_ops.moe operator
        fused_moe = FusedMoE(
            num_experts=num_experts,
            expert_weights=expert_weights_pd,
            hidden_states_scale=hidden_states_scale_pd,
            intermediate_states_scales=intermediate_states_scales_pd,
            weights_scales=weight_scales_pd,
            activation=activation,
            permuted_weights=permuted_weights,
            fused_weights=fused_weights,
            slice_max_expert=slice_max_expert,
            logger=logger,
            ep_rank=ep_rank,
            ep_size=ep_size,
            ep_group=ep_group,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            dtype=dtype,
            block_size=block_size,
        )

        final_hidden_states, amax_per_expert = fused_moe.forward(
            hidden_states=hidden_states_pd,
            router_weights=router_weights_pd,
            routing_table=routing_table_pd,
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

        required_similarity = 0.98
        similar = check_using_cosine_similarity(
            final_hidden_states.to("float32").cpu().numpy(),
            final_hidden_states_ref_np,
            required_similarity,
            ep_rank=ep_rank,
            tp_rank=tp_rank,
            logger=logger,
        )
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
