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
import pytest

import logging
import numpy as np

import paddle
import paddle.nn.functional as F
import paddle.distributed as dist
import paddlenlp_ops


class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging(rank, enable_logging=False):

    logger = logging.getLogger(f"moe_rank_{rank}")
    if enable_logging or os.getenv("ENABLE_LOGGING") == "1":

        log_file = f"test_logs_rank_{rank}.log"
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] Rank %(rank)d: %(message)s")
        )
        logger.addHandler(file_handler)

        stream_handler = FlushStreamHandler(sys.stdout)
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] Rank %(rank)d: %(message)s")
        )
        logger.addHandler(stream_handler)

    logger.info("Logging initialized for rank %d", rank, extra={"rank": rank})
    return logger


def init_distributed():

    if not dist.is_initialized():
        try:
            dist.init_parallel_env()
        except Exception as e:
            print(f"Failed to initialize distributed environment: {str(e)}")
            raise

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return rank, world_size


def check_using_cosine_similarity(
    final_states, final_states_ref, required_similarity, rank, logger
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
        f"Cosine similarity: {cos_sim}, "
        f"required_similarity: {required_similarity}, ",
        extra={"rank": rank},
    )
    return cos_sim >= required_similarity


def generate_moe_params(
    num_tokens,
    hidden_dim,
    ffn_dim,
    top_k,
    num_experts,
    permuted_weights,
    dtype="bfloat16",
):
    if dtype == "float32":
        paddle_dtype = paddle.float32
    elif dtype == "float16":
        paddle_dtype = paddle.float16
    elif dtype == "bfloat16":
        paddle_dtype = paddle.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    def to_bfloat16_numpy(arr):
        arr_f32 = arr.astype(np.float32)
        arr_u32 = arr_f32.view(np.uint32)
        bfloat16_u32 = arr_u32 & 0xFFFF0000
        return bfloat16_u32.view(np.float32)

    hidden_states_numpy = np.random.randn(num_tokens, hidden_dim).astype(np.float32)
    hidden_states_numpy = to_bfloat16_numpy(hidden_states_numpy)
    w1 = np.random.randn(num_experts, hidden_dim, ffn_dim).astype(np.float32)
    w1 = to_bfloat16_numpy(w1)
    w2 = np.random.randn(num_experts, hidden_dim, ffn_dim).astype(np.float16)
    w2 = to_bfloat16_numpy(w2)
    w3 = np.random.randn(num_experts, ffn_dim, hidden_dim).astype(np.float16)
    w3 = to_bfloat16_numpy(w3)
    expert_weights_numpy = (w1, w2, w3)

    hidden_states_paddle = paddle.to_tensor(hidden_states_numpy, dtype=paddle_dtype)
    w1_paddle = [
        paddle.to_tensor(w1[i], dtype=paddle_dtype) for i in range(num_experts)
    ]
    w2_paddle = [
        paddle.to_tensor(w2[i], dtype=paddle_dtype) for i in range(num_experts)
    ]
    w3_paddle = [
        paddle.to_tensor(w3[i], dtype=paddle_dtype) for i in range(num_experts)
    ]
    expert_weights_paddle = (w1_paddle, w2_paddle, w3_paddle)
    if permuted_weights:
        w1_paddle = [w.transpose([1, 0]) for w in w1_paddle]
        w2_paddle = [w.transpose([1, 0]) for w in w2_paddle]
        w3_paddle = [w.transpose([1, 0]) for w in w3_paddle]
        expert_weights_paddle = (w1_paddle, w2_paddle, w3_paddle)

    router_logits_paddle = paddle.randn([num_tokens, num_experts], dtype=paddle_dtype)
    router_probs_paddle = F.softmax(router_logits_paddle, axis=-1)
    router_weights_paddle, routing_table_paddle = paddle.topk(
        router_probs_paddle, k=top_k, axis=-1
    )
    router_weights_paddle = router_weights_paddle / (
        paddle.sum(router_weights_paddle, axis=-1, keepdim=True) + 1e-10
    )

    router_weights_numpy = (
        (paddle.cast(router_weights_paddle, dtype="float32")).numpy().astype(np.float32)
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


def generate_moe_params_static(
    num_tokens,
    hidden_dim,
    ffn_dim,
    top_k,
    num_experts,
    permuted_weights,
    dtype="bfloat16",
):
    if dtype == "float32":
        paddle_dtype = paddle.float32
    elif dtype == "float16":
        paddle_dtype = paddle.float16
    elif dtype == "bfloat16":
        paddle_dtype = paddle.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    def to_bfloat16_numpy(arr):
        arr_f32 = arr.astype(np.float32)
        arr_u32 = arr_f32.view(np.uint32)
        bfloat16_u32 = arr_u32 & 0xFFFF0000
        return bfloat16_u32.view(np.float32)

    hidden_states_numpy = np.linspace(
        -0.1, 0.1, num_tokens * hidden_dim, dtype=np.float32
    ).reshape(num_tokens, hidden_dim)
    hidden_states_numpy = to_bfloat16_numpy(hidden_states_numpy)

    # Fixed expert weights
    w1 = np.full((num_experts, hidden_dim, ffn_dim), 1.0, dtype=np.float32)
    w1 = to_bfloat16_numpy(w1)
    w2 = np.full((num_experts, hidden_dim, ffn_dim), 2.0, dtype=np.float32)
    w2 = to_bfloat16_numpy(w2)
    w3 = np.full((num_experts, ffn_dim, hidden_dim), 3.0, dtype=np.float32)
    w3 = to_bfloat16_numpy(w3)
    expert_weights_numpy = (w1, w2, w3)

    # Convert to Paddle tensors
    hidden_states_paddle = paddle.to_tensor(hidden_states_numpy, dtype=paddle_dtype)
    w1_paddle = [
        paddle.to_tensor(w1[i], dtype=paddle_dtype) for i in range(num_experts)
    ]
    w2_paddle = [
        paddle.to_tensor(w2[i], dtype=paddle_dtype) for i in range(num_experts)
    ]
    w3_paddle = [
        paddle.to_tensor(w3[i], dtype=paddle_dtype) for i in range(num_experts)
    ]
    expert_weights_paddle = (w1_paddle, w2_paddle, w3_paddle)
    if permuted_weights:
        w1_paddle = [w.transpose([1, 0]) for w in w1_paddle]
        w2_paddle = [w.transpose([1, 0]) for w in w2_paddle]
        w3_paddle = [w.transpose([1, 0]) for w in w3_paddle]
        expert_weights_paddle = (w1_paddle, w2_paddle, w3_paddle)

    # Fixed router logits (linearly increasing)
    router_logits_numpy = np.arange(num_tokens * num_experts, dtype=np.float32).reshape(
        num_tokens, num_experts
    )
    router_logits_numpy = to_bfloat16_numpy(router_logits_numpy)
    router_logits_paddle = paddle.to_tensor(router_logits_numpy, dtype=paddle_dtype)
    router_probs_paddle = F.softmax(router_logits_paddle, axis=-1)
    router_weights_paddle, routing_table_paddle = paddle.topk(
        router_probs_paddle, k=top_k, axis=-1
    )
    router_weights_paddle = router_weights_paddle / (
        paddle.sum(router_weights_paddle, axis=-1, keepdim=True) + 1e-10
    )

    # Convert to numpy for output
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
        activation,
        rank,
        ep_size,
        permuted_weights,
        fused_weights,
        slice_max_expert,
        logger,
    ):
        self.num_experts = num_experts
        self.permuted_weights = permuted_weights
        self.fused_weights = fused_weights
        self.activation = activation
        self.ep_rank = rank
        self.ep_size = ep_size
        self.logger = logger

        self.fn = paddlenlp_ops.mixture_of_experts

        self.w1, self.w2, self.w3 = expert_weights
        self.w12 = [
            paddle.concat((w1, w2), axis=0 if self.permuted_weights else 1)
            for w1, w2 in zip(self.w1, self.w2)
        ]

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

        final_hidden_states = paddle.zeros_like(hidden_states)
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
                    self.w12[slice_experts_min : slice_experts_max + 1],
                    self.w3[slice_experts_min : slice_experts_max + 1],
                )
                if self.fused_weights
                else (
                    self.w1[slice_experts_min : slice_experts_max + 1]
                    + self.w2[slice_experts_min : slice_experts_max + 1],
                    self.w3[slice_experts_min : slice_experts_max + 1],
                )
            )

            if compute_amax:
                slice_result, slice_amax = self.fn(
                    *common_inputs, *slice_weights, *common_params, True
                )
                amax_per_expert[slice_experts_min : slice_experts_max + 1] = slice_amax
            else:
                slice_result, _ = self.fn(
                    *common_inputs, *slice_weights, *common_params, False
                )
            final_hidden_states += slice_result

        if self.ep_size > 1 and dist.is_initialized():
            try:
                dist.all_reduce(final_hidden_states, op=dist.ReduceOp.SUM)
                self.logger.info(
                    "All-reduce for MoE successfully.", extra={"rank": self.ep_rank}
                )
                if compute_amax:
                    dist.all_reduce(amax_per_expert, op=dist.ReduceOp.MAX)
                    self.logger.info(
                        "All-reduce for AMax successfully.",
                        extra={"rank": self.ep_rank},
                    )
            except Exception as e:
                self.logger.error(
                    f"Failed to perform All-reduce: {str(e)}",
                    extra={"rank": self.ep_rank},
                )
                raise

        return final_hidden_states, amax_per_expert


rank, world_size = None, None
logger = None


def global_init():
    global rank, world_size, logger
    if rank is None or world_size is None:
        rank, world_size = init_distributed()
    if logger is None:
        logger = setup_logging(rank, False)


NUM_TOKENS = [32, 64]
HIDDEN_DIMS = [128]
FFN_DIMS = [256]
TOP_K = [8]
NUM_EXPERTS = [32, 64]
SLICE_MAX_EXPERT = [32, 64]
FUSED_WEIGHTS = [True, False]
ACTIVATIONS = ["silu"]
PERMUTED_WEIGHTS = [True, False]
COMPUTE_AMAX = [True, False]
DTYPES = ["bfloat16"]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("ffn_dim", FFN_DIMS)
@pytest.mark.parametrize("top_k", TOP_K)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("slice_max_expert", SLICE_MAX_EXPERT)
@pytest.mark.parametrize("fused_weights", FUSED_WEIGHTS)
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("permuted_weights", PERMUTED_WEIGHTS)
@pytest.mark.parametrize("compute_amax", COMPUTE_AMAX)
@pytest.mark.parametrize("dtype", DTYPES)
def test_mixture_of_experts(
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
    dtype,
):

    global_init()

    paddle.seed(rank + 102)
    device = "intel_hpu"

    logger.info(
        f"\n\n======================================="
        f"`test_mixture_of_experts`: \n"
        f" num_tokens={num_tokens}, hidden_dim={hidden_dim}, ffn_dim={ffn_dim}, \n"
        f" top_k={top_k}, num_experts={num_experts}, slice_max_expert={slice_max_expert}, \n"
        f" fused_weights={fused_weights}, permuted_weights={permuted_weights}, activation={activation}, \n"
        f" compute_amax={compute_amax}, dtype={dtype}, \n",
        extra={"rank": rank},
    )

    if world_size == 1:
        numpy_data, paddle_data = generate_moe_params(
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            top_k=top_k,
            num_experts=num_experts,
            permuted_weights=permuted_weights,
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

    print("===== Mixtral Moe numpy ref Output =====")
    print("Final Hidden States (ref):")
    print(f"{final_hidden_states_ref_np}, shape:{final_hidden_states_ref_np.shape}")
    print("AMAX per Expert (ref):")
    print(amax_per_expert_ref_np)
    print("=========================================")

    # paddlenlp_ops.moe operator
    fused_moe = FusedMoE(
        num_experts=num_experts,
        expert_weights=expert_weights_pd,
        activation=activation,
        rank=rank,
        ep_size=world_size,
        permuted_weights=permuted_weights,
        fused_weights=fused_weights,
        slice_max_expert=slice_max_expert,
        logger=logger,
    )

    final_hidden_states, amax_per_expert = fused_moe.forward(
        hidden_states=hidden_states_pd,
        router_weights=router_weights_pd,
        routing_table=routing_table_pd,
        compute_amax=compute_amax,
    )

    print("\n===== paddlenlp_ops.mixture_of_experts Output =====")
    print("Final Hidden States (paddlenlp_ops.mixture_of_experts):")
    print(final_hidden_states)
    print("AMAX per Expert (paddlenlp_ops.mixture_of_experts):")
    print(amax_per_expert)
    print("=========================================")

    required_similarity = 0.98
    similar = check_using_cosine_similarity(
        final_hidden_states.to("float32").cpu().numpy(),
        final_hidden_states_ref_np,
        required_similarity,
        rank=0,
        logger=logger,
    )
    assert similar, f"Cosine similarity check failed: {similar}"

    if compute_amax:
        assert device in str(amax_per_expert.place)
        mask = amax_per_expert_ref_np != 0
        fused_op_vals = amax_per_expert.to("cpu").numpy()[mask]
        ref_vals = amax_per_expert_ref_np[mask]
        print(f"amax_per_expert: {fused_op_vals}, ref: {ref_vals}")
        rtol = 0.01
        atol = 0.01
        if mask.any():
            logger.info(
                f"Comparing amax: "
                f"fused_moe={fused_op_vals.tolist()}, "
                f"ref_mixtral_moe={ref_vals.tolist()}",
                extra={"rank": rank},
            )
            np.testing.assert_allclose(fused_op_vals, ref_vals, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
