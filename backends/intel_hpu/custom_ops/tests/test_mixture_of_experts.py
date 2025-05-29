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
import pytest

import logging
import math
import numpy as np

import paddle
import paddle.nn.functional as F
import paddlenlp_ops


NUM_TOKENS = [16, 32, 64, 128]
HIDDEN_DIMS = [128]
FFN_DIMS = [256]
NUM_EXPERTS = [16]
SLICE_MAX_EXPERT = [16]
FUSED_WEIGHTS = [True, False]
ACTIVATIONS = ["silu"]
PERMUTED_WEIGHTS = [True, False]
COMPUTE_AMAX = [True, False]
DTYPES = ["bfloat16"]


class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging(rank, enable_logging=False):
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.CRITICAL + 1)

    if enable_logging or os.getenv("ENABLE_LOGGING") == "1":
        log_file = f"test_logs_rank_{rank}.log"
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] Rank %(rank)d: %(message)s")
        )

        stream_handler = FlushStreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] Rank %(rank)d: %(message)s")
        )

        logger.setLevel(logging.DEBUG)
        logger.handlers = [stream_handler, file_handler]
        logger.propagate = False

    logger.rank = rank
    return logger


def setup_device(device, rank, logger):
    paddle.seed(102)

    hpu_available = paddle.is_compiled_with_custom_device("intel_hpu")
    visible_modules_str = os.environ.get("HABANA_VISIBLE_MODULES", "0")
    visible_modules = (
        list(range(8))
        if visible_modules_str == "all"
        else [int(m) for m in visible_modules_str.split(",")]
    )

    if not visible_modules:
        logger.error("No HPU modules specified", extra={"rank": rank})
        raise RuntimeError("No HPU modules specified")
    if len(visible_modules) <= rank:
        logger.error(
            f"Insufficient HPU modules for rank {rank}: {len(visible_modules)} available",
            extra={"rank": rank},
        )
        raise RuntimeError(f"Insufficient HPU modules for rank {rank}")
    try:
        target_device = visible_modules[rank]
        paddle.device.set_device(f"{device}:{target_device}")
        logger.info(
            f"Rank {rank} assigned to HPU device {target_device}", extra={"rank": rank}
        )
    except Exception as e:
        logger.error(
            f"Failed to set HPU device: {str(e)}. Falling back to CPU",
            extra={"rank": rank},
        )
        hpu_available = False

    return hpu_available


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
        f"Cosine similarity: {cos_sim}, required_similarity: {required_similarity}, rank: {rank}"
    )
    return cos_sim >= required_similarity


def generate_moe_params_np(
    num_tokens, hidden_dim, ffn_dim, num_experts, permuted_weights, dtype="bfloat16"
):
    if dtype == "float32":
        paddle_dtype = paddle.float32
    elif dtype == "float16":
        paddle_dtype = paddle.float16
    elif dtype == "bfloat16":
        paddle_dtype = paddle.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    hidden_states_pd = paddle.randn([num_tokens, hidden_dim], dtype=paddle_dtype)
    router_logits_pd = paddle.randn([num_tokens, num_experts], dtype=paddle_dtype)
    router_probs_pd = F.softmax(router_logits_pd, axis=-1)
    router_weights_pd, routing_table_pd = paddle.topk(
        router_probs_pd, k=num_experts, axis=-1
    )

    w1_pd = [
        paddle.randn([hidden_dim, ffn_dim], dtype=paddle_dtype)
        for _ in range(num_experts)
    ]
    w2_pd = [
        paddle.randn([hidden_dim, ffn_dim], dtype=paddle_dtype)
        for _ in range(num_experts)
    ]
    w3_pd = [
        paddle.randn([ffn_dim, hidden_dim], dtype=paddle_dtype)
        for _ in range(num_experts)
    ]
    expert_weights_pd = (w1_pd, w2_pd, w3_pd)

    hidden_states_np = hidden_states_pd.cast(paddle.float32).numpy()
    router_weights_np = router_weights_pd.cast(paddle.float32).numpy()
    routing_table_np = routing_table_pd.numpy()
    w1_np = [w.cast(paddle.float32).numpy() for w in w1_pd]
    w2_np = [w.cast(paddle.float32).numpy() for w in w2_pd]
    w3_np = [w.cast(paddle.float32).numpy() for w in w3_pd]
    expert_weights_np = (w1_np, w2_np, w3_np)

    if permuted_weights:
        w1_pd = [w.transpose([1, 0]) for w in w1_pd]
        w2_pd = [w.transpose([1, 0]) for w in w2_pd]
        w3_pd = [w.transpose([1, 0]) for w in w3_pd]
        expert_weights_pd = (w1_pd, w2_pd, w3_pd)

    numpy_data = (
        hidden_states_np,
        router_weights_np,
        routing_table_np,
        expert_weights_np,
    )
    paddle_data = (
        hidden_states_pd,
        router_weights_pd,
        routing_table_pd,
        expert_weights_pd,
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


class FusedMoEOp:
    def __init__(
        self,
        num_experts,
        expert_weights,
        activation,
        rank,
        permuted_weights,
        fused_weights,
        slice_max_expert,
    ):
        self.num_experts = num_experts
        self.permuted_weights = permuted_weights
        self.fused_weights = fused_weights
        self.activation = activation
        self.rank = rank
        self.logger = logging.getLogger()

        self.fn = paddlenlp_ops.mixture_of_experts

        self.w1, self.w2, self.w3 = expert_weights
        self.expert_slice = math.ceil(num_experts / slice_max_expert)
        self.expert_chunk = math.ceil(num_experts / self.expert_slice)

    def forward(self, hidden_states, router_weights, routing_table, compute_amax=False):

        common_inputs = (hidden_states, routing_table, router_weights)
        final_hidden_states = paddle.zeros_like(hidden_states)
        amax_per_expert = (
            paddle.zeros(self.num_experts, dtype="float32") if compute_amax else None
        )

        for idx in range(self.expert_slice):
            experts_min = self.expert_chunk * idx
            experts_max = min(experts_min + self.expert_chunk, self.num_experts)

            if self.fused_weights:
                w12 = [
                    paddle.concat((w1, w2), axis=0 if self.permuted_weights else 1)
                    for w1, w2 in zip(self.w1, self.w2)
                ]
                slice_weights = (
                    w12[experts_min:experts_max],
                    self.w3[experts_min:experts_max],
                )
            else:
                slice_weights = (
                    self.w1[experts_min:experts_max] + self.w2[experts_min:experts_max],
                    self.w3[experts_min:experts_max],
                )

            common_params = (
                self.permuted_weights,
                self.activation,
                experts_min,
                experts_max - 1,
            )

            if compute_amax:
                slice_result, slice_amax = self.fn(
                    *common_inputs, *slice_weights, *common_params, True
                )
                amax_per_expert[experts_min:experts_max] = slice_amax
            else:
                slice_result, _ = self.fn(
                    *common_inputs, *slice_weights, *common_params, False
                )
            final_hidden_states += slice_result

        return final_hidden_states, amax_per_expert


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_dim", HIDDEN_DIMS)
@pytest.mark.parametrize("ffn_dim", FFN_DIMS)
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
    num_experts,
    slice_max_expert,
    fused_weights,
    activation,
    permuted_weights,
    compute_amax,
    dtype,
):
    logger = setup_logging(rank=0)

    # Set HPU device
    device = "intel_hpu"
    hpu_available = setup_device(device, rank=0, logger=logger)

    if not hpu_available or not hasattr(paddlenlp_ops, "mixture_of_experts"):
        logger.error("HPU MoE operation not available", extra={"rank": 0})
        pytest.skip("HPU MoE operation not available")

    logger.info(
        f"Test: num_tokens={num_tokens}, num_experts={num_experts}, slice_max_expert={slice_max_expert}, "
        f"fused_weights={fused_weights}, permuted_weights={permuted_weights}",
        extra={"rank": 0},
    )

    numpy_data, paddle_data = generate_moe_params_np(
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        ffn_dim=ffn_dim,
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
    fused_moe = FusedMoEOp(
        num_experts=num_experts,
        expert_weights=expert_weights_pd,
        activation=activation,
        rank=0,
        permuted_weights=permuted_weights,
        fused_weights=fused_weights,
        slice_max_expert=slice_max_expert,
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

    logger = logging.getLogger()
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
                f"Comparing amax: indices={np.where(mask)[0].tolist()}, "
                f"fused_moe={fused_op_vals.tolist()}, "
                f"cpu={ref_vals.tolist()}",
                extra={"rank": 0},
            )
            np.testing.assert_allclose(fused_op_vals, ref_vals, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
