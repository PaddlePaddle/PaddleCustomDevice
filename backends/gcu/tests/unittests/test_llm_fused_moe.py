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

""" test for moe ops """
import unittest
import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.incubate.nn.functional import swiglu
from paddle_custom_device.gcu import ops as gcu_ops

# Set random seeds for reproducibility
paddle.seed(42)
np.random.seed(42)


class Expert(nn.Layer):
    """A single expert layer using SwiGLU activation."""

    def __init__(self, d_model, d_feedforward):
        super().__init__()
        self.fc1 = nn.Linear(
            d_model, d_feedforward * 2, bias_attr=False
        )  # *2 for SwiGLU
        self.fc2 = nn.Linear(d_feedforward, d_model, bias_attr=False)

    def forward(self, x):
        """forward"""
        x = self.fc1(x)
        x = swiglu(x)
        return self.fc2(x)


class TestFusedMoeConsistency(unittest.TestCase):
    """Test case for verifying consistency between baseline and fused MoE implementations."""

    @classmethod
    def setUpClass(cls):
        """Class-level setup that runs once before all tests."""
        cls.set_config()
        paddle.set_default_dtype(cls.dtype)

    @classmethod
    def set_config(cls):
        """Set the configuration parameters for the test."""
        cls.dtype = "bfloat16"
        cls.batch_size = 8
        cls.seq_len = 128
        cls.num_experts = 5
        cls.d_model = 8192
        cls.d_feedforward = 128
        cls.top_k = 4
        cls.rtol = 1e-2
        cls.atol = 1e-2

    def setUp(self):
        """Test-level setup that runs before each test."""
        self.init_experts()
        self.prepare_data()

    def init_experts(self):
        """Initialize expert layers and gate weights."""
        self.experts = nn.LayerList(
            [Expert(self.d_model, self.d_feedforward) for _ in range(self.num_experts)]
        )

        # Initialize gate weights
        self.gate = nn.Linear(self.d_model, self.num_experts)
        self.gate_weight = self.gate.weight.cast("float32")

    def prepare_data(self):
        """Prepare input data and expert parameters."""
        # Input tensor
        self.x = paddle.randn(
            [self.batch_size, self.seq_len, self.d_model], dtype=self.dtype
        )

        # Stack expert parameters for fused operations
        self.w0 = paddle.stack([e.fc1.weight for e in self.experts]).astype(self.dtype)
        # self.b0 = paddle.stack([e.fc1.bias for e in self.experts]
        #               ).reshape([self.num_experts, 1, -1]).astype(self.dtype)
        self.w1 = paddle.stack([e.fc2.weight for e in self.experts]).astype(self.dtype)
        # self.b1 = paddle.stack([e.fc2.bias for e in self.experts]
        #               ).reshape([self.num_experts, 1, -1]).astype(self.dtype)

        # shape (E, K, N) -> (E, N, K)
        #     E is the number of experts
        #     K is the input feature dimension
        #     N is the output feature dimension
        self.w0_trans = paddle.transpose(self.w0, [0, 2, 1])
        self.w1_trans = paddle.transpose(self.w1, [0, 2, 1])

    def baseline_forward(self, hidden_states):
        """Baseline implementation processing experts sequentially."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape([-1, hidden_dim])

        # Routing computation
        logits = paddle.matmul(hidden_states.cast("float32"), self.gate_weight)
        weights = F.softmax(logits, axis=-1)
        routing_weights, selected_experts = paddle.topk(weights, self.top_k, axis=-1)
        # Initialize output
        final_hidden_states = paddle.zeros_like(hidden_states)
        expert_mask = paddle.transpose(
            F.one_hot(selected_experts, num_classes=self.num_experts), [2, 1, 0]
        )

        # Process each expert
        for expert_id in range(self.num_experts):
            idx, top_x = paddle.where(expert_mask[expert_id])
            if top_x.size == 0:  # Skip if no tokens for this expert
                continue

            current_state = paddle.index_select(hidden_states, top_x, axis=0)
            expert_out = self.experts[expert_id](current_state)
            current_hidden_states = expert_out * routing_weights[top_x, idx].reshape(
                [-1, 1]
            )
            paddle.index_add_(
                x=final_hidden_states,
                index=top_x.squeeze(),
                axis=0,
                value=current_hidden_states.to(hidden_states.dtype),
            )

        return final_hidden_states.reshape([batch_size, seq_len, hidden_dim])

    def fused_moe_forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Fused MoE.
        """
        batch_size, seq_len, hidden_size = x.shape
        top_k = self.top_k
        moe_intermediate_size = self.d_feedforward
        num_experts = self.num_experts

        x = x.reshape([-1, hidden_size])
        token_num = x.shape[0]
        gate_out = paddle.matmul(x.cast("float32"), self.gate_weight)

        topk_weights = paddle.empty([token_num, top_k], dtype=gate_out.dtype)
        topk_indices = paddle.empty([token_num, top_k], dtype="int32")
        token_expert_indices = paddle.empty(
            [token_num, top_k],
            dtype="int32",
        )
        gcu_ops.topk_softmax(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gate_out,
            norm_topk_prob=False,
        )

        config = {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
        }

        block_size = config["BLOCK_SIZE_M"]
        # max_num_tokens_padded = topk_indices.numel() + num_experts * (block_size - 1)
        max_num_tokens_padded = np.prod(topk_indices.shape) + num_experts * (
            block_size - 1
        )
        max_num_m_blocks = max_num_tokens_padded // block_size
        sorted_token_ids = paddle.empty([max_num_tokens_padded], dtype="int32")
        expert_ids_np = np.zeros(shape=[max_num_m_blocks], dtype=np.int32)
        expert_ids = paddle.to_tensor(expert_ids_np, dtype="int32")
        num_tokens_post_pad = paddle.empty([1], dtype="int32")

        (
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
        ) = gcu_ops.moe_align_block_size(
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            topk_indices,
            num_experts,
            block_size,
        )

        intermediate_cache1 = paddle.empty(
            [token_num, top_k, moe_intermediate_size * 2],
            dtype=x.dtype,
        )

        # w0 = paddle.transpose(self.w0, [0, 2, 1])

        gcu_ops.invoke_fused_moe_kernel(
            x,  # input
            self.w0_trans,  # weight
            intermediate_cache1,  # output
            None,  # A_scale
            None,  # B_scale
            None,  # B_zp
            topk_weights,
            topk_indices,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            False,  # mul_routed_weight
            top_k,
            config,
            False,  # use_int4_w4a16
            None,  # block_shape
        )

        intermediate_cache2 = paddle.empty(
            (token_num, top_k, moe_intermediate_size),
            dtype=x.dtype,
        )

        intermediate_cache2 = paddle.incubate.nn.functional.swiglu(intermediate_cache1)

        intermediate_cache2 = intermediate_cache2.reshape([-1, moe_intermediate_size])

        intermediate_cache3 = paddle.empty(
            (token_num, top_k, hidden_size),
            dtype=x.dtype,
        )

        gcu_ops.invoke_fused_moe_kernel(
            intermediate_cache2,  # input
            self.w1_trans,  # paddle.transpose(self.w1, [0, 2, 1]),  # weight
            intermediate_cache3,  # output
            None,  # A_scale
            None,  # B_scale
            None,  # B_zp
            topk_weights,
            topk_indices,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            True,  # mul_routed_weight
            1,
            config,
            None,  # use_int4_w4a16
            None,  # block_shape
        )

        intermediate_cache3.reshape_([token_num, top_k, hidden_size])
        out = intermediate_cache3.sum(axis=1)
        return out.reshape([batch_size, seq_len, hidden_size])

    def test_consistency(self):
        """Test consistency between all three implementations."""
        # Compute outputs
        base_out = self.baseline_forward(self.x)
        fused_out = self.fused_moe_forward(self.x)

        # Convert to float32 for comparison
        base_out = base_out.cast("float32").numpy()
        fused_out = fused_out.cast("float32").numpy()
        # Compare baseline vs fused
        np.testing.assert_allclose(
            base_out,
            fused_out,
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Baseline and fused outputs differ",
        )


if __name__ == "__main__":
    unittest.main()
