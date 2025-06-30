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

import paddle
import paddlenlp_ops
import os
import unittest
from parameterized import parameterized

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)
paddle.device.set_device(f"intel_hpu:{intel_hpus_module_id}")

paddle.seed(105)


def ref_result(
    query_states,
    key_states,
    value_states,
    attention_mask,
    linear_weights,
    scaling_factor,
):
    bsz, q_len, num_heads, head_dim = query_states.shape
    attn_output = paddle.incubate.nn.functional.fused_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
        0.0,
        attention_mask is None,
        scaling_factor,
        False,
    )
    attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])

    out_linear_out = paddle.matmul(attn_output, linear_weights)

    return out_linear_out


HEAD_DIM = [128]
NUM_HEAD = [32]
BATCH_SIZE = [4, 8]
SEQ_LEN = [1, 25]
KV_SEQ_LEN = [25]
MAX_SEQ_LENGTH = [2048]


class SDPA_PROJ_v2_Test(unittest.TestCase):
    @parameterized.expand(
        [
            (
                head_dim,
                num_head,
                batch_size,
                seq_len,
                kv_seq_len,
                max_seq_length,
            )
            for head_dim in HEAD_DIM
            for num_head in NUM_HEAD
            for batch_size in BATCH_SIZE
            for seq_len in SEQ_LEN
            for kv_seq_len in KV_SEQ_LEN
            for max_seq_length in MAX_SEQ_LENGTH
        ]
    )
    def test_sdpa_proj_v2(
        self,
        head_dim,
        num_head,
        batch_size,
        seq_len,
        kv_seq_len,
        max_seq_length,
    ):

        kv_num_heads = num_head
        hidden_size = num_head * head_dim

        query_states = paddle.rand(
            [batch_size, num_head, seq_len, head_dim], dtype=paddle.float32
        ).to(paddle.bfloat16)
        key_states = paddle.rand(
            [batch_size, kv_num_heads, kv_seq_len, head_dim], dtype=paddle.float32
        ).to(paddle.bfloat16)
        value_states = paddle.rand(
            [batch_size, kv_num_heads, kv_seq_len, head_dim], dtype=paddle.float32
        ).to(paddle.bfloat16)

        attn_mask = paddle.ones(
            [1, 1, max_seq_length, max_seq_length], dtype=paddle.bfloat16
        )
        attn_mask = paddle.tril(attn_mask)
        attn_mask = (1.0 - attn_mask) * -10000.0

        linear_weights = paddle.rand(
            [hidden_size, hidden_size], dtype=paddle.float32
        ).to(paddle.bfloat16)

        attention_mask = attn_mask[..., :seq_len, :kv_seq_len]
        attention_mask = attention_mask.astype(query_states.dtype)

        out_linear_out_op = paddlenlp_ops.fused_sdpa_proj(
            query_states,
            key_states,
            value_states,
            attention_mask,
            linear_weights,
            scaling_factor=head_dim**-0.5,
        )

        out_linear_out_ref = ref_result(
            query_states.transpose([0, 2, 1, 3]),
            key_states.transpose([0, 2, 1, 3]),
            value_states.transpose([0, 2, 1, 3]),
            attention_mask,
            linear_weights,
            scaling_factor=head_dim**-0.5,
        )

        key_value_states = paddle.stack([key_states, value_states], axis=0)

        # Test is causal && mask=None
        # seq_len = kv_seq_len
        attention_mask = None
        out_linear_out_op_v2 = paddlenlp_ops.fused_sdpa_proj_v2(
            query_states,
            key_value_states,
            attention_mask,
            linear_weights,
            scaling_factor=head_dim**-0.5,
            causal=attention_mask is None,
        )

        print((out_linear_out_ref == out_linear_out_op).all())
        print(
            paddle.allclose(
                out_linear_out_ref.to("cpu").to("float32"),
                out_linear_out_op_v2.to("cpu").to("float32"),
                rtol=1e-2,
            )
        )


if __name__ == "__main__":
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(SDPA_PROJ_v2_Test)

    # Create a test runner with the desired verbosity level
    runner = unittest.TextTestRunner(
        verbosity=2
    )  # Set verbosity to 2 for detailed output

    # Run the test suite
    runner.run(suite)
