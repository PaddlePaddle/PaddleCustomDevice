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
from paddlenlp.transformers.llama.modeling import LlamaRotaryEmbedding
import paddlenlp_ops

paddle.device.set_device("intel_hpu:1")

paddle.seed(102)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    if position_ids is None:
        # Note: Only for LlamaForCausalLMPipe model pretraining
        cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
        sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
    else:
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TestFusedRmsQkvRope:
    def __init__(self):
        self.head_dim = 128
        self.num_head = 32
        self.kv_num_heads = 32
        self.hidden_size = 4096

        self.epsilon = 1e-06

        self.use_neox = True
        self.position_offset = 0
        self.rope_theta = 10000

    def init_block_prefill_params(self):
        self.test_name = "TestFusedRmsQkvRopeBlockPrefill"
        self.batch_size = 1
        self.seq_len = 34
        position_id = paddle.arange(self.seq_len, dtype=paddle.int64).to(paddle.int64)
        self.position_ids = paddle.expand(
            position_id, shape=[self.batch_size, self.seq_len]
        )

    def init_decode_params(self):
        self.test_name = "TestFusedRmsQkvRopeDecode"
        self.batch_size = 16
        self.seq_len = 1
        position_id = paddle.to_tensor([80])
        self.position_ids = paddle.expand(
            position_id, shape=[self.batch_size, self.seq_len]
        )

    def init_left_padding_params(self):
        self.test_name = "TestFusedRmsQkvRopePadding"
        self.batch_size = 4
        self.seq_len = 64
        self.position_ids = paddle.randint(
            1, self.seq_len - 1, [self.batch_size, self.seq_len], dtype=paddle.int64
        )

    def create_tensors(self):
        self.input_ids = paddle.zeros(
            [self.batch_size, self.seq_len], dtype=paddle.bfloat16
        )
        self.src = paddle.rand(
            [self.batch_size, self.seq_len, self.hidden_size], dtype=paddle.bfloat16
        )
        self.ln_scales = paddle.rand([self.hidden_size], dtype=paddle.bfloat16)
        self.qkv_weights = paddle.rand(
            [self.hidden_size * 3, self.hidden_size], dtype=paddle.float32
        ).to(paddle.bfloat16)
        self.head_dim_shape_tensor = paddle.ones(self.head_dim, dtype="int8")

        self.new_rope = paddlenlp_ops.fused_get_rotary_embedding(
            self.input_ids,
            self.position_ids,
            self.head_dim_shape_tensor,
            self.position_offset,
            self.rope_theta,
            self.use_neox,
        ).to(paddle.bfloat16)

    def ref_result(self):
        hidden_states = paddle.incubate.nn.functional.fused_rms_norm(
            self.src, self.ln_scales, None, self.epsilon, 2
        )[0]

        qkv_out = paddle.matmul(hidden_states, self.qkv_weights, False, True)

        fused_hidden_size = qkv_out.shape[2]
        kv_num_heads = (
            (fused_hidden_size - self.num_head * self.head_dim) // self.head_dim // 2
        )
        num_groups = self.num_head // kv_num_heads
        target_shape = [0, 0, (num_groups + 2) * kv_num_heads, self.head_dim]

        qkv_out = paddle.reshape_(qkv_out, target_shape)

        qkv_out = paddle.transpose(qkv_out, [0, 2, 1, 3])

        query_states, key_states, value_states = paddle.split(
            qkv_out,
            num_or_sections=[self.num_head, kv_num_heads, kv_num_heads],
            axis=1,
        )

        rotary_emb = LlamaRotaryEmbedding(self.head_dim)
        cos, sin = rotary_emb(self.src)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, self.position_ids
        )

        return query_states, key_states, value_states

    def run_test(self):
        # =============== Reference Result ==============
        query_states_ref, key_states_ref, value_states_ref = self.ref_result()

        # =============== fused_rms_qkv_rope_v2 ==============
        (
            query_states_opv2,
            key_states_opv2,
            value_states_opv2,
        ) = paddlenlp_ops.fused_rms_qkv_rope_v2(
            self.src,
            self.ln_scales,
            self.qkv_weights,
            self.new_rope,
            self.epsilon,
            self.head_dim,
            self.num_head,
        )
        # print((query_states_ref-query_states_opv2).abs().max())
        # print(((query_states_ref-query_states_opv2) != 0).sum())
        # print(query_states_ref.shape)
        assert paddle.allclose(
            query_states_ref.to("cpu").to("float32"),
            query_states_opv2.to("cpu").to("float32"),
            rtol=1e-2,
        ), f"Test failed for {self.test_name} fused_rms_qkv_rope_v2 query_states"

        assert paddle.allclose(
            key_states_ref.to("cpu").to("float32"),
            key_states_opv2.to("cpu").to("float32"),
            rtol=1e-2,
        ), f"Test failed for {self.test_name} fused_rms_qkv_rope_v2 key_states"

        assert (
            (value_states_ref == value_states_opv2).all().item()
        ), f"Test failed for {self.test_name} fused_rms_qkv_rope_v2 value_states"

        # =============== key_value_states_opv3 ==============
        query_states_opv3, key_value_states_opv3 = paddlenlp_ops.fused_rms_qkv_rope_v3(
            self.src,
            self.ln_scales,
            self.qkv_weights,
            self.new_rope,
            self.epsilon,
            self.head_dim,
            self.num_head,
        )

        key_value_states_opv2 = paddle.stack(
            [key_states_opv2, value_states_opv2], axis=0
        )
        assert (
            (query_states_opv2 == query_states_opv3).all().item()
        ), f"Test failed for {self.test_name} fused_rms_qkv_rope_v3 query_states"
        assert (
            (key_value_states_opv2 == key_value_states_opv3).all().item()
        ), f"Test failed for {self.test_name} fused_rms_qkv_rope_v3 key_value_states"

        # =============== fused_rms_qkv_rope_t ==============
        query_states_opt, key_value_states_opt = paddlenlp_ops.fused_rms_qkv_rope_t(
            self.src,
            self.ln_scales,
            self.qkv_weights,
            self.new_rope.transpose([0, 1, 3, 2, 4]),
            self.epsilon,
            self.head_dim,
            self.num_head,
        )

        query_states_opv2_t = query_states_opv2.transpose([0, 2, 1, 3])
        key_value_states_opv2_t = key_value_states_opv2.transpose([0, 1, 3, 2, 4])
        assert (
            (query_states_opv2_t == query_states_opt).all().item()
        ), f"Test failed for {self.test_name} fused_rms_qkv_rope_t query_states"
        assert (
            (key_value_states_opv2_t == key_value_states_opt).all().item()
        ), f"Test failed for {self.test_name} fused_rms_qkv_rope_t key_value_states"

        # ===============summary==============
        print(f"Test Pass for {self.test_name} testcase")


class test_case_padding(TestFusedRmsQkvRope):
    def __init__(self):
        super().__init__()
        self.init_left_padding_params()
        self.create_tensors()


class test_case_block_prefill(TestFusedRmsQkvRope):
    def __init__(self):
        super().__init__()
        self.init_block_prefill_params()
        self.create_tensors()


class test_case_decode(TestFusedRmsQkvRope):
    def __init__(self):
        super().__init__()
        self.init_decode_params()
        self.create_tensors()


if __name__ == "__main__":
    test_1 = test_case_padding()
    test_1.run_test()

    test_2 = test_case_block_prefill()
    test_2.run_test()

    test_3 = test_case_decode()
    test_3.run_test()
