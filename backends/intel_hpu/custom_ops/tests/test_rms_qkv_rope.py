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

import numpy as np

import paddle
from paddlenlp.transformers.llama.modeling import LlamaRotaryEmbedding
import paddlenlp_ops

paddle.device.set_device("intel_hpu:5")

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
    def __init__(self, test_GQA=False, with_bias=False):
        self.head_dim = 128
        self.num_head = 32
        self.kv_num_heads = self.num_head
        if test_GQA is True:
            self.kv_num_heads = 8
        self.hidden_size = self.head_dim * self.num_head
        self.kv_hidden_size = self.head_dim * self.kv_num_heads

        self.epsilon = 1e-06

        self.use_neox = True
        self.position_offset = 0
        self.rope_theta = 10000

        self.with_bias = with_bias

    def init_block_prefill_params(self):
        self.test_name = "TestFusedRmsQkvRopeBlockPrefill"
        if self.kv_num_heads != self.num_head:
            self.test_name += "-GQA"
        else:
            self.test_name += "-MHA"
        if self.with_bias:
            self.test_name += "-bias"
        self.batch_size = 1
        self.seq_len = 34
        position_id = paddle.arange(self.seq_len, dtype=paddle.int64).to(paddle.int64)
        self.position_ids = paddle.expand(
            position_id, shape=[self.batch_size, self.seq_len]
        )

    def init_decode_params(self):
        self.test_name = "TestFusedRmsQkvRopeDecode"
        if self.kv_num_heads != self.num_head:
            self.test_name += "-GQA"
        else:
            self.test_name += "-MHA"
        if self.with_bias:
            self.test_name += "-bias"
        self.batch_size = 16
        self.seq_len = 1
        position_id = paddle.to_tensor([80])
        self.position_ids = paddle.expand(
            position_id, shape=[self.batch_size, self.seq_len]
        )

    def init_left_padding_params(self):
        self.test_name = "TestFusedRmsQkvRopePadding"
        if self.kv_num_heads != self.num_head:
            self.test_name += "-GQA"
        else:
            self.test_name += "-MHA"
        if self.with_bias:
            self.test_name += "-bias"
        self.batch_size = 4
        self.seq_len = 64
        self.position_ids = paddle.randint(
            1, self.seq_len - 1, [self.batch_size, self.seq_len], dtype=paddle.int64
        )

    def create_tensors(self):
        device = paddle.get_device()
        self.input_ids = paddle.zeros(
            [self.batch_size, self.seq_len], dtype=paddle.bfloat16
        )
        np_src = np.random.rand(self.batch_size, self.seq_len, self.hidden_size).astype(
            "float32"
        )
        self.src = (
            paddle.to_tensor(np_src, place=paddle.CPUPlace())
            .to(paddle.bfloat16)
            .to(device)
        )

        np_residual = np.random.rand(
            self.batch_size, self.seq_len, self.hidden_size
        ).astype("float32")
        self.residual = (
            paddle.to_tensor(np_residual, place=paddle.CPUPlace())
            .to(paddle.bfloat16)
            .to(device)
        )

        self.src_residual = self.src + self.residual

        np_ln_scales = np.random.rand(self.hidden_size).astype("float32")
        self.ln_scales = (
            paddle.to_tensor(np_ln_scales, place=paddle.CPUPlace())
            .to(paddle.bfloat16)
            .to(device)
        )

        np_qkv_weights = np.random.rand(
            self.hidden_size + 2 * self.kv_hidden_size, self.hidden_size
        ).astype("float32")
        self.qkv_weights = (
            paddle.to_tensor(np_qkv_weights, place=paddle.CPUPlace())
            .to(paddle.bfloat16)
            .to(device)
        )

        if self.with_bias:
            np_qkv_biases = np.random.rand(
                self.hidden_size + 2 * self.kv_hidden_size
            ).astype("float32")
            self.qkv_biases = (
                paddle.to_tensor(np_qkv_biases, place=paddle.CPUPlace())
                .to(paddle.bfloat16)
                .to(device)
            )
        else:
            self.qkv_biases = None
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
            self.src_residual, self.ln_scales, None, self.epsilon, 2
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

    def ref_bias_result(self):
        hidden_states = paddle.incubate.nn.functional.fused_rms_norm(
            self.src_residual, self.ln_scales, None, self.epsilon, 2
        )[0]

        qkv_out = paddle.matmul(hidden_states, self.qkv_weights, False, True)
        if self.qkv_biases is not None:
            qkv_out = paddle.add(qkv_out, self.qkv_biases)

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
        (
            query_states_bias_ref,
            key_states_bias_ref,
            value_states_bias_ref,
        ) = self.ref_bias_result()

        # =============== fused_rms_qkv_rope_v2 ==============
        (
            query_states_opv2,
            key_states_opv2,
            value_states_opv2,
        ) = paddlenlp_ops.fused_rms_qkv_rope_v2(
            self.src_residual,
            self.ln_scales,
            self.qkv_weights,
            self.new_rope,
            self.epsilon,
            self.head_dim,
            self.num_head,
        )

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
            self.src_residual,
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
            self.qkv_biases,
            self.new_rope.transpose([0, 1, 3, 2, 4]),
            self.residual,
            self.epsilon,
            self.head_dim,
            self.num_head,
        )

        (
            query_states_bias_ref,
            key_states_bias_ref,
            value_states_bias_ref,
        ) = self.ref_bias_result()
        query_states_bias_ref = query_states_bias_ref.transpose([0, 2, 1, 3])
        key_value_states_bias_ref = paddle.stack(
            [key_states_bias_ref, value_states_bias_ref], axis=0
        )
        key_value_states_bias_ref = key_value_states_bias_ref.transpose([0, 1, 3, 2, 4])
        assert paddle.allclose(
            query_states_bias_ref.to("cpu").to("float32"),
            query_states_opt.to("cpu").to("float32"),
            rtol=1e-2,
        ), f"Test failed for {self.test_name} fused_rms_qkv_rope_t query_states"
        assert paddle.allclose(
            key_value_states_bias_ref.to("cpu").to("float32"),
            key_value_states_opt.to("cpu").to("float32"),
            rtol=1e-2,
        ), f"Test failed for {self.test_name} fused_rms_qkv_rope_t key_value_states"
        assert paddle.allclose(
            self.src_residual.to("cpu").to("float32"),
            self.residual.to("cpu").to("float32"),
            rtol=1e-2,
        ), f"Test failed for {self.test_name} fused_rms_qkv_rope_t residual"

        # ===============summary==============
        print(f"Test Pass for {self.test_name} testcase")


class test_case_padding(TestFusedRmsQkvRope):
    def __init__(self, test_GQA=False, with_bias=False):
        super().__init__(test_GQA, with_bias)
        self.init_left_padding_params()
        self.create_tensors()


class test_case_block_prefill(TestFusedRmsQkvRope):
    def __init__(self, test_GQA=False, with_bias=False):
        super().__init__(test_GQA, with_bias)
        self.init_block_prefill_params()
        self.create_tensors()


class test_case_decode(TestFusedRmsQkvRope):
    def __init__(self, test_GQA=False, with_bias=False):
        super().__init__(test_GQA, with_bias)
        self.init_decode_params()
        self.create_tensors()


if __name__ == "__main__":
    test_1 = test_case_padding(test_GQA=True)
    test_1.run_test()

    test_2 = test_case_padding()
    test_2.run_test()

    test_3 = test_case_block_prefill(test_GQA=True)
    test_3.run_test()

    test_4 = test_case_block_prefill()
    test_4.run_test()

    test_5 = test_case_decode(test_GQA=True)
    test_5.run_test()

    test_6 = test_case_decode()
    test_6.run_test()

    test_7 = test_case_decode(test_GQA=True, with_bias=True)
    test_7.run_test()

    test_8 = test_case_decode(with_bias=True)
    test_8.run_test()
