# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import unittest
import paddle
import paddle_sdaa
import time

import time
import numpy as np

SEED = 2023

np.random.seed(SEED)
paddle.seed(SEED)


def rotate_half(x):

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


def apply_rotary_pos_emb(q, k, cos, sin):

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoPE_test(unittest.TestCase):
    def test_rope(self):
        paddle.set_device("sdaa")

        batch_size = 2
        seq_len = 512
        num_heads = 4
        head_dim = 128

        # prepare q, k
        mix_layer = paddle.randn(
            [batch_size, seq_len, num_heads, head_dim * 3], dtype="float32"
        )
        query_states, key_states, _ = paddle.split(
            mix_layer, num_or_sections=3, axis=-1
        )
        query_states_rope = query_states.clone()
        key_states_rope = key_states.clone()
        query_states.stop_gradient = False
        key_states.stop_gradient = False
        query_states_rope.stop_gradient = False
        key_states_rope.stop_gradient = False

        x = paddle.randn([1, seq_len, 1, head_dim], dtype="float32")
        y = paddle.randn([1, seq_len, 1, head_dim], dtype="float32")
        sin = paddle.sin(x)
        cos = paddle.cos(y)

        cos = cos[:, : query_states.shape[1], :, :]
        sin = sin[:, : query_states.shape[1], :, :]

        origin_forward_time = []
        rope_forward_time = []

        for i in range(200):
            origin_t0 = time.time()
            query_states_out_gt, key_states_out_gt = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
            origin_t1 = time.time()
            origin_forward_time.append(origin_t1 - origin_t0)
            sum_gt = query_states_out_gt + key_states_out_gt
            sum_gt.backward()

        for i in range(200):
            rope_t0 = time.time()
            query_states_trans = paddle.transpose(query_states_rope, [1, 0, 2, 3])
            key_states_trans = paddle.transpose(key_states_rope, [1, 0, 2, 3])
            (
                query_states_out_device,
                key_states_out_device,
            ) = paddle_sdaa.ops.fused_rotary_position_embedding(
                query_states_trans, key_states_trans, cos, sin
            )
            query_states_out_device = paddle.transpose(
                query_states_out_device, [1, 0, 2, 3]
            )
            key_states_out_device = paddle.transpose(
                key_states_out_device, [1, 0, 2, 3]
            )
            rope_t1 = time.time()
            rope_forward_time.append(rope_t1 - rope_t0)
            sum_device = query_states_out_device + key_states_out_device
            sum_device.backward()

        origin_forward_time_arvg = np.mean(origin_forward_time) * 1000
        rope_forward_time_arvg = np.mean(rope_forward_time) * 1000
        print(
            origin_forward_time_arvg,
            rope_forward_time_arvg,
            origin_forward_time_arvg / rope_forward_time_arvg,
        )

        np.testing.assert_allclose(
            query_states_out_gt.numpy(),
            query_states_out_device.numpy(),
            rtol=1e-5,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            key_states_out_gt.numpy(),
            key_states_out_device.numpy(),
            rtol=1e-5,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            query_states.grad.numpy(),
            query_states_rope.grad.numpy(),
            rtol=1e-5,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            key_states.grad.numpy(), key_states_rope.grad.numpy(), rtol=1e-5, atol=1e-8
        )


if __name__ == "__main__":
    unittest.main()
