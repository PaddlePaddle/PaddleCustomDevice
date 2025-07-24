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

import paddle
import numpy as np
import unittest


import paddle.nn.functional as F


def attention_naive_with_mask(q, k, v, attn_bias):
    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(qt, kt.transpose([0, 1, 3, 2]))
    s = paddle.scale(s, scale)
    p = paddle.nn.functional.softmax(s + attn_bias)
    o = paddle.matmul(p, vt)
    return paddle.transpose(o, [0, 2, 1, 3])


class TestBF16ScaledDotProductAttentionWithMask(unittest.TestCase):
    def setUp(self):
        self.shape = (2, 128, 8, 32)  # (batch, seq_len, num_heads, head_dim)
        self.dtype = "bfloat16"
        self.place = paddle.CustomPlace("metax_gpu", 0)
        self.rtol = 1e-2
        self.atol = 1e-2

    def _create_inputs(self):
        query = paddle.to_tensor(
            np.random.randn(*self.shape).astype(np.float16),
            dtype=self.dtype,
            stop_gradient=False,
        )
        key = paddle.to_tensor(
            np.random.randn(*self.shape).astype(np.float16),
            dtype=self.dtype,
            stop_gradient=False,
        )
        value = paddle.to_tensor(
            np.random.randn(*self.shape).astype(np.float16),
            dtype=self.dtype,
            stop_gradient=False,
        )

        mask_shape = (self.shape[0], 1, self.shape[1], self.shape[1])
        attention_mask = paddle.to_tensor(
            np.random.randn(*mask_shape).astype(np.float16),
            dtype=self.dtype,
            stop_gradient=False,
        )

        out_grad = paddle.to_tensor(
            np.random.randn(*self.shape).astype(np.float16), dtype=self.dtype
        )

        return query, key, value, attention_mask, out_grad

    def test_dygraph_forward(self):
        paddle.disable_static(self.place)

        q, k, v, attn_mask, _ = self._create_inputs()

        ref_out = attention_naive_with_mask(q, k, v, attn_mask)

        sdp_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False
        )

        np.testing.assert_allclose(
            sdp_out.numpy(),
            ref_out.numpy(),
            rtol=self.rtol,
            atol=self.atol,
            err_msg="BF16 SDP with mask forward mismatch",
        )

    def test_dygraph_backward(self):
        paddle.disable_static(self.place)

        q, k, v, attn_mask, out_grad = self._create_inputs()

        q_ref = q.detach().clone()
        k_ref = k.detach().clone()
        v_ref = v.detach().clone()
        attn_mask_ref = attn_mask.detach().clone()

        q_ref.stop_gradient = False
        k_ref.stop_gradient = False
        v_ref.stop_gradient = False
        attn_mask_ref.stop_gradient = False

        ref_out = attention_naive_with_mask(q_ref, k_ref, v_ref, attn_mask_ref)
        ref_out.backward(out_grad)

        sdp_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False
        )
        sdp_out.backward(out_grad)

        self.assertIsNotNone(q_ref.grad, "q_ref.grad should not be None")
        self.assertIsNotNone(k_ref.grad, "k_ref.grad should not be None")
        self.assertIsNotNone(v_ref.grad, "v_ref.grad should not be None")
        self.assertIsNotNone(
            attn_mask_ref.grad, "attn_mask_ref.grad should not be None"
        )

        np.testing.assert_allclose(
            q.grad.numpy(),
            q_ref.grad.numpy(),
            rtol=self.rtol,
            atol=self.atol,
            err_msg="BF16 SDP with mask Q gradient mismatch",
        )

        np.testing.assert_allclose(
            k.grad.numpy(),
            k_ref.grad.numpy(),
            rtol=self.rtol,
            atol=self.atol,
            err_msg="BF16 SDP with mask K gradient mismatch",
        )

        np.testing.assert_allclose(
            v.grad.numpy(),
            v_ref.grad.numpy(),
            rtol=self.rtol,
            atol=self.atol,
            err_msg="BF16 SDP with mask V gradient mismatch",
        )


if __name__ == "__main__":
    unittest.main()
