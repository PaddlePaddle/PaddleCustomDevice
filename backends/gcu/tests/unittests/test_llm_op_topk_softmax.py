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
import paddle.nn.functional as F
import numpy as np
import unittest
from ddt import ddt, data, unpack
from api_base import TestAPIBase

from paddle_custom_device.gcu import ops as gcu_ops


# The table retains its original format for better comparison of parameter settings.
# fmt: off
TOPK_CASE = [
    {"input_dtype": np.float32, "num_tokens": 10, "num_experts": 20, "topk": 1},
    {"input_dtype": np.float32, "num_tokens": 3000, "num_experts": 128, "topk": 3},
    # {"input_dtype": np.float32, "num_tokens": 32000, "num_experts": 512, "topk": 5},

    # {"input_dtype": np.float16, "num_tokens": 10, "num_experts": 20, "topk": 1},
    # {"input_dtype": np.float16, "num_tokens": 3000, "num_experts": 128, "topk": 3},
    # {"input_dtype": np.float16, "num_tokens": 32000, "num_experts": 512, "topk": 5},

    # {"input_dtype": np.float64, "num_tokens": 10, "num_experts": 20, "topk": 1},
    # {"input_dtype": np.float64, "num_tokens": 3000, "num_experts": 128, "topk": 3},
    # {"input_dtype": np.float64, "num_tokens": 32000, "num_experts": 512, "topk": 5},
]
# fmt: on

NATIVE_IMPL_DEV = "cpu"


@ddt
class TestTopkSoftmax(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.input_dtype = np.float32
        self.num_tokens = 10
        self.topk = 5
        self.num_experts = 20

    def prepare_data(self):
        self.gating_output = self.generate_data(
            [self.num_tokens, self.num_experts], self.input_dtype
        )

        self.topk_weights = np.zeros(
            shape=[
                self.num_tokens,
                self.topk,
            ],
            dtype=self.input_dtype,
        )

        self.topk_indices = np.zeros(
            shape=[
                self.num_tokens,
                self.topk,
            ],
            dtype="int32",
        )

        self.token_expert_indices = np.zeros(
            shape=[
                self.num_tokens,
                self.topk,
            ],
            dtype="int32",
        )

    def forward(self):
        topk_weights = paddle.to_tensor(self.topk_weights, dtype=self.input_dtype)
        topk_indices = paddle.to_tensor(self.topk_indices, dtype="int32")
        token_expert_indices = paddle.to_tensor(
            self.token_expert_indices, dtype="int32"
        )
        gating_output = paddle.to_tensor(self.gating_output, dtype=self.input_dtype)

        gcu_ops.topk_softmax(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            norm_topk_prob=False,
        )

        return [topk_weights, topk_indices]

    def topk_softmax_impl(self, dtype):
        paddle.set_device(NATIVE_IMPL_DEV)
        logits = paddle.to_tensor(self.gating_output, dtype=dtype)

        weights = F.softmax(logits, axis=-1)
        routing_weights, selected_experts = paddle.topk(weights, self.topk, axis=-1)
        return routing_weights, selected_experts

    def expect_output(self):
        if NATIVE_IMPL_DEV == "cpu" and self.input_dtype == np.float16:
            out = self.topk_softmax_impl(np.float32)
        else:
            out = self.topk_softmax_impl(self.input_dtype)
        return out

    @data(*TOPK_CASE)
    @unpack
    def test_check_output(self, input_dtype, num_tokens, topk, num_experts):
        self.input_dtype = input_dtype
        self.num_tokens = num_tokens
        self.topk = topk
        self.num_experts = num_experts
        self.check_output_gcu_with_customized(self.forward, self.expect_output)


if __name__ == "__main__":
    unittest.main()
