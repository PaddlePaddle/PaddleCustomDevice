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
from ddt import ddt, data, unpack
from api_base import TestAPIBase

from paddle_custom_device.gcu import ops as gcu_ops

# The table retains its original format for better comparison of parameter settings.
# fmt: off
TOP_P_SAMPLING_CASE = [

    {"batch_size": 2, "vocab_size": 10000, "top_p": 0.6, "dtype": np.float32},
    {"batch_size": 2, "vocab_size": 32000, "top_p": 0.6, "dtype": np.float32},
    {"batch_size": 6, "vocab_size": 32000, "top_p": 0.9, "dtype": np.float32},


]
# fmt: on

NATIVE_IMPL_DEV = "gcu"


def native_top_p_process(probs, top_p):
    sorted_probs = paddle.sort(probs, descending=True)
    sorted_indices = paddle.argsort(probs, descending=True)
    cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)

    # Remove tokens with cumulative probs above the top_p, But keep at
    # least min_tokens_to_keep tokens
    sorted_indices_to_remove = cumulative_probs > top_p

    # Keep the first token
    sorted_indices_to_remove = paddle.cast(sorted_indices_to_remove, dtype="int64")

    sorted_indices_to_remove = paddle.static.setitem(
        sorted_indices_to_remove,
        (slice(None), slice(1, None)),
        sorted_indices_to_remove[:, :-1].clone(),
    )
    sorted_indices_to_remove = paddle.static.setitem(
        sorted_indices_to_remove, (slice(None), 0), 0
    )

    # Scatter sorted tensors to original indexing
    sorted_indices = (
        sorted_indices + paddle.arange(probs.shape[0]).unsqueeze(-1) * probs.shape[-1]
    )
    condition = paddle.scatter(
        sorted_indices_to_remove.flatten(),
        sorted_indices.flatten(),
        sorted_indices_to_remove.flatten(),
    )
    condition = paddle.cast(condition, "bool").reshape(probs.shape)
    probs = paddle.where(condition, paddle.full_like(probs, 0.0), probs)
    next_tokens = paddle.multinomial(probs)
    next_scores = paddle.index_sample(probs, next_tokens)
    return next_scores, next_tokens


@ddt
class TestTopPSampling(TestAPIBase):
    def setUp(self):
        self.init_attrs()

    def init_attrs(self):
        self.batch_size = 2
        self.vocab_size = 10000
        self.top_p = 0.6
        self.dtype = np.float32

    def prepare_data(self):
        self.input_probs = self.generate_data(
            shape=[
                self.batch_size,
                self.vocab_size,
            ],
            dtype=self.dtype,
        )

        self.top_p = np.array(
            [self.top_p],
            dtype=self.dtype,
        )

        # print(f"TESTCASE_DEBUG In prepare_data, self.input_probs\n:{self.input_probs}", flush=True)
        # print(f"TESTCASE_DEBUG In prepare_data, self.top_p\n:{self.top_p}", flush=True)

    def forward(self):
        input_probs = paddle.to_tensor(self.input_probs, dtype=self.dtype)
        top_p = paddle.to_tensor(self.top_p, dtype=self.dtype)

        # next_scores, next_tokens = paddle.tensor.top_p_sampling(input_probs, top_p)
        next_scores, next_tokens = gcu_ops.top_p_sampling(input_probs, top_p, 2)

        # print(f"TESTCASE_DEBUG In prepare_data, next_scores:\n{next_scores}", flush=True)
        # print(f"TESTCASE_DEBUG In prepare_data, next_tokens:\n{next_tokens}", flush=True)

        return [next_scores, next_tokens]

    def ref_top_p_sampling_impl(self, dtype):
        paddle.set_device(NATIVE_IMPL_DEV)
        input_probs = paddle.to_tensor(self.input_probs, dtype=self.dtype)
        top_p = paddle.to_tensor(self.top_p, dtype=self.dtype)

        next_scores, next_tokens = native_top_p_process(input_probs, top_p)

        if dtype != self.dtype:
            return [next_scores.astype(self.dtype), next_tokens.astype(self.dtype)]
        return [next_scores, next_tokens]

    def expect_output(self):
        if NATIVE_IMPL_DEV == "cpu" and self.dtype == np.float16:
            out = self.ref_top_p_sampling_impl(np.float32)
        else:
            out = self.ref_top_p_sampling_impl(self.dtype)
        return out

    @data(*TOP_P_SAMPLING_CASE)
    @unpack
    def test_check_output(
        self,
        batch_size,
        vocab_size,
        top_p,
        dtype,
    ):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.top_p = top_p
        self.dtype = dtype

        rtol = 1e-5
        atol = 1e-5
        if dtype == np.float16:
            rtol = 1e-3
            atol = 1e-3
        self.check_output_gcu_with_customized(
            self.forward, self.expect_output, rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    unittest.main()
