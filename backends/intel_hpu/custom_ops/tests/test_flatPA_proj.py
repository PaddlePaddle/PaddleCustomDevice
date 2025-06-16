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

from __future__ import print_function

import unittest

import paddle
import paddlenlp_ops
import os

intel_hpus_module_id = os.environ.get("FLAGS_selected_intel_hpus", 0)
# paddle.device.set_device(f"intel_hpu:{intel_hpus_module_id}")

# paddle.seed(102)


def get_similarity(x, y):
    x = x.cpu().to("float32")
    y = y.cpu().to("float32")
    return paddle.nn.functional.cosine_similarity(
        x.flatten(), y.flatten(), axis=0
    ).item()


class Test_Fused_Flatpa_Proj_OP(unittest.TestCase):
    def setUp(self):
        self.set_hpu()

        self.testcase = ("Test_Fused_Flatpa_Proj_OP",)
        self.batch_size = (8,)
        self.q_head = (32,)
        self.kv_head = (32,)
        self.head_dim = (128,)
        self.total_block_num = (40,)
        self.block_size = (64,)
        self.num_of_block = (12,)
        self.out_features = (4096,)

    def set_hpu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("intel_hpu", int(intel_hpus_module_id))
        paddle.seed(102)

    def HPU_Fused_Flatpa_Proj_OP(
        self,
        testcase,
        batch_size=8,
        q_head=32,
        kv_head=32,
        head_dim=128,
        total_block_num=40,
        block_size=64,
        num_of_block=12,
        out_features=4096,
    ):
        hidden_size = q_head * head_dim
        scaling_factor = head_dim**-0.5

        paddle.set_device("cpu")

        query = paddle.rand([batch_size, 1, q_head, head_dim], dtype=paddle.bfloat16)
        query = query.to(paddle.bfloat16)

        block_list = paddle.rand([num_of_block], dtype=paddle.float32) * (
            num_of_block - 1
        )
        block_list = block_list.to(paddle.int32)

        block_groups = paddle.rand([num_of_block], dtype=paddle.float32) * (
            num_of_block - 1
        )
        block_groups = block_groups.to(paddle.int32)

        block_mapping = paddle.rand([num_of_block, batch_size], dtype=paddle.bfloat16)
        block_mapping = block_mapping.to(paddle.bfloat16)

        attn_bias = paddle.rand([num_of_block, block_size], dtype=paddle.bfloat16)
        attn_bias = attn_bias.to(paddle.bfloat16)

        linear_weights = paddle.rand([hidden_size, out_features], dtype=paddle.bfloat16)
        linear_weights = linear_weights.to(paddle.bfloat16)

        key_cache = paddle.rand(
            [total_block_num, block_size, kv_head, head_dim], dtype=paddle.bfloat16
        )
        key_cache = key_cache.to(paddle.bfloat16)

        value_cache = paddle.rand(
            [total_block_num, block_size, kv_head, head_dim], dtype=paddle.bfloat16
        )
        value_cache = value_cache.to(paddle.bfloat16)

        paddle.set_device(f"intel_hpu:{intel_hpus_module_id}")

        scale_one = paddle.to_tensor([1.0])

        hpu = scale_one.place
        query = query.to(hpu)
        block_list = block_list.to(hpu)
        block_groups = block_groups.to(hpu)
        block_mapping = block_mapping.to(hpu)
        attn_bias = attn_bias.to(hpu)
        linear_weights = linear_weights.to(hpu)
        key_cache = key_cache.to(hpu)
        value_cache = value_cache.to(hpu)

        out_linear_ref = paddlenlp_ops.fused_flatpa_proj_ref(
            query,
            key_cache,
            value_cache,
            block_groups,
            block_list,
            block_mapping,
            attn_bias,
            linear_weights,
            scaling_factor=scaling_factor,
        )

        out_linear_out = paddlenlp_ops.fused_flatpa_proj(
            query,
            key_cache,
            value_cache,
            block_groups,
            block_list,
            block_mapping,
            attn_bias,
            linear_weights,
            scaling_factor=scaling_factor,
        )

        similarity = get_similarity(out_linear_ref, out_linear_out)
        # print(f"similarity = {similarity}")
        assert (
            abs(1 - similarity) < 1e-4
        ), "similarity check fails between fused_flatpa_proj and fused_flatpa_proj_ref outputs"

        assert paddle.allclose(
            out_linear_ref.to(paddle.float32).cpu(),
            out_linear_out.to(paddle.float32).cpu(),
            rtol=5e-2,
            atol=5e-2,
        ), "out_linear_out outputs not equal to bf16 outputs"

        print(f"Test Pass for {testcase} testcase")


class Test_Fused_Flatpa_Proj_MHA(Test_Fused_Flatpa_Proj_OP):
    def setUp(self):
        self.testcase = "MHA"

    def test(self):
        self.HPU_Fused_Flatpa_Proj_OP(
            self.testcase,
            kv_head=32,
        )


class Test_Fused_Flatpa_Proj_GQA(Test_Fused_Flatpa_Proj_OP):
    def setUp(self):
        self.testcase = "GQA"

    def test(self):
        self.HPU_Fused_Flatpa_Proj_OP(
            self.testcase,
            kv_head=8,
        )


class Test_Fused_Flatpa_Proj_65B(Test_Fused_Flatpa_Proj_OP):
    def setUp(self):
        self.testcase = "65B"

    def test(self):
        self.HPU_Fused_Flatpa_Proj_OP(
            self.testcase,
            q_head=16,
            kv_head=16,
            out_features=8192,
        )


if __name__ == "__main__":
    unittest.main()
