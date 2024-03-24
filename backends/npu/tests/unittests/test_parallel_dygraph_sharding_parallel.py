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

import os
import unittest

from . import test_parallel_dygraph_dataparallel

TestMultipleAccelerators = test_parallel_dygraph_dataparallel.TestMultipleAccelerators


linked_test_file_path_prefix = "../../../../Paddle/python/tests/"
# linked_test_file_path_prefix=""


class TestHybridParallel(TestMultipleAccelerators):
    # check sharding logic as well as the accuracy with single mode
    # def test_hybrid_parallel_sharding_logic(self):
    #     # test shard v2
    #     os.environ["FLAGS_shard_use_reduce"] = "1"
    #     os.environ["FLAGS_shard_norm_align_dp"] = "0"
    #     os.environ["FLAGS_shard_split_param"] = "1"
    #     self.run_mnist_2accelerators(
    #         linked_test_file_path_prefix + "hybrid_parallel_sharding_model.py",
    #         accelerator_type="npu",
    #     )
    #     # test shard grad reduce
    #     os.environ["FLAGS_shard_use_reduce"] = "1"
    #     os.environ["FLAGS_shard_norm_align_dp"] = "0"
    #     os.environ["FLAGS_shard_split_param"] = "0"
    #     self.run_mnist_2accelerators(
    #         linked_test_file_path_prefix + "hybrid_parallel_sharding_model.py",
    #         accelerator_type="npu",
    #     )
    #     # test shard grad allreduce
    #     os.environ["FLAGS_shard_use_reduce"] = "0"
    #     os.environ["FLAGS_shard_norm_align_dp"] = "1"
    #     os.environ["FLAGS_shard_split_param"] = "0"
    #     self.run_mnist_2accelerators(
    #         linked_test_file_path_prefix + "hybrid_parallel_sharding_model.py",
    #         accelerator_type="npu",
    #     )

    def test_hybrid_parallel_sharding_tensor_fusion(self):
        os.environ["FLAGS_shard_split_param"] = "0"
        self.run_mnist_2accelerators(
            linked_test_file_path_prefix
            + "hybrid_parallel_sharding_model_with_fusion.py",
            accelerator_type="npu",
        )

    # def test_hybrid_parallel_sharding_tensor_fusion_amp(self):
    #     os.environ["FLAGS_shard_split_param"] = "0"
    #     self.run_mnist_2accelerators(
    #         linked_test_file_path_prefix
    #         + "hybrid_parallel_sharding_model_with_fusion_amp.py",
    #         accelerator_type="npu",
    #     )

    # def test_hybrid_parallel_sharding_state_dict(self):
    #     os.environ["FLAGS_shard_split_param"] = "0"
    #     self.run_mnist_2accelerators(
    #         linked_test_file_path_prefix + "hybrid_parallel_sharding_state_dict.py",
    #         accelerator_type="npu",
    #     )

    # def test_group_param_tensor_fusion(self):
    #     self.run_mnist_2accelerators(
    #         linked_test_file_path_prefix
    #         + "hybrid_parallel_tensor_fusion_with_group.py",
    #         accelerator_type="npu",
    #     )


if __name__ == "__main__":
    unittest.main()
