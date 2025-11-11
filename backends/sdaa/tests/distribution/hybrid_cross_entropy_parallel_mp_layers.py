# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import random
import unittest

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    fleet.meta_parallel.model_parallel_random_seed(seed)


class TestDistTraning(unittest.TestCase):
    def setUp(self):
        import os

        os.environ["ENABLE_PARALLEL_TP"] = "1"
        self.__class__.use_custom_device = True
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": self.model_parallel_size,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def test_parallel_cross_entropy(self):
        batch_size = 84
        seq_length = 10
        class_size_per_card = 1
        vocab_size = class_size_per_card * self.model_parallel_size
        seed = 100

        set_random_seed(seed)
        rank_id = dist.get_rank()

        # model_a
        model_a = fleet.meta_parallel.ParallelCrossEntropy()

        model_b = paddle.nn.CrossEntropyLoss(reduction="none")

        paddle.seed(rank_id * 10)
        random.seed(seed)
        np.random.seed(seed)
        check_group = dist.new_group(list(range(self.model_parallel_size)))

        for _ in range(1):
            np_label = np.random.randint(
                0, vocab_size, (batch_size, seq_length)
            )  # [batch_size, seq_length]
            label = paddle.to_tensor(np_label, dtype="int64")

            data = paddle.randn(
                shape=[batch_size, seq_length, class_size_per_card],
                dtype="float32",
            )
            data.stop_gradient = False
            integral_data = []
            partial_data = data.clone().detach()
            paddle.distributed.all_gather(
                integral_data, partial_data, group=check_group
            )
            integral_data = paddle.concat(integral_data, axis=-1)
            integral_data = integral_data.detach().clone()
            integral_data.stop_gradient = False

            loss_a = model_a(data, label).sum() / batch_size

            loss_b = model_b(integral_data, label).sum() / batch_size
            print("loss_a: ", loss_a.numpy(), "loss_b: ", loss_b.numpy())

            np.testing.assert_allclose(loss_a.numpy(), loss_b.numpy(), rtol=1e-2)

            loss_a.backward()
            loss_b.backward()

            integral_grad = []
            partial_grad = data.grad.clone().detach()

            # print(partial_grad)
            paddle.distributed.all_gather(
                integral_grad, partial_grad, group=check_group
            )
            integral_grad = paddle.concat(integral_grad, axis=-1)

            np.testing.assert_allclose(
                integral_data.grad.numpy(), integral_grad.numpy(), rtol=1e-2
            )


if __name__ == "__main__":
    unittest.main()
