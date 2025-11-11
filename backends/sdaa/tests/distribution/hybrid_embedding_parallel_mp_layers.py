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


class EmbeddingNet(paddle.nn.Layer):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
            vocab_size, hidden_size
        )

    def forward(self, x):
        output = self.embedding(x)
        return output


class SimpleEmbedding(paddle.nn.Layer):
    def __init__(self, vocab_size, hidden_size, weight):
        super().__init__()
        self.embedding = paddle.nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                name="origin_embedding",
                initializer=paddle.nn.initializer.Assign(weight),
            ),
        )

    def forward(self, x):
        output = self.embedding(x)
        return output


class TestDistTraning(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": self.model_parallel_size,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def test_parallel_embedding(self):
        batch_size = 17
        seq_length = 23
        vocab_size_per_card = 2
        vocab_size = vocab_size_per_card * self.model_parallel_size
        hidden_size = 2
        seed = 1236

        set_random_seed(seed)
        rank_id = dist.get_rank()

        # model_a
        model_a = EmbeddingNet(vocab_size, hidden_size)

        # model_b
        check_group = dist.new_group(list(range(self.model_parallel_size)))
        integral_w = []
        partial_w = model_a.embedding.weight.clone().detach()
        paddle.distributed.all_gather(integral_w, partial_w, group=check_group)
        result_w = []
        for idx in range(len(integral_w)):
            tmp = paddle.gather(
                integral_w[idx],
                paddle.to_tensor(list(range(vocab_size_per_card))),
            )
            result_w.append(tmp)
        integral_w = paddle.concat(result_w, axis=0)

        model_b = SimpleEmbedding(vocab_size, hidden_size, integral_w)

        optimizer_a = paddle.optimizer.SGD(
            learning_rate=0.001, parameters=model_a.parameters()
        )

        optimizer_b = paddle.optimizer.SGD(
            learning_rate=0.001, parameters=model_b.parameters()
        )

        for _ in range(5):
            np_input_data = np.random.randint(0, vocab_size, (batch_size, seq_length))
            input_data = paddle.to_tensor(np_input_data, dtype="int32")

            output_a = model_a(input_data)
            loss_a = output_a.mean()

            output_b = model_b(input_data)
            loss_b = output_b.mean()

            loss_a.backward()
            loss_b.backward()

            optimizer_a.step()
            optimizer_b.step()
            print("embedding loss")
            print(loss_a.numpy(), loss_b.numpy())

            np.testing.assert_allclose(loss_a.numpy(), loss_b.numpy(), rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
