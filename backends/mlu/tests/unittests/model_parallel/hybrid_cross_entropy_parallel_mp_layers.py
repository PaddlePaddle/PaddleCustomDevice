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


class ColumnLinearNet(paddle.nn.Layer):
    def __init__(self, input_size, output_size, global_dtype):
        super().__init__()
        self.parallel_linear = fleet.meta_parallel.ColumnParallelLinear(
            in_features=input_size,
            out_features=output_size,
            weight_attr=None,
            has_bias=True,
            gather_output=True,
            name="test_column_linear",
        )

    def forward(self, x):
        output = self.parallel_linear(x)
        return output


class RowLinearNet(paddle.nn.Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.parallel_linear = fleet.meta_parallel.RowParallelLinear(
            in_features=input_size,
            out_features=output_size,
            has_bias=True,
            input_is_parallel=False,
            name="test_row_linear",
        )

    def forward(self, x):
        output = self.parallel_linear(x)
        return output


class EmbeddingNet(paddle.nn.Layer):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = fleet.meta_parallel.VocabParallelEmbedding(
            vocab_size, hidden_size
        )

    def forward(self, x):
        output = self.embedding(x)
        return output


class SimpleMatmul(paddle.nn.Layer):
    def __init__(self, weight, output_size, global_dtype):
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=weight.shape,
            dtype=global_dtype,
            attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(weight)),
        )
        self.bias = self.create_parameter(
            shape=[output_size],
            dtype=global_dtype,
            attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(0.0)),
        )

    def forward(self, x):
        output = paddle.matmul(x, self.weight) + self.bias
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

    def test_parallel_cross_entropy(self):
        batch_size = 82
        seq_length = 3
        class_size_per_card = 2
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
        for _ in range(5):
            np_label = np.random.randint(0, vocab_size, (batch_size, seq_length))
            label = paddle.to_tensor(np_label, dtype="int64")

            data = paddle.randn(
                shape=[batch_size, seq_length, class_size_per_card],
                dtype="float32",
            )
            data.stop_gradient = False

            # check_group = dist.new_group(list(range(self.model_parallel_size)))
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

            np.testing.assert_allclose(loss_a.numpy(), loss_b.numpy(), rtol=1e-1)

            loss_a.backward()
            loss_b.backward()

            integral_grad = []
            partial_grad = data.grad.clone().detach()
            paddle.distributed.all_gather(
                integral_grad, partial_grad, group=check_group
            )
            integral_grad = paddle.concat(integral_grad, axis=-1)

            np.testing.assert_allclose(
                integral_data.grad.numpy(), integral_grad.numpy(), rtol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
