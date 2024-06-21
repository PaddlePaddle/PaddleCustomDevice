# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at #
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest
from numpy import linalg as LA
from op_test import OpTest, skip_check_grad_ci
import paddle
from paddle import _C_ops

import paddle.nn.functional as F
from paddle import nn
from paddle.io import DataLoader, Dataset

paddle.enable_static()
SEED = 2021


def test_squared_l2_norm(x):
    return _C_ops.squared_l2_norm(x)


@skip_check_grad_ci("tecodnn have not implement grad.")
class TestL2LossOp(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.init_kernel_dtype()
        self.init_input()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "squared_l2_norm"
        self.python_api = test_squared_l2_norm
        self.max_relative_error = 0.05
        self.x[np.abs(self.x) < self.max_relative_error] = 0.1
        self.inputs = {"X": self.x}
        # NOTE(liaotianju): It's crucial to use double to calculate for large shape
        self.outputs = {
            "Out": np.array(
                [np.square(LA.norm(self.x.astype(np.float64))).astype(self.dtype)]
            )
        }

    def init_input(self):
        self.x = np.random.uniform(-1, 1, (13, 19)).astype(self.dtype)

    def init_kernel_dtype(self):
        self.dtype = np.float32

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(place=self.place)


class TestL2LossOp1(TestL2LossOp):
    def init_input(self):
        self.x = np.random.uniform(-1, 1, (30522, 1024)).astype(self.dtype)


class TestL2LossOp2(TestL2LossOp):
    def init_kernel_dtype(self):
        self.dtype = np.float16


class RandomDataset(Dataset):
    def __init__(self, input_size, num_samples):
        self.num_samples = num_samples
        self.input_size = input_size

    def __getitem__(self, idx):
        image = np.random.random([self.input_size]).astype("float32")
        label = np.random.randint(0, self.input_size - 1, (1,)).astype("int64")
        return image, label

    def __len__(self):
        return self.num_samples


class SimpleLayer(nn.Layer):
    def __init__(self, input_size):
        super().__init__()
        shape1 = [input_size, 768]
        shape2 = [768, input_size]
        self.linear1 = nn.Linear(*shape1)
        self.silu1 = nn.Silu()
        self.linear2 = nn.Linear(*shape2)
        self.silu2 = nn.Silu()

    def forward(self, image, lable=None):
        t1 = self.silu1(self.linear1(image))
        return self.silu2(self.linear2(t1))


class SimpleNet(nn.Layer):
    def __init__(self, input_size, num_layer):
        super().__init__()
        self.layer_list = [SimpleLayer(input_size) for i in range(num_layer)]
        self.layers = nn.LayerList(self.layer_list)

    def forward(self, image, label=None):
        for idx, (layer) in enumerate(self.layers):
            image = layer(image)

        return image


# squared_l2_norm stability check
class TestSimpleNet(unittest.TestCase):
    def run_simple_net_training(self):
        paddle.set_device("sdaa")
        paddle.disable_static()

        paddle.seed(SEED)
        np.random.seed(SEED)

        input_size = 512
        num_layer = 128
        bs = 1
        dataset = RandomDataset(input_size, 20 * bs)
        simple_net = SimpleNet(input_size, num_layer)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
        opt = paddle.optimizer.Adam(
            learning_rate=1e-3,
            parameters=simple_net.parameters(),
            use_multi_tensor=True,
            grad_clip=nn.ClipGradByGlobalNorm(1.0),
        )

        losses = []
        for i, (image, label) in enumerate(loader()):
            out = simple_net(image)

            loss = F.cross_entropy(out, label)
            avg_loss = paddle.mean(loss)
            avg_loss.backward()

            opt.step()

            simple_net.clear_gradients()

            loss_cpu = avg_loss.numpy()
            losses.append(loss_cpu)

        paddle.enable_static()

        return losses

    def test(self):
        # same input run twice, loss must totally equal.
        losses1 = self.run_simple_net_training()
        losses2 = self.run_simple_net_training()

        for i in range(0, len(losses1)):
            loss1 = losses1[i]
            loss2 = losses2[i]

            print(f"loss1: {loss1}, loss2: {loss2}")

            np.testing.assert_equal(loss1, loss2)


if __name__ == "__main__":
    unittest.main()
