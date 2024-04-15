#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np

import paddle

paddle.set_device("npu")


class TestApiOneDim(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def test_add(self):
        x_np = np.random.rand(1)
        y_np = np.random.rand()
        out_np = np.add(x_np, y_np)
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        out = paddle.add(x, y)
        np.testing.assert_allclose(out.numpy(), out_np)

        x_np = np.random.rand(1)
        y_np = np.random.rand(1)
        out_np = np.add(x_np, y_np)
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        out = paddle.add(x, y)
        np.testing.assert_array_equal(out.numpy(), out_np)

    def test_maximum(self):
        x_np = np.random.rand(1)
        y_np = np.random.rand()
        out_np = np.maximum(x_np, y_np)
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        out = paddle.maximum(x, y)
        np.testing.assert_allclose(out.numpy(), out_np)

    def test_mean(self):
        x_np = np.random.rand(8)
        out_np = np.mean(x_np, axis=0)
        x = paddle.to_tensor(x_np)
        out = paddle.mean(x, axis=0)
        np.testing.assert_allclose(out.numpy(), out_np)

    def test_sqrt(self):
        x_np = np.random.rand(1)
        out_np = np.sqrt(x_np)
        x = paddle.to_tensor(x_np)
        out = paddle.sqrt(x)
        np.testing.assert_array_equal(out.numpy(), out_np)

    def test_stack(self):
        x_np_list = [np.random.rand(1) for _ in range(3)]
        out_np = np.stack(x_np_list)
        x_list = [paddle.to_tensor(x) for x in x_np_list]
        out = paddle.stack(x_list)
        np.testing.assert_array_equal(out.numpy(), out_np)

        x_np_list = [np.random.rand(1) for _ in range(6)]
        out_np = np.stack(x_np_list)
        x_list = [paddle.to_tensor(x) for x in x_np_list]
        out = paddle.stack(x_list)
        np.testing.assert_array_equal(out.numpy(), out_np)

    def test_mul(self):
        x_np = np.random.rand(1)
        y_np = np.random.rand(1)
        out_np = np.multiply(x_np, y_np)
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        out = paddle.multiply(x, y)
        np.testing.assert_array_equal(out.numpy(), out_np)

    def test_split(self):
        x_np = np.random.rand(8).astype(np.float32)
        out_np = np.split(x_np, 1, axis=0)
        x = paddle.to_tensor(x_np)
        out = paddle.split(x, 1, axis=0)
        np.testing.assert_array_equal(out[0].numpy(), out_np[0])


if __name__ == "__main__":
    unittest.main()
