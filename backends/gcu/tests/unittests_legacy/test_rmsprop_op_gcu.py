# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import paddle

paddle.enable_static()


class TestRMSProp(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.__class__.use_custom_device = True
        paddle.set_device("gcu")

    def test_rmsprop_dygraph(self):
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        # This can be any optimizer supported by dygraph.
        opt = paddle.optimizer.RMSProp(
            learning_rate=0.01,
            parameters=linear.parameters(),
            weight_decay=0.01,
        )
        out = linear(a)
        out.backward()
        opt.step()
        opt.clear_gradients()

    def test_raise_error(self):
        self.assertRaises(ValueError, paddle.optimizer.RMSProp, None)
        self.assertRaises(
            ValueError, paddle.optimizer.RMSProp, learning_rate=0.1, rho=None
        )
        self.assertRaises(
            ValueError,
            paddle.optimizer.RMSProp,
            learning_rate=0.1,
            epsilon=None,
        )
        self.assertRaises(
            ValueError,
            paddle.optimizer.RMSProp,
            learning_rate=0.1,
            momentum=None,
        )

    def test_rmsprop_op_invalid_input(self):
        paddle.disable_static()
        linear = paddle.nn.Linear(10, 10)
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.RMSProp(
                0.1, epsilon=-1, parameters=linear.parameters()
            )
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.RMSProp(
                0.1, momentum=-1, parameters=linear.parameters()
            )
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.RMSProp(0.1, rho=-1, parameters=linear.parameters())


class TestRMSPropGroup(TestRMSProp):
    def test_rmsprop_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        # This can be any optimizer supported by dygraph.
        opt = paddle.optimizer.RMSProp(
            learning_rate=0.01,
            parameters=[
                {"params": linear_1.parameters()},
                {"params": linear_2.parameters(), "weight_decay": 0.001},
            ],
            weight_decay=0.01,
        )
        out = linear_1(a)
        out = linear_2(out)
        out.backward()
        opt.step()
        opt.clear_gradients()


if __name__ == "__main__":
    unittest.main()
