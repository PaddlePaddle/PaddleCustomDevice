#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from tests.op_test import OpTest

paddle.enable_static()


def calculate_momentum_by_numpy(
    param,
    grad,
    mu,
    velocity,
    use_nesterov,
    learning_rate,
    regularization_method=None,
    regularization_coeff=1.0,
):
    if regularization_method == "l2_decay":
        grad = grad + regularization_coeff * param

        velocity_out = mu * velocity + grad
        if use_nesterov:
            param_out = param - (grad + velocity_out * mu) * learning_rate
        else:
            param_out = param - learning_rate * velocity_out
    else:
        velocity_out = mu * velocity + grad
        if use_nesterov:
            param_out = param - grad * learning_rate - velocity_out * mu * learning_rate
        else:
            param_out = param - learning_rate * velocity_out

    return param_out, velocity_out


class TestMomentumOp1(OpTest):
    def set_device(self):
        self.__class__.use_custom_device = True

    def setUp(self):
        self.set_device()
        self.op_type = "momentum"
        self.init_dtype()
        self.init_case()

        param = np.random.random(self.shape).astype(self.dtype)
        grad = np.random.random(self.shape).astype(self.dtype)
        velocity = np.zeros(self.shape).astype(self.dtype)
        learning_rate = np.array([0.001]).astype(np.float32)
        mu = 0.0001

        self.inputs = {
            "Param": param,
            "Grad": grad,
            "Velocity": velocity,
            "LearningRate": learning_rate,
        }

        self.attrs = {"mu": mu, "use_nesterov": self.use_nesterov}

        param_out, velocity_out = calculate_momentum_by_numpy(
            param=param,
            grad=grad,
            mu=mu,
            velocity=velocity,
            use_nesterov=self.use_nesterov,
            learning_rate=learning_rate,
        )

        self.outputs = {"ParamOut": param_out, "VelocityOut": velocity_out}

    def init_case(self):
        self.shape = (123, 321)
        self.use_nesterov = False

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(paddle.CustomPlace("gcu", 0))


class TestMomentumOpFp16(TestMomentumOp1):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output(atol=1e-3)


class TestMomentumOp2(TestMomentumOp1):
    def init_case(self):
        self.shape = (123, 321)
        self.use_nesterov = True


class TestMomentumV2(unittest.TestCase):
    def test_momentum_dygraph(self):
        paddle.disable_static(place=paddle.CustomPlace("gcu", 0))
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        # This can be any optimizer supported by dygraph.
        adam = paddle.optimizer.Momentum(
            learning_rate=0.01, momentum=0.9, parameters=linear.parameters()
        )
        out = linear(a)
        out.backward()
        adam.step()
        adam.clear_gradients()

    def test_raise_error(self):
        self.assertRaises(ValueError, paddle.optimizer.Momentum, learning_rate=None)
        self.assertRaises(ValueError, paddle.optimizer.Momentum, momentum=None)


class TestMomentumOpWithDecay(OpTest):
    def set_device(self):
        self.__class__.use_custom_device = True

    def setUp(self):
        self.set_device()
        self.op_type = "momentum"
        self.dtype = np.float32
        self.use_nesterov = True
        self.regularization_method = "l2_decay"
        self.regularization_coeff = 0.9
        self.init_config()

        param = np.random.random((123, 321)).astype(self.dtype)
        grad = np.random.random((123, 321)).astype(self.dtype)
        velocity = np.zeros((123, 321)).astype(self.dtype)
        learning_rate = np.array([0.001]).astype(np.float32)
        mu = 0.0001
        use_nesterov = self.use_nesterov
        regularization_method = self.regularization_method
        regularization_coeff = self.regularization_coeff

        self.inputs = {
            "Param": param,
            "Grad": grad,
            "Velocity": velocity,
            "LearningRate": learning_rate,
        }

        self.attrs = {
            "mu": mu,
            "use_nesterov": use_nesterov,
            "regularization_method": regularization_method,
            "regularization_coeff": regularization_coeff,
        }

        grad = grad + regularization_coeff * param

        param_out, velocity_out = calculate_momentum_by_numpy(
            param=param,
            grad=grad,
            mu=mu,
            velocity=velocity,
            use_nesterov=use_nesterov,
            learning_rate=learning_rate,
        )

        self.outputs = {"ParamOut": param_out, "VelocityOut": velocity_out}

    def init_config(self):
        pass

    def test_check_output(self):
        paddle.enable_static()
        self.check_output_with_place(paddle.CustomPlace("gcu", 0), atol=3e-3)


class TestMomentumOpWithDecayFP16(TestMomentumOpWithDecay):
    def init_config(self):
        self.dtype = np.float16

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(atol=1e-3)


class TestMomentumOpWithDecay2(TestMomentumOpWithDecay):
    def init_config(self):
        self.use_nesterov = False


class TestMomentumOpWithDecayAPI(unittest.TestCase):
    def _test_momentum_dygraph_common(self, regularization):
        paddle.disable_static(paddle.CustomPlace("gcu", 0))
        inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
        linear = paddle.nn.Linear(10, 10)
        inp = paddle.to_tensor(inp)
        out = linear(inp)
        loss = paddle.mean(out)
        # This can be any optimizer supported by dygraph.
        momentum = paddle.optimizer.Momentum(
            learning_rate=0.01,
            momentum=0.9,
            parameters=linear.parameters(),
            weight_decay=regularization,
        )
        momentum.minimize(loss)

    def test_momentum_dygraph_1(self):
        self._test_momentum_dygraph_common(
            regularization=paddle.regularizer.L2Decay(coeff=0.1)
        )


class TestMomentumOpVsMomentumOpWithDecayAPI(unittest.TestCase):
    def __update_params(self, momentum, linear):
        for i in range(10):
            inp = paddle.full(shape=[2, 2], fill_value=i, dtype="float32").astype(
                "float32"
            )
            inp = paddle.to_tensor(inp)
            out = linear(inp)
            loss = paddle.mean(out)
            loss.backward()
            momentum.minimize(loss)
            linear.clear_gradients()

    def __test_vs(self, place=paddle.CustomPlace("gcu", 0)):
        linear_old = paddle.nn.Linear(
            2,
            2,
            weight_attr=paddle.nn.initializer.Constant(value=2.0),
            bias_attr=paddle.nn.initializer.Constant(value=2.0),
        )
        momentum_old = paddle.optimizer.Momentum(
            learning_rate=0.01,
            momentum=0.9,
            parameters=linear_old.parameters(),
            weight_decay=paddle.regularizer.L2Decay(coeff=0.1),
        )
        self.__update_params(momentum=momentum_old, linear=linear_old)

        linear_new = paddle.nn.Linear(
            2,
            2,
            weight_attr=paddle.nn.initializer.Constant(value=2.0),
            bias_attr=paddle.nn.initializer.Constant(value=2.0),
        )
        momentum_new = paddle.optimizer.Momentum(
            learning_rate=0.01,
            momentum=0.9,
            parameters=linear_new.parameters(),
            weight_decay=paddle.regularizer.L2Decay(coeff=0.1),
        )
        self.__update_params(momentum=momentum_new, linear=linear_new)

        self.assertEqual(
            (linear_old.weight.numpy() == linear_new.weight.numpy()).all(),
            True,
            "the param weight updated by two Momentum optimizers should equal",
        )

    def test_vs(self, place=paddle.CustomPlace("gcu", 0)):
        paddle.disable_static()
        self.__test_vs(place=place)
        paddle.enable_static()


class TestMomentumV2Group(TestMomentumV2):
    def test_momentum_dygraph(self):
        paddle.disable_static(place=paddle.CustomPlace("gcu", 0))
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear_1 = paddle.nn.Linear(13, 5)
        linear_2 = paddle.nn.Linear(5, 3)
        # This can be any optimizer supported by dygraph.
        adam = paddle.optimizer.Momentum(
            learning_rate=0.01,
            parameters=[
                {"params": linear_1.parameters()},
                {
                    "params": linear_2.parameters(),
                    "weight_decay": 0.001,
                    "learning_rate": 0.1,
                    "momentum": 0.99,
                },
            ],
            weight_decay=0.1,
            momentum=0.9,
        )
        out = linear_1(a)
        out = linear_2(out)
        out.backward()
        adam.step()
        adam.clear_gradients()


if __name__ == "__main__":
    unittest.main()
