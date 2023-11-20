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
from tests.op_test import OpTest

paddle.enable_static()


def adamw_step(inputs, attributes):
    """
    Simulate one step of the adam optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment1, moment2,
    beta1 power accumulator and beta2 power accumulator
    """
    param = inputs["Param"]
    grad = inputs["Grad"]
    moment1 = inputs["Moment1"]
    moment2 = inputs["Moment2"]
    lr = inputs["LearningRate"]
    beta1_pow = inputs["Beta1Pow"]
    beta2_pow = inputs["Beta2Pow"]

    epsilon = attributes["epsilon"]
    coeff = attributes["coeff"]
    if attributes.get("with_decay", False):
        decay = 1.0 - lr * coeff
        param2 = param * decay
        param = param2.copy()
    if "beta1" in attributes:
        beta1 = attributes["beta1"]
    else:
        beta1 = inputs["Beta1Tensor"][0]
    if "beta2" in attributes:
        beta2 = attributes["beta2"]
    else:
        beta2 = inputs["Beta2Tensor"][0]

    moment1_out = beta1 * moment1 + (1 - beta1) * grad
    moment2_out = beta2 * moment2 + (1 - beta2) * np.square(grad)
    lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)
    param_out = param - lr_t * (moment1_out / (np.sqrt(moment2_out) + epsilon))

    return param_out, moment1_out, moment2_out


class TestAdamW(OpTest):
    def setUp(self):
        self.init_dtype()
        self.set_device()
        self.op_type = "adamw"
        param = np.random.uniform(-1, 1, (105, 102)).astype(self.dtype)
        grad = np.random.uniform(-1, 1, (105, 102)).astype(self.dtype)
        moment1 = np.random.uniform(-1, 1, (105, 102)).astype(self.dtype)
        # The second moment is positive
        moment2 = np.random.random((105, 102)).astype(self.dtype)

        learning_rate = 0.5
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            "Param": param,
            "Grad": grad,
            "Moment1": moment1,
            "Moment2": moment2,
            "LearningRate": np.array([learning_rate]).astype(self.dtype),
            "Beta1Pow": np.array([beta1_pow]).astype(self.dtype),
            "Beta2Pow": np.array([beta2_pow]).astype(self.dtype),
        }

        self.attrs = {
            "epsilon": epsilon,
            "beta1": beta1,
            "beta2": beta2,
            "coeff": 0.9,
            "with_decay": True,
        }

        param_out, moment1_out, moment2_out = adamw_step(self.inputs, self.attrs)

        self.outputs = {
            "Moment1Out": moment1_out,
            "Moment2Out": moment2_out,
            "ParamOut": param_out,
            "Beta1PowOut": np.array([beta1_pow]).astype(self.dtype) * beta1,
            "Beta2PowOut": np.array([beta2_pow]).astype(self.dtype) * beta2,
        }

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)


class TestAdamOpWithSkipUpdate(OpTest):
    def setUp(self):
        self.init_dtype()
        self.set_device()
        self.op_type = "adamw"
        param = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        grad = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype(self.dtype)

        learning_rate = 0.004
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            "Param": param,
            "Grad": grad,
            "Moment1": moment1,
            "Moment2": moment2,
            "LearningRate": np.array([learning_rate]).astype(self.dtype),
            "Beta1Pow": np.array([beta1_pow]).astype(self.dtype),
            "Beta2Pow": np.array([beta2_pow]).astype(self.dtype),
            "Beta1Tensor": np.array([beta1]).astype(self.dtype),
            "Beta2Tensor": np.array([beta2]).astype(self.dtype),
            "EpsilonTensor": np.array([epsilon]).astype(self.dtype),
            "SkipUpdate": np.array([True]).astype("bool"),
        }

        self.attrs = {"epsilon": epsilon, "coeff": 0.02, "with_decay": True}

        self.outputs = {
            "Moment1Out": moment1,
            "Moment2Out": moment2,
            "ParamOut": param,
            "Beta1PowOut": self.inputs["Beta1Pow"],
            "Beta2PowOut": self.inputs["Beta2Pow"],
        }

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)


class TestAdamOpWithoutDecay(OpTest):
    def setUp(self):
        self.init_dtype()
        self.set_device()
        self.op_type = "adamw"
        param = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        grad = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype(self.dtype)
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype(self.dtype)

        learning_rate = 0.004
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            "Param": param,
            "Grad": grad,
            "Moment1": moment1,
            "Moment2": moment2,
            "LearningRate": np.array([learning_rate]).astype(self.dtype),
            "Beta1Pow": np.array([beta1_pow]).astype(self.dtype),
            "Beta2Pow": np.array([beta2_pow]).astype(self.dtype),
            "Beta1Tensor": np.array([beta1]).astype(self.dtype),
            "Beta2Tensor": np.array([beta2]).astype(self.dtype),
            "EpsilonTensor": np.array([epsilon]).astype(self.dtype),
            "SkipUpdate": np.array([True]).astype("bool"),
        }

        self.attrs = {"epsilon": epsilon, "coeff": 0.02, "with_decay": False}

        self.outputs = {
            "Moment1Out": moment1,
            "Moment2Out": moment2,
            "ParamOut": param,
            "Beta1PowOut": self.inputs["Beta1Pow"],
            "Beta2PowOut": self.inputs["Beta2Pow"],
        }

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
