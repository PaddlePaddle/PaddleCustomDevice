# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest

import numpy as np
import paddle
from tests.op_test import OpTest

paddle.enable_static()
SEED = 2024


def log_softmax(x):
    max_x = np.max(x, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(x - max_x), axis=1, keepdims=True)) + max_x
    return x - logsumexp


def nll_loss(log_probs, labels, weights=None):
    if weights is None:
        weights = np.ones_like(log_probs.shape[1] * [1])

    sample_weights = weights[labels]
    nll = -np.sum(sample_weights * log_probs[np.arange(len(labels)), labels])
    total_weight = np.sum(sample_weights)
    return nll / total_weight, total_weight


def nll_loss_4d(log_softmax, target, weights=None):
    N, C, H, W = log_softmax.shape

    target_flat = target.flatten()
    log_prob = log_softmax.transpose(0, 2, 3, 1).reshape(-1, C)
    log_prob = log_prob[np.arange(log_prob.shape[0]), target_flat]

    if weights is None:
        weights = np.ones_like(log_softmax.shape[1] * [1])
    weight_flat = weights[target_flat]
    log_prob = log_prob * weight_flat
    total_weight = np.sum(weight_flat)

    loss = -np.sum(log_prob) / total_weight
    return loss, total_weight


def nll_loss_backward(log_softmax, target, weight=None):
    N, C = log_softmax.shape
    grad = np.zeros_like(log_softmax)

    if weight is not None:
        grad_flat = -weight[target] / N
    else:
        grad_flat = -1.0 / N

    indices = np.arange(N)
    grad[indices, target] = grad_flat

    return grad


def nll_loss_4d_backward(log_softmax, target, weight=None):
    N, C, H, W = log_softmax.shape

    grad = np.zeros_like(log_softmax)
    target_flat = target.flatten()

    if weight is not None:
        weight_flat = weight[target_flat]
        grad_flat = -weight_flat / (N * H * W)
    else:
        grad_flat = -1.0 / (N * H * W)

    indices = np.arange(target_flat.size)
    grad = grad.transpose(0, 2, 3, 1).reshape(-1, C)
    grad[indices, target_flat] = grad_flat

    grad = grad.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return grad


class TestNLLloss(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.CustomPlace("npu", 0)
        self.op_type = "nll_loss"
        self.init_shape()
        self.init_dtype()
        self.init_weight()

        input_x = np.random.random(self.shape).astype(self.dtype)
        self.log_out = log_softmax(input_x)
        self.labels = np.array([0, 2, 1, 8, 6, 3, 5, 9, 7, 4]).astype(np.int64)

        np_out, total_weight = nll_loss(self.log_out, self.labels, self.weight)
        self.inputs = {"X": self.log_out, "Label": self.labels}
        self.attrs = {}
        self.outputs = {"Out": np_out, "Total_weight": total_weight}

        self.update_inputs()

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_shape(self):
        self.shape = [10, 10]

    def init_dtype(self):
        self.dtype = np.float32

    def init_weight(self):
        self.weight = None

    def update_inputs(self):
        pass

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out", max_relative_error=0.01)


class TestNLLlossFP64(TestNLLloss):
    def init_dtype(self):
        self.dtype = np.float64

    def test_check_grad(self):
        x_grad = nll_loss_backward(self.log_out, self.labels, self.weight)
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            max_relative_error=0.01,
            user_defined_grads=[x_grad],
        )


class TestNLLlossWeight(TestNLLloss):
    def init_weight(self):
        self.weight = np.array(
            [1.0, 0.5, 2.0, 0.8, 0.6, 1.3, 1.6, 0.7, 0.9, 1.6]
        ).astype(np.float32)

    def update_inputs(self):
        self.inputs = {"X": self.log_out, "Label": self.labels, "Weight": self.weight}


class TestNLLloss4d(OpTest):
    def setUp(self):
        self.set_npu()
        # self.place = paddle.CustomPlace("npu", 0)
        self.place = paddle.set_device("cpu")
        self.op_type = "nll_loss"
        self.init_shape()
        self.init_dtype()
        self.init_weight()

        input_x = np.random.random(self.shape).astype(self.dtype)
        self.log_out = log_softmax(input_x)
        self.labels = np.random.randint(0, 3, (5, 4, 4)).astype("int64")

        np_out, total_weight = nll_loss_4d(self.log_out, self.labels, self.weight)
        self.inputs = {"X": self.log_out, "Label": self.labels}
        self.attrs = {}
        self.outputs = {"Out": np_out, "Total_weight": total_weight}

        self.update_inputs()

    def set_npu(self):
        self.__class__.use_custom_device = True

    def init_shape(self):
        self.shape = [5, 3, 4, 4]

    def init_dtype(self):
        self.dtype = np.float32

    def init_weight(self):
        self.weight = None

    def update_inputs(self):
        pass

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out", max_relative_error=0.01)


class TestNLLloss4dFP64(TestNLLloss4d):
    def init_dtype(self):
        self.dtype = np.float64

    def test_check_grad(self):
        x_grad = nll_loss_4d_backward(self.log_out, self.labels, self.weight)
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            max_relative_error=0.01,
            user_defined_grads=[x_grad],
        )


class TestNLLloss4dWeight(TestNLLloss4d):
    def init_weight(self):
        self.weight = np.array([1.0, 0.5, 2.0]).astype(np.float32)

    def update_inputs(self):
        self.inputs = {"X": self.log_out, "Label": self.labels, "Weight": self.weight}


if __name__ == "__main__":
    unittest.main()
