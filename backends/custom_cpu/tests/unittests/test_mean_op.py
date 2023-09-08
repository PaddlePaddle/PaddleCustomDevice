#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.base as base
from paddle.base import Program, program_guard

np.random.seed(10)


def get_places(self):
    return [paddle.CustomPlace("custom_cpu", 0)]


OpTest._get_places = get_places


def mean_wrapper(x, axis=None, keepdim=False, reduce_all=False):
    # flake8: noqa
    if reduce_all == True:
        return paddle.mean(x, range(len(x.shape)), keepdim)
    return paddle.mean(x, axis, keepdim)


def reduce_mean_wrapper(x, axis=0, keepdim=False, reduce_all=False):
    # flake8: noqa
    if reduce_all == True:
        return paddle.mean(x, range(len(x.shape)), keepdim)
    return paddle.mean(x, axis, keepdim)


class TestMeanOp(OpTest):
    def setUp(self):
        self.op_type = "mean"
        self.python_api = base.layers.mean
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {"X": np.random.random((10, 10)).astype(self.dtype)}
        self.outputs = {"Out": np.mean(self.inputs["X"])}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_checkout_grad(self):
        self.check_grad(["X"], "Out", check_eager=False)


class TestMeanOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of mean_op must be Variable.
            input1 = 12
            self.assertRaises(TypeError, base.layers.mean, input1)
            # The input dtype of mean_op must be float16, float32, float64.
            input2 = paddle.static.data(
                name="input2", shape=[-1, 12, 10], dtype="int32"
            )
            self.assertRaises(TypeError, base.layers.mean, input2)
            input3 = paddle.static.data(name="input3", shape=[-1, 4], dtype="float16")
            base.layers.softmax(input3)


def ref_reduce_mean(x, axis=None, keepdim=False, reduce_all=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    if reduce_all:
        axis = None
    return np.mean(x, axis=axis, keepdims=keepdim)


def ref_reduce_mean_grad(x, axis, dtype, reduce_all):
    if reduce_all:
        axis = list(range(x.ndim))

    shape = [x.shape[i] for i in axis]
    return (1.0 / np.prod(shape) * np.ones(shape)).astype(dtype)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
