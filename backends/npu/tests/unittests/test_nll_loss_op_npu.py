#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import sys

from tests.op_test import OpTest
import paddle
import paddle.fluid as fluid

paddle.enable_static()
SEED = 2021


def nll_loss_1d(logs, targets, weight=None, reduction='mean',
                ignore_index=-100):
    input_shape = logs.shape
    N = input_shape[0]
    C = input_shape[1]
    out = np.zeros_like(targets).astype(np.float64)
    total_weight = 0
    for i in range(N):
        cur_target = targets[i]
        if cur_target == ignore_index:
            out[i] = 0
            continue
        cur_weight = weight[cur_target] if weight is not None else 1
        total_weight += cur_weight
        out[i] = -logs[i][cur_target] * cur_weight
    if reduction == 'sum':
        return np.sum(out), np.array([total_weight]).astype('float64')
    elif reduction == 'mean':
        return out.sum() / total_weight, np.array(
            [total_weight]).astype('float64')
    elif reduction == 'none':
        return out


class TestNLLLossOp1DWithReduce(OpTest):
    def set_npu(self):
        self.__class__.use_custom_device = True

    def initParams(self):
        self.set_npu()
        self.init_test_case()
        self.op_type = "nll_loss"
        self.place = paddle.CustomPlace('npu', 0)
        self.python_api = paddle.nn.functional.nll_loss
        self.python_out_sig = ["Out"]
        self.with_weight = False
        self.python_api = paddle.nn.functional.nll_loss
        self.python_out_sig = ["Out"]

    def setUp(self):
        self.initParams()

        np.random.seed(200)
        input_np = np.random.uniform(0.1, 0.8,
                                     self.input_shape).astype("float32")
        np.random.seed(200)
        label_np = np.random.randint(0, self.input_shape[1],
                                     self.label_shape).astype("int64")
        output_np, total_weight_np = nll_loss_1d(input_np, label_np)
        self.inputs = {'X': input_np, 'Label': label_np}
        if self.with_weight:
            np.random.seed(200)
            weight_np = np.random.uniform(0.1, 0.8,
                                          self.input_shape[1]).astype("float32")
            output_np, total_weight_np = nll_loss_1d(
                input_np, label_np, weight=weight_np)
            self.inputs['Weight'] = weight_np

        self.outputs = {'Out': output_np, 'Total_weight': total_weight_np}
        self.attrs = {'reduction': 'mean', 'ignore_index': -100}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_output_with_weight(self):
        self.with_weight = True
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.with_weight = True
        self.check_grad_with_place(self.place, ['X'], 'Out')

    def init_test_case(self):
        self.input_shape = [10, 10]
        self.label_shape = [10]


if __name__ == '__main__':
    unittest.main()
