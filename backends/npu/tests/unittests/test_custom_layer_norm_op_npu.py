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

from __future__ import print_function, division

import os
import numpy as np
import unittest
import paddle

paddle.enable_static()
if os.getenv("CUSTOM_DEVICE_ROOT") is not None:
    for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
        if lib.endswith(".so"):
            paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
                lib
            )
from tests.op_test import OpTest
from functools import reduce
from operator import mul


def _reference_layer_norm_naive(x, scale, beta, epsilon, begin_norm_axis=1):
    x_shape = x.shape
    N = reduce(mul, x_shape[0:begin_norm_axis], 1)
    D = reduce(mul, x_shape[begin_norm_axis : len(x_shape)], 1)
    x.shape = [N, D]

    mean = np.mean(x, axis=1)
    var = np.var(x, axis=1) + epsilon
    output = np.divide((x - mean.reshape([N, 1])), (np.sqrt(var)).reshape([N, 1]))
    if scale is not None:
        output = scale.reshape([1, D]) * output
    if beta is not None:
        output = output + beta.reshape([1, D])

    x.shape, output.shape = x_shape, x_shape
    return output, mean, var


class TestNPULayerNorm(OpTest):
    def setUp(self):
        self.op_type = "custom_layer_norm"
        self.set_npu()
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [1, 32, 128]).astype(self.dtype)
        weight = np.random.uniform(
            -1,
            1,
            [
                128,
            ],
        ).astype(self.dtype)
        bias = np.random.uniform(
            -1,
            1,
            [
                128,
            ],
        ).astype(self.dtype)
        epsilon = 1e-5
        begin_norm_axis = 2

        out = _reference_layer_norm_naive(x, weight, bias, epsilon, begin_norm_axis)[0]

        self.inputs = {"X": x, "Scale": weight, "Bias": bias}
        self.attrs = {"begin_norm_axis": begin_norm_axis, "epsilon": epsilon}
        self.outputs = {"Out": out}

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-02)


if __name__ == "__main__":
    unittest.main()
