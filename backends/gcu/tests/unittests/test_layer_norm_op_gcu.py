#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from functools import reduce
from operator import mul

import numpy as np

import paddle
from paddle.base import Program, program_guard

paddle.enable_static()

np.random.seed(123)
paddle.seed(123)


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


def layer_norm_wrapper(x, scale=None, bias=None, epsilon=1e-05, begin_norm_axis=1):
    input_shape = list(x.shape)
    normalized_shape = input_shape[begin_norm_axis:]
    return paddle.nn.functional.layer_norm(
        x, normalized_shape, weight=scale, bias=bias, epsilon=epsilon
    )


# NOTE：The submodule needs to be updated.
# The 'no_check_set_white_list' must contain 'layer_norm' to enable the following cases.

# class TestLayerNormOpByOpTest(OpTest):
#     def setUp(self):
#         self.python_api = layer_norm_wrapper
#         self.public_python_api = layer_norm_wrapper
#         self.op_type = "layer_norm"
#         self.set_device()
#         self.initConfig()
#         self.initTestCase()


#     def set_device(self):
#         self.__class__.use_custom_device = True
#         self.place = paddle.CustomPlace("gcu", 0)

#     def test_check_output(self):
#         self.check_output_with_place(self.place,
#             no_check_set=["Mean", "Variance"],
#             atol=self.ori_atol,
#             rtol=self.ori_rtol)

#     def test_check_grad(self):
#         self.check_grad_with_place(self.place,
#             self.check_grad_input_list,
#             ['Y'],
#             max_relative_error=self.max_relative_error)

#     def initConfig(self):
#         self.ori_atol = 1e-4
#         self.ori_rtol = 1e-4

#         self.max_relative_error = 1e-2
#         self.dtype = "float32"
#         self.x_shape = [2, 6, 6, 3]
#         self.epsilon = 0.00001
#         self.begin_norm_axis = 1
#         self.has_scale = True
#         self.has_bias = True

#     def initTestCase(self):
#         np.random.seed(123)

#         self.D = reduce(
#             mul, self.x_shape[self.begin_norm_axis : len(self.x_shape)], 1
#         )
#         self.scale_shape = [self.D]
#         x = np.random.random(self.x_shape).astype(self.dtype)
#         scale = (
#             np.random.random(self.scale_shape).astype(self.dtype)
#             if self.has_scale
#             else None
#         )
#         bias = (
#             np.random.random(self.scale_shape).astype(self.dtype)
#             if self.has_bias
#             else None
#         )
#         self.inputs = {
#             "X": x,
#         }
#         self.check_grad_input_list = ['X']

#         if self.has_scale:
#             self.inputs.update({"Scale": scale})
#             self.check_grad_input_list.append('Scale')
#         if self.has_bias:
#             self.inputs.update({"Bias": bias})
#             self.check_grad_input_list.append('Bias')

#         self.attrs = {
#             "epsilon": self.epsilon,
#             "begin_norm_axis": self.begin_norm_axis,
#         }
#         y, mean, variance = _reference_layer_norm_naive(
#             x, scale, bias, self.epsilon, self.begin_norm_axis
#         )
#         self.outputs = {
#             "Y": y,
#             "Mean": mean,
#             "Variance": variance,
#         }


# class TestLayerNormOpByOpTestFP32_case1(TestLayerNormOpByOpTest):
#     def initConfig(self):
#         self.ori_atol = 1e-4
#         self.ori_rtol = 1e-4
#         self.max_relative_error = 1e-2

#         self.dtype = "float32"
#         self.x_shape = [2, 100]
#         self.epsilon = 0.00001
#         self.begin_norm_axis = 1
#         self.has_scale = True
#         self.has_bias = True


# class TestLayerNormOpByOpTestFP32_case2(TestLayerNormOpByOpTest):
#     def initConfig(self):
#         self.ori_atol = 1e-4
#         self.ori_rtol = 1e-4
#         self.max_relative_error = 1e-5

#         self.dtype = "float32"
#         self.x_shape = [2, 6, 6, 3]
#         self.epsilon = 0.00001
#         self.begin_norm_axis = 1
#         self.has_scale = False
#         self.has_bias = False


# class TestLayerNormOpByOpTestFP32_case3(TestLayerNormOpByOpTest):
#     def initConfig(self):
#         self.ori_atol = 1e-4
#         self.ori_rtol = 1e-4
#         self.max_relative_error = 3e-3

#         self.dtype = "float32"
#         self.x_shape = [2, 6, 6, 3]
#         self.epsilon = 0.00001
#         self.begin_norm_axis = 1
#         self.has_scale = True
#         self.has_bias = False


# class TestLayerNormOpByOpTestFP32_case4(TestLayerNormOpByOpTest):
#     def initConfig(self):
#         self.ori_atol = 1e-4
#         self.ori_rtol = 1e-4
#         self.max_relative_error = 1e-3

#         self.dtype = "float32"
#         self.x_shape = [2, 6, 6, 3]
#         self.epsilon = 0.00001
#         self.begin_norm_axis = 1
#         self.has_scale = False
#         self.has_bias = True


class TestDygraphLayerNormAPIError(unittest.TestCase):
    def test_errors(self):
        paddle.set_device("gcu")
        with program_guard(Program(), Program()):
            paddle.enable_static()

            layer_norm = paddle.nn.LayerNorm([32, 32])
            # the input of LayerNorm must be Variable.
            x1 = np.random.random((3, 32, 32)).astype("float32")
            self.assertRaises(TypeError, layer_norm, x1)

            # the input dtype of LayerNorm must be float32 or float64
            # float16 only can be set on GPU place
            x2 = paddle.static.data(name="x2", shape=[-1, 3, 32, 32], dtype="int32")
            self.assertRaises(TypeError, layer_norm, x2)
        with paddle.pir_utils.IrGuard(), program_guard(Program(), Program()):
            layer_norm = paddle.nn.LayerNorm([32, 32])
            # the input of LayerNorm must be Variable.
            x1 = np.random.random((3, 32, 32)).astype("float32")
            self.assertRaises(ValueError, layer_norm, x1)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
