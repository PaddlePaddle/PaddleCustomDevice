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
import copy

from op_test import OpTest
import paddle
from paddle.static import Program, program_guard
from op_test import skip_check_grad_ci

paddle.enable_static()


@skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
class TestScale(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "scale"
        self.python_api = paddle.scale
        self.init_dtype()
        self.init_input()
        self.attrs = {
            "scale": self.scale,
            "bias": self.bias,
            "bias_after_scale": self.bias_after_scale,
        }
        self.inputs = {"X": self.x}
        self.outputs = {
            "Out": self.x * self.scale
            + (self.bias if self.bias_after_scale else self.scale * self.bias)
        }

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_input(self):
        self.scale = 2.0
        self.bias = 1.0
        self.bias_after_scale = False
        self.x = np.random.random((2, 3, 4)).astype(self.dtype)

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestScaleOpScaleVariable(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "scale"
        self.python_api = paddle.scale
        self.place = paddle.CustomPlace("sdaa", 0)
        self.dtype = np.float32
        self.init_dtype_type()
        self.scale = -2.3
        self.bias = 1.0
        self.bias_after_scale = False
        self.x = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.attrs = {
            "scale": self.scale,
            "bias": self.bias,
            "bias_after_scale": self.bias_after_scale,
        }
        self.inputs = {
            "X": self.x,
            "ScaleTensor": np.array([self.scale]).astype(self.dtype),
        }
        self.outputs = {
            "Out": self.x * self.scale
            + (self.bias if self.bias_after_scale else self.scale * self.bias)
        }

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype_type(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestScaleApiDygraph(unittest.TestCase):
    def _executed_api(self, x, scale=1.0, bias=0.0):
        return paddle.scale(x, scale, bias)

    def init_dtype(self):
        self.dtype = np.float32

    def test_api(self):
        self.init_dtype()
        paddle.disable_static()
        paddle.set_device("sdaa")
        input = np.random.random([2, 25]).astype(self.dtype)
        x = paddle.to_tensor(input)
        input_copy = copy.deepcopy(x)
        out = self._executed_api(x, scale=2.0, bias=3.0)
        np.testing.assert_array_equal(out.numpy(), input * 2.0 + 3.0)
        np.testing.assert_array_equal(
            x.numpy(), input_copy
        )  # To check if the input value has been modified
        paddle.enable_static()


class TestScaleInplaceApiDygraph(unittest.TestCase):
    def _executed_api(self, x, scale=1.0, bias=0.0):
        return x.scale_(scale, bias)

    def init_dtype(self):
        self.dtype = np.float32

    def test_api(self):
        self.init_dtype()
        paddle.disable_static()
        paddle.set_device("sdaa")
        input = np.random.random([2, 25]).astype(self.dtype)
        x = paddle.to_tensor(input)
        out = self._executed_api(x, scale=2.0, bias=3.0)
        np.testing.assert_array_equal(out.numpy(), input * 2.0 + 3.0)
        paddle.enable_static()


class TestScaleApiStatic(unittest.TestCase):
    def _executed_api(self, x, scale=1.0, bias=0.0):
        return paddle.scale(x, scale, bias)

    def init_dtype(self):
        self.dtype = np.float32

    def test_api(self):
        self.init_dtype()
        paddle.enable_static()
        input = np.random.random([2, 25]).astype(self.dtype)
        main_prog = Program()
        with program_guard(main_prog, Program()):
            x = paddle.static.data(name="x", shape=[2, 25], dtype="float32")
            out = self._executed_api(x, scale=2.0, bias=3.0)

        exe = paddle.static.Executor(place=paddle.CustomPlace("sdaa", 0))
        out = exe.run(main_prog, feed={"x": input}, fetch_list=[out, x])
        np.testing.assert_array_equal(out[0], input * 2.0 + 3.0)
        np.testing.assert_array_equal(
            out[1], input
        )  # To check if the input value has been modified


class TestScaleInplaceApiStatic(unittest.TestCase):
    def _executed_api(self, x, scale=1.0, bias=0.0):
        return x.scale_(scale, bias)

    def init_dtype(self):
        self.dtype = np.float32

    def test_api(self):
        self.init_dtype()
        paddle.enable_static()
        input = np.random.random([2, 25]).astype(self.dtype)
        main_prog = Program()
        with program_guard(main_prog, Program()):
            x = paddle.static.data(name="x", shape=[2, 25], dtype="float32")
            out = self._executed_api(x, scale=2.0, bias=3.0)

        exe = paddle.static.Executor(place=paddle.CustomPlace("sdaa", 0))
        out = exe.run(main_prog, feed={"x": input}, fetch_list=[out, x])
        np.testing.assert_array_equal(out[0], input * 2.0 + 3.0)


class TestScaleOpFp16(OpTest):
    def setUp(self):
        self.op_type = "scale"
        self.python_api = paddle.scale
        self.place = paddle.CustomPlace("sdaa", 0)
        self.dtype = np.float16
        self.scale = -2.3
        self.x_fp16 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.attrs = {"scale": self.scale, "bias": 0.0, "bias_after_scale": False}
        self.inputs = {"X": self.x_fp16}
        self.outputs = {"Out": self.x_fp16 * self.scale}

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-2)


class TestScaleOpDouble(TestScale):
    def init_dtype(self):
        self.dtype = np.double


class TestScaleOpUint8(TestScale):
    def init_dtype(self):
        self.dtype = np.uint8


class TestScaleOpInt8(TestScale):
    def init_dtype(self):
        self.dtype = np.int8


class TestScaleOpInt16(TestScale):
    def init_dtype(self):
        self.dtype = np.int16


class TestScaleOpInt32(TestScale):
    def init_dtype(self):
        self.dtype = np.int32


class TestScaleOpInt64(TestScale):
    def init_dtype(self):
        self.dtype = np.int64


class TestScaleInplaceApiDygraphFp16(TestScaleInplaceApiDygraph):
    def init_dtype(self):
        self.dtype = np.float16


class TestScaleInplaceApiDygraphDouble(TestScaleInplaceApiDygraph):
    def init_dtype(self):
        self.dtype = np.double


class TestScaleInplaceApiDygraphUint8(TestScaleInplaceApiDygraph):
    def init_dtype(self):
        self.dtype = np.uint8


class TestScaleInplaceApiDygraphInt8(TestScaleInplaceApiDygraph):
    def init_dtype(self):
        self.dtype = np.int8


class TestScaleInplaceApiDygraphInt16(TestScaleInplaceApiDygraph):
    def init_dtype(self):
        self.dtype = np.int16


class TestScaleInplaceApiDygraphInt32(TestScaleInplaceApiDygraph):
    def init_dtype(self):
        self.dtype = np.int32


class TestScaleInplaceApiDygraphInt64(TestScaleInplaceApiDygraph):
    def init_dtype(self):
        self.dtype = np.int64


class TestScaleOpInf(TestScale):
    def init_input(self):
        self.scale = np.inf
        self.bias = 1.0
        self.bias_after_scale = False
        self.x = np.random.random((2, 3, 4)).astype(self.dtype)


class TestScaleOpNegInf(TestScale):
    def init_input(self):
        self.scale = -np.inf
        self.bias = 1.0
        self.bias_after_scale = False
        self.x = np.random.random((2, 3, 4)).astype(self.dtype)


class TestBiasAfterScale(TestScale):
    def init_input(self):
        self.scale = 2.0
        self.bias = 1.0
        self.bias_after_scale = True
        self.x = np.random.random((2, 3, 4)).astype(self.dtype)


class TestBiasAfterScaleFp16(TestBiasAfterScale):
    def init_dtype(self):
        self.dtype = np.float16


class TestBiasAfterScaleDouble(TestBiasAfterScale):
    def init_dtype(self):
        self.dtype = np.double


class TestBiasAfterScaleUint8(TestBiasAfterScale):
    def init_dtype(self):
        self.dtype = np.uint8


class TestBiasAfterScaleInt8(TestBiasAfterScale):
    def init_dtype(self):
        self.dtype = np.int8


class TestBiasAfterScaleInt16(TestBiasAfterScale):
    def init_dtype(self):
        self.dtype = np.int16


class TestBiasAfterScaleInt32(TestBiasAfterScale):
    def init_dtype(self):
        self.dtype = np.int32


class TestBiasAfterScaleInt64(TestBiasAfterScale):
    def init_dtype(self):
        self.dtype = np.int64


class TestScaleOpZeroNumelVariable(unittest.TestCase):
    def test_check_zero_numel(self):
        paddle.set_device("sdaa")
        data = paddle.ones([0, 1])
        out = paddle.scale(data, 2)
        self.assertEqual(out, data)


class TestScaleRaiseError(unittest.TestCase):
    def test_errors(self):
        def test_type():
            paddle.set_device("sdaa")
            paddle.scale([10])

        self.assertRaises(TypeError, test_type)


if __name__ == "__main__":
    unittest.main()
