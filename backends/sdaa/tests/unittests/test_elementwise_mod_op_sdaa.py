# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import random
import unittest

import numpy as np
from op_test import OpTest

import paddle
from paddle import base


class TestElementwiseModOp(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "elementwise_mod"
        self.python_api = paddle.remainder
        self.axis = -1
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(self.x),
            "Y": OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.attrs = {"axis": self.axis, "use_mkldnn": self.use_mkldnn}
        self.outputs = {"Out": self.out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def init_input_output(self):
        self.x = np.random.uniform(1, 10000, [10, 10]).astype(self.dtype)
        self.y = np.random.uniform(1, 1000, [10, 10]).astype(self.dtype)
        self.out = np.mod(self.x, self.y)

    def init_dtype(self):
        self.dtype = np.int32

    def init_kernel_type(self):
        self.use_mkldnn = False

    def init_axis(self):
        pass


class TestElementwiseModOp_ZeroDim1(TestElementwiseModOp):
    def init_input_output(self):
        self.x = np.random.uniform(1, 10000, []).astype(self.dtype)
        self.y = np.random.uniform(1, 1000, []).astype(self.dtype)
        self.out = np.mod(self.x, self.y)


class TestElementwiseModOp_ZeroDim2(TestElementwiseModOp):
    def init_input_output(self):
        self.x = np.random.uniform(1, 10000, [10, 10]).astype(self.dtype)
        self.y = np.random.uniform(1, 1000, []).astype(self.dtype)
        self.out = np.mod(self.x, self.y)


class TestElementwiseModOp_ZeroDim3(TestElementwiseModOp):
    def init_input_output(self):
        self.x = np.random.uniform(1, 10000, []).astype(self.dtype)
        self.y = np.random.uniform(1, 1000, [10, 10]).astype(self.dtype)
        self.out = np.mod(self.x, self.y)


@unittest.skip(
    "Intermediate calculations of tecodnnremainder would exceed Int32 range, to be fixed by tecodnn."
)
class TestElementwiseModOp_scalar(TestElementwiseModOp):
    def init_input_output(self):
        scale_x = random.randint(0, 10000000)
        scale_y = random.randint(1, 10000000)
        self.x = (np.random.uniform(1, 100, [2, 3, 4]) * scale_x).astype(self.dtype)
        self.y = (np.random.uniform(1, 10, [1]) * scale_y + 1).astype(self.dtype)
        self.out = np.mod(self.x, self.y)


class TestElementwiseModOp_broadcast_0(TestElementwiseModOp):
    def init_input_output(self):
        self.x = np.random.uniform(1, 100, [100, 2, 3]).astype(self.dtype)
        self.y = np.random.uniform(1, 1000, [100, 1, 1]).astype(self.dtype)
        self.out = np.mod(self.x, self.y.reshape(100, 1, 1))


class TestElementwiseModOp_broadcast_1(TestElementwiseModOp):
    def init_input_output(self):
        self.x = np.random.uniform(1, 1000, [2, 100, 3]).astype(self.dtype)
        self.y = np.random.uniform(1, 10000, [1, 100, 1]).astype(self.dtype)
        self.out = np.mod(self.x, self.y.reshape(1, 100, 1))


class TestElementwiseModOp_broadcast_2(TestElementwiseModOp):
    def init_input_output(self):
        self.x = np.random.uniform(1, 10, [2, 3, 100]).astype(self.dtype)
        self.y = np.random.uniform(1, 100, [1, 1, 100]).astype(self.dtype)
        self.out = np.mod(self.x, self.y.reshape(1, 1, 100))


class TestElementwiseModOpFloat(TestElementwiseModOp):
    def init_dtype(self):
        self.dtype = np.float32

    def init_input_output(self):
        self.x = np.random.uniform(-1000, 1000, [10, 10]).astype(self.dtype)
        self.y = np.random.uniform(-100, 100, [10, 10]).astype(self.dtype)
        self.out = np.fmod(self.y + np.fmod(self.x, self.y), self.y)

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestElementwiseModFP16Op(TestElementwiseModOp):
    def init_dtype(self):
        self.dtype = np.float16

    def init_input_output(self):
        self.x = np.random.uniform(-1000, 1000, [10, 10]).astype(self.dtype)
        self.y = np.random.uniform(-100, 100, [10, 10]).astype(self.dtype)
        self.out = np.fmod(self.y + np.fmod(self.x, self.y), self.y)

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestElementwiseModFP16Op_ZeroDim1(TestElementwiseModFP16Op):
    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, []).astype(np.float16)
        self.y = np.random.uniform(0, 1000, []).astype(np.float16)
        self.out = np.fmod(self.y + np.fmod(self.x, self.y), self.y)


class TestElementwiseModFP16Op_ZeroDim2(TestElementwiseModFP16Op):
    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, [10, 10]).astype(np.float16)
        self.y = np.random.uniform(0, 1000, []).astype(np.float16)
        self.out = np.fmod(self.y + np.fmod(self.x, self.y), self.y)


class TestElementwiseModFP16Op_ZeroDim3(TestElementwiseModFP16Op):
    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, []).astype(np.float16)
        self.y = np.random.uniform(0, 1000, [10, 10]).astype(np.float16)
        self.out = np.fmod(self.y + np.fmod(self.x, self.y), self.y)


class TestRemainderOp(unittest.TestCase):
    def _executed_api(self, x, y, name=None):
        return paddle.remainder(x, y, name)

    def test_name(self):
        paddle.set_device("sdaa")
        with base.program_guard(base.Program()):
            x = paddle.static.data(name="x", shape=[2, 3], dtype="int32")
            y = paddle.static.data(name="y", shape=[2, 3], dtype="int32")

            y_1 = self._executed_api(x, y, name="div_res")
            self.assertEqual(("div_res" in y_1.name), True)

    def test_dygraph(self):
        paddle.set_device("sdaa")
        with base.dygraph.guard():
            np_x = np.array([2, 3, 8, 7]).astype("int32")
            np_y = np.array([1, 5, 3, 3]).astype("int32")
            x = paddle.to_tensor(np_x)
            y = paddle.to_tensor(np_y)
            z = self._executed_api(x, y)
            np_z = z.numpy()
            z_expected = np.array([0, 3, 2, 1])
            self.assertEqual((np_z == z_expected).all(), True)

            np_x = np.array([-3.3, 11.5, -2, 3.5])
            np_y = np.array([-1.2, 2.0, 3.3, -2.3])
            x = paddle.to_tensor(np_x)
            y = paddle.to_tensor(np_y)
            z = x % y
            z_expected = np.array([-0.9, 1.5, 1.3, -1.1])
            np.testing.assert_allclose(z_expected, z.numpy(), rtol=1e-05)

            np_x = np.array([-3, 11, -2, 3])
            np_y = np.array([-1, 2, 3, -2])
            x = paddle.to_tensor(np_x, dtype="int64")
            y = paddle.to_tensor(np_y, dtype="int64")
            z = x % y
            z_expected = np.array([0, 1, 1, -1])
            np.testing.assert_allclose(z_expected, z.numpy(), rtol=1e-05)


class TestRemainderInplaceOp(TestRemainderOp):
    def _executed_api(self, x, y, name=None):
        return x.remainder_(y, name)


class TestRemainderInplaceBroadcastSuccess(unittest.TestCase):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 3, 4).astype("float")
        self.y_numpy = np.random.rand(3, 4).astype("float")

    def test_broadcast_success(self):
        paddle.disable_static()
        self.init_data()
        x = paddle.to_tensor(self.x_numpy)
        y = paddle.to_tensor(self.y_numpy)
        inplace_result = x.remainder_(y)
        numpy_result = self.x_numpy % self.y_numpy
        self.assertEqual((inplace_result.numpy() == numpy_result).all(), True)
        paddle.enable_static()


class TestRemainderInplaceBroadcastSuccess2(TestRemainderInplaceBroadcastSuccess):
    def init_data(self):
        self.x_numpy = np.random.rand(1, 2, 3, 1).astype("float")
        self.y_numpy = np.random.rand(3, 1).astype("float")


class TestRemainderInplaceBroadcastSuccess3(TestRemainderInplaceBroadcastSuccess):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 3, 1, 5).astype("float")
        self.y_numpy = np.random.rand(1, 3, 1, 5).astype("float")


if __name__ == "__main__":
    unittest.main()
