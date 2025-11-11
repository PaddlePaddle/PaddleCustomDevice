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

from __future__ import print_function
import paddle

import numpy as np
import unittest
from op_test import OpTest

paddle.enable_static()
SEED = 2022


class TestElementwisePowOp(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "elementwise_pow"
        self.python_api = paddle.pow

        self.init_dtype()
        self.init_input_output()
        self.init_axis()

        self.inputs = {
            "X": self.x,
            "Y": self.y,
        }
        self.attrs = {"axis": self.axis}
        self.outputs = {"Out": self.out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def init_axis(self):
        self.axis = -1

    def init_input_output(self):
        np.random.seed(SEED)
        self.x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        self.out = np.power(self.x, self.y)

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestElementwisePowOp1(TestElementwisePowOp):
    def init_input_output(self):
        # for llama-13b
        np.random.seed(SEED)
        self.x = np.random.uniform(1, 2, [64]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [64]).astype(self.dtype)
        self.out = np.power(self.x, self.y)


class TestElementwisePowOp2(TestElementwisePowOp):
    def init_input_output(self):
        # for llama-13b
        np.random.seed(SEED)
        self.x = np.random.uniform(0, 1, [64]).astype(self.dtype)
        self.y = np.random.uniform(0, 1, [64]).astype(self.dtype)
        self.out = np.power(self.x, self.y)


class TestElementwisePowOp3(TestElementwisePowOp):
    def init_input_output(self):
        # for llama-13b
        np.random.seed(SEED)
        self.x = np.random.uniform(0, 0.1, [64]).astype(self.dtype)
        self.y = np.random.uniform(0, 0.1, [64]).astype(self.dtype)
        self.out = np.power(self.x, self.y)


@unittest.skip(
    "accuracy test failure, because tecodnnElementwisePow does not support broadcast."
)
class TestElementwisePowOp_ZeroDim2(TestElementwisePowOp):
    def init_input_output(self):
        np.random.seed(SEED)
        self.x = np.random.uniform(1, 2, [20, 5]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, []).astype(self.dtype)
        self.out = np.power(self.x, self.y)


@unittest.skip("tecodnnElementwisePow only support y broadcast to x.")
class TestElementwisePowOp_ZeroDim3(TestElementwisePowOp):
    def init_input_output(self):
        np.random.seed(SEED)
        self.x = np.random.uniform(1, 2, []).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [20, 5]).astype(self.dtype)
        self.out = np.power(self.x, self.y)


class TestElementwisePowOp_big_shape_1(TestElementwisePowOp):
    def init_input_output(self):
        np.random.seed(SEED)
        self.x = np.random.uniform(1, 2, [10, 10]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [10, 10]).astype(self.dtype)
        self.out = np.power(self.x, self.y)


class TestElementwisePowOp_big_shape_2(TestElementwisePowOp):
    def init_input_output(self):
        np.random.seed(SEED)
        self.x = np.random.uniform(1, 2, [10, 10]).astype(self.dtype)
        self.y = np.random.uniform(0.2, 2, [10, 10]).astype(self.dtype)
        self.out = np.power(self.x, self.y)


@unittest.skip(
    "accuracy test failure, because tecodnnElementwisePow does not support broadcast."
)
class TestElementwisePowOp_scalar(TestElementwisePowOp):
    def init_input_output(self):
        np.random.seed(SEED)
        self.x = np.random.uniform(0.1, 1, [3, 3, 4]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [1]).astype(self.dtype)
        self.out = np.power(self.x, self.y)


class TestElementwisePowOp_tensor(TestElementwisePowOp):
    def init_input_output(self):
        np.random.seed(SEED)
        self.x = np.random.uniform(0.1, 1, [100]).astype(self.dtype)
        self.y = np.random.uniform(1, 3, [100]).astype(self.dtype)
        self.out = np.power(self.x, self.y)


@unittest.skip(
    "accuracy test failure, because tecodnnElementwisePow does not support broadcast."
)
class TestElementwisePowOp_broadcast_0(TestElementwisePowOp):
    def init_input_output(self):
        np.random.seed(SEED)
        self.x = np.random.uniform(0.1, 1, [2, 1, 100]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [100]).astype(self.dtype)
        self.out = np.power(self.x, self.y)


@unittest.skip(
    "accuracy test failure, because tecodnnElementwisePow does not support broadcast."
)
class TestElementwisePowOp_broadcast_4(TestElementwisePowOp):
    def init_input_output(self):
        np.random.seed(SEED)
        self.x = np.random.uniform(0.1, 1, [2, 10, 3, 5]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [2, 10, 1, 5]).astype(self.dtype)
        self.out = np.power(self.x, self.y)


class TestElementwisePowFp16(TestElementwisePowOp):
    def init_input_output(self):
        np.random.seed(SEED)
        self.x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        self.out = np.power(self.x, self.y)

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
