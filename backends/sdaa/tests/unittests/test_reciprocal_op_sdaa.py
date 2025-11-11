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

import numpy as np
import unittest

from op_test import OpTest
import paddle

paddle.enable_static()
SEED = 1024


class TestReciprocal(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "reciprocal"
        self.python_api = paddle.reciprocal
        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_dtype()
        self.init_shape()

        np.random.seed(SEED)
        self.x = np.random.uniform(1, 2, self.shape).astype(self.dtype)
        self.out = np.reciprocal(self.x)

        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(self.x)}
        self.outputs = {"Out": self.out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [11, 17]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        dx = np.divide(np.ones_like(self.inputs, dtype=self.dtype), self.x.size)
        dx = -dx * self.out * self.out
        user_defined_grad = [dx]
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            max_relative_error=0.01,
            user_defined_grads=user_defined_grad,
        )


class TestReciprocalShape1(TestReciprocal):
    def init_shape(self):
        self.shape = [11, 12, 8, 6]


class TestReciprocalShape2(TestReciprocal):
    def init_shape(self):
        self.shape = [120]


class TestReciprocalFp16(TestReciprocal):
    def init_dtype(self):
        self.dtype = np.float16


class TestReciprocalDouble(TestReciprocal):
    def init_dtype(self):
        self.dtype = np.double

    def test_check_grad(self):
        pass


if __name__ == "__main__":
    unittest.main()
