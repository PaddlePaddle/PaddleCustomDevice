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

import paddle


def fill_inplace(x, value):
    paddle._C_ops.fill_(x, value)


class TestFillOp(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.shape = [2, 3, 4, 5]
        self.value = 0.0
        self.atol = 1e-6
        self.rtol = 1e-6
        self.init()

        self.inputs = np.random.random(self.shape).astype(self.dtype)

    def init(self):
        paddle.set_device("sdaa:0")

    def test_fill_inplace(self):
        self.inputs.fill(self.value)

        sdaa_inputs = paddle.to_tensor(self.inputs)
        fill_inplace(sdaa_inputs, self.value)
        np.testing.assert_allclose(
            sdaa_inputs.numpy(), self.inputs, atol=self.atol, rtol=self.rtol
        )


class TestFillOpInt8(TestFillOp):
    def init(self):
        self.dtype = np.int8
        self.value = 8


class TestFillOpInt16(TestFillOp):
    def init(self):
        self.dtype = np.int16
        self.shape = [2, 7, 1, 5]
        self.value = 16


class TestFillOpInt32(TestFillOp):
    def init(self):
        self.dtype = np.int32
        self.value = 32


class TestFillOpFloat16(TestFillOp):
    def init(self):
        self.dtype = np.float16
        self.value = 0.121
        self.atol = 1e-3
        self.rtol = 1e-3


class TestFillOpBool(TestFillOp):
    def init(self):
        self.dtype = bool
        self.value = True


class TestFillOpValue(TestFillOp):
    def init(self):
        self.value = 123.66


if __name__ == "__main__":
    unittest.main()
