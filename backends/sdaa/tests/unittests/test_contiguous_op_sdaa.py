# BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
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
import unittest
import numpy as np


class TestContiguous(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.input = np.random.random([3, 3]).astype(self.dtype)

    def run_test(self, use_sdaa):
        if not use_sdaa:
            paddle.set_device("cpu")
        else:
            paddle.set_device("sdaa")

        x = paddle.to_tensor(self.input)
        assert x.is_contiguous()

        y = paddle.as_strided(x, [3, 3], [1, 3])
        assert not y.is_contiguous()

        z = y.contiguous()
        assert z.is_contiguous()
        return x.strides, y, y.strides, z, z.strides

    def test_contiguous(self):
        (
            input_strides,
            output0,
            output0_strides,
            output1,
            output1_strides,
        ) = self.run_test(False)
        (
            input_strides_sdaa,
            output0_sdaa,
            output0_strides_sdaa,
            output1_sdaa,
            output1_strides_sdaa,
        ) = self.run_test(True)

        np.testing.assert_allclose(input_strides, input_strides_sdaa)
        np.testing.assert_allclose(output0, output0_sdaa)
        np.testing.assert_allclose(output0_strides, output0_strides_sdaa)
        np.testing.assert_allclose(output1, output1_sdaa)
        np.testing.assert_allclose(output1_strides, output1_strides_sdaa)


class TestContiguousDouble(TestContiguous):
    def setUp(self):
        self.dtype = "double"
        self.input = np.random.random([3, 3]).astype(self.dtype)


class TestContiguousFP16(TestContiguous):
    def setUp(self):
        self.dtype = "float16"
        self.input = np.random.random([3, 3]).astype(self.dtype)


class TestContiguousBOOL(TestContiguous):
    def setUp(self):
        self.dtype = "bool"
        self.input = np.random.random([3, 3]).astype(self.dtype)


class TestContiguousINT32(TestContiguous):
    def setUp(self):
        self.dtype = "int32"
        self.input = np.random.random([3, 3]).astype(self.dtype)


class TestContiguousINT64(TestContiguous):
    def setUp(self):
        self.dtype = "int64"
        self.input = np.random.random([3, 3]).astype(self.dtype)


if __name__ == "__main__":
    unittest.main()
