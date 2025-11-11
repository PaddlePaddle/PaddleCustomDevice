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

import unittest
import numpy as np

from op_test import OpTest
import paddle

paddle.enable_static()


class TestEmpty(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.init_dtype()
        self.op_type = "is_empty"
        self.set_data()

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def set_data(self):
        self.inputs = {"X": np.array([1, 2, 3]).astype(self.dtype)}
        self.outputs = {"Out": np.array(False)}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestNotEmpty(TestEmpty):
    def set_data(self):
        self.inputs = {"X": np.array([])}
        self.outputs = {"Out": np.array(True)}


class TestIsEmptyOpError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            input_data = np.random.random((3, 2)).astype("float32")

            def test_Variable():
                # the input type must be Variable
                paddle.is_empty(x=input_data)

            self.assertRaises(TypeError, test_Variable)

            def test_type():
                x3 = paddle.static.data(name="x3", shape=[4, 32, 32], dtype="bool")
                res = paddle.is_empty(x=x3)

            self.assertRaises(TypeError, test_type)

            def test_name_type():
                # name type must be string.
                x4 = paddle.static.data(name="x4", shape=[3, 2], dtype="float32")
                res = paddle.is_empty(x=x4, name=1)

            self.assertRaises(TypeError, test_name_type)


class TestIsEmptyOpDygraph(unittest.TestCase):
    def test_dygraph(self):
        paddle.disable_static(paddle.CustomPlace("sdaa", 0))
        input = paddle.rand(shape=[4, 32, 32], dtype="float32")
        res = paddle.is_empty(x=input)


if __name__ == "__main__":
    unittest.main()
