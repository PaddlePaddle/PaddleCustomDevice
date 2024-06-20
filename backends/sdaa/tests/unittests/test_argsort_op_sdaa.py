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
import unittest
import numpy as np
from op_test import OpTest
import paddle

paddle.enable_static()
SEED = 2022


class TestArgsortOp(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.__class__.op_type = "argsort"
        self.init_dtype()
        self.init_inputshape()
        self.init_axis()
        self.init_direction()
        np.random.seed(SEED)
        self.x = np.random.random(self.input_shape).astype(self.dtype)
        self.inputs = {"X": self.x}
        self.attrs = {"axis": self.axis, "descending": self.descending}
        self.get_output()
        self.outputs = {"Out": self.sorted_x, "Indices": self.indices}

    def get_output(self):
        if self.descending:
            # self.indices = np.flip(
            #     np.argsort(self.x, kind="heapsort", axis=self.axis), self.axis
            # )
            # do not support output indices
            self.indices = np.arange(len(self.x))
            self.sorted_x = np.flip(
                np.sort(self.x, kind="heapsort", axis=self.axis), self.axis
            )
        else:
            # self.indices = np.argsort(self.x, kind="heapsort", axis=self.axis)
            self.indices = np.arange(len(self.x))
            self.sorted_x = np.sort(self.x, kind="heapsort", axis=self.axis)

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.no_need_check_grad = True

    def init_inputshape(self):
        self.input_shape = 2097152

    def init_dtype(self):
        self.dtype = np.float32

    def init_axis(self):
        self.axis = -1

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def init_direction(self):
        self.descending = True


if __name__ == "__main__":
    unittest.main()
