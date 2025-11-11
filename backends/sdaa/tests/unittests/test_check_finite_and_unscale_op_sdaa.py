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
import paddle.static.amp.amp_nn as amp_nn

paddle.enable_static()


def check_finite_and_unscale_wrapper(x, scale):
    _, found_inf = amp_nn.check_finite_and_unscale([x], scale)
    return x, found_inf


class TestCheckFiniteAndUnscaleOp(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "check_finite_and_unscale"
        self.python_api = check_finite_and_unscale_wrapper
        self.python_out_sig = ["out0", "FoundInfinite"]
        self.init_dtype()
        x = np.random.random((1024, 1024)).astype(self.dtype)
        scale = np.random.random((1)).astype(self.dtype)

        self.inputs = {"X": [("x0", x)], "Scale": scale}
        self.outputs = {
            "FoundInfinite": np.array([0]),
            "Out": [("out0", x / scale)],
        }

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)


class TestCheckFiniteAndUnscaleOpWithNan(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "check_finite_and_unscale"
        self.python_api = check_finite_and_unscale_wrapper
        self.python_out_sig = ["out0", "FoundInfinite"]
        self.init_dtype()
        x = np.random.random((1024, 1024)).astype(self.dtype)
        x[128][128] = np.nan
        scale = np.random.random((1)).astype(self.dtype)

        self.inputs = {"X": [("x0", x)], "Scale": scale}
        self.outputs = {
            "FoundInfinite": np.array([1]),
            "Out": [("out0", x)],
        }

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        # When input contains nan, do not check the output,
        # since the output may be nondeterministic and will be discarded.
        self.check_output_with_place(
            self.place,
            no_check_set=["Out"],
        )


class TestCheckFiniteAndUnscaleOpWithInf(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "check_finite_and_unscale"
        self.python_api = check_finite_and_unscale_wrapper
        self.python_out_sig = ["out0", "FoundInfinite"]
        self.init_dtype()
        x = np.random.random((1024, 1024)).astype(self.dtype)
        x[128][128] = np.inf
        scale = np.random.random((1)).astype(self.dtype)

        self.inputs = {"X": [("x0", x)], "Scale": scale}
        self.outputs = {
            "FoundInfinite": np.array([1]),
            "Out": [("out0", x)],
        }

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        # When input contains inf, do not check the output,
        # since the output may be nondeterministic and will be discarded.
        self.check_output_with_place(
            self.place,
            no_check_set=["Out"],
        )


if __name__ == "__main__":
    unittest.main()
