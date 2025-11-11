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

import numpy as np
import unittest
from op_test import OpTest
import paddle

paddle.enable_static()
SEED = 2021


def ref_hardsigmoid(x, slope=0.166666666666667, offset=0.5):
    return np.maximum(np.minimum(x * slope + offset, 1.0), 0.0).astype(x.dtype)


class TestSDAAHardSigmoid(OpTest):
    def setUp(self):
        paddle.enable_static()

        self.op_type = "hard_sigmoid"
        self.python_api = paddle.nn.functional.hardsigmoid
        self.set_sdaa()
        self.init_dtype()
        self.set_attrs()

        x = np.random.uniform(-5, 5, [10, 12]).astype(self.dtype)
        lower_threshold = -self.offset / self.slope
        upper_threshold = (1.0 - self.offset) / self.slope

        # Same reason as TestAbs
        delta = 0.005
        x[np.abs(x - lower_threshold) < delta] = lower_threshold - 0.02
        x[np.abs(x - upper_threshold) < delta] = upper_threshold - 0.02

        out = ref_hardsigmoid(x, self.slope, self.offset)

        self.attrs = {"slope": self.slope, "offset": self.offset}
        self.inputs = {"X": x}
        self.outputs = {"Out": out}

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def init_dtype(self):
        self.dtype = np.float32

    def set_attrs(self):
        self.slope = 0.166666666666667
        self.offset = 0.5


class TestSDAAHardSigmoid2(TestSDAAHardSigmoid):
    def set_attrs(self):
        self.slope = 0.2
        self.offset = 0.5


class TestSDAAHardSigmoid3(TestSDAAHardSigmoid):
    def set_attrs(self):
        self.slope = 0.2
        self.offset = 0.4


class TestSDAAHardSigmoidFp16(TestSDAAHardSigmoid):
    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)

    def init_dtype(self):
        self.dtype = np.float16


if __name__ == "__main__":
    unittest.main()
