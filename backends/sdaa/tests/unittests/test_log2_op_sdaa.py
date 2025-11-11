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

import unittest
import numpy as np

from op_test import OpTest
import paddle
import paddle.base as base


class TestLog2(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "log2"
        self.python_api = paddle.log2
        self.check_dygraph = True
        self.init_dtype()
        self.init_shape()

        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.log2(x)

        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {"Out": out}

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [11, 17]

    def test_check_output(self):
        check_dygraph = False
        if hasattr(self, "check_dygraph"):
            check_dygraph = self.check_dygraph
        self.check_output_with_place(self.place, check_dygraph=check_dygraph)

    def test_check_grad_with_place(self):
        self.check_grad_with_place(self.place, ["X"], "Out")

    def test_api(self):
        with base.dygraph.guard(self.place):
            np_x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
            data_x = paddle.to_tensor(np_x)
            z = paddle.log2(data_x)
            np_z = z.numpy()
            z_expected = np.array(np.log2(np_x))
        rtol = 1e-5
        if self.dtype == np.float16:
            rtol = 1e-3
        np.testing.assert_allclose(np_z, z_expected, rtol=rtol)


class TestLog2_ZeroDim(TestLog2):
    def init_shape(self):
        self.shape = []


def create_test_fp16_class(parent):
    class TestLog2OpFp16Case(parent):
        def init_dtype(self):
            self.dtype = np.float16

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestLog2OpFp16Case.__name__ = cls_name
    globals()[cls_name] = TestLog2OpFp16Case


create_test_fp16_class(TestLog2)

if __name__ == "__main__":
    unittest.main()
