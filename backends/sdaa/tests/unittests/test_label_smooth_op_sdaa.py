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
SEED = 1234


class TestLabelSmoothOp(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "label_smooth"
        self.python_api = paddle.nn.functional.label_smooth
        self.place = paddle.CustomPlace("sdaa", 0)

        self.init_dtype()
        np.random.seed(SEED)

        self.set_inputs()
        self.set_attrs()
        self.set_outputs()

    def calc_out(self, label, epsilon, dist=None):
        label_dim = label.shape[-1]
        y = (1 - epsilon) * label
        if dist is not None:
            y += epsilon * dist
        else:
            y += epsilon / label_dim
        return y.astype(self.dtype)

    def set_inputs(self):
        batch_size, label_dim = 64, 102
        x = np.zeros((batch_size, label_dim)).astype(self.dtype)
        nonzero_index = np.random.randint(label_dim, size=(batch_size))
        x[np.arange(batch_size), nonzero_index] = 1
        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(x)}

    def set_attrs(self):
        epsilon = 0.1
        self.attrs = {"epsilon": epsilon}

    def set_outputs(self):
        dist = None if "PriorDist" not in self.inputs else self.inputs["PriorDist"]
        out = self.calc_out(self.inputs["X"], self.attrs["epsilon"], dist)
        self.outputs = {"Out": out}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(self.place, ["X"], "Out")
        else:
            self.check_grad_with_place(self.place, ["X"], "Out")


class TestLabelSmoothOpWithPriorDist(TestLabelSmoothOp):
    def set_inputs(self):
        super(TestLabelSmoothOpWithPriorDist, self).set_inputs()
        label_dim = self.inputs["X"].shape[-1]
        dist = np.random.random((1, label_dim)).astype(self.dtype)
        self.inputs["PriorDist"] = dist


class TestLabelSmoothOp3D(TestLabelSmoothOp):
    def set_inputs(self):
        super(TestLabelSmoothOp3D, self).set_inputs()
        self.inputs["X"].reshape([2, -1, self.inputs["X"].shape[-1]])


class TestLabelSmoothOpWithPriorDist3D(TestLabelSmoothOpWithPriorDist):
    def set_inputs(self):
        super(TestLabelSmoothOpWithPriorDist3D, self).set_inputs()
        self.inputs["X"].reshape([2, -1, self.inputs["X"].shape[-1]])


class TestLabelSmoothOpFP16(TestLabelSmoothOp):
    def init_dtype(self):
        self.dtype = np.float16


class TestLabelSmoothOpWithPriorDistFP16(TestLabelSmoothOpWithPriorDist):
    def init_dtype(self):
        self.dtype = np.float16


class TestLabelSmoothOp3DFP16(TestLabelSmoothOp3D):
    def init_dtype(self):
        self.dtype = np.float16


class TestLabelSmoothOpWithPriorDist3DFP16(TestLabelSmoothOpWithPriorDist3D):
    def init_dtype(self):
        self.dtype = np.float16


if __name__ == "__main__":
    unittest.main()
