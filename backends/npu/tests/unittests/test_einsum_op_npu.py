#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest

from tests.op_test import OpTest
import paddle

SEED = 2023

paddle.enable_static()


def einsum_wrapper(a, b):
    if not isinstance(a, list):
        a = [a]
    ret = paddle._C_ops.einsum(a, b)
    # ret include list: [Tensor(Not initialized)], skip the list
    return ret[0]


class TestEinsumBinary(OpTest):
    def setUp(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)
        self.op_type = "einsum"
        self.python_api = einsum_wrapper
        self.python_out_sig = ["Out"]
        self.disable = False
        self.init_dtype()
        self.set_mandatory()
        self.init_input()
        np.random.seed(123)
        out = np.einsum(self.equation, *self.inputs)
        self.operands = []
        for idx, inp in enumerate(self.inputs):
            self.operands.append(("x" + str(idx), inp))
        self.inputs = {"Operands": self.operands}
        self.attrs = {"equation": self.equation}
        self.outputs = {
            "Out": out,
            "InnerCache": [
                ("cache_" + str(i), np.array([1.0])) for i in range(len(self.operands))
            ],
            "XShape": [
                ("xshape_" + str(i), np.array([1.0])) for i in range(len(self.operands))
            ],
        }

    def init_dtype(self):
        self.dtype = np.float16

    def init_input(self):
        self.inputs = []
        for t, s in zip(self.types, self.shapes):
            input_data = np.random.random(s).astype(t)
            self.inputs.append(input_data)

    def set_mandatory(self):
        self.shapes = [(10, 10, 20), (20, 6)]
        self.types = [self.dtype, self.dtype]
        self.equation = "mij,jk->ki"

    def test_check_output(self):
        if not self.disable:
            self.check_output_with_place(
                self.place, atol=1e-3, rtol=2e-2, no_check_set=["InnerCache", "XShape"]
            )

    def test_grad(self):
        if not self.disable:
            self.check_grad_with_place(
                self.place, [op[0] for op in self.operands], ["Out"]
            )


class TestEinsum1(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(20, 3, 3), (20, 3, 3)]
        self.types = [self.dtype, self.dtype]
        self.equation = "mij,mjk->mik"


class TestEinsum2(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(20, 3, 3), (20, 3, 3)]
        self.types = [self.dtype, self.dtype]
        self.equation = "mij,mjk->ikm"


class TestEinsum3(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 10), (10, 10)]
        self.types = [self.dtype, self.dtype]
        self.equation = "ij,jk->ik"  # }}}


class TestEinsumWithReduction(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 3, 5), (5, 30)]
        self.types = [self.dtype, self.dtype]
        self.equation = "ijk,kl->jl"


class TestEinsumWithReduction1(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 3, 3, 5), (10, 5, 10, 10)]
        self.types = [self.dtype, self.dtype]
        self.equation = "mijk,mklh->ljm"


class TestEinsumWithUnary(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 10, 3, 5)]
        self.types = [self.dtype]
        self.equation = "mijk->mi"


class TestEinsumWithUnary1(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(5, 10, 3, 3), (3, 6, 3, 10)]
        self.types = [self.dtype, self.dtype]
        self.equation = "imjl,jklm->imk"


class TestEinsumWithBroadcast1(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(5, 10, 3, 3)]
        self.types = [self.dtype]
        self.equation = "i...->..."


class TestEinsumWithBroadcast2(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 11), (3, 4, 5, 10)]
        self.types = [self.dtype, self.dtype]
        self.equation = "...ij,...i->j..."


# class TestEinsumWithBroadcast3(TestEinsumBinary):
#     def set_mandatory(self):
#         self.shapes = [(10, 3, 2, 3, 4), (12, 10)]
#         self.types = [self.dtype, self.dtype]
#         self.equation = "k...,...jk->...k"


# class TestEinsumWithBroadcast4(TestEinsumBinary):
#     def set_mandatory(self):
#         self.shapes = [(10, 3, 2, 3, 4), (12, 10)]
#         self.types = [self.dtype, self.dtype]
#         self.equation = "a...d,...cb->...abcd"


class TestEinsumWithBroadcast5(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(3, 2, 2, 10), (10, 3, 2, 2)]
        self.types = [self.dtype, self.dtype]
        self.equation = "...a,a...->..."


class TestEinsumWithBroadcast6(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(100), (100)]
        self.types = [self.dtype, self.dtype]
        self.equation = "i,i->"


class TestEinsumWithDiagonal(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 10)]
        self.types = [self.dtype]
        self.equation = "ii->"


class TestEinsumWithDiagonal2(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 3, 10)]
        self.types = [self.dtype]
        self.equation = "iji->j"


class TestEinsumWithDiagonal3(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(5, 3, 2, 1, 4, 5)]
        self.types = [self.dtype]
        self.equation = "a...a->..."


class TestEinsumWithDiagonal4(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(5, 3, 2, 1, 4, 5)]
        self.types = [self.dtype]
        self.equation = "a...a->a..."


class TestEinsumWithDiagonal5(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(8, 8, 8)]
        self.types = [self.dtype]
        self.equation = "aaa->a"


class TestEinsumWithDiagonal6(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(3, 5, 7, 3), (5, 7, 5, 7)]
        self.types = [self.dtype, self.dtype]
        self.equation = "ijki,jkjk->ik"


class TestEinsumWithDiagonal8(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(3, 5, 7, 3), (5, 7, 5, 7)]
        self.types = [self.dtype, self.dtype]
        self.equation = "ijki,jkjk->"


class TestEinsumFP32Op(TestEinsumBinary):
    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        if not self.disable:
            self.check_output_with_place(
                self.place, atol=2e-2, rtol=1e-3, no_check_set=["InnerCache", "XShape"]
            )

    def test_grad(self):
        if not self.disable:
            self.check_grad_with_place(
                self.place,
                [op[0] for op in self.operands],
                ["Out"],
                numeric_place=paddle.CPUPlace(),
                max_relative_error=3e-2,
            )


if __name__ == "__main__":
    unittest.main()
