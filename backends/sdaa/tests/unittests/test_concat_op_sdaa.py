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
from op_test import OpTest, skip_check_grad_ci
import paddle
from white_list import no_grad_set_white_list

no_grad_set_white_list.NEED_TO_FIX_OP_LIST.append("concat")

paddle.enable_static()
SEED = 2021


class TestConcatOp(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "concat"
        self.python_api = paddle.concat
        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_dtype()
        self.init_test_data()

        self.inputs = {"X": [("x0", self.x0), ("x1", self.x1), ("x2", self.x2)]}
        self.attrs = {"axis": self.axis}
        if self.axis < 0:
            self.actual_axis = self.axis + len(self.x0.shape)
            self.actual_axis = self.actual_axis if self.actual_axis > 0 else 0
        else:
            self.actual_axis = self.axis

        self.outputs = {
            "Out": np.concatenate((self.x0, self.x1, self.x2), axis=self.actual_axis)
        }

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_test_data(self):
        self.x0 = np.random.random((1, 4, 50)).astype(self.dtype)
        self.x1 = np.random.random((2, 4, 50)).astype(self.dtype)
        self.x2 = np.random.random((3, 4, 50)).astype(self.dtype)
        self.axis = 0

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["x0", "x2"],
            "Out",
        )
        self.check_grad_with_place(self.place, ["x1"], "Out")
        self.check_grad_with_place(self.place, ["x2"], "Out")


class TestConcatOp2(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x1 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x2 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.axis = 1


@skip_check_grad_ci(reason="The function 'check_grad' for large inputs is too slow.")
class TestConcatOp3(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((1, 256, 170, 256)).astype(self.dtype)
        self.x1 = np.random.random((1, 128, 170, 256)).astype(self.dtype)
        self.x2 = np.random.random((1, 128, 170, 256)).astype(self.dtype)
        self.axis = 1

    def test_check_grad(self):
        pass


@unittest.skip("0D tensor allocate memory failed in sdaa.")
class TestConcatOp4(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x1 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x2 = np.random.random((0, 3, 4, 5)).astype(self.dtype)
        self.axis = 0


class TestConcatOp5(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((5, 1, 4, 5)).astype(self.dtype)
        self.x1 = np.random.random((5, 2, 4, 5)).astype(self.dtype)
        self.x2 = np.random.random((5, 3, 4, 5)).astype(self.dtype)
        self.axis = -3


class TestConcatOp6(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x1 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x2 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.axis = 0

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["x0", "x2"], "Out", no_grad_set=set(["x1"])
        )
        self.check_grad_with_place(
            self.place, ["x0", "x2"], "Out", no_grad_set=set(["x0"])
        )


# ----------------Concat Fp16----------------
def create_test_fp16(parent):
    class TestConcatFp16(parent):
        def init_dtype(self):
            self.dtype = np.float16

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestConcatFp16.__name__ = cls_name
    globals()[cls_name] = TestConcatFp16


# ----------------Concat Int64----------------
def create_test_int64(parent):
    class TestConcatInt64(parent):
        def init_dtype(self):
            self.dtype = np.int64

        def test_check_grad(self):
            pass

    cls_name = "{0}_{1}".format(parent.__name__, "Int64")
    TestConcatInt64.__name__ = cls_name
    globals()[cls_name] = TestConcatInt64


# ----------------Concat Int32----------------
def create_test_int32(parent):
    class TestConcatInt32(parent):
        def init_dtype(self):
            self.dtype = np.int32

        def test_check_grad(self):
            pass

    cls_name = "{0}_{1}".format(parent.__name__, "Int32")
    TestConcatInt32.__name__ = cls_name
    globals()[cls_name] = TestConcatInt32


# ----------------Concat Bool----------------
def create_test_bool(parent):
    class TestConcatBool(parent):
        def init_dtype(self):
            self.dtype = np.bool_

        def test_check_grad(self):
            pass

    cls_name = "{0}_{1}".format(parent.__name__, "Bool")
    TestConcatBool.__name__ = cls_name
    globals()[cls_name] = TestConcatBool


create_test_fp16(TestConcatOp)
create_test_fp16(TestConcatOp2)
create_test_fp16(TestConcatOp3)
create_test_fp16(TestConcatOp4)
create_test_fp16(TestConcatOp5)
create_test_fp16(TestConcatOp6)

create_test_int32(TestConcatOp)
create_test_int32(TestConcatOp2)
create_test_int32(TestConcatOp3)
create_test_int32(TestConcatOp4)
create_test_int32(TestConcatOp5)
create_test_int32(TestConcatOp6)

create_test_int64(TestConcatOp)
create_test_int64(TestConcatOp2)
create_test_int64(TestConcatOp3)
create_test_int64(TestConcatOp4)
create_test_int64(TestConcatOp6)

create_test_bool(TestConcatOp)
create_test_bool(TestConcatOp2)
create_test_bool(TestConcatOp3)
create_test_bool(TestConcatOp4)
create_test_bool(TestConcatOp5)
create_test_bool(TestConcatOp6)
if __name__ == "__main__":
    unittest.main()
