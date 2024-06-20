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
SEED = 2021


class SDAAOpTest(OpTest):
    def set_plugin(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("sdaa", 0)


class TestCase1(SDAAOpTest):
    def setUp(self):
        self.set_plugin()
        self.init_dtype()
        self.set_example()
        self.op_type = "split"
        self.python_api = paddle.split
        self.python_out_sig = ["out0", "out1"]
        ipt = self.x.astype(self.dtype)
        axis = self.axis if isinstance(self.axis, int) else int(self.axis[0])
        tmp_outs = np.split(ipt, axis=axis, indices_or_sections=self.num_or_sections)
        tmp_outs = [o.astype(self.dtype) for o in tmp_outs]
        self.outputs = {"Out": []}
        self.outs = []
        for i, o in enumerate(tmp_outs):
            self.outputs["Out"].append((str(i), o))
            self.outs.append(str(i))

        self.attrs = {"axis": self.axis, "num": self.num_or_sections}
        self.inputs = {}
        self.inputs.update({"X": ipt.astype(self.dtype)})

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_example(self):
        self.x = np.random.random((2, 4, 6)).astype(self.dtype)
        self.axis = 1
        self.num_or_sections = 2


class TestCase2(TestCase1):
    def set_example(self):
        self.x = np.random.random((20, 4, 50)).astype(self.dtype)
        self.axis = 0
        self.num_or_sections = 4

    def setUp(self):
        super().setUp()
        self.python_out_sig = ["out0", "out1", "out2", "out3"]


class TestCase3(TestCase1):
    def set_example(self):
        self.x = np.random.random((4, 50, 20)).astype(self.dtype)
        self.axis = 2
        self.num_or_sections = 4

    def setUp(self):
        super().setUp()
        self.python_out_sig = ["out0", "out1", "out2", "out3"]


# Test Sections
class TestCase4(TestCase1):
    def set_example(self):
        super().set_example()
        self.x = np.random.random((2, 10, 4)).astype(self.dtype)
        self.axis = 1
        self.num_or_sections = [2, 4, 8]

    def setUp(self):
        super().setUp()
        self.attrs.update({"sections": [2, 2, 4, 2], "num": 0})
        self.python_out_sig = ["out0", "out1", "out2", "out3"]


class TestCase5(TestCase1):
    def set_example(self):
        self.x = np.random.random((20, 4, 50)).astype(self.dtype)
        self.axis = 0
        self.num_or_sections = 1

    def setUp(self):
        super().setUp()
        self.python_out_sig = ["out0"]


# ----------------Split Fp16----------------
def create_test_fp16(parent):
    class TestSplitFp16(parent):
        def init_dtype(self):
            self.dtype = np.float16

        def test_check_grad(self):
            pass

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestSplitFp16.__name__ = cls_name
    globals()[cls_name] = TestSplitFp16


create_test_fp16(TestCase1)
create_test_fp16(TestCase2)
create_test_fp16(TestCase3)
create_test_fp16(TestCase4)


# ----------------Split Int64----------------
def create_test_int64(parent):
    class TestSplitInt64(parent):
        def init_dtype(self):
            self.dtype = np.int64

        def test_check_grad(self):
            pass

    cls_name = "{0}_{1}".format(parent.__name__, "Int64")
    TestSplitInt64.__name__ = cls_name
    globals()[cls_name] = TestSplitInt64


create_test_int64(TestCase1)
create_test_int64(TestCase2)
create_test_int64(TestCase3)
create_test_int64(TestCase4)


# ----------------Split Int32----------------
def create_test_int32(parent):
    class TestSplitInt32(parent):
        def init_dtype(self):
            self.dtype = np.int32

        def test_check_grad(self):
            pass

    cls_name = "{0}_{1}".format(parent.__name__, "Int32")
    TestSplitInt32.__name__ = cls_name
    globals()[cls_name] = TestSplitInt32


create_test_int32(TestCase1)
create_test_int32(TestCase2)
create_test_int32(TestCase3)
create_test_int32(TestCase4)


# ----------------Split Bool----------------
def create_test_bool(parent):
    class TestSplitBool(parent):
        def init_dtype(self):
            self.dtype = np.bool_

        def test_check_grad(self):
            pass

    cls_name = "{0}_{1}".format(parent.__name__, "Bool")
    TestSplitBool.__name__ = cls_name
    globals()[cls_name] = TestSplitBool


create_test_bool(TestCase1)
create_test_bool(TestCase2)
create_test_bool(TestCase3)
create_test_bool(TestCase4)


# ----------------Split Uint8----------------
def create_test_uint8(parent):
    class TestSplitUint8(parent):
        def init_dtype(self):
            self.dtype = np.uint8

        def test_check_grad(self):
            pass

    cls_name = "{0}_{1}".format(parent.__name__, "Uint8")
    TestSplitUint8.__name__ = cls_name
    globals()[cls_name] = TestSplitUint8


create_test_uint8(TestCase1)
create_test_uint8(TestCase2)
create_test_uint8(TestCase3)
create_test_uint8(TestCase4)


# attr(axis) is Tensor
class TestSplitOp_AxisTensor(SDAAOpTest):
    def setUp(self):
        self.set_plugin()
        self._set_op_type()
        self.python_api = paddle.split
        self.python_out_sig = ["out0", "out1", "out2"]
        self.dtype = self.get_dtype()
        self.init_data()
        self.inputs = {"X": self.x, "AxisTensor": np.array([self.axis]).astype("int32")}
        self.attrs = {"sections": self.sections, "num": self.num}

        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {"Out": [("out%d" % i, out[i]) for i in range(len(out))]}

    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype(self.dtype)
        self.axis = 2
        self.sections = []
        self.num = 3
        self.indices_or_sections = 3

    def get_dtype(self):
        return np.float32

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestSplitOp_SectionsTensor(SDAAOpTest):
    def setUp(self):
        self.set_plugin()
        self._set_op_type()
        self.python_api = paddle.split
        self.python_out_sig = ["out0", "out1", "out2"]
        self.dtype = self.get_dtype()
        self.init_data()
        self.inputs = {"X": self.x}

        sections_tensor = []
        for index, ele in enumerate(self.sections):
            sections_tensor.append(
                ("x" + str(index), np.ones((1)).astype("int32") * ele)
            )

        self.inputs["SectionsTensorList"] = sections_tensor

        self.attrs = {
            "axis": self.axis,
            "sections": self.sections_infer,
            "num": self.num,
        }

        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {"Out": [("out%d" % i, out[i]) for i in range(len(out))]}

    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype(self.dtype)
        self.axis = 1
        self.sections = [2, 1, 2]
        self.sections_infer = [-1, -1, -1]
        self.num = 0
        self.indices_or_sections = [2, 3]

    def get_dtype(self):
        return np.float32

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == "__main__":
    unittest.main()
