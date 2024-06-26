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
import sys
import inspect

paddle.enable_static()


class TestTransposeOp(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "transpose2"
        self.python_api = paddle.transpose
        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_dtype()
        self.init_shape_axis()

        self.inputs = {"X": np.random.random(self.shape).astype(self.dtype)}
        self.attrs = {"axis": self.axis, "data_format": "AnyLayout"}
        self.outputs = {"Out": self.inputs["X"].transpose(self.axis)}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_shape_axis(self):
        self.shape = (3, 40)
        self.axis = (1, 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")

    def check_grad_with_place(
        self,
        place,
        inputs_to_check,
        output_names,
        no_grad_set=None,
        numeric_grad_delta=0.005,
        in_place=False,
        max_relative_error=0.005,
        user_defined_grads=None,
        user_defined_grad_outputs=None,
        check_dygraph=True,
        numeric_place=None,
    ):
        if self.dtype == np.float32:
            numeric_place = paddle.CPUPlace()

        super().check_grad_with_place(
            place,
            inputs_to_check,
            output_names,
            no_grad_set,
            numeric_grad_delta,
            in_place,
            max_relative_error,
            user_defined_grads,
            user_defined_grad_outputs,
            check_dygraph,
            numeric_place=numeric_place,
        )


class TestCase0(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (100,)
        self.axis = (0,)


class TestCase1(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (3, 4, 10)
        self.axis = (0, 2, 1)


class TestCase2(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (3, 4, 10)
        self.axis = (1, 0, 2)


class TestCase3(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (3, 4, 10)
        self.axis = (1, 2, 0)


class TestCase4(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 4, 5)
        self.axis = (0, 2, 3, 1)


@unittest.skip("tecodnn not support the dimension size greater than 5.")
class TestCase5(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 4, 5, 6, 1)
        self.axis = (4, 2, 3, 1, 0, 5)


@unittest.skip("tecodnn not support the dimension size greater than 5.")
class TestCase6(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 4, 5, 6, 1)
        self.axis = (4, 2, 3, 1, 0, 5)


class TestCase7(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 10, 12, 16)
        self.axis = (3, 1, 2, 0)


@unittest.skip("tecodnn not support the dimension size greater than 5.")
class TestCase8(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 2, 3, 2, 4, 3, 3)
        self.axis = (0, 1, 3, 2, 4, 5, 6, 7)


@unittest.skip("tecodnn not support the dimension size greater than 5.")
class TestCase9(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 2, 3, 2, 4, 3, 3)
        self.axis = (6, 1, 3, 5, 0, 2, 4, 7)


class TestCase10(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 6, 9, 10)
        self.axis = (0, 1, 3, 4, 2)


class TestCase11(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 4, 5, 8)
        self.axis = (2, 3, 4, 0, 1)


class TestCase12(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 4, 6, 7)
        self.axis = (0, 1, 4, 2, 3)


class TestCase13(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 8, 4, 6, 8)
        self.axis = (4, 2, 3, 0, 1)


class TestCase14(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 9, 4, 6, 8)
        self.axis = (0, 1, 3, 2, 4)


class TestCase16(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 2, 6, 8)
        self.axis = (0, 3, 4, 1, 2)


class TestCase17(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 4, 3, 8)
        self.axis = (1, 2, 3, 4, 0)


class TestCase18(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 4, 2, 8)
        self.axis = (0, 4, 1, 2, 3)


class TestCase19(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 4, 6, 5)
        self.axis = (4, 1, 2, 3, 0)


class TestCase20(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 4, 6, 8)
        self.axis = (0, 3, 1, 2, 4)


class TestCase21(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (5, 3, 4, 6, 8)
        self.axis = (4, 0, 1, 2, 3)


class TestCase22(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (6, 3, 4, 2, 8)
        self.axis = (0, 2, 3, 4, 1)


class TestCase23(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 4, 6, 3)
        self.axis = (0, 2, 3, 1, 4)


class TestCase24(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 4, 6, 2)
        self.axis = (3, 4, 1, 2, 0)


class TestCase25(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 4, 6, 1)
        self.axis = (0, 2, 1, 3, 4)


class TestCase26(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 4, 6, 8)
        self.axis = (3, 4, 0, 1, 2)


class TestCase27(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (3, 40)
        self.axis = (0, 1)


class TestCase28(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (3, 40, 6)
        self.axis = (0, 1, 2)


class TestCase29(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 3, 4, 5)
        self.axis = (0, 1, 2, 3)


class TestCase30(TestTransposeOp):
    def init_shape_axis(self):
        self.shape = (2, 4, 5, 5, 6)
        self.axis = (0, 1, 2, 3, 4)


class TestTransposeOpFP16(OpTest):
    def setUp(self):
        self.init_op_type()
        self.initTestCase()
        self.dtype = np.float16
        self.python_api = paddle.transpose
        self.public_python_api = paddle.transpose
        self.place = paddle.CustomPlace("sdaa", 0)
        x = np.random.random(self.shape).astype(self.dtype)

        self.inputs = {"X": x}
        self.attrs = {
            "axis": list(self.axis),
        }
        self.outputs = {
            "XShape": np.random.random(self.shape).astype(self.dtype),
            "Out": self.inputs["X"].transpose(self.axis),
        }

    def init_op_type(self):
        self.op_type = "transpose2"

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output(self.place, no_check_set=["XShape"])

    def test_check_grad(self):
        self.check_grad(self.place, ["X"], "Out")

    def initTestCase(self):
        self.shape = (3, 40)
        self.axis = (1, 0)


class TestTransposeOpInt64(TestTransposeOp):
    def init_dtype(self):
        self.dtype = np.int64

    def init_shape_axis(self):
        self.shape = (7, 7)
        self.axis = (1, 0)

    def test_check_grad(self):
        pass


# ----------------Transpose Int64----------------
def create_test_int64(parent):
    class TestTransposeInt64(parent):
        def init_dtype(self):
            self.dtype = np.int64

        def test_check_grad(self):
            pass

    cls_name = "{0}_{1}".format(parent.__name__, "Int64")
    TestTransposeInt64.__name__ = cls_name
    globals()[cls_name] = TestTransposeInt64


def create_test_dtype():
    current_module = sys.modules[__name__]
    cls_collections = filter(
        lambda cls_name: cls_name != "OpTest"
        and inspect.isclass(getattr(current_module, cls_name))
        and issubclass(getattr(current_module, cls_name), OpTest)
        and hasattr(getattr(current_module, cls_name), "init_dtype"),
        dir(current_module),
    )
    dtype_collections = {
        "bool_",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
    }
    for cls_name in cls_collections:
        base_cls = getattr(current_module, cls_name)

        for dtype_name in dtype_collections:
            dtype = getattr(np, dtype_name)

            import functools

            def init_dtype(self, cls_dtype):
                self.dtype = cls_dtype

            cls = type(f"{cls_name}_{dtype_name}", (base_cls,), {})

            @functools.wraps(cls.test_check_grad)
            def test_check_grad(self):
                pass

            if dtype_name == "bool_" or dtype_name.startswith("int"):
                cls.test_check_grad = test_check_grad

            cls.init_dtype = functools.wraps(cls.init_dtype)(
                functools.partialmethod(init_dtype, cls_dtype=dtype)
            )
            globals()[cls.__name__] = cls


create_test_int64(TestTransposeOp)
create_test_int64(TestCase0)
create_test_int64(TestCase1)
create_test_int64(TestCase2)
create_test_int64(TestCase3)
create_test_int64(TestCase4)

create_test_dtype()

if __name__ == "__main__":
    unittest.main()
