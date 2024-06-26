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
from paddle.static import Program, program_guard

paddle.enable_static()


def huber_loss_forward(val, delta):
    abs_val = abs(val)
    if abs_val <= delta:
        return 0.5 * val * val
    else:
        return delta * (abs_val - 0.5 * delta)


class TestHuberLossOp(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.init_dtype()
        self.op_type = "huber_loss"
        self.python_api = paddle.nn.functional.smooth_l1_loss
        self.python_out_sig = ["Out"]
        self.delta = 1.0
        self.shape = self.set_shape()
        self.init_input()
        residual = self.inputs["Y"] - self.inputs["X"]
        loss = np.vectorize(huber_loss_forward)(residual, self.delta).astype(self.dtype)
        self.attrs = {"delta": self.delta}
        self.outputs = {"Residual": residual, "Out": loss.reshape(self.shape)}

    def init_input(self):
        self.inputs = {
            "X": np.random.uniform(0, 1.0, self.shape).astype(self.dtype),
            "Y": np.random.uniform(0, 1.0, self.shape).astype(self.dtype),
        }

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        # self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def set_shape(self):
        return (100, 1)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        self.check_grad_with_place(self.place, ["X", "Y"], "Out")

    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(
            self.place,
            ["Y"],
            "Out",
            no_grad_set=set("X"),
        )

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(
            self.place,
            ["X"],
            "Out",
            no_grad_set=set("Y"),
        )


class TestHuberLossOp1(TestHuberLossOp):
    def set_shape(self):
        return (1, 128)


class TestHuberLossOp2(TestHuberLossOp):
    def set_shape(self):
        return (16, 8)


class TestHuberLossOp3(TestHuberLossOp):
    def set_shape(self):
        return (16, 12, 1)


class TestHuberLossOp4(TestHuberLossOp):
    def set_shape(self):
        return 120


def create_test_fp64_class(parent):
    class TestHuberLossOpFp64Case(parent):
        def init_dtype(self):
            self.dtype = np.float64

        def test_check_grad_normal(self):
            pass

        def test_check_grad_ingore_x(self):
            pass

        def test_check_grad_ingore_y(self):
            pass

    cls_name = "{0}_{1}".format(parent.__name__, "Fp64")
    TestHuberLossOpFp64Case.__name__ = cls_name
    globals()[cls_name] = TestHuberLossOpFp64Case


create_test_fp64_class(TestHuberLossOp)
create_test_fp64_class(TestHuberLossOp1)
create_test_fp64_class(TestHuberLossOp2)
create_test_fp64_class(TestHuberLossOp3)
create_test_fp64_class(TestHuberLossOp4)


def create_test_fp16_class(parent):
    class TestHuberLossOpFp16Case(parent):
        def init_dtype(self):
            self.dtype = np.float16

        def test_check_grad_normal(self):
            pass

        def test_check_grad_ingore_x(self):
            pass

        def test_check_grad_ingore_y(self):
            pass

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestHuberLossOpFp16Case.__name__ = cls_name
    globals()[cls_name] = TestHuberLossOpFp16Case


create_test_fp16_class(TestHuberLossOp)
create_test_fp16_class(TestHuberLossOp1)
create_test_fp16_class(TestHuberLossOp2)
create_test_fp16_class(TestHuberLossOp3)
create_test_fp16_class(TestHuberLossOp4)


class TestHuberLossOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # the input and label must be Variable
            xw = np.random.random((6, 6)).astype("float32")
            xr = paddle.static.data(name="xr", shape=[None, 6], dtype="float32")
            lw = np.random.random((6, 6)).astype("float32")
            lr = paddle.static.data(name="lr", shape=[None, 6], dtype="float32")
            delta = 1.0
            self.assertRaises(
                TypeError, paddle.nn.functional.smooth_l1_loss, xr, lw, delta
            )
            self.assertRaises(
                TypeError, paddle.nn.functional.smooth_l1_loss, xw, lr, delta
            )

            # the dtype of input and label must be float32 or float64
            xw2 = paddle.static.data(name="xw2", shape=[None, 6], dtype="int32")
            lw2 = paddle.static.data(name="lw2", shape=[None, 6], dtype="int32")
            self.assertRaises(
                TypeError, paddle.nn.functional.smooth_l1_loss, xw2, lr, delta
            )
            self.assertRaises(
                TypeError, paddle.nn.functional.smooth_l1_loss, xr, lw2, delta
            )


if __name__ == "__main__":
    unittest.main()
