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
import paddle.base as base
import paddle.tensor as tensor
from paddle.base.framework import Program, program_guard

paddle.enable_static()


class TestTrilTriu(OpTest):
    """the base class of other op testcases"""

    def setUp(self):
        self.set_sdaa()
        self.init_dtype()
        self.initTestCase()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.real_np_op = getattr(np, self.real_op_type)

        self.op_type = "tril_triu"
        self.python_api = paddle.tril if self.real_op_type == "tril" else paddle.triu

        self.inputs = {"X": self.X}
        self.attrs = {
            "diagonal": self.diagonal,
            "lower": True if self.real_op_type == "tril" else False,
        }
        self.outputs = {
            "Out": self.real_np_op(self.X, self.diagonal)
            if self.diagonal
            else self.real_np_op(self.X)
        }

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float32

    def initTestCase(self):
        self.real_op_type = np.random.choice(["triu", "tril"])
        self.diagonal = None
        self.X = np.arange(1, 101, dtype=self.dtype).reshape([10, -1])


def case_generator(op_type, Xshape, diagonal, expected):
    """
    Generate testcases with the params shape of X, diagonal and op_type.
    If arg `expercted` is 'success', it will register an Optest case and expect to pass.
    Otherwise, it will register an API case and check the expect failure.
    """
    cls_name = "{0}_{1}_shape_{2}_diag_{3}".format(expected, op_type, Xshape, diagonal)
    errmsg = {
        "diagonal: TypeError": "diagonal in {} must be a python Int".format(op_type),
        "input: ValueError": "x shape in {} must be at least 2-D".format(op_type),
    }

    class FailureCase(unittest.TestCase):
        def test_failure(self):
            paddle.enable_static()

            data = paddle.static.data(shape=Xshape, dtype="float32", name=cls_name)
            with self.assertRaisesRegexp(
                eval(expected.split(":")[-1]), errmsg[expected]
            ):
                getattr(tensor, op_type)(x=data, diagonal=diagonal)

    class SuccessCase(TestTrilTriu):
        def initTestCase(self):
            paddle.enable_static()

            self.real_op_type = op_type
            self.diagonal = diagonal
            self.X = np.random.random(Xshape).astype("float32")

    CLASS = locals()["SuccessCase" if expected == "success" else "FailureCase"]
    CLASS.__name__ = cls_name
    globals()[cls_name] = CLASS


# NOTE: meaningful diagonal is [1 - min(H, W), max(H, W) -1]
# test the diagonal just at the border, upper/lower the border,
# negative/positive integer within range and a zero
cases = {
    "success": {
        (2, 2, 3, 4, 5): [-100, -3, -1, 0, 2, 4, 100],  # normal shape
        (10, 10, 1, 1): [-100, -1, 0, 1, 100],  # small size of matrix
    },
    "diagonal: TypeError": {
        (20, 20): [
            "2020",
            [20],
            {20: 20},
            (20, 20),
            20.20,
        ],  # str, list, dict, tuple, float
    },
    "input: ValueError": {
        (2020,): [None],
    },
}
for _op_type in ["tril", "triu"]:
    for _expected, _params in cases.items():
        for _Xshape, _diaglist in _params.items():
            list(
                map(
                    lambda _diagonal: case_generator(
                        _op_type, _Xshape, _diagonal, _expected
                    ),
                    _diaglist,
                )
            )


class TestTrilTriuOpAPI(unittest.TestCase):
    """test case by using API and has -1 dimension"""

    def test_api(self):
        paddle.enable_static()

        dtypes = ["float16", "float32", "int64", "float64", "bool"]
        for dtype in dtypes:
            prog = Program()
            startup_prog = Program()
            with program_guard(prog, startup_prog):
                if dtype == "bool":
                    data = np.random.choice(
                        [False, True], size=(1 * 9 * 9 * 4)
                    ).reshape([1, 9, 9, 4])
                else:
                    data = np.random.random([1, 9, 9, 4]).astype(dtype)
                x = paddle.static.data(shape=[1, 9, -1, 4], dtype=dtype, name="x")
                tril_out, triu_out = tensor.tril(x), tensor.triu(x)

                place = paddle.CustomPlace("sdaa", 0)
                exe = base.Executor(place)
                tril_out, triu_out = exe.run(
                    base.default_main_program(),
                    feed={"x": data},
                    fetch_list=[tril_out, triu_out],
                )
                np.testing.assert_allclose(tril_out, np.tril(data))
                np.testing.assert_allclose(triu_out, np.triu(data))

    def test_api_with_dygraph(self):
        paddle.disable_static(paddle.CustomPlace("sdaa", 0))

        dtypes = ["float16", "float32", "int64", "float64"]
        for dtype in dtypes:
            with base.dygraph.guard():
                if dtype == "bool":
                    data = np.random.choice(
                        [False, True], size=(1 * 9 * 9 * 4)
                    ).reshape([1, 9, 9, 4])
                else:
                    data = np.random.random([1, 9, 9, 4]).astype(dtype)
                x = base.dygraph.to_variable(data)
                tril_out, triu_out = tensor.tril(x).numpy(), tensor.triu(x).numpy()
                np.testing.assert_allclose(tril_out, np.tril(data))
                np.testing.assert_allclose(triu_out, np.triu(data))

    def test_base_api(self):
        paddle.enable_static()

        dtypes = ["float16", "float32", "int64", "float64"]
        for dtype in dtypes:
            prog = Program()
            startup_prog = Program()
            with program_guard(prog, startup_prog):
                data = np.random.random([1, 9, 9, 4]).astype(dtype)
                x = paddle.static.data(shape=[1, 9, -1, 4], dtype=dtype, name="x")
                triu_out = paddle.triu(x)

                place = paddle.CustomPlace("sdaa", 0)
                exe = base.Executor(place)
                triu_out = exe.run(
                    base.default_main_program(), feed={"x": data}, fetch_list=[triu_out]
                )


class TestTrilTriuFP16(TestTrilTriu):
    def init_dtype(self):
        self.dtype = np.float16


class TestTrilTriuDouble(TestTrilTriu):
    def init_dtype(self):
        self.dtype = np.double


class TestTrilTriuINT64(TestTrilTriu):
    def init_dtype(self):
        self.dtype = np.int64


class TestTrilTriuBool(TestTrilTriu):
    def test_check_output(self):
        self.check_output_with_place(self.place)

    def init_dtype(self):
        self.dtype = np.bool_

    def initTestCase(self):
        self.real_op_type = np.random.choice(["triu", "tril"])
        self.diagonal = None
        self.X = np.random.choice([False, True], size=(100)).reshape([10, -1])


if __name__ == "__main__":
    unittest.main()
