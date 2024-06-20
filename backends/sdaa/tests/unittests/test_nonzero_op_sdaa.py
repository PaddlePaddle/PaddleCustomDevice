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
import paddle
import paddle.base as base
from paddle.base import Program, program_guard
from op_test import OpTest

paddle.enable_static()


class TestWhereIndexOp(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "where_index"
        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_config()

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def init_config(self):
        self.inputs = {
            "Condition": np.array([True, False, True]),
        }

        self.outputs = {"Out": np.array([[0], [2]], dtype="int64")}

    def set_sdaa(self):
        self.__class__.use_custom_device = True


class TestNotBool(TestWhereIndexOp):
    def init_config(self):
        self.inputs = {
            "Condition": np.array([1, 0, 8]),
        }

        self.outputs = {"Out": np.array([[0], [2]], dtype="int64")}


class TestAllFalse(TestWhereIndexOp):
    def init_config(self):
        self.inputs = {
            "Condition": np.array([False, False, False]),
        }

        self.outputs = {"Out": np.array([], dtype="int64")}


class TestRank2(TestWhereIndexOp):
    def init_config(self):
        self.inputs = {
            "Condition": np.array([[True, False], [False, True]]),
        }

        self.outputs = {"Out": np.array([[0, 0], [1, 1]], dtype="int64")}


class TestRank3(TestWhereIndexOp):
    def init_config(self):
        self.inputs = {
            "Condition": np.array(
                [
                    [[True, False], [False, True]],
                    [[False, True], [True, False]],
                    [[False, False], [False, True]],
                ]
            ),
        }

        self.outputs = {
            "Out": np.array(
                [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [2, 1, 1]], dtype="int64"
            )
        }


class TestWhereOpError(unittest.TestCase):
    def test_api(self):
        with program_guard(Program(), Program()):
            cond = paddle.static.data(name="cond", shape=[-1, 4], dtype="bool")
            result = paddle.nonzero(cond)

            exe = base.Executor(paddle.CustomPlace("sdaa", 0))
            exe.run(base.default_startup_program())
            cond_i = np.array([True, False, False, False]).astype("bool")
            out = exe.run(base.default_main_program(), feed={"cond": cond_i})


class TestNonZeroAPI(unittest.TestCase):
    def test_nonzero_api_as_tuple(self):
        data = np.array([[True, False], [False, True]])
        with program_guard(Program(), Program()):
            x = paddle.static.data(name="x", shape=[-1, 2], dtype="float32")
            x.desc.set_need_check_feed(False)
            y = paddle.nonzero(x, as_tuple=True)
            self.assertEqual(type(y), tuple)
            self.assertEqual(len(y), 2)
            z = paddle.concat(list(y), axis=1)
            exe = base.Executor(paddle.CustomPlace("sdaa", 0))

            (res,) = exe.run(feed={"x": data}, fetch_list=[z.name], return_numpy=False)
        expect_out = np.array([[0, 0], [1, 1]])
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

        data = np.array([True, True, False])
        with program_guard(Program(), Program()):
            x = paddle.static.data(name="x", shape=[-1], dtype="float32")
            x.desc.set_need_check_feed(False)
            y = paddle.nonzero(x, as_tuple=True)
            self.assertEqual(type(y), tuple)
            self.assertEqual(len(y), 1)
            z = paddle.concat(list(y), axis=1)
            exe = base.Executor(paddle.CustomPlace("sdaa", 0))
            (res,) = exe.run(feed={"x": data}, fetch_list=[z.name], return_numpy=False)
        expect_out = np.array([[0], [1]])
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

    def test_nonzero_api(self):
        data = np.array([[True, False], [False, True]])
        with program_guard(Program(), Program()):
            x = paddle.static.data(name="x", shape=[-1, 2], dtype="float32")
            x.desc.set_need_check_feed(False)
            y = paddle.nonzero(x)
            exe = base.Executor(paddle.CustomPlace("sdaa", 0))
            (res,) = exe.run(feed={"x": data}, fetch_list=[y.name], return_numpy=False)
        expect_out = np.array([[0, 0], [1, 1]])
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

        data = np.array([True, True, False])
        with program_guard(Program(), Program()):
            x = paddle.static.data(name="x", shape=[-1], dtype="float32")
            x.desc.set_need_check_feed(False)
            y = paddle.nonzero(x)
            exe = base.Executor(paddle.CustomPlace("sdaa", 0))
            (res,) = exe.run(feed={"x": data}, fetch_list=[y.name], return_numpy=False)
        expect_out = np.array([[0], [1]])
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

    def test_dygraph_api(self):
        data_x = np.array([[True, False], [False, True]])
        with base.dygraph.guard(paddle.CustomPlace("sdaa", 0)):
            x = base.dygraph.to_variable(data_x)
            z = paddle.nonzero(x)
            np_z = z.numpy()
        expect_out = np.array([[0, 0], [1, 1]])
        np.testing.assert_allclose(expect_out, np.array(np_z), rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
