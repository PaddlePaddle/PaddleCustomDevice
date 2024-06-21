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
from paddle import base
from paddle.base import Program, program_guard

paddle.enable_static()
SEED = 1024


@skip_check_grad_ci(reason="Haven not implement index_select_grad kernel.")
class TestIndexSelectOp(OpTest):
    def setUp(self):
        self.python_api = paddle.index_select
        self.public_python_api = paddle.index_select
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "index_select"
        self.init_dtype_type()

        index_np = np.random.randint(
            low=0, high=self.x_shape[self.dim], size=self.index_size
        )
        x_np = np.random.random(self.x_shape).astype(self.x_type)
        self.inputs = {"X": x_np, "Index": index_np}
        self.attrs = {"dim": self.dim}
        outer_loop = np.prod(self.x_shape[: self.dim])
        x_reshape = [outer_loop] + list(self.x_shape[self.dim :])
        x_np_reshape = np.reshape(x_np, tuple(x_reshape))
        out_list = []
        for i in range(outer_loop):
            for j in range(self.index_size):
                out_list.append(x_np_reshape[i, index_np[j]])
        self.out_shape = list(self.x_shape)
        self.out_shape[self.dim] = self.index_size
        self.out_shape = tuple(self.out_shape)

        out = np.reshape(out_list, self.out_shape)
        self.outputs = {"Out": out}

    def init_dtype_type(self):
        self.dim = 1
        self.x_type = np.float32
        self.index_type = np.int32
        self.x_shape = (100, 4, 5)
        self.index_size = 100

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestIndexSelectOpCase2(TestIndexSelectOp):
    def init_dtype_type(self):
        self.x_type = np.float32
        self.index_type = np.int64
        self.dim = 1
        self.x_shape = (100, 4, 5)
        self.index_size = 100


class TestIndexSelectOpDouble(TestIndexSelectOp):
    def init_dtype_type(self):
        self.x_type = np.double
        self.index_type = np.int32
        self.dim = 1
        self.x_shape = (100, 4, 5)
        self.index_size = 100


class TestIndexSelectOpDoubleCase2(TestIndexSelectOp):
    def init_dtype_type(self):
        self.x_type = np.double
        self.index_type = np.int64
        self.dim = 1
        self.x_shape = (100, 4, 5)
        self.index_size = 100


class TestIndexSelectFP16OP(TestIndexSelectOp):
    def init_dtype_type(self):
        self.x_type = np.float16
        self.index_type = np.int32
        self.dim = 1
        self.x_shape = (100, 4, 5)
        self.index_size = 100


class TestIndexSelectFP16OPCase2(TestIndexSelectOp):
    def init_dtype_type(self):
        self.x_type = np.float16
        self.index_type = np.int64
        self.dim = 1
        self.x_shape = (100, 4, 5)
        self.index_size = 100


class TestIndexSelectOpInt32(TestIndexSelectOp):
    def init_dtype_type(self):
        self.x_type = np.int32
        self.index_type = np.int32
        self.dim = 1
        self.x_shape = (100, 4, 5)
        self.index_size = 100


class TestIndexSelectOpInt32Case2(TestIndexSelectOp):
    def init_dtype_type(self):
        self.x_type = np.int32
        self.index_type = np.int64
        self.dim = 1
        self.x_shape = (100, 4, 5)
        self.index_size = 100


class TestIndexSelectOpInt64(TestIndexSelectOp):
    def init_dtype_type(self):
        self.x_type = np.int64
        self.index_type = np.int32
        self.dim = 1
        self.x_shape = (100, 4, 5)
        self.index_size = 100


class TestIndexSelectOpInt64Case2(TestIndexSelectOp):
    def init_dtype_type(self):
        self.x_type = np.int64
        self.index_type = np.int64
        self.dim = 1
        self.x_shape = (100, 4, 5)
        self.index_size = 100


class TestIndexSelectAPI(unittest.TestCase):
    def input_data(self):
        self.data_x = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        ).astype("float32")
        self.data_index = np.array([0, 1, 1]).astype("int32")

    def test_index_select_api(self):
        paddle.enable_static()
        self.input_data()

        # case 1:
        with program_guard(Program(), Program()):
            x = paddle.static.data(name="x", shape=[-1, 4])
            index = paddle.static.data(name="index", shape=[3], dtype="int32")
            z = paddle.index_select(x, index, axis=1)
            exe = base.Executor(paddle.CustomPlace("sdaa", 0))
            (res,) = exe.run(
                feed={"x": self.data_x, "index": self.data_index},
                fetch_list=[z.name],
                return_numpy=False,
            )
        expect_out = np.array([[1.0, 2.0, 2.0], [5.0, 6.0, 6.0], [9.0, 10.0, 10.0]])
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

        # case 2:
        with program_guard(Program(), Program()):
            x = paddle.static.data(name="x", shape=[-1, 4])
            index = paddle.static.data(name="index", shape=[3], dtype="int32")
            z = paddle.index_select(x, index)
            exe = base.Executor(paddle.CustomPlace("sdaa", 0))
            (res,) = exe.run(
                feed={"x": self.data_x, "index": self.data_index},
                fetch_list=[z.name],
                return_numpy=False,
            )
        expect_out = np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [5.0, 6.0, 7.0, 8.0]]
        )
        np.testing.assert_allclose(expect_out, np.array(res), rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static()
        paddle.device.set_device("sdaa")
        self.input_data()
        # case 1:
        with base.dygraph.guard():
            x = base.dygraph.to_variable(self.data_x)
            index = base.dygraph.to_variable(self.data_index)
            z = paddle.index_select(x, index)
            np_z = z.numpy()
        expect_out = np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [5.0, 6.0, 7.0, 8.0]]
        )
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)

        # case 2:
        with base.dygraph.guard():
            x = base.dygraph.to_variable(self.data_x)
            index = base.dygraph.to_variable(self.data_index)
            z = paddle.index_select(x, index, axis=1)
            np_z = z.numpy()
        expect_out = np.array([[1.0, 2.0, 2.0], [5.0, 6.0, 6.0], [9.0, 10.0, 10.0]])
        np.testing.assert_allclose(expect_out, np_z, rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
