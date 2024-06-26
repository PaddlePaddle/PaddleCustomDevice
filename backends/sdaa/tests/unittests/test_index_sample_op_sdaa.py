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

paddle.enable_static()
SEED = 1024


@skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
class TestIndexSampleOp(OpTest):
    def setUp(self):
        self.op_type = "index_sample"
        self.python_api = paddle.index_sample
        self.place = paddle.CustomPlace("sdaa", 0)
        self.config()
        xnp = np.random.random(self.x_shape).astype(self.x_type)
        indexnp = np.random.randint(
            low=0, high=self.x_shape[1], size=self.index_shape
        ).astype(self.index_type)
        self.inputs = {"X": xnp, "Index": indexnp}
        index_array = []
        for i in range(self.index_shape[0]):
            for j in indexnp[i]:
                index_array.append(xnp[i, j])
        index_array = np.array(index_array).astype(self.x_type)
        out = np.reshape(index_array, self.index_shape)
        self.outputs = {"Out": out}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (10, 20)
        self.x_type = "float64"
        self.index_shape = (10, 10)
        self.index_type = "int32"


class TestCase1(TestIndexSampleOp):
    def config(self):
        """
        For one dimension input
        """
        self.x_shape = (100, 1)
        self.x_type = "float64"
        self.index_shape = (100, 1)
        self.index_type = "int32"


class TestCase2(TestIndexSampleOp):
    def config(self):
        """
        For int64_t index type
        """
        self.x_shape = (10, 100)
        self.x_type = "float64"
        self.index_shape = (10, 10)
        self.index_type = "int64"


class TestCase3(TestIndexSampleOp):
    def config(self):
        """
        For int index type
        """
        self.x_shape = (10, 100)
        self.x_type = "float64"
        self.index_shape = (10, 10)
        self.index_type = "int32"


class TestCase4(TestIndexSampleOp):
    def config(self):
        """
        For int64 index type
        """
        self.x_shape = (10, 128)
        self.x_type = "float64"
        self.index_shape = (10, 64)
        self.index_type = "int64"


class TestCase5(TestIndexSampleOp):
    def config(self):
        """
        For double x type
        """
        self.x_shape = (10, 128)
        self.x_type = "double"
        self.index_shape = (10, 64)
        self.index_type = "int32"


class TestCase6(TestIndexSampleOp):
    def config(self):
        """
        For double x type
        """
        self.x_shape = (10, 128)
        self.x_type = "double"
        self.index_shape = (10, 64)
        self.index_type = "int64"


class TestCase7(TestIndexSampleOp):
    def config(self):
        """
        For float16 x type
        """
        self.x_shape = (10, 128)
        self.x_type = "float16"
        self.index_shape = (10, 64)
        self.index_type = "int32"


class TestCase8(TestIndexSampleOp):
    def config(self):
        """
        For float16 x type
        """
        self.x_shape = (10, 128)
        self.x_type = "float16"
        self.index_shape = (10, 64)
        self.index_type = "int64"


class TestCase9(TestIndexSampleOp):
    def config(self):
        """
        For int32 x type
        """
        self.x_shape = (10, 128)
        self.x_type = "int32"
        self.index_shape = (10, 64)
        self.index_type = "int32"


class TestCase10(TestIndexSampleOp):
    def config(self):
        """
        For int32 x type
        """
        self.x_shape = (10, 128)
        self.x_type = "int32"
        self.index_shape = (10, 64)
        self.index_type = "int64"


class TestCase11(TestIndexSampleOp):
    def config(self):
        """
        For int64 x type
        """
        self.x_shape = (10, 128)
        self.x_type = "int64"
        self.index_shape = (10, 64)
        self.index_type = "int32"


class TestCase12(TestIndexSampleOp):
    def config(self):
        """
        For int64 x type
        """
        self.x_shape = (10, 128)
        self.x_type = "int64"
        self.index_shape = (10, 64)
        self.index_type = "int64"


class TestIndexSampleShape(unittest.TestCase):
    def test_shape(self):
        paddle.enable_static()
        paddle.device.set_device("sdaa")
        # create x value
        x_shape = (2, 5)
        x_type = "float64"
        x_np = np.random.random(x_shape).astype(x_type)

        # create index value
        index_shape = (2, 3)
        index_type = "int32"
        index_np = np.random.randint(low=0, high=x_shape[1], size=index_shape).astype(
            index_type
        )

        x = paddle.static.data(name="x", shape=[-1, 5], dtype="float64")
        index = paddle.static.data(name="index", shape=[-1, 3], dtype="int32")
        output = paddle.index_sample(x=x, index=index)

        place = paddle.CustomPlace("sdaa", 0)
        exe = base.Executor(place=place)
        exe.run(base.default_startup_program())

        feed = {"x": x_np, "index": index_np}
        res = exe.run(feed=feed, fetch_list=[output])


class TestIndexSampleDynamic(unittest.TestCase):
    def test_result(self):
        paddle.device.set_device("sdaa")
        with base.dygraph.guard():
            x = paddle.to_tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ],
                dtype="float32",
            )
            index = paddle.to_tensor([[0, 1, 2], [1, 2, 3], [0, 0, 0]], dtype="int32")
            out_z1 = paddle.index_sample(x, index)

            except_output = np.array(
                [[1.0, 2.0, 3.0], [6.0, 7.0, 8.0], [9.0, 9.0, 9.0]]
            )
            assert out_z1.numpy().all() == except_output.all()


if __name__ == "__main__":
    unittest.main()
