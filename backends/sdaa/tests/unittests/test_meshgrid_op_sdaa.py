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
import paddle.base as base

paddle.enable_static()
SEED = 1024


def meshgrid_wrapper(x):
    return paddle.tensor.meshgrid(x[0], x[1])


@skip_check_grad_ci(reason="The backward test is not supported yet on sdaa.")
class TestMeshgridOp(OpTest):
    def setUp(self):
        self.op_type = "meshgrid"
        self.python_api = meshgrid_wrapper
        self.public_python_api = meshgrid_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_data_type()
        self.init_inputs_and_outputs()
        self.python_out_sig = ["out0", "out1"]

    def init_data_type(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def init_inputs_and_outputs(self):
        self.shape = self.get_x_shape()
        ins = []
        outs = []
        for i in range(len(self.shape)):
            ins.append(np.random.random((self.shape[i],)).astype(self.dtype))

        for i in range(len(self.shape)):
            out_reshape = [1] * len(self.shape)
            out_reshape[i] = self.shape[i]
            out_temp = np.reshape(ins[i], out_reshape)
            outs.append(np.broadcast_to(out_temp, self.shape))
        self.inputs = {"X": [("x%d" % i, ins[i]) for i in range(len(ins))]}
        self.outputs = {"Out": [("out%d" % i, outs[i]) for i in range(len(outs))]}

    def get_x_shape(self):
        return [100, 200]


class TestMeshgridOpFp16(TestMeshgridOp):
    def init_data_type(self):
        self.dtype = np.float16


class TestMeshgridOpDouble(TestMeshgridOp):
    def init_data_type(self):
        self.dtype = np.double


class TestMeshgridOpInt32(TestMeshgridOp):
    def init_data_type(self):
        self.dtype = np.int32


class TestMeshgridOpInt64(TestMeshgridOp):
    def init_data_type(self):
        self.dtype = np.int64


class TestMeshgridOp2(TestMeshgridOp):
    def get_x_shape(self):
        return [100, 300]


class TestMeshgridOp3(TestMeshgridOp):
    def get_x_shape(self):
        return [100, 20, 5, 2, 16]


class TestMeshgridOp4(unittest.TestCase):
    def test_api(self):
        paddle.device.set_device("sdaa")
        x = paddle.static.data(shape=[100], dtype="int32", name="x")
        y = paddle.static.data(shape=[200], dtype="int32", name="y")

        input_1 = np.random.randint(
            0,
            100,
            [
                100,
            ],
        ).astype("int32")
        input_2 = np.random.randint(
            0,
            100,
            [
                200,
            ],
        ).astype("int32")

        out_1 = np.reshape(input_1, [100, 1])
        out_1 = np.broadcast_to(out_1, [100, 200])
        out_2 = np.reshape(input_2, [1, 200])
        out_2 = np.broadcast_to(out_2, [100, 200])

        exe = base.Executor(place=paddle.CustomPlace("sdaa", 0))
        grid_x, grid_y = paddle.tensor.meshgrid(x, y)
        res_1, res_2 = exe.run(
            base.default_main_program(),
            feed={"x": input_1, "y": input_2},
            fetch_list=[grid_x, grid_y],
        )
        assert np.array_equal(res_1, out_1)
        assert np.array_equal(res_2, out_2)


class TestMeshgridOp5(unittest.TestCase):
    def test_list_input(self):
        paddle.device.set_device("sdaa")
        x = paddle.static.data(shape=[100], dtype="int32", name="x")
        y = paddle.static.data(shape=[200], dtype="int32", name="y")

        input_1 = np.random.randint(
            0,
            100,
            [
                100,
            ],
        ).astype("int32")
        input_2 = np.random.randint(
            0,
            100,
            [
                200,
            ],
        ).astype("int32")

        out_1 = np.reshape(input_1, [100, 1])
        out_1 = np.broadcast_to(out_1, [100, 200])
        out_2 = np.reshape(input_2, [1, 200])
        out_2 = np.broadcast_to(out_2, [100, 200])

        exe = base.Executor(place=paddle.CustomPlace("sdaa", 0))
        grid_x, grid_y = paddle.tensor.meshgrid([x, y])
        res_1, res_2 = exe.run(
            base.default_main_program(),
            feed={"x": input_1, "y": input_2},
            fetch_list=[grid_x, grid_y],
        )

        assert np.array_equal(res_1, out_1)
        assert np.array_equal(res_2, out_2)


class TestMeshgridOp6(unittest.TestCase):
    def test_tuple_input(self):
        paddle.device.set_device("sdaa")
        x = paddle.static.data(shape=[100], dtype="int32", name="x")
        y = paddle.static.data(shape=[200], dtype="int32", name="y")

        input_1 = np.random.randint(
            0,
            100,
            [
                100,
            ],
        ).astype("int32")
        input_2 = np.random.randint(
            0,
            100,
            [
                200,
            ],
        ).astype("int32")

        out_1 = np.reshape(input_1, [100, 1])
        out_1 = np.broadcast_to(out_1, [100, 200])
        out_2 = np.reshape(input_2, [1, 200])
        out_2 = np.broadcast_to(out_2, [100, 200])

        exe = base.Executor(place=paddle.CustomPlace("sdaa", 0))
        grid_x, grid_y = paddle.tensor.meshgrid((x, y))
        res_1, res_2 = exe.run(
            base.default_main_program(),
            feed={"x": input_1, "y": input_2},
            fetch_list=[grid_x, grid_y],
        )

        assert np.array_equal(res_1, out_1)
        assert np.array_equal(res_2, out_2)


class TestMeshgridOp7(unittest.TestCase):
    def test_api_with_dygraph(self):
        paddle.device.set_device("sdaa")
        input_3 = np.random.randint(
            0,
            100,
            [
                100,
            ],
        ).astype("int32")
        input_4 = np.random.randint(
            0,
            100,
            [
                200,
            ],
        ).astype("int32")

        with base.dygraph.guard():
            tensor_3 = base.dygraph.to_variable(input_3)
            tensor_4 = base.dygraph.to_variable(input_4)
            res_3, res_4 = paddle.tensor.meshgrid(tensor_3, tensor_4)

            assert np.array_equal(res_3.shape, [100, 200])
            assert np.array_equal(res_4.shape, [100, 200])


class TestMeshgridOp8(unittest.TestCase):
    def test_api_with_dygraph_list_input(self):
        paddle.device.set_device("sdaa")
        input_3 = np.random.randint(
            0,
            100,
            [
                100,
            ],
        ).astype("int32")
        input_4 = np.random.randint(
            0,
            100,
            [
                200,
            ],
        ).astype("int32")

        with base.dygraph.guard():
            tensor_3 = base.dygraph.to_variable(input_3)
            tensor_4 = base.dygraph.to_variable(input_4)
            res_3, res_4 = paddle.tensor.meshgrid([tensor_3, tensor_4])

            assert np.array_equal(res_3.shape, [100, 200])
            assert np.array_equal(res_4.shape, [100, 200])


class TestMeshgridOp9(unittest.TestCase):
    def test_api_with_dygraph_tuple_input(self):
        paddle.device.set_device("sdaa")
        input_3 = np.random.randint(
            0,
            100,
            [
                100,
            ],
        ).astype("int32")
        input_4 = np.random.randint(
            0,
            100,
            [
                200,
            ],
        ).astype("int32")

        with base.dygraph.guard():
            tensor_3 = base.dygraph.to_variable(input_3)
            tensor_4 = base.dygraph.to_variable(input_4)
            res_3, res_4 = paddle.tensor.meshgrid((tensor_3, tensor_4))

            assert np.array_equal(res_3.shape, [100, 200])
            assert np.array_equal(res_4.shape, [100, 200])


class TestMeshGrid_ZeroDim(TestMeshgridOp):
    def init_inputs_and_outputs(self):
        paddle.device.set_device("sdaa")
        self.shape = self.get_x_shape()
        ins = []
        outs = []
        ins.append(np.random.random([]).astype(self.dtype))
        ins.append(np.random.random([2]).astype(self.dtype))
        ins.append(np.random.random([3]).astype(self.dtype))
        for i in range(len(self.shape)):
            out_reshape = [1] * len(self.shape)
            out_reshape[i] = self.shape[i]
            out_temp = np.reshape(ins[i], out_reshape)
            outs.append(np.broadcast_to(out_temp, self.shape))
        self.inputs = {"X": [("x%d" % i, ins[i]) for i in range(len(ins))]}
        self.outputs = {"Out": [("out%d" % i, outs[i]) for i in range(len(outs))]}

    def get_x_shape(self):
        return [1, 2, 3]


class TestMeshgridEager(unittest.TestCase):
    def test_dygraph_api(self):
        paddle.device.set_device("sdaa")
        input_1 = np.random.randint(
            0,
            100,
            [
                100,
            ],
        ).astype("int32")
        input_2 = np.random.randint(
            0,
            100,
            [
                200,
            ],
        ).astype("int32")

        with base.dygraph.guard():
            tensor_1 = base.dygraph.to_variable(input_1)
            tensor_2 = base.dygraph.to_variable(input_2)
            tensor_1.stop_gradient = False
            tensor_2.stop_gradient = False
            res_1, res_2 = paddle.tensor.meshgrid((tensor_1, tensor_2))
            sum = paddle.add_n([res_1, res_2])
            sum.backward()
            tensor_eager_1 = base.dygraph.to_variable(input_1)
            tensor_eager_2 = base.dygraph.to_variable(input_2)
            tensor_eager_1.stop_gradient = False
            tensor_eager_2.stop_gradient = False
            res_eager_1, res_eager_2 = paddle.tensor.meshgrid(
                (tensor_eager_1, tensor_eager_2)
            )
            sum_eager = paddle.add_n([res_eager_1, res_eager_2])
            sum_eager.backward()
            self.assertEqual(
                (tensor_1.grad.numpy() == tensor_eager_1.grad.numpy()).all(),
                True,
            )
            self.assertEqual(
                (tensor_2.grad.numpy() == tensor_eager_2.grad.numpy()).all(),
                True,
            )


if __name__ == "__main__":
    unittest.main()
