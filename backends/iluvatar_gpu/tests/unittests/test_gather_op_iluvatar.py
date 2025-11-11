#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
from op_test import OpTest

import numpy as np
import paddle

paddle.enable_static()


def gather_numpy(x, index, axis):
    x_transpose = np.swapaxes(x, 0, axis)
    tmp_gather = x_transpose[index, ...]
    gather = np.swapaxes(tmp_gather, 0, axis)
    return gather


class TestGatherOp(OpTest):
    def setUp(self):
        self.op_type = "gather"
        self.place = paddle.CustomPlace("iluvatar_gpu", 0)
        self.__class__.use_custom_device = True
        self.python_api = paddle.gather
        self.config()
        xnp = np.random.random(self.x_shape).astype(self.x_type)
        self.inputs = {"X": xnp, "Index": np.array(self.index).astype(self.index_type)}
        self.outputs = {"Out": self.inputs["X"][self.inputs["Index"]]}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")

    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (10, 20)
        self.x_type = "float32"
        self.index = [1, 3, 5]
        self.index_type = "int32"


class TestCase1(TestGatherOp):
    def config(self):
        """
        For one dimension input
        """
        self.x_shape = 100
        self.x_type = "float32"
        self.index = [1, 3, 5]
        self.index_type = "int32"


class TestCase2(TestGatherOp):
    def config(self):
        """
        For int64_t index type
        """
        self.x_shape = 100
        self.x_type = "float32"
        self.index = [1, 3, 5]
        self.index_type = "int64"


class API_TestDygraphGather(unittest.TestCase):
    def test_out1(self):
        paddle.set_device("iluvatar_gpu")
        paddle.disable_static()
        input_1 = np.array([[1, 2], [3, 4], [5, 6]]).astype("int32")
        index_1 = np.array([1, 2])
        input = paddle.to_tensor(input_1)
        index = paddle.to_tensor(index_1)
        output = paddle.gather(input, index)
        output_np = output.numpy()
        expected_output = np.array([[3, 4], [5, 6]]).astype("int32")
        np.testing.assert_allclose(output_np, expected_output)
        paddle.enable_static()

    def test_out12(self):
        paddle.set_device("iluvatar_gpu")
        paddle.disable_static()
        input_1 = np.array([[1, 2], [3, 4], [5, 6]]).astype("int32")
        index_1 = np.array([1, 2])
        x = paddle.to_tensor(input_1)
        index = paddle.to_tensor(index_1)
        output = paddle.gather(x, index, axis=0)
        output_np = output.numpy()
        expected_output = gather_numpy(input_1, index_1, axis=0)
        np.testing.assert_allclose(output_np, expected_output)
        paddle.enable_static()

    def test_zero_index(self):
        paddle.set_device("iluvatar_gpu")
        paddle.disable_static()
        x = paddle.to_tensor([[1, 2], [3, 4]]).astype("int32")
        index = paddle.to_tensor(np.array([]).astype("int64"))
        for axis in range(len(x.shape)):
            out = paddle.gather(x, index, axis)
            expected_shape = list(x.shape)
            expected_shape[axis] = 0
            self.assertEqual(list(out.shape), expected_shape)
        paddle.enable_static()


class TestGathertError(unittest.TestCase):
    def setUp(self) -> None:
        self.place = paddle.CustomPlace("iluvatar_gpu", 0)
        paddle.set_device("iluvatar_gpu:0")

    def test_error1(self):
        paddle.enable_static()
        if not paddle.framework.use_pir_api():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):

                input_shape = [8, 9, 6]
                index_shape = [4]
                x_int8 = paddle.static.data(
                    shape=input_shape, dtype="int8", name="x_int8"
                )
                x_float32 = paddle.static.data(
                    shape=input_shape, dtype="float32", name="x_float32"
                )
                axis = paddle.static.data(shape=[1], dtype="float32", name="axis")
                index = paddle.static.data(
                    shape=index_shape, dtype="int32", name="index"
                )
                index_float = paddle.static.data(
                    shape=index_shape, dtype="float32", name="index_float"
                )

                def test_x_type():
                    paddle.gather(x_int8, index)

                self.assertRaises(TypeError, test_x_type)

                def test_index_type():
                    paddle.gather(x_float32, index_float)

                self.assertRaises(TypeError, test_index_type)

                def test_axis_dtype():
                    paddle.gather(x_float32, index, axis=1.11)

                self.assertRaises(TypeError, test_axis_dtype)

                def test_axis_dtype1():
                    paddle.gather(x_float32, index, axis=axis)

                self.assertRaises(TypeError, test_axis_dtype1)
        else:
            paddle.set_device("iluvatar_gpu")
            input_shape = [8, 9, 6]
            index_shape = [4]

            def test_index_type():
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    x = paddle.static.data(shape=input_shape, dtype="float32", name="x")
                    index = paddle.static.data(
                        shape=index_shape, dtype="float32", name="index_float"
                    )
                    out = paddle.gather(x, index)
                    exe = paddle.static.Executor(place=self.place)
                    exe.run(paddle.static.default_startup_program())
                    self.assertRaises(
                        ValueError,
                        exe.run,
                        paddle.static.default_main_program(),
                        feed={
                            "x": np.random.random(input_shape).astype("float32"),
                            "index_float": np.random.random(index_shape).astype(
                                "float32"
                            ),
                        },
                    )

            def test_axis_scalar_dtype():
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    x = paddle.static.data(shape=input_shape, dtype="float32", name="x")
                    index = paddle.static.data(
                        shape=index_shape, dtype="int32", name="index"
                    )
                    axis = paddle.static.data(shape=[1], dtype="int32", name="axis")
                    self.assertRaises(TypeError, paddle.gather, x, index, axis=1.11)

            def test_axis_tensor_dtype():
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    x = paddle.static.data(shape=input_shape, dtype="float32", name="x")
                    index = paddle.static.data(
                        shape=index_shape, dtype="int32", name="index"
                    )
                    axis = paddle.static.data(shape=[1], dtype="float32", name="axis")
                    y = paddle.gather(x, index, axis=axis)
                    exe = paddle.static.Executor(place=self.place)
                    exe.run(paddle.static.default_startup_program())
                    self.assertRaises(
                        ValueError,
                        exe.run,
                        paddle.static.default_main_program(),
                        feed={
                            "x": np.random.random(input_shape).astype("float32"),
                            "index": np.random.randint(0, 8, index_shape).astype(
                                "int32"
                            ),
                            "axis": np.array([1.11]).astype("float32"),
                        },
                    )

            test_index_type()
            test_axis_scalar_dtype()
            # test_axis_tensor_dtype()


if __name__ == "__main__":
    unittest.main()
