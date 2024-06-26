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
from op_test import OpTest, skip_check_grad_ci
from paddle.framework import core

paddle.enable_static()
SEED = 2021


def gather_numpy(x, index, axis):
    x_transpose = np.swapaxes(x, 0, axis)
    tmp_gather = x_transpose[index, ...]
    gather = np.swapaxes(tmp_gather, 0, axis)
    return gather


class TestGatherOp(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "gather"
        self.python_api = paddle.gather
        self.init_dtype()
        self.config()
        xnp = np.random.random(self.x_shape).astype(self.x_type)
        self.inputs = {"X": xnp, "Index": np.array(self.index).astype(self.index_type)}
        self.outputs = {"Out": self.inputs["X"][self.inputs["Index"]]}

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")

    def init_dtype(self):
        self.x_type = np.float32

    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (10, 20)
        self.index = [1, 3, 5]
        self.index_type = np.int32


class TestGatherOp2(TestGatherOp):
    def config(self):
        """
        For one dimension input
        """
        self.x_shape = 100
        self.index = [1, 3, 5]
        self.index_type = np.int32


class TestGatherOp3(TestGatherOp):
    def config(self):
        """
        For index int64
        """
        self.x_shape = (10, 20)
        self.index = [1, 3, 5]
        self.index_type = np.int64


@skip_check_grad_ci(reason="Gather backward index int16 is not supported yet on sdaa.")
class TestGatherOp4(TestGatherOp):
    def config(self):
        """
        For index int16
        """
        self.x_shape = (10, 20)
        self.index = [1, 3, 5]
        self.index_type = np.int16

    def test_check_grad(self):
        pass


class TestGatherOpFp16(TestGatherOp):
    def init_dtype(self):
        self.x_type = np.float16


class TestGatherOp2Fp16(TestGatherOp2):
    def init_dtype(self):
        self.x_type = np.float16


class TestGatherOp3Fp16(TestGatherOp3):
    def init_dtype(self):
        self.x_type = np.float16


class TestGatherOp4Fp16(TestGatherOp4):
    def init_dtype(self):
        self.x_type = np.float16


class TestGatherOpDouble(TestGatherOp):
    def init_dtype(self):
        self.x_type = np.double


class TestGatherOp2Double(TestGatherOp2):
    def init_dtype(self):
        self.x_type = np.double


class TestGatherOp3Double(TestGatherOp3):
    def init_dtype(self):
        self.x_type = np.double


class TestGatherOp4Double(TestGatherOp4):
    def init_dtype(self):
        self.x_type = np.double


@skip_check_grad_ci(reason="The backward test is not supported for uint8.")
class TestGatherOpUint8(TestGatherOp):
    def init_dtype(self):
        self.x_type = np.uint8

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for uint8.")
class TestGatherOp2Uint8(TestGatherOp2):
    def init_dtype(self):
        self.x_type = np.uint8

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for uint8.")
class TestGatherOp3Uint8(TestGatherOp3):
    def init_dtype(self):
        self.x_type = np.uint8

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for uint8.")
class TestGatherOp4Uint8(TestGatherOp4):
    def init_dtype(self):
        self.x_type = np.uint8

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for int8.")
class TestGatherOpInt8(TestGatherOp):
    def init_dtype(self):
        self.x_type = np.int8

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for int8.")
class TestGatherOp2Int8(TestGatherOp2):
    def init_dtype(self):
        self.x_type = np.int8

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for int8.")
class TestGatherOp3Int8(TestGatherOp3):
    def init_dtype(self):
        self.x_type = np.int8

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for int8.")
class TestGatherOp4Int8(TestGatherOp4):
    def init_dtype(self):
        self.x_type = np.int8

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for int16.")
class TestGatherOpInt16(TestGatherOp):
    def init_dtype(self):
        self.x_type = np.int16

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for int16.")
class TestGatherOp2Int16(TestGatherOp2):
    def init_dtype(self):
        self.x_type = np.int16

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for int16.")
class TestGatherOp3Int16(TestGatherOp3):
    def init_dtype(self):
        self.x_type = np.int16

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for int16.")
class TestGatherOp4Int16(TestGatherOp4):
    def init_dtype(self):
        self.x_type = np.int16

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for int32.")
class TestGatherOpInt32(TestGatherOp):
    def init_dtype(self):
        self.x_type = np.int32

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for int32.")
class TestGatherOp2Int32(TestGatherOp2):
    def init_dtype(self):
        self.x_type = np.int32

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for int32.")
class TestGatherOp3Int32(TestGatherOp3):
    def init_dtype(self):
        self.x_type = np.int32

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for int32.")
class TestGatherOp4Int32(TestGatherOp4):
    def init_dtype(self):
        self.x_type = np.int32

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for int64.")
class TestGatherOpInt64(TestGatherOp):
    def init_dtype(self):
        self.x_type = np.int64

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for int64.")
class TestGatherOp2Int64(TestGatherOp2):
    def init_dtype(self):
        self.x_type = np.int64

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for int64.")
class TestGatherOp3Int64(TestGatherOp3):
    def init_dtype(self):
        self.x_type = np.int64

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for int64.")
class TestGatherOp4Int64(TestGatherOp4):
    def init_dtype(self):
        self.x_type = np.int64

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for bool.")
class TestGatherOpBool(TestGatherOp):
    def init_dtype(self):
        self.x_type = np.bool_

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for bool.")
class TestGatherOp2Bool(TestGatherOp2):
    def init_dtype(self):
        self.x_type = np.bool_

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for bool.")
class TestGatherOp3Bool(TestGatherOp3):
    def init_dtype(self):
        self.x_type = np.bool_

    def test_check_grad(self):
        pass


@skip_check_grad_ci(reason="The backward test is not supported for bool.")
class TestGatherOp4Bool(TestGatherOp4):
    def init_dtype(self):
        self.x_type = np.bool_

    def test_check_grad(self):
        pass


class API_TestGather(unittest.TestCase):
    def test_out1(self):
        with base.program_guard(base.Program(), base.Program()):
            data1 = paddle.static.data(name="data1", shape=[-1, 2], dtype="float32")
            index = paddle.static.data(name="index", shape=[-1, 1], dtype="int32")
            out = paddle.gather(data1, index)
            place = paddle.CustomPlace("sdaa", 0)
            exe = base.Executor(place)
            input = np.array([[1, 2], [3, 4], [5, 6]]).astype("float32")
            index_1 = np.array([1, 2]).astype("int32")
            (result,) = exe.run(
                feed={"data1": input, "index": index_1}, fetch_list=[out]
            )
            expected_output = np.array([[3, 4], [5, 6]])
        self.assertTrue(np.allclose(result, expected_output))

    def test_out2(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(name="x", shape=[-1, 2], dtype="float32")
            index = paddle.static.data(name="index", shape=[-1, 1], dtype="int32")
            out = paddle.gather(x, index)
            place = paddle.CustomPlace("sdaa", 0)
            exe = paddle.static.Executor(place)
            x_np = np.array([[1, 2], [3, 4], [5, 6]]).astype("float32")
            index_np = np.array([1, 1]).astype("int32")
            (result,) = exe.run(feed={"x": x_np, "index": index_np}, fetch_list=[out])
            expected_output = gather_numpy(x_np, index_np, axis=0)
        self.assertTrue(np.allclose(result, expected_output))


class API_TestDygraphGather(unittest.TestCase):
    def test_out1(self):
        paddle.disable_static()
        paddle.device.set_device("sdaa")
        input_1 = np.array([[1, 2], [3, 4], [5, 6]])
        index_1 = np.array([1, 2])
        input = paddle.to_tensor(input_1)
        index = paddle.to_tensor(index_1)
        output = paddle.gather(input, index)
        output_np = output.numpy()
        expected_output = np.array([[3, 4], [5, 6]])
        np.testing.assert_allclose(output_np, expected_output, rtol=1e-05)
        paddle.enable_static()

    def test_out12(self):
        paddle.disable_static()
        paddle.device.set_device("sdaa")
        input_1 = np.array([[1, 2], [3, 4], [5, 6]])
        index_1 = np.array([1, 2])
        x = paddle.to_tensor(input_1)
        index = paddle.to_tensor(index_1)
        output = paddle.gather(x, index, axis=0)
        output_np = output.numpy()
        expected_output = gather_numpy(input_1, index_1, axis=0)
        np.testing.assert_allclose(output_np, expected_output, rtol=1e-05)
        paddle.enable_static()


class TestGathertError(unittest.TestCase):
    def test_error1(self):
        paddle.device.set_device("sdaa")
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):

            shape = [8, 9, 6]
            x = paddle.static.data(shape=shape, dtype="int8", name="x")
            axis = paddle.static.data(shape=[1], dtype="float32", name="axis")
            index = paddle.static.data(shape=shape, dtype="int32", name="index")
            index_float = paddle.static.data(
                shape=shape, dtype="float32", name="index_float"
            )

            def test_x_type():
                paddle.gather(x, index)

            self.assertRaises(TypeError, test_x_type)

            def test_index_type():
                paddle.gather(x, index_float)

            self.assertRaises(TypeError, test_index_type)

            def test_axis_dtype():
                paddle.gather(x, index, axis=1.11)

            self.assertRaises(TypeError, test_axis_dtype)

            def test_axis_dtype1():
                paddle.gather(x, index, axis=axis)

            self.assertRaises(TypeError, test_axis_dtype1)

    def test_error2(self):
        paddle.device.set_device("sdaa")
        with base.program_guard(base.Program(), base.Program()):

            shape = [8, 9, 6]
            x = paddle.static.data(shape=shape, dtype="int8", name="x")
            index = paddle.static.data(shape=shape, dtype="int32", name="mask")
            index_float = paddle.static.data(
                shape=shape, dtype="float32", name="index_float"
            )

            def test_x_type():
                paddle.gather(x, index)

            self.assertRaises(TypeError, test_x_type)

            def test_index_type():
                paddle.gather(x, index_float)

            self.assertRaises(TypeError, test_index_type)

    def test_error3(self):
        paddle.device.set_device("sdaa")
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):

            shape = [8, 9, 6]
            x = paddle.static.data(shape=shape, dtype="int32", name="x")
            axis = paddle.static.data(shape=[1], dtype="int32", name="axis")
            index = paddle.static.data(shape=shape, dtype="int32", name="index")
            index_float = paddle.static.data(
                shape=shape, dtype="float32", name="index_float"
            )

            def test_axis_minsize():
                paddle.gather(x, index, axis=-1)

            self.assertRaises(ValueError, test_axis_minsize)

            def test_axis_maxsize():
                paddle.gather(x, index, axis=512)

            self.assertRaises(ValueError, test_axis_maxsize)


class TestCheckOutType(unittest.TestCase):
    def test_out_type(self):
        paddle.device.set_device("sdaa")
        data = paddle.static.data(shape=[16, 10], dtype="int64", name="x")
        index = paddle.static.data(shape=[4], dtype="int64", name="index")
        out = paddle.gather(data, index)
        self.assertTrue(out.dtype == core.VarDesc.VarType.INT64)


if __name__ == "__main__":
    unittest.main()
