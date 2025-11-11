# BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
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

import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle import base, tensor


class TestUnbind(unittest.TestCase):
    def test_unbind(self):

        paddle.enable_static()
        x_1 = paddle.static.data(shape=[2, 3], dtype="float32", name="x_1")
        [out_0, out_1] = tensor.unbind(input=x_1, axis=0)
        input_1 = np.random.random([2, 3]).astype("float32")
        axis = paddle.static.data(shape=[], dtype="int32", name="axis")
        exe = base.Executor(place=paddle.CustomPlace("sdaa", 0))

        [res_1, res_2] = exe.run(
            base.default_main_program(),
            feed={"x_1": input_1, "axis": 0},
            fetch_list=[out_0, out_1],
        )

        assert np.array_equal(res_1, input_1[0, 0:100])
        assert np.array_equal(res_2, input_1[1, 0:100])

    def test_unbind_static_fp16_sdaa(self):
        place = paddle.CustomPlace("sdaa", 0)
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            input = np.random.random([2, 3]).astype("float16")

            x = paddle.static.data(name="x", shape=[2, 3], dtype="float16")
            y = paddle.unbind(x)

            exe = paddle.static.Executor(place)
            res = exe.run(
                paddle.static.default_main_program(),
                feed={
                    "x": input,
                },
                fetch_list=[y],
            )

            assert np.array_equal(res[0], input[0, :])
            assert np.array_equal(res[1], input[1, :])

    def test_unbind_dygraph(self):
        with base.dygraph.guard(paddle.CustomPlace("sdaa", 0)):
            np_x = np.random.random([2, 3]).astype("float32")
            x = paddle.to_tensor(np_x)
            x.stop_gradient = False
            [res_1, res_2] = paddle.unbind(x, 0)
            np.testing.assert_array_equal(res_1, np_x[0, 0:100])
            np.testing.assert_array_equal(res_2, np_x[1, 0:100])

            out = paddle.add_n([res_1, res_2])

            np_grad = np.ones(x.shape, np.float32)
            out.backward()
            np.testing.assert_array_equal(x.grad.numpy(False), np_grad)


class TestLayersUnbind(unittest.TestCase):
    def test_layers_unbind(self):
        paddle.enable_static()

        x_1 = paddle.static.data(shape=[2, 3], dtype="float32", name="x_1")
        [out_0, out_1] = paddle.unbind(input=x_1, axis=0)
        input_1 = np.random.random([2, 3]).astype("float32")
        axis = paddle.static.data(shape=[], dtype="int32", name="axis")
        exe = base.Executor(place=paddle.CustomPlace("sdaa", 0))

        [res_1, res_2] = exe.run(
            base.default_main_program(),
            feed={"x_1": input_1, "axis": 0},
            fetch_list=[out_0, out_1],
        )

        assert np.array_equal(res_1, input_1[0, 0:100])
        assert np.array_equal(res_2, input_1[1, 0:100])


class TestUnbindOp(OpTest):
    def initParameters(self):
        pass

    def outReshape(self):
        self.out[0] = self.out[0].reshape((2, 2))
        self.out[1] = self.out[1].reshape((2, 2))
        self.out[2] = self.out[2].reshape((2, 2))

    def setAxis(self):
        pass

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def setUp(self):
        self.set_sdaa()
        self._set_op_type()
        self.dtype = self.get_dtype()
        self.axis = 0
        self.num = 3
        self.initParameters()
        x = np.arange(12).reshape(3, 2, 2).astype(self.dtype)
        self.out = np.split(x, self.num, self.axis)
        self.outReshape()
        self.inputs = {"X": x}
        self.attrs = {"axis": self.axis}
        self.setAxis()
        self.outputs = {
            "Out": [("out%d" % i, self.out[i]) for i in range(len(self.out))]
        }
        self.python_api = paddle.unbind
        self.python_out_sig = ["out%d" % i for i in range(len(self.out))]

    def get_dtype(self):
        return "float32"

    def _set_op_type(self):
        self.op_type = "unbind"

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], ["out0", "out1", "out2"])


class TestUnbindBF16Op(OpTest):
    def initParameters(self):
        pass

    def outReshape(self):
        self.out[0] = self.out[0].reshape((2, 2))
        self.out[1] = self.out[1].reshape((2, 2))
        self.out[2] = self.out[2].reshape((2, 2))

    def setAxis(self):
        pass

    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def setUp(self):
        self.set_sdaa()
        self._set_op_type()
        self.dtype = self.get_dtype()
        self.axis = 0
        self.num = 3
        self.initParameters()
        x = np.arange(12).reshape(3, 2, 2).astype(np.float32)
        self.out = np.split(x, self.num, self.axis)
        self.outReshape()
        self.inputs = {"X": convert_float_to_uint16(x)}
        self.attrs = {"axis": self.axis}
        self.setAxis()
        self.outputs = {
            "Out": [
                ("out%d" % i, convert_float_to_uint16(self.out[i]))
                for i in range(len(self.out))
            ]
        }
        self.python_api = paddle.unbind
        self.python_out_sig = ["out%d" % i for i in range(len(self.out))]

    def get_dtype(self):
        return np.uint16

    def _set_op_type(self):
        self.op_type = "unbind"

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        with base.dygraph.guard():
            x = paddle.to_tensor(self.inputs["X"])
            x.stop_gradient = False
            y = paddle.unbind(x, axis=self.attrs["axis"])
            dx = paddle.grad(y, x)[0].numpy()
            dx_expected = convert_float_to_uint16(
                np.ones(self.inputs["X"].shape, np.float32)
            )
            np.testing.assert_array_equal(dx, dx_expected)


class TestUnbindOp1(TestUnbindOp):
    def initParameters(self):
        self.axis = 1
        self.num = 2

    def outReshape(self):
        self.out[0] = self.out[0].reshape((3, 2))
        self.out[1] = self.out[1].reshape((3, 2))

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], ["out0", "out1"])


class TestUnbindOp2(TestUnbindOp):
    def initParameters(self):
        self.axis = 2
        self.num = 2

    def outReshape(self):
        self.out[0] = self.out[0].reshape((3, 2))
        self.out[1] = self.out[1].reshape((3, 2))

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], ["out0", "out1"])


class TestUnbindOp3(TestUnbindOp):
    def initParameters(self):
        self.axis = 2
        self.num = 2

    def setAxis(self):
        self.attrs = {"axis": -1}

    def outReshape(self):
        self.out[0] = self.out[0].reshape((3, 2))
        self.out[1] = self.out[1].reshape((3, 2))

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], ["out0", "out1"])


class TestUnbindOp4(TestUnbindOp):
    def initParameters(self):
        self.axis = 1
        self.num = 2

    def setAxis(self):
        self.attrs = {"axis": -2}

    def outReshape(self):
        self.out[0] = self.out[0].reshape((3, 2))
        self.out[1] = self.out[1].reshape((3, 2))

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], ["out0", "out1"])


class TestUnbindFP16Op(TestUnbindOp):
    def get_dtype(self):
        return np.float16

    def test_check_grad(self):
        pass


class TestUnbindInt32(TestUnbindOp):
    def get_dtype(self):
        return np.int32

    def test_check_grad(self):
        pass


class TestUnbindInt64(TestUnbindOp):
    def get_dtype(self):
        return np.int64

    def test_check_grad(self):
        pass


if __name__ == "__main__":
    unittest.main()
