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

import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16, paddle_static_guard

import paddle
from paddle import base


class TestBmmOp(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "bmm"
        self.python_api = paddle.tensor.bmm
        X = np.random.random((10, 3, 4)).astype("float32")
        Y = np.random.random((10, 4, 5)).astype("float32")
        self.inputs = {"X": X, "Y": Y}
        Out = np.matmul(X, Y)
        self.outputs = {"Out": Out}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ["X", "Y"], "Out", numeric_place=paddle.CPUPlace()
        )


class TestBmmFP16Op(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "bmm"
        self.dtype = np.float16
        self.python_api = paddle.tensor.bmm
        X = np.random.random((10, 3, 4)).astype("float16")
        Y = np.random.random((10, 4, 5)).astype("float16")
        self.inputs = {"X": X, "Y": Y}
        Out = np.matmul(X, Y)
        self.outputs = {"Out": Out}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X", "Y"], "Out")


class TestBmmBF16Op(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("sdaa", 0)

    def setUp(self):
        self.set_sdaa()
        self.op_type = "bmm"
        self.dtype = np.uint16
        self.python_api = paddle.tensor.bmm
        X = np.random.random((10, 3, 4)).astype("float32")
        Y = np.random.random((10, 4, 5)).astype("float32")
        self.inputs = {"X": X, "Y": Y}
        Out = np.matmul(X, Y)
        self.outputs = {"Out": Out}

        self.inputs["X"] = convert_float_to_uint16(self.inputs["X"])
        self.inputs["Y"] = convert_float_to_uint16(self.inputs["Y"])
        self.outputs["Out"] = convert_float_to_uint16(self.outputs["Out"])

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X", "Y"], "Out")


class API_TestBmm(unittest.TestCase):
    def test_out(self):
        with paddle_static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                data1 = paddle.static.data("data1", shape=[-1, 3, 4], dtype="float32")
                data2 = paddle.static.data("data2", shape=[-1, 4, 5], dtype="float32")
                result_bmm = paddle.bmm(data1, data2)
                place = paddle.CustomPlace("sdaa", 0)
                exe = base.Executor(place)
                input1 = np.random.random([10, 3, 4]).astype("float32")
                input2 = np.random.random([10, 4, 5]).astype("float32")
                (result,) = exe.run(
                    feed={"data1": input1, "data2": input2},
                    fetch_list=[result_bmm],
                )
                expected_result = np.matmul(input1, input2)
            np.testing.assert_allclose(expected_result, result, rtol=1e-05)


class API_TestDygraphBmm(unittest.TestCase):
    def test_out(self):
        input1 = np.array(
            [
                [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]],
            ]
        )
        input2 = np.array(
            [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
            ]
        )
        place = paddle.CustomPlace("sdaa", 0)
        with base.dygraph.guard(place):
            x = paddle.to_tensor(input1)
            y = paddle.to_tensor(input2)
            out = paddle.bmm(x, y)
            out_np = out.numpy()
        expected_result = np.matmul(input1, input2)
        np.testing.assert_allclose(expected_result, out_np, rtol=1e-05)


class TestBmmAPIError(unittest.TestCase):
    def test_api_error(self):
        x_data = np.arange(24, dtype="float32").reshape((2, 3, 4))
        y_data = np.arange(16, dtype="float32").reshape((2, 4, 2))
        y_data_wrong1 = np.arange(16, dtype="float32").reshape((2, 2, 4))
        y_data_wrong2 = np.arange(16, dtype="float32").reshape((2, 2, 2, 2))
        y_data_wrong3 = np.arange(24, dtype="float32").reshape((3, 4, 2))
        self.assertRaises(ValueError, paddle.bmm, x_data, y_data_wrong1)
        self.assertRaises(ValueError, paddle.bmm, x_data, y_data_wrong2)
        self.assertRaises(ValueError, paddle.bmm, x_data, y_data_wrong3)


if __name__ == "__main__":
    unittest.main()
