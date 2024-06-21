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

# from __future__ import print_function

from __future__ import print_function

import numpy as np
import unittest
from op_test import OpTest
import paddle

paddle.enable_static()


def np_masked_select(x, mask):
    result = np.empty(shape=(0), dtype=x.dtype)
    for ele, ma in zip(np.nditer(x), np.nditer(mask)):
        if ma:
            result = np.append(result, ele)
    return result.flatten()


class TestMaskedSelectOp(OpTest):
    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def setUp(self):
        self.set_sdaa()
        self.init_shape()
        self.init_dtype()
        self.python_api = paddle.masked_select
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "masked_select"
        x = np.random.random(self.shape).astype(self.dtype)
        mask = np.array(np.random.randint(2, size=self.shape, dtype=bool))
        out = np_masked_select(x, mask)
        self.inputs = {"X": x, "Mask": mask}
        self.outputs = {"Y": out}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Y")

    def init_shape(self):
        self.shape = (50, 3)

    def init_dtype(self):
        self.dtype = np.float32


class TestMaskedSelectOp1(TestMaskedSelectOp):
    def init_shape(self):
        self.shape = (6, 8, 9, 18)


class TestMaskedSelectOp2(TestMaskedSelectOp):
    def init_shape(self):
        self.shape = (168,)


@unittest.skip("tecodnn may failure when x_dims not equal to mask_dims")
class TestMaskedSelectDimsless(TestMaskedSelectOp):
    def init_shape(self):
        self.shape = (2, 3, 4)
        self.shape_ = (3, 4)

    def setUp(self):
        self.set_sdaa()
        self.init_shape()
        self.init_dtype()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "masked_select"
        x = np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
        ).astype(self.dtype)
        mask = np.array([[True, False, True, False]]).astype(bool)
        out = np.array([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]]).astype(self.dtype)
        self.inputs = {"X": x, "Mask": mask}
        self.outputs = {"Y": out}


class TestMaskedSelectOpFp16(TestMaskedSelectOp):
    def init_dtype(self):
        self.dtype = np.float16


class TestMaskedSelectAPI(unittest.TestCase):
    def test_imperative_mode(self):
        paddle.disable_static(paddle.CustomPlace("sdaa", 0))
        shape = (88, 6, 8)
        np_x = np.random.random(shape).astype("float32")
        np_mask = np.array(np.random.randint(2, size=shape, dtype=bool))
        x = paddle.to_tensor(np_x)
        mask = paddle.to_tensor(np_mask)
        out = paddle.masked_select(x, mask)
        np_out = np_masked_select(np_x, np_mask)
        self.assertEqual(np.allclose(out.numpy(), np_out), True)
        paddle.enable_static()

    def test_static_mode(self):
        paddle.enable_static()
        shape = [8, 9, 6]
        x = paddle.static.data(shape=shape, dtype="float32", name="x")
        mask = paddle.static.data(shape=shape, dtype="bool", name="mask")
        np_x = np.random.random(shape).astype("float32")
        np_mask = np.array(np.random.randint(2, size=shape, dtype=bool))

        out = paddle.masked_select(x, mask)
        np_out = np_masked_select(np_x, np_mask)

        exe = paddle.static.Executor(place=paddle.CustomPlace("sdaa", 0))

        res = exe.run(
            paddle.static.default_main_program(),
            feed={"x": np_x, "mask": np_mask},
            fetch_list=[out],
        )
        self.assertEqual(np.allclose(res, np_out), True)


class TestMaskedSelectError(unittest.TestCase):
    def test_error(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):

            shape = [8, 9, 6]
            x = paddle.static.data(shape=shape, dtype="float32", name="x")
            mask = paddle.static.data(shape=shape, dtype="bool", name="mask")
            mask_float = paddle.static.data(
                shape=shape, dtype="float32", name="mask_float"
            )
            np_x = np.random.random(shape).astype("float32")
            np_mask = np.array(np.random.randint(2, size=shape, dtype=bool))

            def test_x_type():
                paddle.masked_select(np_x, mask)

            self.assertRaises(TypeError, test_x_type)

            def test_mask_type():
                paddle.masked_select(x, np_mask)

            self.assertRaises(TypeError, test_mask_type)

            def test_mask_dtype():
                paddle.masked_select(x, mask_float)

            self.assertRaises(TypeError, test_mask_dtype)


if __name__ == "__main__":
    unittest.main()
