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
import paddle


class TestSetValueBase(unittest.TestCase):
    def set_sdaa(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True

    def setUp(self):
        paddle.enable_static()
        self.set_sdaa()
        self.set_dtype()
        self.set_value()
        self.set_shape()
        self.data = np.ones(self.shape).astype(self.dtype)
        self.program = paddle.static.Program()

    def set_shape(self):
        self.shape = [2, 3, 4]

    def set_value(self):
        self.value = 6

    def set_dtype(self):
        self.dtype = "float32"

    def _call_setitem(self, x):
        zero = paddle.full([1], 0, dtype="int32")
        x[zero] = self.value

    def _get_answer(self):
        self.data[0] = self.value


class TestSetValueApi(TestSetValueBase):
    def _run_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(self.program):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            self._call_setitem(x)

        exe = paddle.static.Executor(self.place)
        out = exe.run(self.program, fetch_list=[x])
        paddle.disable_static()
        return out

    def _run_dynamic(self):
        paddle.disable_static()
        x = paddle.ones(shape=self.shape, dtype=self.dtype)
        self._call_setitem(x)
        out = x.numpy()
        paddle.enable_static()
        return out

    def test_api(self):
        static_out = self._run_static()
        dynamic_out = self._run_dynamic()
        self._get_answer()

        error_msg = "\nIn {} mode: \nExpected res = \n{}, \n\nbut received : \n{}"
        self.assertTrue(
            (self.data == static_out).all(),
            msg=error_msg.format("static", self.data, static_out),
        )

        self.assertTrue(
            (self.data == dynamic_out).all(),
            msg=error_msg.format("dynamic", self.data, dynamic_out),
        )


class TestSetValueValueCase1(TestSetValueApi):
    def set_value(self):
        self.value = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]).astype(
            self.dtype
        )  # shape is (3,4)

    def _call_setitem(self, x):
        x[0] = paddle.assign(self.value)  # x is Paddle.Tensor

    def _get_answer(self):
        self.data[0] = self.value


class TestSetValueValueCase2(TestSetValueApi):
    def set_value(self):
        self.value = np.array([3, 3, 3]).astype(self.dtype)

    def set_shape(self):
        self.shape = [3, 4]

    def _call_setitem(self, x):
        x[:, 0] = paddle.assign(self.value)  # x is Paddle.Tensor

    def _get_answer(self):
        self.data[:, 0] = self.value


def create_test_value_int64(parent):
    class TestValueInt(parent):
        def set_dtype(self):
            self.dtype = "int64"

    cls_name = "{}_{}".format(parent.__name__, "ValueInt64")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_int64(TestSetValueApi)
create_test_value_int64(TestSetValueValueCase1)
create_test_value_int64(TestSetValueValueCase2)


class TestSetValueItemNone(TestSetValueApi):
    def set_value(self):
        self.value = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]).astype(
            self.dtype
        )  # shape is (3,4)

    def _call_setitem(self, x):
        x[0, None] = paddle.assign(self.value)  # x is Paddle.Tensor

    def _get_answer(self):
        self.data[0, None] = self.value


class TestSetValueItemNone1(TestSetValueApi):
    def set_value(self):
        self.value = np.array([[1, 1, 1, 1]]).astype(self.dtype)  # shape is (1,4)

    def _call_setitem(self, x):
        x[0, 0, None] = paddle.assign(self.value)  # x is Paddle.Tensor

    def _get_answer(self):
        self.data[0, 0, None] = self.value


class TestSetValueValueShape(TestSetValueApi):
    def set_value(self):
        self.value = np.ones((1, 4)) * 5

    def set_shape(self):
        self.shape = [4, 4]

    def _call_setitem(self, x):
        x[:, 0] = self.value  # x is Paddle.Tensor

    def _get_answer(self):
        self.data[:, 0] = self.value


class TestSetValueItemSliceInWhile(TestSetValueApi):
    def _call_setitem(self, x):
        def cond(i, x):
            return i < 1

        def body(i, x):
            x[i] = self.value
            i = i + 1
            return i, x

        i = paddle.zeros(shape=(1,), dtype="int32")
        i, x = paddle.static.nn.while_loop(cond, body, [i, x])

    def _call_setitem_static_api(self, x):
        def cond(i, x):
            return i < 1

        def body(i, x):
            x = paddle.static.setitem(x, i, self.value)
            i = i + 1
            return i, x

        i = paddle.zeros(shape=(1,), dtype="int32")
        i, x = paddle.static.nn.while_loop(cond, body, [i, x])
        return x

    def _get_answer(self):
        self.data[0] = self.value


class TestSetValueItemSliceNegetiveStep(TestSetValueApi):
    def set_shape(self):
        self.shape = [5, 2]

    def set_value(self):
        self.value = np.array([3, 4])

    def _call_setitem(self, x):
        x[5:2:-1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, slice(5, 2, -1), self.value)
        return x

    def _get_answer(self):
        self.data[5:2:-1] = self.value


class TestSetValueItemSliceNegetiveStep2(TestSetValueApi):
    def set_shape(self):
        self.shape = [3, 4, 5]

    def _call_setitem(self, x):
        x[2:0:-1, 0:2, ::-1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x, (slice(2, 0, -1), slice(0, 2), slice(None, None, -1)), self.value
        )
        return x

    def _get_answer(self):
        self.data[2:0:-1, 0:2, ::-1] = self.value


if __name__ == "__main__":
    unittest.main()
