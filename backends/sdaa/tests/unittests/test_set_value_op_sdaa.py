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
        x[0, 0] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (0, 0), self.value)
        return x

    def _get_answer(self):
        self.data[0, 0] = self.value


class TestSetValueApi(TestSetValueBase):
    def _run_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(self.program):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            x = self._call_setitem_static_api(x)

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


# 1. Test different type of item: int, Python slice, Paddle Tensor
# 1.1 item is int
class TestSetValueItemInt(TestSetValueApi):
    def _call_setitem(self, x):
        x[0] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, 0, self.value)
        return x

    def _get_answer(self):
        self.data[0] = self.value


# 1.2 item is slice
# 1.2.1 step is 1
class TestSetValueItemSlice(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:2] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, slice(0, 2), self.value)
        return x

    def _get_answer(self):
        self.data[0:2] = self.value


class TestSetValueItemSlice2(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:-1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, slice(0, -1), self.value)
        return x

    def _get_answer(self):
        self.data[0:-1] = self.value


class TestSetValueItemSlice3(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:-1, 0:2] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (slice(0, -1), slice(0, 2)), self.value)
        return x

    def _get_answer(self):
        self.data[0:-1, 0:2] = self.value


class TestSetValueItemSlice4(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:, 1:2, :] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x, (slice(0, None), slice(1, 2), slice(None)), self.value
        )
        return x

    def _get_answer(self):
        self.data[0:, 1:2, :] = self.value


class TestSetValueItemSlice5(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:, 1:1, :] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x, (slice(0, None), slice(1, 1), slice(None)), self.value
        )
        return x

    def _get_answer(self):
        self.data[0:, 1:1, :] = self.value


class TestSetValueItemSliceInWhile(TestSetValueApi):
    def _call_setitem(self, x):
        def cond(i, x):
            return i < 1

        def body(i, x):
            x[i] = self.value
            i = i + 1
            return i, x

        i = paddle.zeros(shape=(1,), dtype="int64")
        i, x = paddle.static.nn.while_loop(cond, body, [i, x])

    def _call_setitem_static_api(self, x):
        def cond(i, x):
            return i < 1

        def body(i, x):
            x = paddle.static.setitem(x, i, self.value)
            i = i + 1
            return i, x

        i = paddle.zeros(shape=(1,), dtype="int64")
        i, x = paddle.static.nn.while_loop(cond, body, [i, x])
        return x

    def _get_answer(self):
        self.data[0] = self.value

    # 重写 test_api 方法，只运行静态图测试
    def test_api(self):
        static_out = self._run_static()
        self._get_answer()

        error_msg = "\nIn {} mode: \nExpected res = \n{}, \n\nbut received : \n{}"
        self.assertTrue(
            (self.data == static_out).all(),
            msg=error_msg.format("static", self.data, static_out),
        )


# 1.2.2 step > 1
class TestSetValueItemSliceStep(TestSetValueApi):
    def set_shape(self):
        self.shape = [5, 5, 5]

    def _call_setitem(self, x):
        x[0:2:2] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, slice(0, 2, 2), self.value)
        return x

    def _get_answer(self):
        self.data[0:2:2] = self.value


class TestSetValueItemSliceStep2(TestSetValueApi):
    def set_shape(self):
        self.shape = [7, 5, 5]

    def _call_setitem(self, x):
        x[0:-1:3] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, slice(0, -1, 3), self.value)
        return x

    def _get_answer(self):
        self.data[0:-1:3] = self.value


class TestSetValueItemSliceStep3(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:-1, 0:2, ::2] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x, (slice(0, -1), slice(0, 2), slice(None, None, 2)), self.value
        )
        return x

    def _get_answer(self):
        self.data[0:-1, 0:2, ::2] = self.value


class TestSetValueItemSliceStep4(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:, 1:2:2, :] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x, (slice(0, None), slice(1, 2, 2), slice(None)), self.value
        )
        return x

    def _get_answer(self):
        self.data[0:, 1:2:2, :] = self.value


# 1.2.3 step < 0
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
        self.shape = [5]

    def set_value(self):
        self.value = np.array([3, 4])

    def _call_setitem(self, x):
        x[1::-1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, slice(1, None, -1), self.value)
        return x

    def _get_answer(self):
        self.data[1::-1] = self.value


class TestSetValueItemSliceNegetiveStep3(TestSetValueApi):
    def set_shape(self):
        self.shape = [3]

    def set_value(self):
        self.value = np.array([3, 4, 5])

    def _call_setitem(self, x):
        x[::-1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, slice(None, None, -1), self.value)
        return x

    def _get_answer(self):
        self.data[::-1] = self.value


class TestSetValueItemSliceNegetiveStep4(TestSetValueApi):
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


# 1.3 item is Ellipsis
class TestSetValueItemEllipsis1(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:, ..., 1:] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (slice(0, None), ..., slice(1, None)), self.value)
        return x

    def _get_answer(self):
        self.data[0:, ..., 1:] = self.value


class TestSetValueItemEllipsis2(TestSetValueApi):
    def _call_setitem(self, x):
        x[0:, ...] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (slice(0, None), ...), self.value)
        return x

    def _get_answer(self):
        self.data[0:, ...] = self.value


class TestSetValueItemEllipsis3(TestSetValueApi):
    def _call_setitem(self, x):
        x[..., 1:] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (..., slice(1, None)), self.value)
        return x

    def _get_answer(self):
        self.data[..., 1:] = self.value


class TestSetValueItemEllipsis4(TestSetValueApi):
    def _call_setitem(self, x):
        x[...] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, ..., self.value)
        return x

    def _get_answer(self):
        self.data[...] = self.value


# 1.4 item is Paddle Tensor
class TestSetValueItemTensor(TestSetValueApi):
    def _call_setitem(self, x):
        zero = paddle.full([], 0, dtype="int32")
        x[zero] = self.value

    def _call_setitem_static_api(self, x):
        zero = paddle.full([], 0, dtype="int32")
        x = paddle.static.setitem(x, zero, self.value)
        return x

    def _get_answer(self):
        self.data[0] = self.value


class TestSetValueItemTensor2(TestSetValueApi):
    def _call_setitem(self, x):
        zero = paddle.full([], 0, dtype="int32")
        two = paddle.full([], 2, dtype="int64")
        x[zero:two] = self.value

    def _call_setitem_static_api(self, x):
        zero = paddle.full([], 0, dtype="int32")
        two = paddle.full([], 2, dtype="int64")
        x = paddle.static.setitem(x, slice(zero, two), self.value)
        return x

    def _get_answer(self):
        self.data[0:2] = self.value


class TestSetValueItemTensor3(TestSetValueApi):
    def _call_setitem(self, x):
        zero = paddle.full([], 0, dtype="int32")
        two = paddle.full([], 2, dtype="int64")
        x[zero:-1, 0:two] = self.value

    def _call_setitem_static_api(self, x):
        zero = paddle.full([], 0, dtype="int32")
        two = paddle.full([], 2, dtype="int64")
        x = paddle.static.setitem(x, (slice(zero, -1), slice(0, two)), self.value)
        return x

    def _get_answer(self):
        self.data[0:-1, 0:2] = self.value


class TestSetValueItemTensor4(TestSetValueApi):
    def _call_setitem(self, x):
        zero = paddle.full([], 0, dtype="int32")
        two = paddle.full([], 2, dtype="int64")
        x[0:-1, zero:2, 0:6:two] = self.value

    def _call_setitem_static_api(self, x):
        zero = paddle.full([], 0, dtype="int32")
        two = paddle.full([], 2, dtype="int64")
        x = paddle.static.setitem(
            x, (slice(0, -1), slice(zero, 2), slice(0, 6, two)), self.value
        )
        return x

    def _get_answer(self):
        self.data[0:-1, 0:2, ::2] = self.value


class TestSetValueItemTensor5(TestSetValueApi):
    def _call_setitem(self, x):
        zero = paddle.full([], 0, dtype="int32")
        two = paddle.full([], 2, dtype="int64")
        x[zero:, 1:2:two, :] = self.value

    def _call_setitem_static_api(self, x):
        zero = paddle.full([], 0, dtype="int32")
        two = paddle.full([], 2, dtype="int64")
        x = paddle.static.setitem(x, (slice(zero, None), slice(1, 2, two)), self.value)
        return x

    def _get_answer(self):
        self.data[0:, 1:2:2, :] = self.value


class TestSetValueItemTensor6(TestSetValueApi):
    def set_shape(self):
        self.shape = [3, 4, 5]

    def _call_setitem(self, x):
        minus1 = paddle.full([], -1, dtype="int32")
        zero = paddle.full([], 0, dtype="int32")
        x[2:zero:minus1, 0:2, 10:-6:minus1] = self.value

    def _call_setitem_static_api(self, x):
        minus1 = paddle.full([], -1, dtype="int32")
        zero = paddle.full([], 0, dtype="int64")
        x = paddle.static.setitem(
            x,
            (slice(2, zero, minus1), slice(0, 2), slice(10, -6, minus1)),
            self.value,
        )
        return x

    def _get_answer(self):
        self.data[2:0:-1, 0:2, ::-1] = self.value


# 1.5 item is None
class TestSetValueItemNone1(TestSetValueApi):
    def _call_setitem(self, x):
        x[None] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, None, self.value)
        return x

    def _get_answer(self):
        self.data[None] = self.value


class TestSetValueItemNone2(TestSetValueApi):
    def _call_setitem(self, x):
        x[0, None, 1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (0, None, 1), self.value)
        return x

    def _get_answer(self):
        self.data[0, None, 1] = self.value


class TestSetValueItemNone3(TestSetValueApi):
    def _call_setitem(self, x):
        x[:, None, None, 1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (slice(None), None, None, 1), self.value)
        return x

    def _get_answer(self):
        self.data[:, None, None, 1] = self.value


class TestSetValueItemNone4(TestSetValueApi):
    def _call_setitem(self, x):
        x[0, 0, None, 1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (0, 0, None, 1), self.value)
        return x

    def _get_answer(self):
        self.data[0, 0, None, 1] = self.value


class TestSetValueItemNone5(TestSetValueApi):
    def _call_setitem(self, x):
        x[0, None, 0, None, 1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (0, None, 0, None, 1), self.value)
        return x

    def _get_answer(self):
        self.data[0, None, 0, None, 1] = self.value


class TestSetValueItemNone6(TestSetValueApi):
    def _call_setitem(self, x):
        x[None, 0, 0, None, 0] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (None, 0, 0, None, 0), self.value)
        return x

    def _get_answer(self):
        self.data[None, 0, 0, None, 0] = self.value


class TestSetValueItemNone7(TestSetValueApi):
    def _call_setitem(self, x):
        x[:, None, 1] = np.zeros(self.shape)[:, None, 0]

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x, (slice(None), None, 1), np.zeros(self.shape)[:, None, 0]
        )
        return x

    def _get_answer(self):
        self.data[:, None, 1] = np.zeros(self.shape)[:, None, 0]


class TestSetValueItemNone8(TestSetValueApi):
    def _call_setitem(self, x):
        x[:, 1, None] = np.zeros(self.shape)[:, 0, None]

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x, (slice(None), 1, None), np.zeros(self.shape)[:, 0, None]
        )
        return x

    def _get_answer(self):
        self.data[:, 1, None] = np.zeros(self.shape)[:, 0, None]


class TestSetValueItemNone9(TestSetValueApi):
    def _call_setitem(self, x):
        x[None, :, 1, ..., None] = np.zeros(self.shape)[0, 0, :, None]

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x,
            (None, slice(None), 1, ..., None),
            np.zeros(self.shape)[0, 0, :, None],
        )
        return x

    def _get_answer(self):
        self.data[None, :, 1, ..., None] = np.zeros(self.shape)[0, 0, :, None]


class TestSetValueItemNone10(TestSetValueApi):
    def _call_setitem(self, x):
        x[..., None, :, None] = np.zeros(self.shape)[..., None, :, None]

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(
            x,
            (..., None, slice(None), None),
            np.zeros(self.shape)[..., None, :, None],
        )
        return x

    def _get_answer(self):
        self.data[..., None, :, None] = np.zeros(self.shape)[..., None, :, None]


# 1.5 item is list or Tensor of bool
# NOTE(zoooo0820): Currently, 1-D List is same to Tuple.
# The semantic of index will be modified later.
class TestSetValueItemBool1(TestSetValueApi):
    def _call_setitem(self, x):
        x[[True, False]] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, [True, False], self.value)
        return x

    def _get_answer(self):
        self.data[[True, False]] = self.value

    def test_api(self):
        dynamic_out = self._run_dynamic()
        self._get_answer()

        error_msg = "\nIn {} mode: \nExpected res = \n{}, \n\nbut received : \n{}"
        self.assertTrue(
            (self.data == dynamic_out).all(),
            msg=error_msg.format("dynamic", self.data, dynamic_out),
        )


class TestSetValueItemBool2(TestSetValueApi):
    def _call_setitem(self, x):
        x[[False, False]] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, [False, False], self.value)
        return x

    def _get_answer(self):
        self.data[[False, False]] = self.value


class TestSetValueItemBool3(TestSetValueItemBool1):
    def _call_setitem(self, x):
        x[[False, True]] = np.zeros(self.shape[2])

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, [False, True], np.zeros(self.shape[2]))
        return x

    def _get_answer(self):
        self.data[[False, True]] = np.zeros(self.shape[2])


class TestSetValueItemBool4(TestSetValueItemBool1):
    def _call_setitem(self, x):
        idx = paddle.assign(np.array([False, True]))
        x[idx] = np.zeros(self.shape[2])

    def _call_setitem_static_api(self, x):
        idx = paddle.assign(np.array([False, True]))
        x = paddle.static.setitem(x, idx, np.zeros(self.shape[2]))
        return x

    def _get_answer(self):
        self.data[np.array([False, True])] = np.zeros(self.shape[2])


class TestSetValueItemBool5(TestSetValueItemBool1):
    def _call_setitem(self, x):
        idx = paddle.assign(np.array([[False, True, False], [True, True, False]]))
        x[idx] = self.value

    def _call_setitem_static_api(self, x):
        idx = paddle.assign(np.array([[False, True, False], [True, True, False]]))
        x = paddle.static.setitem(x, idx, self.value)
        return x

    def _get_answer(self):
        self.data[np.array([[False, True, False], [True, True, False]])] = self.value


class TestSetValueItemBool6(TestSetValueItemBool1):
    def _call_setitem(self, x):
        x[0, ...] = 0
        x[x > 0] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (0, ...), 0)
        x = paddle.static.setitem(x, x > 0, self.value)
        return x

    def _get_answer(self):
        self.data[0, ...] = 0
        self.data[self.data > 0] = self.value


def create_test_value_int32(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = 7

        def set_dtype(self):
            self.dtype = "int32"

    cls_name = "{}_{}".format(parent.__name__, "ValueInt32")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_int32(TestSetValueItemInt)
create_test_value_int32(TestSetValueItemSlice)
create_test_value_int32(TestSetValueItemSlice2)
create_test_value_int32(TestSetValueItemSlice3)
create_test_value_int32(TestSetValueItemSlice4)


def create_test_value_int64(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = 7

        def set_dtype(self):
            self.dtype = "int64"

    cls_name = "{}_{}".format(parent.__name__, "ValueInt64")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_int64(TestSetValueItemInt)
create_test_value_int64(TestSetValueItemSlice)
create_test_value_int64(TestSetValueItemSlice2)
create_test_value_int64(TestSetValueItemSlice3)
create_test_value_int64(TestSetValueItemSlice4)


def create_test_value_fp16(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = 3.7

        def set_dtype(self):
            self.dtype = "float16"

    cls_name = "{}_{}".format(parent.__name__, "Valuefp16")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_fp16(TestSetValueItemInt)
create_test_value_fp16(TestSetValueItemSlice)
create_test_value_fp16(TestSetValueItemSlice2)
create_test_value_fp16(TestSetValueItemSlice3)
create_test_value_fp16(TestSetValueItemSlice4)


def create_test_value_fp32(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = 3.3

        def set_dtype(self):
            self.dtype = "float32"

    cls_name = "{}_{}".format(parent.__name__, "ValueFp32")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_fp32(TestSetValueItemInt)
create_test_value_fp32(TestSetValueItemSlice)
create_test_value_fp32(TestSetValueItemSlice2)
create_test_value_fp32(TestSetValueItemSlice3)
create_test_value_fp32(TestSetValueItemSlice4)


# 2.2 value is numpy.array (int32, int64, float32, float64, bool)
def create_test_value_numpy_int32(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = np.array([5])

        def set_dtype(self):
            self.dtype = "int32"

    cls_name = "{}_{}".format(parent.__name__, "ValueNumpyInt32")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_numpy_int32(TestSetValueItemInt)
create_test_value_numpy_int32(TestSetValueItemSlice)
create_test_value_numpy_int32(TestSetValueItemSlice2)
create_test_value_numpy_int32(TestSetValueItemSlice3)
create_test_value_numpy_int32(TestSetValueItemSlice4)


def create_test_value_numpy_int64(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = np.array([1])

        def set_dtype(self):
            self.dtype = "int64"

    cls_name = "{}_{}".format(parent.__name__, "ValueNumpyInt64")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_numpy_int64(TestSetValueItemInt)
create_test_value_numpy_int64(TestSetValueItemSlice)
create_test_value_numpy_int64(TestSetValueItemSlice2)
create_test_value_numpy_int64(TestSetValueItemSlice3)
create_test_value_numpy_int64(TestSetValueItemSlice4)


def create_test_value_numpy_fp32(parent):
    class TestValueInt(parent):
        def set_value(self):
            self.value = np.array([1])

        def set_dtype(self):
            self.dtype = "float32"

    cls_name = "{}_{}".format(parent.__name__, "ValueNumpyFp32")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_numpy_fp32(TestSetValueItemInt)
create_test_value_numpy_fp32(TestSetValueItemSlice)
create_test_value_numpy_fp32(TestSetValueItemSlice2)
create_test_value_numpy_fp32(TestSetValueItemSlice3)
create_test_value_numpy_fp32(TestSetValueItemSlice4)


# 2.3 value is a Paddle Tensor (int32, int64, float32, float64, bool)
def create_test_value_tensor_int32(parent):
    class TestValueInt(parent):
        def set_dtype(self):
            self.dtype = "int32"

        def _call_setitem(self, x):
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x[0, 1] = value

        def _call_setitem_static_api(self, x):
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x = paddle.static.setitem(x, (0, 1), value)
            return x

        def _get_answer(self):
            self.data[0, 1] = 3

    cls_name = "{}_{}".format(parent.__name__, "ValueTensorInt32")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_tensor_int32(TestSetValueItemInt)
create_test_value_tensor_int32(TestSetValueItemSlice)
create_test_value_tensor_int32(TestSetValueItemSlice2)
create_test_value_tensor_int32(TestSetValueItemSlice3)
create_test_value_tensor_int32(TestSetValueItemSlice4)


def create_test_value_tensor_int64(parent):
    class TestValueInt(parent):
        def set_dtype(self):
            self.dtype = "int64"

        def _call_setitem(self, x):
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x[0, 1] = value

        def _call_setitem_static_api(self, x):
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x = paddle.static.setitem(x, (0, 1), value)
            return x

        def _get_answer(self):
            self.data[0, 1] = 3

    cls_name = "{}_{}".format(parent.__name__, "ValueTensorInt64")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_tensor_int64(TestSetValueItemInt)
create_test_value_tensor_int64(TestSetValueItemSlice)
create_test_value_tensor_int64(TestSetValueItemSlice2)
create_test_value_tensor_int64(TestSetValueItemSlice3)
create_test_value_tensor_int64(TestSetValueItemSlice4)


def create_test_value_tensor_fp32(parent):
    class TestValueInt(parent):
        def set_dtype(self):
            self.dtype = "float32"

        def _call_setitem(self, x):
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x[0, 1] = value

        def _call_setitem_static_api(self, x):
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x = paddle.static.setitem(x, (0, 1), value)
            return x

        def _get_answer(self):
            self.data[0, 1] = 3

    cls_name = "{}_{}".format(parent.__name__, "ValueTensorFp32")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_tensor_fp32(TestSetValueItemInt)
create_test_value_tensor_fp32(TestSetValueItemSlice)
create_test_value_tensor_fp32(TestSetValueItemSlice2)
create_test_value_tensor_fp32(TestSetValueItemSlice3)
create_test_value_tensor_fp32(TestSetValueItemSlice4)


# 3. Test different shape of value
class TestSetValueValueShape1(TestSetValueApi):
    def set_value(self):
        self.value = np.array([3, 4, 5, 6])  # shape is (4,)

    def _call_setitem(self, x):
        x[0] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, 0, self.value)
        return x

    def _get_answer(self):
        self.data[0] = self.value


class TestSetValueValueShape2(TestSetValueApi):
    def set_value(self):
        self.value = np.array([[3, 4, 5, 6]])  # shape is (1,4)

    def _call_setitem(self, x):
        x[0:1] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, slice(0, 1), self.value)
        return x

    def _get_answer(self):
        self.data[0:1] = self.value


class TestSetValueValueShape3(TestSetValueApi):
    def set_value(self):
        self.value = np.array(
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
        )  # shape is (3,4)

    def _call_setitem(self, x):
        x[0] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, 0, self.value)
        return x

    def _get_answer(self):
        self.data[0] = self.value


class TestSetValueValueShape4(TestSetValueApi):
    def set_value(self):
        self.value = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]).astype(
            self.dtype
        )  # shape is (3,4)

    def _call_setitem(self, x):
        x[0] = paddle.assign(self.value)  # x is Paddle.Tensor

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, 0, paddle.assign(self.value))
        return x

    def _get_answer(self):
        self.data[0] = self.value


class TestSetValueValueShape5(TestSetValueApi):
    def set_value(self):
        self.value = np.array([3, 3, 3]).astype(self.dtype)

    def set_shape(self):
        self.shape = [3, 4]

    def _call_setitem(self, x):
        x[:, 0] = paddle.assign(self.value)  # x is Paddle.Tensor

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (slice(None), 0), paddle.assign(self.value))
        return x

    def _get_answer(self):
        self.data[:, 0] = self.value


# This is to test case which dims of indexed Tensor is
# less than value Tensor on CPU / GPU.
class TestSetValueValueShape6(TestSetValueApi):
    def set_value(self):
        self.value = np.ones((1, 4)) * 5

    def set_shape(self):
        self.shape = [4, 4]

    def _call_setitem(self, x):
        x[:, 0] = self.value  # x is Paddle.Tensor

    def _get_answer(self):
        self.data[:, 0] = self.value

    def _call_setitem_static_api(self, x):
        x = paddle.static.setitem(x, (slice(None), 0), self.value)
        return x


# 4. Test error
class TestError(TestSetValueBase):
    def _value_type_error(self):
        with self.assertRaisesRegex(
            TypeError,
            "Only support to assign an integer, float, numpy.ndarray or paddle.Tensor",
        ):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            value = [1]
            if paddle.in_dynamic_mode():
                x[0] = value
            else:
                x = paddle.static.setitem(x, 0, value)

    def _dtype_error(self):
        with self.assertRaisesRegex(
            TypeError,
            "When assign a numpy.ndarray, integer or float to a paddle.Tensor, ",
        ):
            y = paddle.ones(shape=self.shape, dtype="float16")
            y[0] = 1

    def _step_error(self):
        with self.assertRaisesRegex(ValueError, "step can not be 0"):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            if paddle.in_dynamic_mode():
                x[0:1:0] = self.value
            else:
                x = paddle.static.setitem(x, slice(0, 1, 0), self.value)

    def _ellipsis_error(self):
        with self.assertRaisesRegex(
            IndexError, "An index can only have a single ellipsis"
        ):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            x[..., ...] = self.value
        with self.assertRaisesRegex(ValueError, "the start or end is None"):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            one = paddle.ones([1])
            x[::one] = self.value

    def _bool_list_error(self):
        with self.assertRaises(IndexError):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            if paddle.in_dynamic_mode():
                x[[True, False], [True, False]] = 0
            else:
                x = paddle.static.setitem(x, ([True, False], [True, False]), 0)

    def _bool_tensor_error(self):
        with self.assertRaises(IndexError):
            x = paddle.ones(shape=self.shape, dtype=self.dtype)
            idx = paddle.assign([True, False, True])
            if paddle.in_dynamic_mode():
                x[idx] = 0
            else:
                x = paddle.static.setitem(x, idx, 0)

    def test_error(self):
        paddle.enable_static()
        with paddle.static.program_guard(self.program):
            self._value_type_error()
            self._bool_list_error()
            self._bool_tensor_error()


class TestSetValueInplace(unittest.TestCase):
    def test_inplace(self):
        paddle.disable_static()
        with paddle.base.dygraph.guard():
            paddle.seed(100)
            a = paddle.rand(shape=[1, 4])
            a.stop_gradient = False
            b = a[:] * 1
            c = b
            b[paddle.zeros([], dtype="int64")] = 1.0

            self.assertTrue(id(b) == id(c))
            np.testing.assert_array_equal(b.numpy(), c.numpy())
            self.assertEqual(b.inplace_version, 1)

        paddle.enable_static()


class TestSetValueInplaceLeafVar(unittest.TestCase):
    def test_inplace_var_become_leaf_var(self):

        paddle.set_device("sdaa")
        paddle.disable_static()

        a_grad_1, b_grad_1, a_grad_2, b_grad_2 = 0, 1, 2, 3
        with paddle.base.dygraph.guard():
            paddle.seed(100)
            a = paddle.rand(shape=[1, 4])
            b = paddle.rand(shape=[1, 4])
            a.stop_gradient = False
            b.stop_gradient = False
            c = a / b
            c.sum().backward()
            a_grad_1 = a.grad.numpy()
            b_grad_1 = b.grad.numpy()

        with paddle.base.dygraph.guard():
            paddle.seed(100)
            a = paddle.rand(shape=[1, 4])
            b = paddle.rand(shape=[1, 4])
            a.stop_gradient = False
            b.stop_gradient = False
            c = a / b
            d = paddle.zeros((4, 4))
            self.assertTrue(d.stop_gradient)
            d[0, :] = c
            self.assertFalse(d.stop_gradient)
            d[0, :].sum().backward()
            a_grad_2 = a.grad.numpy()
            b_grad_2 = b.grad.numpy()

        np.testing.assert_array_equal(a_grad_1, a_grad_2)
        np.testing.assert_array_equal(b_grad_1, b_grad_2)
        paddle.enable_static()


class TestSetValueIsSamePlace(unittest.TestCase):
    def test_is_same_place(self):
        paddle.disable_static()
        paddle.seed(100)
        paddle.set_device("sdaa")
        a = paddle.rand(shape=[2, 3, 4])
        origin_place = a.place
        a[[0, 1], 1] = 10
        self.assertEqual(origin_place._type(), a.place._type())
        paddle.enable_static()


# Test customized fallback
def create_test_value_int32_fallback(parent):
    class TestValueInt(parent):
        def set_shape(self):
            self.shape = [200, 10, 4]

        def set_value(self):
            self.value = 7

        def set_dtype(self):
            self.dtype = "int32"

    cls_name = "{}_{}".format(parent.__name__, "ValueInt32")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_int32_fallback(TestSetValueItemInt)
create_test_value_int32_fallback(TestSetValueItemSlice)
create_test_value_int32_fallback(TestSetValueItemSlice2)
create_test_value_int32_fallback(TestSetValueItemSlice3)
create_test_value_int32_fallback(TestSetValueItemSlice4)


def create_test_value_int64_fallback(parent):
    class TestValueInt(parent):
        def set_shape(self):
            self.shape = [200, 10, 4]

        def set_value(self):
            self.value = 7

        def set_dtype(self):
            self.dtype = "int64"

    cls_name = "{}_{}".format(parent.__name__, "ValueInt64")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_int64_fallback(TestSetValueItemInt)
create_test_value_int64_fallback(TestSetValueItemSlice)
create_test_value_int64_fallback(TestSetValueItemSlice2)
create_test_value_int64_fallback(TestSetValueItemSlice3)
create_test_value_int64_fallback(TestSetValueItemSlice4)


def create_test_value_fp16_fallback(parent):
    class TestValueInt(parent):
        def set_shape(self):
            self.shape = [200, 10, 4]

        def set_value(self):
            self.value = 3.7

        def set_dtype(self):
            self.dtype = "float16"

    cls_name = "{}_{}".format(parent.__name__, "Valuefp16")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_fp16_fallback(TestSetValueItemInt)
create_test_value_fp16_fallback(TestSetValueItemSlice)
create_test_value_fp16_fallback(TestSetValueItemSlice2)
create_test_value_fp16_fallback(TestSetValueItemSlice3)
create_test_value_fp16_fallback(TestSetValueItemSlice4)


def create_test_value_fp32_fallback(parent):
    class TestValueInt(parent):
        def set_shape(self):
            self.shape = [200, 10, 4]

        def set_value(self):
            self.value = 3.3

        def set_dtype(self):
            self.dtype = "float32"

    cls_name = "{}_{}".format(parent.__name__, "ValueFp32")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_fp32_fallback(TestSetValueItemInt)
create_test_value_fp32_fallback(TestSetValueItemSlice)
create_test_value_fp32_fallback(TestSetValueItemSlice2)
create_test_value_fp32_fallback(TestSetValueItemSlice3)
create_test_value_fp32_fallback(TestSetValueItemSlice4)


def create_test_value_numpy_int32_fallback(parent):
    class TestValueInt(parent):
        def set_shape(self):
            self.shape = [150, 10, 4]

        def set_value(self):
            self.value = np.array([5])

        def set_dtype(self):
            self.dtype = "int32"

    cls_name = "{}_{}".format(parent.__name__, "ValueNumpyInt32")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_numpy_int32_fallback(TestSetValueItemInt)
create_test_value_numpy_int32_fallback(TestSetValueItemSlice)
create_test_value_numpy_int32_fallback(TestSetValueItemSlice2)
create_test_value_numpy_int32_fallback(TestSetValueItemSlice3)
create_test_value_numpy_int32_fallback(TestSetValueItemSlice4)


def create_test_value_numpy_int64_fallback(parent):
    class TestValueInt(parent):
        def set_shape(self):
            self.shape = [150, 10, 4]

        def set_value(self):
            self.value = np.array([1])

        def set_dtype(self):
            self.dtype = "int64"

    cls_name = "{}_{}".format(parent.__name__, "ValueNumpyInt64")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_numpy_int64_fallback(TestSetValueItemInt)
create_test_value_numpy_int64_fallback(TestSetValueItemSlice)
create_test_value_numpy_int64_fallback(TestSetValueItemSlice2)
create_test_value_numpy_int64_fallback(TestSetValueItemSlice3)
create_test_value_numpy_int64_fallback(TestSetValueItemSlice4)


def create_test_value_numpy_fp32_fallback(parent):
    class TestValueInt(parent):
        def set_shape(self):
            self.shape = [150, 10, 4]

        def set_value(self):
            self.value = np.array([1])

        def set_dtype(self):
            self.dtype = "float32"

    cls_name = "{}_{}".format(parent.__name__, "ValueNumpyFp32")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_numpy_fp32_fallback(TestSetValueItemInt)
create_test_value_numpy_fp32_fallback(TestSetValueItemSlice)
create_test_value_numpy_fp32_fallback(TestSetValueItemSlice2)
create_test_value_numpy_fp32_fallback(TestSetValueItemSlice3)
create_test_value_numpy_fp32_fallback(TestSetValueItemSlice4)


# 2.3 value is a Paddle Tensor (int32, int64, float32, float64, bool)
def create_test_value_tensor_int32_fallback(parent):
    class TestValueInt(parent):
        def set_shape(self):
            self.shape = [150, 10, 4]

        def set_dtype(self):
            self.dtype = "int32"

        def _call_setitem(self, x):
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x[0, 1] = value

        def _call_setitem_static_api(self, x):
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x = paddle.static.setitem(x, (0, 1), value)
            return x

        def _get_answer(self):
            self.data[0, 1] = 3

    cls_name = "{}_{}".format(parent.__name__, "ValueTensorInt32")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_tensor_int32_fallback(TestSetValueItemInt)
create_test_value_tensor_int32_fallback(TestSetValueItemSlice)
create_test_value_tensor_int32_fallback(TestSetValueItemSlice2)
create_test_value_tensor_int32_fallback(TestSetValueItemSlice3)
create_test_value_tensor_int32_fallback(TestSetValueItemSlice4)


def create_test_value_tensor_int64_fallback(parent):
    class TestValueInt(parent):
        def set_shape(self):
            self.shape = [150, 10, 4]

        def set_dtype(self):
            self.dtype = "int64"

        def _call_setitem(self, x):
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x[0, 1] = value

        def _call_setitem_static_api(self, x):
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x = paddle.static.setitem(x, (0, 1), value)
            return x

        def _get_answer(self):
            self.data[0, 1] = 3

    cls_name = "{}_{}".format(parent.__name__, "ValueTensorInt64")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_tensor_int64_fallback(TestSetValueItemInt)
create_test_value_tensor_int64_fallback(TestSetValueItemSlice)
create_test_value_tensor_int64_fallback(TestSetValueItemSlice2)
create_test_value_tensor_int64_fallback(TestSetValueItemSlice3)
create_test_value_tensor_int64_fallback(TestSetValueItemSlice4)


def create_test_value_tensor_fp32_fallback(parent):
    class TestValueInt(parent):
        def set_shape(self):
            self.shape = [150, 10, 4]

        def set_dtype(self):
            self.dtype = "float32"

        def _call_setitem(self, x):
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x[0, 1] = value

        def _call_setitem_static_api(self, x):
            value = paddle.full(shape=[], fill_value=3, dtype=self.dtype)
            x = paddle.static.setitem(x, (0, 1), value)
            return x

        def _get_answer(self):
            self.data[0, 1] = 3

    cls_name = "{}_{}".format(parent.__name__, "ValueTensorFp32")
    TestValueInt.__name__ = cls_name
    globals()[cls_name] = TestValueInt


create_test_value_tensor_fp32_fallback(TestSetValueItemInt)
create_test_value_tensor_fp32_fallback(TestSetValueItemSlice)
create_test_value_tensor_fp32_fallback(TestSetValueItemSlice2)
create_test_value_tensor_fp32_fallback(TestSetValueItemSlice3)
create_test_value_tensor_fp32_fallback(TestSetValueItemSlice4)

if __name__ == "__main__":
    unittest.main()
