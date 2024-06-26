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

from op_test import OpTest
import numpy as np
import unittest
import paddle.base as base
import paddle

paddle.enable_static()


def strided_slice_native_forward(input, axes, starts, ends, strides):
    dim = input.ndim
    start = []
    end = []
    stride = []
    for i in range(dim):
        start.append(0)
        end.append(input.shape[i])
        stride.append(1)

    for i in range(len(axes)):
        start[axes[i]] = starts[i]
        end[axes[i]] = ends[i]
        stride[axes[i]] = strides[i]

    result = {
        1: lambda input, start, end, stride: input[start[0] : end[0] : stride[0]],
        2: lambda input, start, end, stride: input[
            start[0] : end[0] : stride[0], start[1] : end[1] : stride[1]
        ],
        3: lambda input, start, end, stride: input[
            start[0] : end[0] : stride[0],
            start[1] : end[1] : stride[1],
            start[2] : end[2] : stride[2],
        ],
        4: lambda input, start, end, stride: input[
            start[0] : end[0] : stride[0],
            start[1] : end[1] : stride[1],
            start[2] : end[2] : stride[2],
            start[3] : end[3] : stride[3],
        ],
        5: lambda input, start, end, stride: input[
            start[0] : end[0] : stride[0],
            start[1] : end[1] : stride[1],
            start[2] : end[2] : stride[2],
            start[3] : end[3] : stride[3],
            start[4] : end[4] : stride[4],
        ],
        6: lambda input, start, end, stride: input[
            start[0] : end[0] : stride[0],
            start[1] : end[1] : stride[1],
            start[2] : end[2] : stride[2],
            start[3] : end[3] : stride[3],
            start[4] : end[4] : stride[4],
            start[5] : end[5] : stride[5],
        ],
    }[dim](input, start, end, stride)

    return result


class TestStrideSliceOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.initTestDtype()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

        self.op_type = "strided_slice"
        self.python_api = paddle.strided_slice
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )

        self.inputs = {"Input": self.input}
        self.outputs = {"Out": self.output}
        self.attrs = {
            "axes": self.axes,
            "starts": self.starts,
            "ends": self.ends,
            "strides": self.strides,
            "infer_flags": self.infer_flags,
        }

    def test_check_output(self):
        self.check_output_with_place(
            self.place,
        )

    def test_check_grad(self):
        pass

    def initTestCase(self):
        self.input = np.random.rand(100).astype(self.dtype)
        self.axes = [0]
        self.starts = [-4]
        self.ends = [-3]
        self.strides = [1]
        self.infer_flags = [1]

    def initTestDtype(self):
        self.dtype = np.float32


class TestStrideSliceOp1(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(100).astype(self.dtype)
        self.axes = [0]
        self.starts = [3]
        self.ends = [8]
        self.strides = [1]
        self.infer_flags = [1]


class TestStrideSliceOp2(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(100).astype(self.dtype)
        self.axes = [0]
        self.starts = [5]
        self.ends = [0]
        self.strides = [-1]
        self.infer_flags = [1]


class TestStrideSliceOp3(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(100).astype(self.dtype)
        self.axes = [0]
        self.starts = [-1]
        self.ends = [-3]
        self.strides = [-1]
        self.infer_flags = [1]


class TestStrideSliceOp4(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 4, 10).astype(self.dtype)
        self.axes = [0, 1, 2]
        self.starts = [0, -1, 0]
        self.ends = [2, -3, 5]
        self.strides = [1, -1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOp5(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(5, 5, 5).astype(self.dtype)
        self.axes = [0, 1, 2]
        self.starts = [1, 0, 0]
        self.ends = [2, 1, 3]
        self.strides = [1, 1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOp6(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(5, 5, 5).astype(self.dtype)
        self.axes = [0, 1, 2]
        self.starts = [1, -1, 0]
        self.ends = [2, -3, 3]
        self.strides = [1, -1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOp7(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(5, 5, 5).astype(self.dtype)
        self.axes = [0, 1, 2]
        self.starts = [1, 0, 0]
        self.ends = [2, 2, 3]
        self.strides = [1, 1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOp8(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(1, 100, 1).astype(self.dtype)
        self.axes = [1]
        self.starts = [1]
        self.ends = [2]
        self.strides = [1]
        self.infer_flags = [1]


class TestStrideSliceOp9(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(1, 100, 1).astype(self.dtype)
        self.axes = [1]
        self.starts = [-1]
        self.ends = [-2]
        self.strides = [-1]
        self.infer_flags = [1]


class TestStrideSliceOp10(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(10, 10).astype(self.dtype)
        self.axes = [0, 1]
        self.starts = [1, 0]
        self.ends = [2, 2]
        self.strides = [1, 1]
        self.infer_flags = [1, 1]


class TestStrideSliceOp11(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 4).astype(self.dtype)
        self.axes = [0, 1, 2, 3]
        self.starts = [1, 0, 0, 0]
        self.ends = [2, 2, 3, 4]
        self.strides = [1, 1, 1, 2]
        self.infer_flags = [1, 1, 1, 1]


class TestStrideSliceOp12(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 4, 5).astype(self.dtype)
        self.axes = [0, 1, 2, 3, 4]
        self.starts = [1, 0, 0, 0, 0]
        self.ends = [2, 2, 3, 4, 4]
        self.strides = [1, 1, 1, 1, 1]
        self.infer_flags = [1, 1, 1, 1]


class TestStrideSliceOp13(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 6, 7, 8).astype(self.dtype)
        self.axes = [0, 1, 2, 3, 4, 5]
        self.starts = [1, 0, 0, 0, 1, 2]
        self.ends = [2, 2, 3, 1, 2, 8]
        self.strides = [1, 1, 1, 1, 1, 2]
        self.infer_flags = [1, 1, 1, 1, 1]


class TestStrideSliceOp14(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(4, 4, 4, 4).astype(self.dtype)
        self.axes = [1, 2, 3]
        self.starts = [-5, 0, -7]
        self.ends = [-1, 2, 4]
        self.strides = [1, 1, 1]
        self.infer_flags = [1, 1, 1]


def create_test_bool_class(parent):
    class TestStrideSliceOpBoolCase(parent):
        def initTestDtype(self):
            self.dtype = np.bool_

    cls_name = "{0}_{1}".format(parent.__name__, "BOOL")
    TestStrideSliceOpBoolCase.__name__ = cls_name
    globals()[cls_name] = TestStrideSliceOpBoolCase


create_test_bool_class(TestStrideSliceOp)
create_test_bool_class(TestStrideSliceOp1)
create_test_bool_class(TestStrideSliceOp2)
create_test_bool_class(TestStrideSliceOp3)
create_test_bool_class(TestStrideSliceOp4)
create_test_bool_class(TestStrideSliceOp5)
create_test_bool_class(TestStrideSliceOp6)
create_test_bool_class(TestStrideSliceOp7)
create_test_bool_class(TestStrideSliceOp8)
create_test_bool_class(TestStrideSliceOp9)
create_test_bool_class(TestStrideSliceOp10)
create_test_bool_class(TestStrideSliceOp11)
create_test_bool_class(TestStrideSliceOp12)
create_test_bool_class(TestStrideSliceOp13)
create_test_bool_class(TestStrideSliceOp14)


def create_test_fp16_class(parent):
    class TestStrideSliceOpFp16Case(parent):
        def initTestDtype(self):
            self.dtype = np.float16

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestStrideSliceOpFp16Case.__name__ = cls_name
    globals()[cls_name] = TestStrideSliceOpFp16Case


create_test_fp16_class(TestStrideSliceOp)
create_test_fp16_class(TestStrideSliceOp1)
create_test_fp16_class(TestStrideSliceOp2)
create_test_fp16_class(TestStrideSliceOp3)
create_test_fp16_class(TestStrideSliceOp4)
create_test_fp16_class(TestStrideSliceOp5)
create_test_fp16_class(TestStrideSliceOp6)
create_test_fp16_class(TestStrideSliceOp7)
create_test_fp16_class(TestStrideSliceOp8)
create_test_fp16_class(TestStrideSliceOp9)
create_test_fp16_class(TestStrideSliceOp10)
create_test_fp16_class(TestStrideSliceOp11)
create_test_fp16_class(TestStrideSliceOp12)
create_test_fp16_class(TestStrideSliceOp13)
create_test_fp16_class(TestStrideSliceOp14)


def create_test_fp64_class(parent):
    class TestStrideSliceOpFp64Case(parent):
        def initTestDtype(self):
            self.dtype = np.float64

    cls_name = "{0}_{1}".format(parent.__name__, "Fp64")
    TestStrideSliceOpFp64Case.__name__ = cls_name
    globals()[cls_name] = TestStrideSliceOpFp64Case


create_test_fp64_class(TestStrideSliceOp)
create_test_fp64_class(TestStrideSliceOp1)
create_test_fp64_class(TestStrideSliceOp2)
create_test_fp64_class(TestStrideSliceOp3)
create_test_fp64_class(TestStrideSliceOp4)
create_test_fp64_class(TestStrideSliceOp5)
create_test_fp64_class(TestStrideSliceOp6)
create_test_fp64_class(TestStrideSliceOp7)
create_test_fp64_class(TestStrideSliceOp8)
create_test_fp64_class(TestStrideSliceOp9)
create_test_fp64_class(TestStrideSliceOp10)
create_test_fp64_class(TestStrideSliceOp11)
create_test_fp64_class(TestStrideSliceOp12)
create_test_fp64_class(TestStrideSliceOp13)
create_test_fp64_class(TestStrideSliceOp14)


def create_test_int32_class(parent):
    class TestStrideSliceOpInt32Case(parent):
        def initTestDtype(self):
            self.dtype = np.int32

    cls_name = "{0}_{1}".format(parent.__name__, "INT32")
    TestStrideSliceOpInt32Case.__name__ = cls_name
    globals()[cls_name] = TestStrideSliceOpInt32Case


create_test_int32_class(TestStrideSliceOp)
create_test_int32_class(TestStrideSliceOp1)
create_test_int32_class(TestStrideSliceOp2)
create_test_int32_class(TestStrideSliceOp3)
create_test_int32_class(TestStrideSliceOp4)
create_test_int32_class(TestStrideSliceOp5)
create_test_int32_class(TestStrideSliceOp6)
create_test_int32_class(TestStrideSliceOp7)
create_test_int32_class(TestStrideSliceOp8)
create_test_int32_class(TestStrideSliceOp9)
create_test_int32_class(TestStrideSliceOp10)
create_test_int32_class(TestStrideSliceOp11)
create_test_int32_class(TestStrideSliceOp12)
create_test_int32_class(TestStrideSliceOp13)
create_test_int32_class(TestStrideSliceOp14)


def create_test_int64_class(parent):
    class TestStrideSliceOpInt64Case(parent):
        def initTestDtype(self):
            self.dtype = np.int64

    cls_name = "{0}_{1}".format(parent.__name__, "INT64")
    TestStrideSliceOpInt64Case.__name__ = cls_name
    globals()[cls_name] = TestStrideSliceOpInt64Case


create_test_int64_class(TestStrideSliceOp)
create_test_int64_class(TestStrideSliceOp1)
create_test_int64_class(TestStrideSliceOp2)
create_test_int64_class(TestStrideSliceOp3)
create_test_int64_class(TestStrideSliceOp4)
create_test_int64_class(TestStrideSliceOp5)
create_test_int64_class(TestStrideSliceOp6)
create_test_int64_class(TestStrideSliceOp7)
create_test_int64_class(TestStrideSliceOp8)
create_test_int64_class(TestStrideSliceOp9)
create_test_int64_class(TestStrideSliceOp10)
create_test_int64_class(TestStrideSliceOp11)
create_test_int64_class(TestStrideSliceOp12)
create_test_int64_class(TestStrideSliceOp13)
create_test_int64_class(TestStrideSliceOp14)


class TestStridedSliceOp_starts_ListTensor(OpTest):
    def setUp(self):
        self.op_type = "strided_slice"
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.config()

        starts_tensor = []
        for index, ele in enumerate(self.starts):
            starts_tensor.append(("x" + str(index), np.ones((1)).astype("int32") * ele))

        self.inputs = {"Input": self.input, "StartsTensorList": starts_tensor}
        self.outputs = {"Out": self.output}
        self.attrs = {
            "axes": self.axes,
            "starts": self.starts_infer,
            "ends": self.ends,
            "strides": self.strides,
            "infer_flags": self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 1]
        self.infer_flags = [1, -1, 1]
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )

        self.starts_infer = [1, 10, 2]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        pass


class TestStridedSliceOp_ends_ListTensor(OpTest):
    def setUp(self):
        self.op_type = "strided_slice"
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.config()

        ends_tensor = []
        for index, ele in enumerate(self.ends):
            ends_tensor.append(("x" + str(index), np.ones((1)).astype("int32") * ele))

        self.inputs = {"Input": self.input, "EndsTensorList": ends_tensor}
        self.outputs = {"Out": self.output}
        self.attrs = {
            "axes": self.axes,
            "starts": self.starts,
            "ends": self.ends_infer,
            "strides": self.strides,
            "infer_flags": self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 0]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 2]
        self.infer_flags = [1, -1, 1]
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )

        self.ends_infer = [3, 1, 4]

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        pass


class TestStridedSliceOp_starts_Tensor(OpTest):
    def setUp(self):
        self.op_type = "strided_slice"
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.config()
        self.inputs = {
            "Input": self.input,
            "StartsTensor": np.array(self.starts, dtype="int32"),
        }
        self.outputs = {"Out": self.output}
        self.attrs = {
            "axes": self.axes,
            # 'starts': self.starts,
            "ends": self.ends,
            "strides": self.strides,
            "infer_flags": self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 1]
        self.infer_flags = [-1, -1, -1]
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        pass


class TestStridedSliceOp_ends_Tensor(OpTest):
    def setUp(self):
        self.op_type = "strided_slice"
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.config()
        self.inputs = {
            "Input": self.input,
            "EndsTensor": np.array(self.ends, dtype="int32"),
        }
        self.outputs = {"Out": self.output}
        self.attrs = {
            "axes": self.axes,
            "starts": self.starts,
            # 'ends': self.ends,
            "strides": self.strides,
            "infer_flags": self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 1]
        self.infer_flags = [-1, -1, -1]
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        pass


class TestStridedSliceOp_listTensor_Tensor(OpTest):
    def setUp(self):
        self.config()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        ends_tensor = []
        for index, ele in enumerate(self.ends):
            ends_tensor.append(("x" + str(index), np.ones((1)).astype("int32") * ele))
        self.op_type = "strided_slice"

        self.inputs = {
            "Input": self.input,
            "StartsTensor": np.array(self.starts, dtype="int32"),
            "EndsTensorList": ends_tensor,
        }
        self.outputs = {"Out": self.output}
        self.attrs = {
            "axes": self.axes,
            # 'starts': self.starts,
            # 'ends': self.ends,
            "strides": self.strides,
            "infer_flags": self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 1]
        self.infer_flags = [-1, -1, -1]
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        pass


class TestStridedSliceOp_strides_Tensor(OpTest):
    def setUp(self):
        self.op_type = "strided_slice"
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.config()
        self.inputs = {
            "Input": self.input,
            "StridesTensor": np.array(self.strides, dtype="int32"),
        }
        self.outputs = {"Out": self.output}
        self.attrs = {
            "axes": self.axes,
            "starts": self.starts,
            "ends": self.ends,
            # 'strides': self.strides,
            "infer_flags": self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, -1, 2]
        self.ends = [2, 0, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, -1, 1]
        self.infer_flags = [-1, -1, -1]
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides
        )

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad_normal(self):
        pass


# Test python API
class TestStridedSliceAPI(unittest.TestCase):
    def test_1(self):
        input = np.random.random([3, 4, 5, 6]).astype("float32")
        minus_1 = paddle.tensor.fill_constant([], "int32", -1)
        minus_3 = paddle.tensor.fill_constant([], "int32", -3)
        starts = paddle.static.data(name="starts", shape=[3], dtype="int32")
        ends = paddle.static.data(name="ends", shape=[3], dtype="int32")
        strides = paddle.static.data(name="strides", shape=[3], dtype="int32")

        x = paddle.static.data(name="x", shape=[3, 4, 5, 6], dtype="float32")
        out_1 = paddle.strided_slice(
            x, axes=[0, 1, 2], starts=[-3, 0, 2], ends=[3, 100, -1], strides=[1, 1, 1]
        )
        out_2 = paddle.strided_slice(
            x,
            axes=[0, 1, 3],
            starts=[minus_3, 0, 2],
            ends=[3, 100, -1],
            strides=[1, 1, 1],
        )
        out_3 = paddle.strided_slice(
            x,
            axes=[0, 1, 3],
            starts=[minus_3, 0, 2],
            ends=[3, 100, minus_1],
            strides=[1, 1, 1],
        )
        out_4 = paddle.strided_slice(
            x, axes=[0, 1, 2], starts=starts, ends=ends, strides=strides
        )

        out_5 = x[-3:3, 0:100:2, -1:2:-1]
        out_6 = x[minus_3:3:1, 0:100:2, :, minus_1:2:minus_1]
        out_7 = x[minus_1, 0:100:2, :, -1:2:-1]

        exe = base.Executor(place=paddle.CustomPlace("sdaa", 0))
        res_1, res_2, res_3, res_4, res_5, res_6, res_7 = exe.run(
            base.default_main_program(),
            feed={
                "x": input,
                "starts": np.array([-3, 0, 2]).astype("int32"),
                "ends": np.array([3, 2147483647, -1]).astype("int32"),
                "strides": np.array([1, 1, 1]).astype("int32"),
            },
            fetch_list=[out_1, out_2, out_3, out_4, out_5, out_6, out_7],
        )
        assert np.array_equal(res_1, input[-3:3, 0:100, 2:-1, :])
        assert np.array_equal(res_2, input[-3:3, 0:100, :, 2:-1])
        assert np.array_equal(res_3, input[-3:3, 0:100, :, 2:-1])
        assert np.array_equal(res_4, input[-3:3, 0:100, 2:-1, :])
        assert np.array_equal(res_5, input[-3:3, 0:100:2, -1:2:-1, :])
        assert np.array_equal(res_6, input[-3:3, 0:100:2, :, -1:2:-1])
        assert np.array_equal(res_7, input[-1, 0:100:2, :, -1:2:-1])

    def test_dygraph_op(self):
        x = paddle.zeros(shape=[3, 4, 5, 6], dtype="float32")
        axes = [1, 2, 3]
        starts = [-3, 0, 2]
        ends = [3, 2, 4]
        strides_1 = [1, 1, 1]
        sliced_1 = paddle.strided_slice(
            x, axes=axes, starts=starts, ends=ends, strides=strides_1
        )
        assert sliced_1.shape == (3, 2, 2, 2)


if __name__ == "__main__":
    unittest.main()
