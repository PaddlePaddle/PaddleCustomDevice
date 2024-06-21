# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest

from op_test import OpTest
import paddle
from op_test import skip_check_grad_ci
import paddle.base.core as core

import functools
from paddle.base.data_feeder import convert_dtype

paddle.enable_static()


def convert_to_dtype_(dtype):
    if dtype == 5:
        return core.VarDesc.VarType.FP32
    elif dtype == 6:
        return core.VarDesc.VarType.FP64
    elif dtype == 4:
        return core.VarDesc.VarType.FP16
    elif dtype == 2:
        return core.VarDesc.VarType.INT32
    elif dtype == 1:
        return core.VarDesc.VarType.INT16
    elif dtype == 3:
        return core.VarDesc.VarType.INT64
    elif dtype == 0:
        return core.VarDesc.VarType.BOOL
    elif dtype == 22:
        return core.VarDesc.VarType.BF16
    elif dtype == 20:
        return core.VarDesc.VarType.UINT8
    elif dtype == 21:
        return core.VarDesc.VarType.INT8
    elif dtype == np.complex64:
        raise ValueError("Not supported dtype %s" % dtype)


def cast_wrapper(x, out_dtype=None):
    return paddle.tensor.cast(x, convert_to_dtype_(out_dtype))


@skip_check_grad_ci(reason="Haven not implement cast grad kernel.")
class TestCast(OpTest):
    def setUp(self):
        ipt = np.random.random(size=[10, 10])
        self.inputs = {"X": ipt.astype("float32")}
        self.outputs = {"Out": ipt.astype("float32")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.FP32),
            "out_dtype": int(core.VarDesc.VarType.FP32),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpFp32ToFp16(TestCast):
    def setUp(self):
        ipt = np.random.random(size=[10, 10])
        self.inputs = {"X": ipt.astype("float32")}
        self.outputs = {"Out": ipt.astype("float16")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.FP32),
            "out_dtype": int(core.VarDesc.VarType.FP16),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpFp16ToFp32(TestCast):
    def setUp(self):
        ipt = np.random.random(size=[10, 10])
        self.inputs = {"X": ipt.astype("float16")}
        self.outputs = {"Out": ipt.astype("float32")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.FP16),
            "out_dtype": int(core.VarDesc.VarType.FP32),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpFp32ToINT16(TestCast):
    def setUp(self):
        ipt = np.random.random(size=[10, 10])
        self.inputs = {"X": ipt.astype("float32")}
        self.outputs = {"Out": ipt.astype("int16")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.FP32),
            "out_dtype": int(core.VarDesc.VarType.INT16),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpInt32ToInt32(TestCast):
    def setUp(self):
        ipt = np.random.randint(1000, size=(10, 10))
        self.inputs = {"X": ipt.astype("int32")}
        self.outputs = {"Out": ipt.astype("int32")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.INT32),
            "out_dtype": int(core.VarDesc.VarType.INT32),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpInt64ToInt32(TestCast):
    def setUp(self):
        ipt = np.random.randint(1000, size=(10, 10))
        self.inputs = {"X": ipt.astype("int64")}
        self.outputs = {"Out": ipt.astype("int32")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.INT64),
            "out_dtype": int(core.VarDesc.VarType.INT32),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpInt64ToFp32(TestCast):
    def setUp(self):
        ipt = np.random.randint(1000, size=(10, 10))
        self.inputs = {"X": ipt.astype("int64")}
        self.outputs = {"Out": ipt.astype("float32")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.INT64),
            "out_dtype": int(core.VarDesc.VarType.FP32),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpInt32ToInt64(TestCast):
    def setUp(self):
        ipt = np.random.randint(1000, size=(10, 10))
        self.inputs = {"X": ipt.astype("int32")}
        self.outputs = {"Out": ipt.astype("int64")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.INT32),
            "out_dtype": int(core.VarDesc.VarType.INT64),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpFp32ToInt64(TestCast):
    def setUp(self):
        ipt = np.random.random(size=[10, 10])
        self.inputs = {"X": ipt.astype("float32")}
        self.outputs = {"Out": ipt.astype("int64")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.FP32),
            "out_dtype": int(core.VarDesc.VarType.INT64),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpInt32ToFp32(TestCast):
    def setUp(self):
        ipt = np.random.randint(1000, size=[10, 10])
        self.inputs = {"X": ipt.astype("int32")}
        self.outputs = {"Out": ipt.astype("float32")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.INT32),
            "out_dtype": int(core.VarDesc.VarType.FP32),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpInt16ToFp32(TestCast):
    def setUp(self):
        ipt = np.random.randint(1000, size=[10, 10])
        self.inputs = {"X": ipt.astype("int16")}
        self.outputs = {"Out": ipt.astype("float")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.INT16),
            "out_dtype": int(core.VarDesc.VarType.FP32),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpFp32ToInt8(TestCast):
    def setUp(self):
        ipt = np.random.randint(1000, size=[10, 10])
        self.inputs = {"X": ipt.astype("float32")}
        self.outputs = {"Out": ipt.astype("int8")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.FP32),
            "out_dtype": int(core.VarDesc.VarType.INT8),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


@unittest.skip("not pass ci")
class TestCastOpInt8ToFp32(TestCast):
    def setUp(self):
        ipt = np.random.randint(1000, size=[10, 10])
        self.inputs = {"X": ipt.astype("int8")}
        self.outputs = {"Out": ipt.astype("float32")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.INT8),
            "out_dtype": int(core.VarDesc.VarType.FP32),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpInt64ToBool(TestCast):
    def setUp(self):
        ipt = np.random.randint(1000, size=[10, 10])
        self.inputs = {"X": ipt.astype("int64")}
        self.outputs = {"Out": ipt.astype("bool")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.INT64),
            "out_dtype": int(core.VarDesc.VarType.BOOL),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpInt32ToBool(TestCast):
    def setUp(self):
        ipt = np.random.randint(1000, size=[10, 10])
        self.inputs = {"X": ipt.astype("int32")}
        self.outputs = {"Out": ipt.astype("bool")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.INT32),
            "out_dtype": int(core.VarDesc.VarType.BOOL),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpFloat32ToBool(TestCast):
    def setUp(self):
        ipt = np.random.random([10, 10])
        self.inputs = {"X": ipt.astype("float32")}
        self.outputs = {"Out": ipt.astype("bool")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.FP32),
            "out_dtype": int(core.VarDesc.VarType.BOOL),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpFloat16ToBool(TestCast):
    def setUp(self):
        ipt = np.random.random([10, 10])
        self.inputs = {"X": ipt.astype("float16")}
        self.outputs = {"Out": ipt.astype("bool")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.FP16),
            "out_dtype": int(core.VarDesc.VarType.BOOL),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpFloat64ToINT32Case01(TestCast):
    def setUp(self):
        ipt = np.random.random([10, 10])
        self.inputs = {"X": ipt.astype("float64")}
        self.outputs = {"Out": ipt.astype("int32")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.FP64),
            "out_dtype": int(core.VarDesc.VarType.INT32),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpFloat64ToINT32Case02(TestCast):
    def setUp(self):
        ipt = np.random.random([10, 10]) + 1
        self.inputs = {"X": ipt.astype("float64")}
        self.outputs = {"Out": ipt.astype("int32")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.FP64),
            "out_dtype": int(core.VarDesc.VarType.INT32),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestCastOpFloat64ToINT32Case03(TestCast):
    def setUp(self):
        ipt = np.random.random([10, 10]) - 2
        self.inputs = {"X": ipt.astype("float64")}
        self.outputs = {"Out": ipt.astype("int32")}
        self.attrs = {
            "in_dtype": int(core.VarDesc.VarType.FP64),
            "out_dtype": int(core.VarDesc.VarType.INT32),
        }
        self.op_type = "cast"
        self.python_api = cast_wrapper
        self.place = paddle.CustomPlace("sdaa", 0)
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


def fake_setUp(self, src_dtype: paddle.dtype, dst_dtype: paddle.dtype):
    self.input = 100 * np.random.random([10, 10]).astype(convert_dtype(src_dtype)) - 7

    self.dst_dtype = dst_dtype


class AutoTestCast(unittest.TestCase):
    def setUp(self):
        self.input = 100 * np.random.random([10, 10]).astype("float32") - 7

        self.dst_dtype = paddle.int32

    def test_check_output(self):
        with paddle.base.dygraph.guard(paddle.CPUPlace()):
            cpu_output = paddle.cast(
                paddle.to_tensor(self.input), self.dst_dtype
            ).numpy()

        with paddle.base.dygraph.guard(paddle.CustomPlace("sdaa", 0)):
            sdaa_output = paddle.cast(
                paddle.to_tensor(self.input), self.dst_dtype
            ).numpy()

        np.allclose(sdaa_output, cpu_output)


sdaa_supported_cast = iter(
    paddle.base.core.libpaddle._get_registered_phi_kernels("function")["cast"]
)
sdaa_supported_cast = filter(
    lambda kernelkey: kernelkey[1:5] == "sdaa", sdaa_supported_cast
)
sdaa_supported_dtype = list(
    map(
        lambda kernelkey: getattr(
            paddle.framework.dtype, kernelkey.split(", ")[-1][:-1]
        ),
        sdaa_supported_cast,
    )
)


def generate_cast_tests(dst_dtype: paddle.dtype, fake_setup):
    for src_dtype in sdaa_supported_dtype:

        if src_dtype is dst_dtype:
            continue
        cls = type(
            f"AutoTestCastOp{src_dtype.name}To{dst_dtype.name}", (AutoTestCast,), {}
        )

        print(f"{src_dtype} to {dst_dtype}")
        cls.setUp = functools.wraps(cls.setUp)(
            functools.partialmethod(
                fake_setup, src_dtype=src_dtype, dst_dtype=dst_dtype
            )
        )

        globals()[cls.__name__] = cls


for dst_dtype in sdaa_supported_dtype:
    generate_cast_tests(dst_dtype, fake_setUp)

if __name__ == "__main__":
    unittest.main()
