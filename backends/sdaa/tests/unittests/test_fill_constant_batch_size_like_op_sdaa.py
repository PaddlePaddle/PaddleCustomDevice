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

from op_test import OpTest
import paddle
from paddle.base.framework import convert_np_dtype_to_dtype_

paddle.enable_static()
SEED = 2021


def fill_constant_batch_size_like(
    input,
    shape,
    value,
    data_type,
    input_dim_idx=0,
    output_dim_idx=0,
    force_cpu=False,
):
    return paddle.base.layers.fill_constant_batch_size_like(
        input, shape, data_type, value, input_dim_idx, output_dim_idx, force_cpu
    )


class TestFillConstantBatchSizeLike(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "fill_constant_batch_size_like"
        self.python_api = fill_constant_batch_size_like
        self.init_shape()
        self.init_value()
        self.init_dtype()
        self.init_force_cpu()
        self.init_dim_idx()

        self.inputs = {"Input": np.random.random(self.input_shape).astype("float32")}
        self.attrs = {
            "shape": self.shape,
            "value": self.value,
            "str_value": self.str_value,
            "dtype": convert_np_dtype_to_dtype_(self.dtype),
            "force_cpu": self.force_cpu,
            "input_dim_idx": self.input_dim_idx,
            "output_dim_idx": self.output_dim_idx,
        }
        self.outputs = {
            "Out": np.full(self.output_shape, self.output_value, self.dtype)
        }

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_shape(self):
        self.input_shape = [4, 5]
        self.shape = [123, 92]
        self.output_shape = (4, 92)

    def init_value(self):
        self.value = 3.8
        self.str_value = ""
        self.output_value = 3.8

    def init_dtype(self):
        self.dtype = np.float32

    def init_force_cpu(self):
        self.force_cpu = False

    def init_dim_idx(self):
        self.input_dim_idx = 0
        self.output_dim_idx = 0

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestFillConstantBatchSizeLike2(TestFillConstantBatchSizeLike):
    def init_shape(self):
        # test shape
        self.input_shape = [4, 5, 6, 7]
        self.shape = [10, 123, 92]
        self.output_shape = (4, 123, 92)


class TestFillConstantBatchSizeLike3(TestFillConstantBatchSizeLike):
    def init_value(self):
        # use 'str_value' rather than 'value'
        self.value = 3.8
        self.str_value = "4.5"
        self.output_value = 4.5


class TestFillConstantBatchSizeLike4(TestFillConstantBatchSizeLike):
    def init_shape(self):
        self.input_shape = [4, 5]
        self.shape = [123, 92]
        self.output_shape = (123, 4)

    def init_dim_idx(self):
        self.input_dim_idx = 0
        self.output_dim_idx = 1


class TestFillConstantBatchSizeFloat16(TestFillConstantBatchSizeLike):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-2)


class TestFillConstantBatchSizeFloat64(TestFillConstantBatchSizeLike):
    def init_dtype(self):
        self.dtype = np.float64


class TestFillConstantBatchSizeBool(TestFillConstantBatchSizeLike):
    def init_dtype(self):
        self.dtype = bool

    def init_value(self):
        self.value = True
        self.str_value = ""
        self.output_value = True


class TestFillConstantBatchSizeUint8(TestFillConstantBatchSizeLike):
    def init_dtype(self):
        self.dtype = np.uint8


class TestFillConstantBatchSizeInt8(TestFillConstantBatchSizeLike):
    def init_dtype(self):
        self.dtype = np.int8


class TestFillConstantBatchSizeInt16(TestFillConstantBatchSizeLike):
    def init_dtype(self):
        self.dtype = np.int16


class TestFillConstantBatchSizeInt32(TestFillConstantBatchSizeLike):
    def init_dtype(self):
        self.dtype = np.int32


class TestFillConstantBatchSizeInt64(TestFillConstantBatchSizeLike):
    def init_dtype(self):
        self.dtype = np.int64


class TestFillConstantBatchSizeCPU(TestFillConstantBatchSizeLike):
    def init_force_cpu(self):
        self.force_cpu = True


class TestFillConstantBatchSizeLikeLodTensor(TestFillConstantBatchSizeLike):
    # test LodTensor
    def setUp(self):
        self.set_sdaa()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "fill_constant_batch_size_like"
        self.python_api = fill_constant_batch_size_like
        self.init_shape()
        self.init_value()
        self.init_dtype()
        self.init_force_cpu()
        self.init_dim_idx()

        lod = [[3, 2, 5]]
        self.inputs = {
            "Input": (np.random.random(self.input_shape).astype("float32"), lod)
        }
        self.attrs = {
            "shape": self.shape,
            "value": self.value,
            "str_value": self.str_value,
            "dtype": convert_np_dtype_to_dtype_(self.dtype),
            "force_cpu": self.force_cpu,
            "input_dim_idx": self.input_dim_idx,
            "output_dim_idx": self.output_dim_idx,
        }
        self.outputs = {
            "Out": np.full(self.output_shape, self.output_value, self.dtype)
        }

    def init_shape(self):
        self.input_shape = [10, 20]
        self.shape = [123, 92]
        self.output_shape = (3, 92)


class TestFillConstantBatchSizeLikeLodTensor2(TestFillConstantBatchSizeLikeLodTensor):
    # test LodTensor with 'input_dim_idx' != 0
    def init_shape(self):
        self.input_shape = [10, 20]
        self.shape = [123, 92]
        self.output_shape = (20, 92)

    def init_dim_idx(self):
        self.input_dim_idx = 1
        self.output_dim_idx = 0


if __name__ == "__main__":
    unittest.main()
