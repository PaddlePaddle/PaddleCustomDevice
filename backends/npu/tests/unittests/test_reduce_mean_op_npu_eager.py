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

import unittest
import os

select_npu = os.environ.get("FLAGS_selected_npus", 0)

import numpy as np
import paddle
from tests.op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float

from npu_utils import check_soc_version


class TestMeanOpBF16(OpTest):
    def set_npu(self):
        self.__class__.use_custom_device = True

    def setUp(self):
        self.set_npu()
        self.init_dtype()
        self.init_data()
        self.op_type = "reduce_mean"
        self.inputs = {"X": self.x}
        self.outputs = {"Out": self.out}

    def init_dtype(self):
        self.dtype = np.uint16

    def init_data(self):
        self.x = convert_float_to_uint16(np.random.random((5, 6, 10)).astype("float32"))
        self.out = convert_uint16_to_float(self.x).mean(axis=0)

    @check_soc_version
    def test_check_output(self):
        self.check_output_with_place(paddle.CustomPlace("npu", select_npu), atol=0.004)

    @check_soc_version
    def test_check_grad(self):
        self.check_grad_with_place(paddle.CustomPlace("npu", select_npu), ["X"], "Out")


class TestMeanOp5DBF16(TestMeanOpBF16):
    def init_data(self):
        self.x = convert_float_to_uint16(
            np.random.random((1, 2, 5, 6, 10)).astype("float32")
        )
        self.out = convert_uint16_to_float(self.x).mean(axis=0)


class TestMeanOp6DBF16(TestMeanOpBF16):
    def init_data(self):
        self.x = convert_float_to_uint16(
            np.random.random((1, 1, 2, 5, 6, 10)).astype("float32")
        )
        self.out = convert_uint16_to_float(self.x).mean(axis=0)


class TestMeanOp8DBF16(TestMeanOpBF16):
    def init_data(self):
        self.x = convert_float_to_uint16(
            np.random.random((1, 3, 1, 2, 5, 6, 10)).astype("float32")
        )
        self.out = convert_uint16_to_float(self.x).mean(axis=0)
        self.attrs = {"dim": (0, 3)}


class Test1DReduceBF16(TestMeanOpBF16):
    def init_data(self):
        self.x = convert_float_to_uint16(np.random.random(120).astype("float32"))
        self.out = convert_uint16_to_float(self.x).mean(axis=0)


class Test2DReduce0BF16(TestMeanOpBF16):
    def init_data(self):
        self.x = convert_float_to_uint16(np.random.random((20, 10)).astype("float32"))
        self.out = convert_uint16_to_float(self.x).mean(axis=0)
        self.attrs = {"dim": [0]}


class Test2DReduce1BF16(TestMeanOpBF16):
    def init_data(self):
        self.attrs = {"dim": [1]}
        self.x = convert_float_to_uint16(np.random.random((20, 10)).astype("float32"))
        self.out = convert_uint16_to_float(self.x).mean(axis=tuple(self.attrs["dim"]))


class Test3DReduce0BF16(TestMeanOpBF16):
    def init_data(self):
        self.attrs = {"dim": [1]}
        self.x = convert_float_to_uint16(np.random.random((5, 6, 7)).astype("float32"))
        self.out = convert_uint16_to_float(self.x).mean(axis=tuple(self.attrs["dim"]))


class Test3DReduce1BF16(TestMeanOpBF16):
    def init_data(self):
        self.attrs = {"dim": [2]}
        self.x = convert_float_to_uint16(np.random.random((5, 6, 7)).astype("float32"))
        self.out = convert_uint16_to_float(self.x).mean(axis=tuple(self.attrs["dim"]))


class Test3DReduce2BF16(TestMeanOpBF16):
    def init_data(self):
        self.attrs = {"dim": [-2]}
        self.x = convert_float_to_uint16(np.random.random((5, 6, 7)).astype("float32"))
        self.out = convert_uint16_to_float(self.x).mean(axis=tuple(self.attrs["dim"]))


class Test3DReduce3BF16(TestMeanOpBF16):
    def init_data(self):
        self.attrs = {"dim": [1, 2]}
        self.x = convert_float_to_uint16(np.random.random((5, 6, 7)).astype("float32"))
        self.out = convert_uint16_to_float(self.x).mean(axis=tuple(self.attrs["dim"]))


class TestKeepDimReduceBF16(TestMeanOpBF16):
    def init_data(self):
        self.attrs = {"dim": [1], "keep_dim": True}
        self.x = convert_float_to_uint16(np.random.random((5, 6, 10)).astype("float32"))
        self.out = convert_uint16_to_float(self.x).mean(
            axis=tuple(self.attrs["dim"]), keepdims=self.attrs["keep_dim"]
        )


class TestKeepDim8DReduceBF16(TestMeanOpBF16):
    def init_data(self):
        self.attrs = {"dim": [3, 4, 5], "keep_dim": True}
        self.x = convert_float_to_uint16(
            np.random.random((2, 5, 4, 2, 2, 3, 4, 2)).astype("float32")
        )
        self.out = convert_uint16_to_float(self.x).mean(
            axis=tuple(self.attrs["dim"]), keepdims=self.attrs["keep_dim"]
        )


class TestReduceAllBF16(TestMeanOpBF16):
    def init_data(self):
        self.attrs = {"reduce_all": True}
        self.x = convert_float_to_uint16(
            np.random.random((5, 6, 2, 10)).astype("float32")
        )
        self.out = convert_uint16_to_float(self.x).mean()


if __name__ == "__main__":
    unittest.main()
