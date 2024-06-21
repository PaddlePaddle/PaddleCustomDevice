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

paddle.enable_static()


def ref_reduce_min(x, axis=None, keepdim=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.min(x, axis=axis, keepdims=keepdim)


@skip_check_grad_ci(reason="Haven not implement interpolate grad kernel.")
class TestMin(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "reduce_min"
        self.python_api = paddle.min
        self.init_dtype()
        self.axis = [0]
        self.keepdim = True
        self.set_attrs()

        x_np = np.random.random((1, 2, 3, 4)).astype(self.dtype)

        out_np = ref_reduce_min(x_np, self.axis, self.keepdim)
        self.inputs = {"X": x_np}
        self.outputs = {"Out": out_np}
        self.attrs = {"dim": self.axis, "keep_dim": self.keepdim}

    def set_attrs(self):
        pass

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(
            paddle.CustomPlace("sdaa", 0),
        )


class TestReduceMinOpAxisAll(TestMin):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]


class TestReduceMinOpAxisNegative(TestMin):
    def set_attrs(self):
        self.axis = [-1]


class TestReduceMinInt64(TestMin):
    def init_dtype(self):
        self.dtype = np.int64


class Test4DReduceMin2(TestMin):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "reduce_min"
        self.python_api = paddle.min
        self.init_dtype()
        self.axis = [0]
        self.keepdim = True
        self.set_attrs()

        x_np = np.random.random((2, 3, 4, 5)).astype(self.dtype)

        out_np = ref_reduce_min(x_np, self.axis, self.keepdim)
        self.inputs = {"X": x_np}
        self.outputs = {"Out": out_np}
        self.attrs = {"dim": self.axis, "keep_dim": self.keepdim}


class TestReduceMinFp16(TestMin):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "reduce_min"
        self.python_api = paddle.min
        self.axis = [0]
        self.keepdim = False
        self.set_attrs()

        x_np = np.random.random((1, 2, 3)).astype("float16")

        out_np = ref_reduce_min(x_np, self.axis, self.keepdim)
        self.inputs = {"X": x_np}
        self.outputs = {"Out": out_np}
        self.attrs = {"dim": self.axis, "keep_dim": self.keepdim}

    def test_check_output(self):
        self.check_output_with_place(
            paddle.CustomPlace("sdaa", 0),
            atol=1e-2,
        )


if __name__ == "__main__":
    unittest.main()
