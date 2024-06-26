#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


def ref_reduce_max(x, axis=None, keepdim=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.max(x, axis=axis, keepdims=keepdim)


@skip_check_grad_ci(reason="Haven not implement interpolate grad kernel.")
class TestMax(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "reduce_max"
        self.python_api = paddle.max
        self.init_dtype()
        self.axis = [0]
        self.keepdim = False
        self.set_attrs()

        x_np = np.random.random((2, 3, 4)).astype(self.dtype)

        out_np = ref_reduce_max(x_np, self.axis, self.keepdim)
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
        self.check_output_with_place(paddle.CustomPlace("sdaa", 0))


class TestReduceMaxOpAxisAll(TestMax):
    def set_attrs(self):
        self.axis = [0, 1, 2]


class TestReduceMaxOpAxisNegative(TestMax):
    def set_attrs(self):
        self.axis = [-1]


class Test4DReduceMax2(TestMax):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "reduce_max"
        self.python_api = paddle.max
        self.init_dtype()
        self.axis = [0]
        self.keepdim = True
        self.set_attrs()

        x_np = np.random.random((2, 3, 4, 5)).astype(self.dtype)

        out_np = ref_reduce_max(x_np, self.axis, self.keepdim)
        self.inputs = {"X": x_np}
        self.outputs = {"Out": out_np}
        self.attrs = {"dim": self.axis, "keep_dim": self.keepdim}


class TestReduceMaxFp16(TestMax):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "reduce_max"
        self.python_api = paddle.max
        self.axis = [0]
        self.keepdim = False
        self.set_attrs()

        x_np = np.random.random((2, 3, 4)).astype("float16")

        out_np = ref_reduce_max(x_np, self.axis, self.keepdim)
        self.inputs = {"X": x_np}
        self.outputs = {"Out": out_np}
        self.attrs = {"dim": self.axis, "keep_dim": self.keepdim}

    def test_check_output(self):
        self.check_output_with_place(paddle.CustomPlace("sdaa", 0), atol=1e-2)


def create_test_fp16_class(parent):
    class TestReduceMaxOpFp16Case(parent):
        def init_dtype(self):
            self.dtype = np.float16

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestReduceMaxOpFp16Case.__name__ = cls_name
    globals()[cls_name] = TestReduceMaxOpFp16Case


create_test_fp16_class(TestReduceMaxOpAxisAll)
create_test_fp16_class(TestReduceMaxOpAxisNegative)
create_test_fp16_class(Test4DReduceMax2)


def create_test_int64_class(parent):
    class TestReduceMaxOpInt64Case(parent):
        def init_dtype(self):
            self.dtype = np.int64

    cls_name = "{0}_{1}".format(parent.__name__, "INT64")
    TestReduceMaxOpInt64Case.__name__ = cls_name
    globals()[cls_name] = TestReduceMaxOpInt64Case


create_test_int64_class(TestMax)
create_test_int64_class(TestReduceMaxOpAxisAll)
create_test_int64_class(TestReduceMaxOpAxisNegative)
create_test_int64_class(Test4DReduceMax2)


def create_test_int32_class(parent):
    class TestReduceMaxOpInt32Case(parent):
        def init_dtype(self):
            self.dtype = np.int32

    cls_name = "{0}_{1}".format(parent.__name__, "INT32")
    TestReduceMaxOpInt32Case.__name__ = cls_name
    globals()[cls_name] = TestReduceMaxOpInt32Case


create_test_int32_class(TestMax)
create_test_int32_class(TestReduceMaxOpAxisAll)
create_test_int32_class(TestReduceMaxOpAxisNegative)
create_test_int32_class(Test4DReduceMax2)


def create_test_fp64_class(parent):
    class TestReduceMaxOpFp64Case(parent):
        def init_dtype(self):
            self.dtype = np.float64

    cls_name = "{0}_{1}".format(parent.__name__, "FP64")
    TestReduceMaxOpFp64Case.__name__ = cls_name
    globals()[cls_name] = TestReduceMaxOpFp64Case


create_test_fp64_class(TestMax)
create_test_fp64_class(TestReduceMaxOpAxisAll)
create_test_fp64_class(TestReduceMaxOpAxisNegative)
create_test_fp64_class(Test4DReduceMax2)

if __name__ == "__main__":
    unittest.main()
