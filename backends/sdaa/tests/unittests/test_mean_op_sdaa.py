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

import numpy as np
import unittest

from op_test import OpTest
import paddle

paddle.enable_static()


def create_test_fp16_class(parent):
    class TestFp16Case(parent):
        def init_dtype(self):
            self.dtype = np.float16

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestFp16Case.__name__ = cls_name
    globals()[cls_name] = TestFp16Case


def ref_reduce_mean(x, axis=None, keepdim=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.mean(x, axis=axis, keepdims=keepdim)


class TestMean(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.op_type = "reduce_mean"
        self.python_api = paddle.mean
        self.init_dtype()
        self.keepdim = False
        self.set_attrs()
        self.place = paddle.CustomPlace("sdaa", 0)
        self.init_input()
        self.out = ref_reduce_mean(self.x, self.axis, self.keepdim)

        self.inputs = {"X": self.x}
        self.outputs = {"Out": self.out}
        self.attrs = {"dim": self.axis, "keep_dim": self.keepdim}

    def init_input(self):
        self.x = np.random.random((20, 6, 4)).astype(self.dtype)

    def set_attrs(self):
        self.axis = [0, 1]
        self.keepdim = True

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(
            paddle.CustomPlace("sdaa", 0),
            atol=0.1,
        )

    def test_check_grad(self):
        dx = np.divide(np.ones_like(self.x, dtype=self.dtype), self.x.size)
        self.check_grad_with_place(
            paddle.CustomPlace("sdaa", 0),
            ["X"],
            "Out",
            user_defined_grads=[dx],
        )


class TestReduceMeanOpAxisNegative(TestMean):
    def set_attrs(self):
        self.axis = [-1]


class Test4DReduceMean2(TestMean):
    def set_attrs(self):
        self.axis = [0]

    def init_input(self):
        self.x = np.random.random((2, 3, 4, 5)).astype(self.dtype)


class Test5DReduceMean(TestMean):
    def set_attrs(self):
        self.axis = [0]

    def init_input(self):
        self.x = np.random.random((1, 2, 5, 6, 10)).astype(self.dtype)


class Test6DReduceMean(TestMean):
    def set_attrs(self):
        self.axis = [0]

    def init_input(self):
        self.x = np.random.random((1, 1, 2, 5, 6, 10)).astype(self.dtype)


class Test8DReduceMean(TestMean):
    def set_attrs(self):
        self.axis = [0, 3]

    def init_input(self):
        self.x = np.random.random((1, 3, 1, 2, 1, 4, 3, 10)).astype(self.dtype)


class Test1DReduceMean(TestMean):
    def set_attrs(self):
        self.axis = [0]

    def init_input(self):
        self.x = np.random.random(120).astype(self.dtype)


class Test2DReduceMean(TestMean):
    def set_attrs(self):
        self.axis = [0]

    def init_input(self):
        self.x = np.random.random((20, 10)).astype(self.dtype)


class Test2DReduceMean1(TestMean):
    def set_attrs(self):
        self.axis = [1]

    def init_input(self):
        self.x = np.random.random((20, 10)).astype(self.dtype)


class Test3DReduceMean(TestMean):
    def set_attrs(self):
        self.axis = [2]

    def init_input(self):
        self.x = np.random.random((5, 6, 7)).astype(self.dtype)


class Test3DReduceMean2(TestMean):
    def set_attrs(self):
        self.axis = [-2]

    def init_input(self):
        self.x = np.random.random((5, 6, 7)).astype(self.dtype)


class Test3DReduceMean3(TestMean):
    def set_attrs(self):
        self.axis = [1, 2]

    def init_input(self):
        self.x = np.random.random((5, 6, 7)).astype(self.dtype)


class Test3DKeepDimReduceMean(TestMean):
    def set_attrs(self):
        self.axis = [1, 2]
        self.keepdim = True

    def init_input(self):
        self.x = np.random.random((5, 6, 10)).astype(self.dtype)


class Test8DKeepDimReduceMean(TestMean):
    def set_attrs(self):
        self.axis = [3, 4, 5]
        self.keepdim = True

    def init_input(self):
        self.x = np.random.random((2, 5, 3, 2, 2, 3, 4, 2)).astype(self.dtype)


create_test_fp16_class(TestMean)
create_test_fp16_class(TestReduceMeanOpAxisNegative)
create_test_fp16_class(Test4DReduceMean2)
create_test_fp16_class(Test5DReduceMean)
create_test_fp16_class(Test6DReduceMean)
create_test_fp16_class(Test8DReduceMean)
create_test_fp16_class(Test1DReduceMean)
create_test_fp16_class(Test2DReduceMean)
create_test_fp16_class(Test2DReduceMean1)
create_test_fp16_class(Test3DReduceMean)
create_test_fp16_class(Test3DReduceMean2)
create_test_fp16_class(Test3DReduceMean3)
create_test_fp16_class(Test3DKeepDimReduceMean)
create_test_fp16_class(Test8DKeepDimReduceMean)

if __name__ == "__main__":
    unittest.main()
