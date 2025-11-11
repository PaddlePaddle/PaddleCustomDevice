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

paddle.enable_static()
SEED = 2021


class Testaddn1(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.init_dtype()
        self.op_type = "sum"
        self.python_api = paddle.add_n
        self.place = paddle.CustomPlace("sdaa", 0)

        x0 = np.random.random((3, 40)).astype(self.dtype)
        x1 = np.random.random((3, 40)).astype(self.dtype)
        x2 = np.random.random((3, 40)).astype(self.dtype)
        print("numpy x0: ", x0)
        print("numpy x1: ", x1)
        print("numpy x2: ", x2)
        self.inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
        y = x0 + x1 + x2
        self.outputs = {"Out": y}

        self.attrs = {"use_mkldnn": False}

    def init_dtype(self):
        self.dtype = np.float32

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)


# This case is sourced from the ppocr-cls model
class Testaddn2(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.init_dtype()
        self.op_type = "sum"
        self.python_api = paddle.add_n
        self.place = paddle.CustomPlace("sdaa", 0)

        x0 = np.random.random((24, 8, 1, 1)).astype(self.dtype)
        x1 = np.random.random((24, 8, 1, 1)).astype(self.dtype)
        self.inputs = {"X": [("x0", x0), ("x1", x1)]}
        # There will be a problem if just using `y=x0+x1+x2+x3` to calculate the
        # summation result as the reference standard result. The reason is that
        # numpy's fp16 data has precision loss when doing `add` operation.
        # For example, the results of `x0+x1+x2+x3` is different from that of
        # `x3+x2+x1+x0` if the dtype is fp16.
        # Therefore, converting the input to fp32 for calculation.
        y = (x0.astype(np.float32) + x1.astype(np.float32)).astype(self.dtype)
        self.outputs = {"Out": y}

        self.attrs = {"use_mkldnn": False}

    def init_dtype(self):
        self.dtype = np.float16

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)


class Testaddn3(OpTest):
    def setUp(self):
        self.set_sdaa()
        self.init_dtype()
        self.op_type = "sum"
        self.python_api = paddle.add_n
        self.place = paddle.CustomPlace("sdaa", 0)

        x0 = np.random.random((3, 3)).astype(self.dtype)

        self.inputs = {"X": [("x0", x0)]}
        y = x0
        self.outputs = {"Out": y}

        self.attrs = {"use_mkldnn": False}

    def init_dtype(self):
        self.dtype = np.float32

    def set_sdaa(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)


class Testaddn4(Testaddn3):
    def init_dtype(self):
        self.dtype = np.float16


# This case is sourced from the ppocr-cls model
class Testaddn5(Testaddn2):
    def setUp(self):
        self.set_sdaa()
        self.init_dtype()
        self.op_type = "sum"
        self.python_api = paddle.add_n
        self.place = paddle.CustomPlace("sdaa", 0)

        x0 = np.random.random((24, 1, 3, 3)).astype(self.dtype)
        x1 = np.random.random((24, 1, 3, 3)).astype(self.dtype)
        self.inputs = {"X": [("x0", x0), ("x1", x1)]}
        # There will be a problem if just using `y=x0+x1+x2+x3` to calculate the
        # summation result as the reference standard result. The reason is that
        # numpy's fp16 data has precision loss when doing `add` operation.
        # For example, the results of `x0+x1+x2+x3` is different from that of
        # `x3+x2+x1+x0` if the dtype is fp16.
        # Therefore, converting the input to fp32 for calculation.
        y = (x0.astype(np.float32) + x1.astype(np.float32)).astype(self.dtype)
        self.outputs = {"Out": y}

        self.attrs = {"use_mkldnn": False}


# This case is sourced from the ppocr-cls model
class Testaddn6(Testaddn2):
    def setUp(self):
        self.set_sdaa()
        self.init_dtype()
        self.op_type = "sum"
        self.python_api = paddle.add_n
        self.place = paddle.CustomPlace("sdaa", 0)

        x0 = np.random.random((16, 32, 1, 1)).astype(self.dtype)
        x1 = np.random.random((16, 32, 1, 1)).astype(self.dtype)
        self.inputs = {"X": [("x0", x0), ("x1", x1)]}
        # There will be a problem if just using `y=x0+x1+x2+x3` to calculate the
        # summation result as the reference standard result. The reason is that
        # numpy's fp16 data has precision loss when doing `add` operation.
        # For example, the results of `x0+x1+x2+x3` is different from that of
        # `x3+x2+x1+x0` if the dtype is fp16.
        # Therefore, converting the input to fp32 for calculation.
        y = (x0.astype(np.float32) + x1.astype(np.float32)).astype(self.dtype)
        self.outputs = {"Out": y}

        self.attrs = {"use_mkldnn": False}


if __name__ == "__main__":
    unittest.main()
