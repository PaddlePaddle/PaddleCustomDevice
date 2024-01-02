# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from tests.op_test import OpTest
import paddle

paddle.enable_static()
SEED = 2021


class TestScale(OpTest):
    def setUp(self):
        self.set_gcu()
        self.op_type = "scale"
        self.place = paddle.CustomPlace("gcu", 0)
        self.init_dtype()

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(
                np.random.random((10, 10)).astype(self.dtype)
            )
        }
        self.attrs = {"scale": -2.3, "bias": 0, "bias_after_scale": True}
        self.outputs = {
            "Out": (self.inputs["X"] * self.dtype(self.attrs["scale"])).astype(
                self.dtype
            )
        }

    def set_gcu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestFP16Scale(TestScale):
    def init_dtype(self):
        self.dtype = np.float16


class TestScaleInt(TestScale):
    def init_dtype(self):
        self.dtype = np.int32


class TestScaleInt64(TestScale):
    def init_dtype(self):
        self.dtype = np.int64


class TestScaleDouble(TestScale):
    def init_dtype(self):
        self.dtype = np.double


class TestBiasAfterScale(OpTest):
    def setUp(self):
        self.set_gcu()
        self.op_type = "scale"
        self.place = paddle.CustomPlace("gcu", 0)
        self.init_dtype()

        self.inputs = {
            "X": OpTest.np_dtype_to_base_dtype(
                np.random.random((10, 10)).astype(self.dtype)
            )
        }
        self.attrs = {"scale": -2.3, "bias": 0, "bias_after_scale": False}
        self.outputs = {"Out": self.inputs["X"] * self.dtype(self.attrs["scale"])}

    def set_gcu(self):
        self.__class__.use_custom_device = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == "__main__":
    unittest.main()
