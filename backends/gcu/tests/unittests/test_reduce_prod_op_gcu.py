#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
from tests.op_test import OpTest
import paddle

paddle.enable_static()


class TestReduceProd(OpTest):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_device()
        self.init_dtype()

        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {"dim": [0]}
        self.outputs = {"Out": self.inputs["X"].prod(axis=tuple(self.attrs["dim"]))}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], ["Out"])

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def init_dtype(self):
        self.dtype = np.float32


class TestReduceProd1(TestReduceProd):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_device()
        self.init_dtype()

        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {}  # default 'dim': [0]
        self.outputs = {"Out": self.inputs["X"].prod(axis=tuple([0]))}


class TestReduceProd2(TestReduceProd):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_device()
        self.init_dtype()

        self.inputs = {"X": np.random.random((32, 8, 50, 2)).astype(self.dtype)}
        self.attrs = {"dim": [-1]}
        self.outputs = {"Out": self.inputs["X"].prod(axis=tuple([-1]))}


class TestReduceAll(TestReduceProd):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.set_device()
        self.init_dtype()

        self.inputs = {"X": np.random.random((5, 6, 10)).astype(self.dtype)}
        self.attrs = {"reduce_all": True}
        self.outputs = {"Out": self.inputs["X"].prod()}


class TestReduceProdInt32(TestReduceProd):
    def init_dtype(self):
        self.dtype = np.int32

    # int32 is not supported for gradient check
    def test_check_grad(self):
        pass


class TestReduceProdInt64(TestReduceProd):
    def init_dtype(self):
        self.dtype = np.int64

    # int64 is not supported for gradient check
    def test_check_grad(self):
        pass


# class TestReduceProd6D(TestReduceProd):
#     def setUp(self):
#         self.op_type = 'reduce_prod'
#         self.set_device()
#         self.init_dtype()

#         self.inputs = {'X': np.random.random((5, 6, 2, 3, 4, 2)).astype(self.dtype)}
#         self.attrs = {'dim': [2, 3, 4]}
#         self.outputs = {'Out': self.inputs['X'].prod(axis=tuple(self.attrs['dim']))}


# class TestReduceProd8D(TestReduceProd):
#     def setUp(self):
#         self.op_type = 'reduce_prod'
#         self.set_device()
#         self.init_dtype()

#         self.inputs = {
#             'X': np.random.random((2, 5, 3, 2, 2, 3, 4, 2)).astype(self.dtype)
#         }
#         self.attrs = {'dim': [2, 3, 4]}
#         self.outputs = {'Out': self.inputs['X'].prod(axis=tuple(self.attrs['dim']))}


if __name__ == "__main__":
    unittest.main()
