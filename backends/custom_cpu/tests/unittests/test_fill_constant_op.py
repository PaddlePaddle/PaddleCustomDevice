#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest

import paddle
import numpy as np


def get_places(self):
    return [paddle.CustomPlace("custom_cpu", 0)]


OpTest._get_places = get_places


# Situation 1: Attr(shape) is a list(without tensor)
class TestFillConstantOp1(OpTest):
    def setUp(self):
        """Test fill_constant op with specified value"""
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {"shape": [123, 92], "value": 3.8}
        self.outputs = {"Out": np.full((123, 92), 3.8)}

    def test_check_output(self):
        self.check_output()


class TestFillConstantOp2(OpTest):
    def setUp(self):
        """Test fill_constant op with default value"""
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {"shape": [123, 92]}
        self.outputs = {"Out": np.full((123, 92), 0.0)}

    def test_check_output(self):
        self.check_output()


class TestFillConstantOp3(OpTest):
    def setUp(self):
        """Test fill_constant op with specified int64 value"""
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {"shape": [123, 92], "value": 10000000000}
        self.outputs = {"Out": np.full((123, 92), 10000000000)}

    def test_check_output(self):
        self.check_output()


class TestFillConstantOp4(OpTest):
    def setUp(self):
        """Test fill_constant op with specified int value"""
        self.op_type = "fill_constant"

        self.inputs = {}
        self.attrs = {"shape": [123, 92], "value": 3}
        self.outputs = {"Out": np.full((123, 92), 3)}

    def test_check_output(self):
        self.check_output()


# class TestFillConstantOpWithSelectedRows(unittest.TestCase):
#     def check_with_place(self, place):
#         scope = core.Scope()
#         # create Out Variable
#         out = scope.var('Out').get_selected_rows()

#         # create and run fill_constant_op operator
#         fill_constant_op = Operator(
#             "fill_constant", shape=[123, 92], value=3.8, Out='Out')
#         fill_constant_op.run(scope, place)

#         # get result from Out
#         result_array = np.array(out.get_tensor())
#         full_array = np.full((123, 92), 3.8, 'float32')

#         self.assertTrue(np.array_equal(result_array, full_array))

#     def test_fill_constant_with_selected_rows(self):
#         places = [core.CustomPlace('custom_cpu', 0)]

#         for place in places:
#             self.check_with_place(place)


# Situation 2: Attr(shape) is a list(with tensor)
class TestFillConstantOp1_ShapeTensorList(OpTest):
    def setUp(self):
        """Test fill_constant op with specified value"""
        self.op_type = "fill_constant"
        self.init_data()
        shape_tensor_list = []
        for index, ele in enumerate(self.shape):
            shape_tensor_list.append(
                ("x" + str(index), np.ones((1)).astype("int32") * ele)
            )

        self.inputs = {"ShapeTensorList": shape_tensor_list}
        self.attrs = {"shape": self.infer_shape, "value": self.value}
        self.outputs = {"Out": np.full(self.shape, self.value)}

    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [-1, 92]
        self.value = 3.8

    def test_check_output(self):
        self.check_output()


class TestFillConstantOp2_ShapeTensorList(OpTest):
    def setUp(self):
        """Test fill_constant op with default value"""
        self.op_type = "fill_constant"
        self.init_data()
        shape_tensor_list = []
        for index, ele in enumerate(self.shape):
            shape_tensor_list.append(
                ("x" + str(index), np.ones((1)).astype("int32") * ele)
            )

        self.inputs = {"ShapeTensorList": shape_tensor_list}
        self.attrs = {"shape": self.infer_shape}
        self.outputs = {"Out": np.full(self.shape, 0.0)}

    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [-1, -1]

    def test_check_output(self):
        self.check_output()


class TestFillConstantOp3_ShapeTensorList(TestFillConstantOp1_ShapeTensorList):
    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [123, -1]
        self.value = 10000000000


class TestFillConstantOp4_ShapeTensorList(TestFillConstantOp1_ShapeTensorList):
    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [123, -1]
        self.value = 3


# Situation 3: shape is a tensor
class TestFillConstantOp1_ShapeTensor(OpTest):
    def setUp(self):
        """Test fill_constant op with specified value"""
        self.op_type = "fill_constant"
        self.init_data()

        self.inputs = {"ShapeTensor": np.array(self.shape).astype("int32")}
        self.attrs = {"value": self.value}
        self.outputs = {"Out": np.full(self.shape, self.value)}

    def init_data(self):
        self.shape = [123, 92]
        self.value = 3.8

    def test_check_output(self):
        self.check_output()


# Situation 4: value is a tensor
class TestFillConstantOp1_ValueTensor(OpTest):
    def setUp(self):
        """Test fill_constant op with specified value"""
        self.op_type = "fill_constant"
        self.init_data()

        self.inputs = {
            "ShapeTensor": np.array(self.shape).astype("int32"),
            "ValueTensor": np.array([self.value]).astype("float32"),
        }
        self.attrs = {"value": self.value + 1.0}
        self.outputs = {"Out": np.full(self.shape, self.value)}

    def init_data(self):
        self.shape = [123, 92]
        self.value = 3.8
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output()


# Situation 5: value is a tensor
class TestFillConstantOp2_ValueTensor(OpTest):
    def setUp(self):
        """Test fill_constant op with specified value"""
        self.op_type = "fill_constant"
        self.init_data()

        self.inputs = {
            "ShapeTensor": np.array(self.shape).astype("int32"),
            "ValueTensor": np.array([self.value]).astype("int32"),
        }
        self.attrs = {"value": self.value, "dtype": 2}
        self.outputs = {"Out": np.full(self.shape, self.value)}

    def init_data(self):
        self.shape = [123, 92]
        self.value = 3
        self.dtype = np.int32

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
