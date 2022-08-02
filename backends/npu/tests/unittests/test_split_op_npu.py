# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function, division

import numpy as np
import unittest
import sys
from tests.op_test import OpTest
import paddle
import paddle.fluid as fluid

paddle.enable_static()

class NPUOpTest(OpTest):
    def set_plugin(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace('ascend', 0)

class TestNPUSplitOp(NPUOpTest):
    def setUp(self):
        self.set_plugin()
        self._set_op_type()
        self.dtype = self.get_dtype()
        axis = 1
        x = np.random.random((4, 5, 6)).astype(self.dtype)
        out = np.split(x, [2, 3], axis)
        self.inputs = {'X': x}
        self.outputs = {'Out': [('out%d' % i, out[i]) \
            for i in range(len(out))]}
        self.attrs = {'axis': axis, 'sections': [2, 1, 2]}

    def get_dtype(self):
        return "float32"

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass

class TestNPUSplitOp_2(NPUOpTest):
    def setUp(self):
        self.set_plugin()
        self._set_op_type()
        self.dtype = self.get_dtype()
        self.init_data()
        self.inputs = {'X': self.x}
        self.attrs = {
            'axis': self.axis,
            'sections': self.sections,
            'num': self.num
        }

        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {'Out': [('out%d' % i, out[i]) \
                                for i in range(len(out))]}

    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype(self.dtype)
        self.axis = 2
        self.sections = []
        self.num = 3
        self.indices_or_sections = 3

    def get_dtype(self):
        return "float32"

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


# # attr(axis) is Tensor
# class TestSplitOp_AxisTensor(NPUOpTest):

#     def setUp(self):
#         self.set_plugin()
#         self._set_op_type()
#         self.dtype = self.get_dtype()
#         self.init_data()
#         self.inputs = {
#             'X': self.x,
#             'AxisTensor': np.array([self.axis]).astype("int32")
#         }
#         self.attrs = {'sections': self.sections, 'num': self.num}

#         out = np.split(self.x, self.indices_or_sections, self.axis)
#         self.outputs = {'Out': [('out%d' % i, out[i]) \
#                                 for i in range(len(out))]}

#     def init_data(self):
#         self.x = np.random.random((4, 5, 6)).astype(self.dtype)
#         self.axis = 2
#         self.sections = []
#         self.num = 3
#         self.indices_or_sections = 3

#     def get_dtype(self):
#         return "float32"

#     def _set_op_type(self):
#         self.op_type = "split"

#     def test_check_output(self):
#         self.check_output_with_place(self.place)

#     def test_check_grad(self):
#         pass


# # attr(sections) is list containing Tensor
# class TestSplitOp_SectionsTensor(NPUOpTest):

#     def setUp(self):
#         self.set_plugin()
#         self._set_op_type()
#         self.dtype = self.get_dtype()
#         self.init_data()
#         self.inputs = {'X': self.x}

#         sections_tensor = []
#         for index, ele in enumerate(self.sections):
#             sections_tensor.append(("x" + str(index), np.ones(
#                 (1)).astype('int32') * ele))

#         self.inputs['SectionsTensorList'] = sections_tensor

#         self.attrs = {
#             'axis': self.axis,
#             'sections': self.sections_infer,
#             'num': self.num
#         }

#         out = np.split(self.x, self.indices_or_sections, self.axis)
#         self.outputs = {'Out': [('out%d' % i, out[i]) \
#                                 for i in range(len(out))]}

#     def init_data(self):
#         self.x = np.random.random((4, 5, 6)).astype(self.dtype)
#         self.axis = 1
#         self.sections = [2, 1, 2]
#         self.sections_infer = [-1, -1, -1]
#         self.num = 0
#         self.indices_or_sections = [2, 3]

#     def get_dtype(self):
#         return "float32"

#     def _set_op_type(self):
#         self.op_type = "split"

#     def test_check_output(self):
#         self.check_output_with_place(self.place)

#     def test_check_grad(self):
#         pass

if __name__ == '__main__':
    unittest.main()
