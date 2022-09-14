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
from tests.op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

paddle.enable_static()
SEED = 2021

class MLUOpTest(OpTest):
    def set_plugin(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace('CustomMLU', 0)

class TestCase1(MLUOpTest):

    def setUp(self):
        self.set_plugin()
        self.set_example()
        self.op_type = "split"
        ipt = self.x.astype(self.dtype)
        axis = self.axis if isinstance(self.axis, int) else int(self.axis[0])
        tmp_outs = np.split(ipt,
                            axis=axis,
                            indices_or_sections=self.num_or_sections)
        tmp_outs = [o.astype(self.dtype) for o in tmp_outs]
        self.outputs = {'Out': []}
        self.outs = []
        for i, o in enumerate(tmp_outs):
            self.outputs["Out"].append((str(i), o))
            self.outs.append(str(i))

        self.attrs = {"axis": self.axis, "num": self.num_or_sections}
        self.inputs = {}
        self.inputs.update({'X': ipt.astype(self.dtype)})

    def set_mlu(self):
        self.__class__.use_custom_device = True
        self.__class__.op_type = "split"

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_example(self):
        self.dtype = "float32"
        self.x = np.random.random((2, 4, 6))
        self.axis = 1
        self.num_or_sections = 2


class TestCase2(TestCase1):

    def set_example(self):
        self.dtype = "float32"
        self.x = np.random.random((20, 4, 50))
        self.axis = 0
        self.num_or_sections = 4


class TestCase4(TestCase1):

    def set_example(self):
        self.dtype = "float16"
        self.x = np.random.random((4, 50, 20))
        self.axis = 2
        self.num_or_sections = 4


# Test Sections
class TestCase5(TestCase1):

    def set_example(self):
        super().set_example()
        self.x = np.random.random((2, 10, 4))
        self.axis = 1
        self.num_or_sections = [2, 4, 8]

    def setUp(self):
        super().setUp()
        self.attrs.update({"sections": [2, 2, 4, 2], "num": 0})

# attr(axis) is Tensor
class TestSplitOp_AxisTensor(MLUOpTest):

    def setUp(self):
        self.set_plugin()
        self._set_op_type()
        self.dtype = self.get_dtype()
        self.init_data()
        self.inputs = {
            'X': self.x,
            'AxisTensor': np.array([self.axis]).astype("int32")
        }
        self.attrs = {'sections': self.sections, 'num': self.num}

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
        return "float"

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestSplitOp_SectionsTensor(MLUOpTest):

    def setUp(self):
        self.set_plugin()
        self._set_op_type()
        self.dtype = self.get_dtype()
        self.init_data()
        self.inputs = {'X': self.x}

        sections_tensor = []
        for index, ele in enumerate(self.sections):
            sections_tensor.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs['SectionsTensorList'] = sections_tensor

        self.attrs = {
            'axis': self.axis,
            'sections': self.sections_infer,
            'num': self.num
        }

        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {'Out': [('out%d' % i, out[i]) \
                                for i in range(len(out))]}

    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype(self.dtype)
        self.axis = 1
        self.sections = [2, 1, 2]
        self.sections_infer = [-1, -1, -1]
        self.num = 0
        self.indices_or_sections = [2, 3]

    def get_dtype(self):
        return "float"

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == '__main__':
    unittest.main()
