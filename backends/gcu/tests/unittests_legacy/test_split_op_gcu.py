# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
from tests.op_test import OpTest
import paddle

paddle.enable_static()


class OpTest(OpTest):
    def set_device(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("gcu", 0)


class TestSplitOp(OpTest):
    def setUp(self):
        self.set_device()
        self._set_op_type()
        self.dtype = self.get_dtype()
        axis = 1
        x = np.random.random((4, 5, 6)).astype(self.dtype)
        out = np.split(x, [2, 3], axis)
        self.inputs = {"X": x}
        self.outputs = {"Out": [("out%d" % i, out[i]) for i in range(len(out))]}
        self.attrs = {"axis": axis, "sections": [2, 1, 2]}

    def set_device(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("gcu", 0)

    def get_dtype(self):
        return "float32"

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


class TestSplitOp_2(OpTest):
    def setUp(self):
        self.set_device()
        self._set_op_type()
        self.dtype = self.get_dtype()
        self.init_data()
        self.inputs = {"X": self.x}
        self.attrs = {"axis": self.axis, "sections": self.sections, "num": self.num}

        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {"Out": [("out%d" % i, out[i]) for i in range(len(out))]}

    def set_device(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("gcu", 0)

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


class TestSplitOpFp16(OpTest):
    def setUp(self):
        self.set_device()
        self._set_op_type()
        self.dtype = self.get_dtype()
        self.init_data()
        self.inputs = {"X": self.x}
        self.attrs = {"axis": self.axis, "sections": self.sections, "num": self.num}

        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {"Out": [("out%d" % i, out[i]) for i in range(len(out))]}

    def set_device(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("gcu", 0)

    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype(self.dtype)
        self.axis = 2
        self.sections = []
        self.num = 3
        self.indices_or_sections = 3

    def get_dtype(self):
        return "float16"

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


# attr(axis) is Tensor
class TestSplitOp_AxisTensor(OpTest):
    def setUp(self):
        self.set_device()
        self._set_op_type()
        self.dtype = self.get_dtype()
        self.init_data()
        self.inputs = {"X": self.x, "AxisTensor": np.array([self.axis]).astype("int32")}
        self.attrs = {"sections": self.sections, "num": self.num}

        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {"Out": [("out%d" % i, out[i]) for i in range(len(out))]}

    def set_device(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("gcu", 0)

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


# attr(sections) is list containing Tensor
class TestSplitOp_SectionsTensor(OpTest):
    def setUp(self):
        self.set_device()
        self._set_op_type()
        self.dtype = self.get_dtype()
        self.init_data()
        self.inputs = {"X": self.x}

        sections_tensor = []
        for index, ele in enumerate(self.sections):
            sections_tensor.append(
                ("x" + str(index), np.ones((1)).astype("int32") * ele)
            )

        self.inputs["SectionsTensorList"] = sections_tensor

        self.attrs = {
            "axis": self.axis,
            "sections": self.sections_infer,
            "num": self.num,
        }

        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {"Out": [("out%d" % i, out[i]) for i in range(len(out))]}

    def set_device(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("gcu", 0)

    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype(self.dtype)
        self.axis = 1
        self.sections = [2, 1, 2]
        self.sections_infer = [-1, -1, -1]
        self.num = 0
        self.indices_or_sections = [2, 3]

    def get_dtype(self):
        return "float32"

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


# split_with_num op cannot be called by set op type because it has not been registered in base OpProto,
# we call paddle.split() in dygraph mode to test it.
class TestSplitWithNumOp(unittest.TestCase):
    def initTestCase(self):
        self.dim = (4, 5, 6)
        self.dtype = "float32"
        self.axis = 2
        self.num = 3

    def setUp(self):
        self.initTestCase()
        self.set_device()

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def test_split_with_num(self):
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            numpy_input = (np.random.random(self.dim)).astype(self.dtype)
            tensor_input = paddle.to_tensor(numpy_input)
            numpy_output = np.split(numpy_input, self.num, self.axis)
            paddle_output = paddle.split(tensor_input, self.num, self.axis)
            for i in range(self.num):
                self.assertEqual(
                    np.allclose(numpy_output[i], paddle_output[i].numpy()), True
                )
            paddle.enable_static()

        run(self.place)


class TestSplitWithNumOp_AxisTensor(TestSplitWithNumOp):
    def initTestCase(self):
        self.dim = (4, 5, 6)
        self.dtype = "float32"
        self.axis = np.array([2]).astype("int32")
        self.num = 3

    def setUp(self):
        self.initTestCase()
        self.set_device()

    def test_split_with_num(self):
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            numpy_input = (np.random.random(self.dim)).astype(self.dtype)
            tensor_input = paddle.to_tensor(numpy_input)
            numpy_output = np.split(numpy_input, self.num, self.axis[0])
            paddle_output = paddle.split(
                tensor_input, self.num, paddle.to_tensor(self.axis)
            )
            for i in range(self.num):
                self.assertEqual(
                    np.allclose(numpy_output[i], paddle_output[i].numpy()), True
                )
            paddle.enable_static()

        run(self.place)


if __name__ == "__main__":
    unittest.main()
