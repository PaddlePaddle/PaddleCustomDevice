# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

from __future__ import print_function
import unittest
import numpy as np
from op_test import OpTest
import paddle.base as base
import paddle

paddle.enable_static()


def test_class1(op_type, typename):
    class TestExpandAsBasic(OpTest):
        def setUp(self):
            self.set_sdaa()
            self.op_type = "expand_as_v2"
            self.python_api = paddle.expand_as
            x = np.random.rand(100).astype(typename)
            target_tensor = np.random.rand(2, 100).astype(typename)
            self.inputs = {"X": x}
            self.attrs = {"target_shape": target_tensor.shape}
            bcast_dims = [2, 1]
            output = np.tile(self.inputs["X"], bcast_dims)
            self.outputs = {"Out": output}

        def set_sdaa(self):
            self.__class__.use_custom_device = True
            self.place = paddle.CustomPlace("sdaa", 0)
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

    cls_name = str(op_type) + "_" + str(typename) + "_1"
    TestExpandAsBasic.__name__ = cls_name
    globals()[cls_name] = TestExpandAsBasic


def test_class2(op_type, typename):
    class TestExpandAsOpRank2(OpTest):
        def setUp(self):
            self.set_sdaa()
            self.op_type = "expand_as_v2"
            self.python_api = paddle.expand_as
            x = np.random.rand(10, 12).astype(typename)
            target_tensor = np.random.rand(10, 12).astype(typename)
            self.inputs = {"X": x}
            self.attrs = {"target_shape": target_tensor.shape}
            bcast_dims = [1, 1]
            output = np.tile(self.inputs["X"], bcast_dims)
            self.outputs = {"Out": output}

        def set_sdaa(self):
            self.__class__.use_custom_device = True
            self.place = paddle.CustomPlace("sdaa", 0)
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

    cls_name = str(op_type) + "_" + str(typename) + "_2"
    TestExpandAsOpRank2.__name__ = cls_name
    globals()[cls_name] = TestExpandAsOpRank2


def test_class3(op_type, typename):
    class TestExpandAsOpRank3(OpTest):
        def setUp(self):
            self.set_sdaa()
            self.op_type = "expand_as_v2"
            self.python_api = paddle.expand_as
            x = np.random.rand(2, 3, 20).astype(typename)
            target_tensor = np.random.rand(2, 3, 20).astype(typename)
            self.inputs = {"X": x}
            self.attrs = {"target_shape": target_tensor.shape}
            bcast_dims = [1, 1, 1]
            output = np.tile(self.inputs["X"], bcast_dims)
            self.outputs = {"Out": output}

        def set_sdaa(self):
            self.__class__.use_custom_device = True
            self.place = paddle.CustomPlace("sdaa", 0)
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

    cls_name = str(op_type) + "_" + str(typename) + "_3"
    TestExpandAsOpRank3.__name__ = cls_name
    globals()[cls_name] = TestExpandAsOpRank3


def test_class4(op_type, typename):
    class TestExpandAsOpRank4(OpTest):
        def setUp(self):
            self.set_sdaa()
            self.op_type = "expand_as_v2"
            self.python_api = paddle.expand_as
            x = np.random.rand(1, 1, 7, 16).astype(typename)
            target_tensor = np.random.rand(4, 6, 7, 16).astype(typename)
            self.inputs = {"X": x}
            self.attrs = {"target_shape": target_tensor.shape}
            bcast_dims = [4, 6, 1, 1]
            output = np.tile(self.inputs["X"], bcast_dims)
            self.outputs = {"Out": output}

        def set_sdaa(self):
            self.__class__.use_custom_device = True
            self.place = paddle.CustomPlace("sdaa", 0)
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            self.check_output_with_place(self.place)

    cls_name = str(op_type) + "_" + str(typename) + "_4"
    TestExpandAsOpRank4.__name__ = cls_name
    globals()[cls_name] = TestExpandAsOpRank4


# Test python API
class TestExpandAsV2API(unittest.TestCase):
    def test_api(self):
        input1 = np.random.random([12, 14]).astype("float32")
        input2 = np.random.random([2, 12, 14]).astype("float32")
        x = paddle.static.data(name="x", shape=[12, 14], dtype="float32")

        y = paddle.static.data(name="target_tensor", shape=[2, 12, 14], dtype="float32")

        out_1 = paddle.expand_as(x, y=y)

        exe = base.Executor(place=paddle.CustomPlace("sdaa", 0))
        res_1 = exe.run(
            base.default_main_program(),
            feed={"x": input1, "target_tensor": input2},
            fetch_list=[out_1],
        )
        assert np.array_equal(res_1[0], np.tile(input1, (2, 1, 1)))


for _typename in {"float16", "float32", "float64", "int64", "int32", "bool"}:
    test_class1("expand_as_v2", _typename)
    test_class2("expand_as_v2", _typename)
    test_class3("expand_as_v2", _typename)
    test_class4("expand_as_v2", _typename)

if __name__ == "__main__":
    unittest.main()
