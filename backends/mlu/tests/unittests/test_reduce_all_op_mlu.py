#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from tests.op_test import OpTest
import paddle

paddle.enable_static()


class TestAllOp(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("mlu", 0)
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {"X": np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.outputs = {"Out": self.inputs["X"].all()}
        self.attrs = {"reduce_all": True}

    def set_mlu(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestAllFloatOp(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("mlu", 0)
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {"X": np.random.randint(0, 2, (5, 6, 10)).astype("float")}
        self.outputs = {"Out": self.inputs["X"].all()}
        self.attrs = {"reduce_all": True}

    def set_mlu(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestAllIntOp(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("mlu", 0)
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {"X": np.random.randint(0, 2, (5, 6, 10)).astype("int")}
        self.outputs = {"Out": self.inputs["X"].all()}
        self.attrs = {"reduce_all": True}

    def set_mlu(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestAllInt64Op(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("mlu", 0)
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {"X": np.random.randint(0, 2, (5, 6, 10)).astype("int64")}
        self.outputs = {"Out": self.inputs["X"].all()}
        self.attrs = {"reduce_all": True}

    def set_mlu(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestAll8DOp(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("mlu", 0)
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {
            "X": np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype("bool")
        }
        self.attrs = {"dim": (2, 3, 4)}
        self.outputs = {"Out": self.inputs["X"].all(axis=self.attrs["dim"])}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestAllOpWithDim(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("mlu", 0)
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {"X": np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.attrs = {"dim": (1,)}
        self.outputs = {"Out": self.inputs["X"].all(axis=self.attrs["dim"])}

    def test_check_output(self):
        self.check_output_with_place(self.place)


# @check_run_big_shape_test()
class TestAllOpWithDim1(TestAllOpWithDim):
    def setUp(self):
        self.place = paddle.CustomPlace("mlu", 0)
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {"X": np.random.randint(0, 2, (2, 1, 4096, 4096)).astype("bool")}
        self.attrs = {"dim": (0, 1, 2, 3)}
        self.outputs = {"Out": self.inputs["X"].all(axis=self.attrs["dim"])}


class TestAll8DOpWithDim(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("mlu", 0)
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {
            "X": np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype("bool")
        }
        self.attrs = {"dim": (1, 3, 4)}
        self.outputs = {"Out": self.inputs["X"].all(axis=self.attrs["dim"])}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestAllOpWithKeepDim(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("mlu", 0)
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {"X": np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.attrs = {"dim": [1], "keep_dim": True}
        self.outputs = {"Out": np.expand_dims(self.inputs["X"].all(axis=1), axis=1)}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestAll8DOpWithKeepDim(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("mlu", 0)
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {
            "X": np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype("bool")
        }
        self.attrs = {"dim": (5,), "keep_dim": True}
        self.outputs = {
            "Out": np.expand_dims(self.inputs["X"].all(axis=self.attrs["dim"]), axis=5)
        }

    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == "__main__":
    unittest.main()
