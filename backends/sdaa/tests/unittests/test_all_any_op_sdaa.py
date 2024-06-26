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

import numpy as np
import unittest
from op_test import OpTest
import paddle

paddle.enable_static()
SEED = 2022


@unittest.skip("do not support reduce_all: False when input is multi-dimension")
class TestAllOp(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {"X": np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.outputs = {"Out": self.inputs["X"].all()}
        self.attrs = {"reduce_all": False}

    def test_check_output(self):
        self.check_output_with_place(self.place)


@unittest.skip("do not support reduce on multi-dimension")
class TestAll8DOp(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {
            "X": np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype("bool")
        }
        self.attrs = {"reduce_all": True, "dim": (2, 3, 4)}
        self.outputs = {"Out": self.inputs["X"].all(axis=self.attrs["dim"])}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestAllOpWithDim(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {"X": np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.attrs = {"dim": (1,)}
        self.outputs = {"Out": self.inputs["X"].all(axis=self.attrs["dim"])}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestAllOpWithoutDim(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {"X": np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.attrs = {"dim": [], "keep_dims": False, "reduce_all": True}
        self.outputs = {"Out": self.inputs["X"].all(axis=None)}

    def test_check_output(self):
        self.check_output_with_place(self.place)


@unittest.skip("do not support reduce on multi-dimension")
class TestAll8DOpWithDim(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
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
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {"X": np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.attrs = {"dim": [1], "keep_dim": True}
        self.outputs = {"Out": np.expand_dims(self.inputs["X"].all(axis=1), axis=1)}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestAll8DOpWithKeepDim(OpTest):
    def setUp(self):
        np.random.seed(2022)
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "reduce_all"
        self.python_api = paddle.all
        self.inputs = {
            "X": np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype("bool")
        }
        self.attrs = {"dim": (5,), "keep_dim": True}
        self.outputs = {
            "Out": self.inputs["X"].all(axis=self.attrs["dim"], keepdims=True)
        }

    def test_check_output(self):
        self.check_output_with_place(self.place)


@unittest.skip("do not support reduce_all: False when input is multi-dimension")
class TestAnyOp(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "reduce_any"
        self.python_api = paddle.any
        self.inputs = {"X": np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.outputs = {"Out": self.inputs["X"].any()}
        self.attrs = {"reduce_all": True}

    def test_check_output(self):
        self.check_output_with_place(self.place)


@unittest.skip("do not support reduce_all: False when input is multi-dimension")
class TestAny8DOp(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "reduce_any"
        self.python_api = paddle.any
        self.inputs = {
            "X": np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype("bool")
        }
        self.attrs = {"reduce_all": True, "dim": (3, 5, 4)}
        self.outputs = {"Out": self.inputs["X"].any(axis=self.attrs["dim"])}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestAnyOpWithDim(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "reduce_any"
        self.python_api = paddle.any
        self.inputs = {"X": np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.attrs = {"dim": [1]}
        self.outputs = {"Out": self.inputs["X"].any(axis=1)}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestAnyOpWithoutDim(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "reduce_any"
        self.python_api = paddle.any
        self.inputs = {"X": np.random.randint(0, 2, (10, 10)).astype("bool")}
        self.attrs = {"dim": [], "keep_dim": False, "reduce_all": True}
        self.outputs = {"Out": self.inputs["X"].any(axis=None)}

    def test_check_output(self):
        self.check_output_with_place(self.place)


@unittest.skip("do not support reduce_any: False when input is multi-dimension")
class TestAny8DOpWithDim(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "reduce_any"
        self.python_api = paddle.any
        self.inputs = {
            "X": np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype("bool")
        }
        self.attrs = {"dim": (3, 6)}
        self.outputs = {"Out": self.inputs["X"].any(axis=self.attrs["dim"])}

    def test_check_output(self):
        self.check_output_with_place(self.place)


@unittest.skip("do not support reduce_any: False when input is multi-dimension")
class TestAny8DOpWithDim(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "reduce_any"
        self.python_api = paddle.any
        self.inputs = {
            "X": np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype("bool")
        }
        self.attrs = {"dim": (3, 6)}
        self.outputs = {"Out": self.inputs["X"].any(axis=self.attrs["dim"])}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestAnyOpWithKeepDim(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "reduce_any"
        self.python_api = paddle.any
        self.inputs = {"X": np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.attrs = {"dim": (1,), "keep_dim": True}
        self.outputs = {
            "Out": np.expand_dims(self.inputs["X"].any(axis=self.attrs["dim"]), axis=1)
        }

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestAny8DOpWithKeepDim(OpTest):
    def setUp(self):
        self.place = paddle.CustomPlace("sdaa", 0)
        self.op_type = "reduce_any"
        self.python_api = paddle.any
        self.inputs = {
            "X": np.random.randint(0, 2, (2, 5, 3, 2, 2, 3, 4, 2)).astype("bool")
        }
        self.attrs = {"dim": (1,), "keep_dim": True}
        self.outputs = {
            "Out": np.expand_dims(self.inputs["X"].any(axis=self.attrs["dim"]), axis=1)
        }

    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
