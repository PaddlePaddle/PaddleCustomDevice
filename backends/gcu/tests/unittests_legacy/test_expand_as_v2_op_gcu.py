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
import paddle.base as base

paddle.enable_static()


class TestExpandAsOpRank1(OpTest):
    def setUp(self):
        self.set_device()
        self.op_type = "expand_as_v2"
        np.random.seed(2023)
        x = np.random.rand(100).astype("float32")
        target_tensor = np.random.rand(2, 100).astype("float32")
        self.inputs = {"X": x}
        self.attrs = {"target_shape": target_tensor.shape}
        bcast_dims = [2, 1]
        output = np.tile(self.inputs["X"], bcast_dims)
        self.outputs = {"Out": output}

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


class TestExpandAsOpRank2(OpTest):
    def setUp(self):
        self.set_device()
        self.op_type = "expand_as_v2"
        np.random.seed(2023)
        x = np.random.rand(10, 12).astype("float32")
        target_tensor = np.random.rand(10, 12).astype("float32")
        self.inputs = {"X": x}
        self.attrs = {"target_shape": target_tensor.shape}
        bcast_dims = [1, 1]
        output = np.tile(self.inputs["X"], bcast_dims)
        self.outputs = {"Out": output}

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


class TestExpandAsOpRank3(OpTest):
    def setUp(self):
        self.set_device()
        self.op_type = "expand_as_v2"
        np.random.seed(2023)
        x = np.random.rand(2, 3, 20).astype("float32")
        target_tensor = np.random.rand(2, 3, 20).astype("float32")
        self.inputs = {"X": x}
        self.attrs = {"target_shape": target_tensor.shape}
        bcast_dims = [1, 1, 1]
        output = np.tile(self.inputs["X"], bcast_dims)
        self.outputs = {"Out": output}

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


class TestExpandAsOpRank4(OpTest):
    def setUp(self):
        self.set_device()
        self.op_type = "expand_as_v2"
        np.random.seed(2023)
        x = np.random.rand(1, 1, 7, 16).astype("float32")
        target_tensor = np.random.rand(4, 6, 7, 16).astype("float32")
        self.inputs = {"X": x}
        self.attrs = {"target_shape": target_tensor.shape}
        bcast_dims = [4, 6, 1, 1]
        output = np.tile(self.inputs["X"], bcast_dims)
        self.outputs = {"Out": output}

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


# Test python API
class TestExpandAsV2API(unittest.TestCase):
    def test_api(self):
        np.random.seed(2023)
        input1 = np.random.random([12, 14]).astype("float32")
        input2 = np.random.random([2, 12, 14]).astype("float32")
        x = paddle.static.data(name="x", shape=[12, 14], dtype="float32")

        y = paddle.static.data(name="target_tensor", shape=[2, 12, 14], dtype="float32")

        out_1 = paddle.expand_as(x, y=y)

        exe = base.Executor(place=base.CustomPlace("gcu", 0))
        res_1 = exe.run(
            base.default_main_program(),
            feed={"x": input1, "target_tensor": input2},
            fetch_list=[out_1],
        )
        assert np.array_equal(res_1[0], np.tile(input1, (2, 1, 1)))


if __name__ == "__main__":
    unittest.main()
