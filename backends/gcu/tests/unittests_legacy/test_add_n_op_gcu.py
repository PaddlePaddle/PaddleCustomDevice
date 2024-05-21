#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class BaseTestAddnOpCse(OpTest):
    def setUp(self):
        self.op_type = "sum"
        self.set_device()
        self.init_data()

        np.random.seed(1024)
        self.x = np.random.random(self.shape).astype("float32")
        self.y = np.random.random(self.shape).astype("float32")
        ipt = [("x", self.x), ("y", self.y)]
        self.inputs = {"X": ipt}
        self.attrs = {"axis": self.axis}
        out = np.sum([self.x, self.y], axis=self.axis)
        self.outputs = {"Out": out}

    def set_device(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("gcu", 0)

    def init_data(self):
        self.dtype = "float32"
        self.shape = [32, 16, 256]
        self.axis = 0

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestAddnFloat16Case1(BaseTestAddnOpCse):
    def init_data(self):
        self.dtype = "float16"
        self.shape = [256]
        self.axis = 0


class TestAddnFloat16Case2(BaseTestAddnOpCse):
    def init_data(self):
        self.dtype = "float16"
        self.shape = [32, 256]
        self.axis = 0


class TestAddnFloat16Case3(BaseTestAddnOpCse):
    def init_data(self):
        self.dtype = "float16"
        self.shape = [16, 32, 24, 256]
        self.axis = 0


class TestAddnFloat32Case1(BaseTestAddnOpCse):
    def init_data(self):
        self.dtype = "float32"
        self.shape = [256]
        self.axis = 0


class TestAddnFloat32Case2(BaseTestAddnOpCse):
    def init_data(self):
        self.dtype = "float32"
        self.shape = [32, 256]
        self.axis = 0


class TestAddnFloat32Case3(BaseTestAddnOpCse):
    def init_data(self):
        self.dtype = "float32"
        self.shape = [16, 32, 24, 256]
        self.axis = 0


class TestAddnOp(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CustomPlace("gcu", 0)

    def test_add_n_fp16(self):
        paddle.disable_static(self.place)

        self.x_np1 = np.random.random([2, 32]).astype("float16")
        self.x_np2 = np.random.random([2, 32]).astype("float16")

        x1 = paddle.to_tensor(self.x_np1)
        x2 = paddle.to_tensor(self.x_np2)

        y = paddle.add_n([x1, x2])

        res_np = np.sum([self.x_np1, self.x_np2], axis=0).astype("float16")

        np.testing.assert_allclose(res_np, y, rtol=1e-05)
        paddle.enable_static()

    def test_add_n_api(self):
        paddle.enable_static()
        x1 = paddle.static.data("X1", [2])
        x2 = paddle.static.data("X2", [2])

        y = paddle.add_n([x1, x2])

        self.x_np1 = np.random.random([2]).astype("float32")
        self.x_np2 = np.random.random([2]).astype("float32")
        self.x_np = np.vstack((self.x_np1, self.x_np2))

        exe = paddle.static.Executor(self.place)
        res = exe.run(feed={"X1": self.x_np1, "X2": self.x_np2}, fetch_list=[y])
        res_np = np.sum(self.x_np, axis=0).astype("float32")
        for r in res:
            self.assertEqual((res_np == r).all(), True)


if __name__ == "__main__":
    unittest.main()
