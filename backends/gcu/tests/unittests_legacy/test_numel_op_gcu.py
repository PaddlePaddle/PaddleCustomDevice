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


class TestNumelOp(OpTest):
    def setUp(self):
        self.op_type = "size"
        self.set_device()
        self.python_api = paddle.numel
        self.init()
        x = np.random.random(self.shape).astype(self.dtype)
        self.inputs = {
            "Input": x,
        }
        self.outputs = {"Out": np.array(np.size(x))}

    def set_device(self):
        self.__class__.use_custom_device = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.CustomPlace("gcu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def init(self):
        self.shape = (6, 56, 8, 55)
        self.dtype = np.float64


class TestNumelOp1(TestNumelOp):
    def init(self):
        self.shape = (11, 66)
        self.dtype = np.float64


class TestNumelOp2(TestNumelOp):
    def init(self):
        self.shape = (0,)
        self.dtype = np.float64


class TestNumelOpFP16(TestNumelOp):
    def init(self):
        self.dtype = np.float16
        self.shape = (6, 56, 8, 55)


class TestNumelOp1FP16(TestNumelOp):
    def init(self):
        self.dtype = np.float16
        self.shape = (11, 66)


class TestNumelOp2FP16(TestNumelOp):
    def init(self):
        self.dtype = np.float16
        self.shape = (0,)


class TestNumelAPI(unittest.TestCase):
    def test_numel_static(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            shape1 = [2, 1, 4, 5]
            shape2 = [1, 4, 5]
            x_1 = paddle.static.data(shape=shape1, dtype="int32", name="x_1")
            x_2 = paddle.static.data(shape=shape2, dtype="int32", name="x_2")
            input_1 = np.random.random(shape1).astype("int32")
            input_2 = np.random.random(shape2).astype("int32")
            out_1 = paddle.numel(x_1)
            out_2 = paddle.numel(x_2)
            exe = paddle.static.Executor(place=paddle.CustomPlace("gcu", 0))
            res_1, res_2 = exe.run(
                feed={
                    "x_1": input_1,
                    "x_2": input_2,
                },
                fetch_list=[out_1, out_2],
            )
            np.testing.assert_array_equal(
                res_1, np.array(np.size(input_1)).astype("int64")
            )
            np.testing.assert_array_equal(
                res_2, np.array(np.size(input_2)).astype("int64")
            )

    def test_numel_imperative(self):
        paddle.disable_static(paddle.CustomPlace("gcu", 0))
        input_1 = np.random.random([2, 1, 4, 5]).astype("int32")
        input_2 = np.random.random([1, 4, 5]).astype("int32")
        x_1 = paddle.to_tensor(input_1)
        x_2 = paddle.to_tensor(input_2)
        out_1 = paddle.numel(x_1)
        out_2 = paddle.numel(x_2)
        np.testing.assert_array_equal(out_1.numpy().item(0), np.size(input_1))
        np.testing.assert_array_equal(out_2.numpy().item(0), np.size(input_2))
        paddle.enable_static()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
