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
import paddle.fluid as fluid
from paddle.fluid import core

paddle.enable_static()


class TestLinspaceOpCommonCase(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "linspace"
        dtype = "float32"
        self.inputs = {
            "Start": np.array([0]).astype(dtype),
            "Stop": np.array([10]).astype(dtype),
            "Num": np.array([11]).astype("int32"),
        }
        self.attrs = {"dtype": int(core.VarDesc.VarType.FP32)}

        self.outputs = {"Out": np.arange(0, 11).astype(dtype)}

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestLinspaceOpFP64(TestLinspaceOpCommonCase):
    def setUp(self):
        self.set_npu()
        self.op_type = "linspace"
        dtype = "float64"
        self.inputs = {
            "Start": np.array([0.1]).astype(dtype),
            "Stop": np.array([10.1]).astype(dtype),
            "Num": np.array([11]).astype("int32"),
        }
        self.attrs = {"dtype": int(core.VarDesc.VarType.FP64)}

        self.outputs = {"Out": np.linspace(0.1, 10.1, 11, dtype=dtype)}


class TestLinspaceOpInt32(TestLinspaceOpCommonCase):
    def setUp(self):
        self.set_npu()
        self.op_type = "linspace"
        dtype = "int32"
        self.inputs = {
            "Start": np.array([0]).astype(dtype),
            "Stop": np.array([10]).astype(dtype),
            "Num": np.array([11]).astype("int32"),
        }
        self.attrs = {"dtype": int(core.VarDesc.VarType.INT32)}

        self.outputs = {"Out": np.linspace(0, 10, 11, dtype=dtype)}


class TestLinspaceOpInt64(TestLinspaceOpCommonCase):
    def setUp(self):
        self.set_npu()
        self.op_type = "linspace"
        dtype = "int64"
        self.inputs = {
            "Start": np.array([0]).astype(dtype),
            "Stop": np.array([10]).astype(dtype),
            "Num": np.array([12]).astype("int32"),
        }
        self.attrs = {"dtype": int(core.VarDesc.VarType.INT64)}

        self.outputs = {"Out": np.linspace(0, 10, 12, dtype=dtype)}


class TestLinspaceOpReverseCase(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "linspace"
        dtype = "float32"
        self.inputs = {
            "Start": np.array([10]).astype(dtype),
            "Stop": np.array([0]).astype(dtype),
            "Num": np.array([11]).astype("int32"),
        }
        self.attrs = {"dtype": int(core.VarDesc.VarType.FP32)}

        self.outputs = {"Out": np.arange(10, -1, -1).astype(dtype)}

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestLinspaceOpNumOneCase(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "linspace"
        dtype = "float32"
        self.inputs = {
            "Start": np.array([10]).astype(dtype),
            "Stop": np.array([0]).astype(dtype),
            "Num": np.array([1]).astype("int32"),
        }
        self.attrs = {"dtype": int(core.VarDesc.VarType.FP32)}

        self.outputs = {"Out": np.array([10], dtype=dtype)}

    def set_npu(self):
        self.__class__.use_custom_device = True
        self.place = paddle.CustomPlace("npu", 0)

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestLinspaceAPI(unittest.TestCase):
    def test_variable_input1(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program()):
            start = paddle.full(shape=[1], fill_value=0, dtype="float32")
            stop = paddle.full(shape=[1], fill_value=10, dtype="float32")
            num = paddle.full(shape=[1], fill_value=5, dtype="int32")
            out = paddle.linspace(start, stop, num, dtype="float32")
            exe = fluid.Executor(paddle.CustomPlace("npu", 0))
            res = exe.run(fluid.default_main_program(), fetch_list=[out])
        np_res = np.linspace(0, 10, 5, dtype="float32")
        self.assertEqual((res == np_res).all(), True)

    def test_variable_input2(self):
        paddle.disable_static(paddle.CustomPlace("npu", 0))
        start = paddle.full(shape=[1], fill_value=0, dtype="float32")
        stop = paddle.full(shape=[1], fill_value=10, dtype="float32")
        num = paddle.full(shape=[1], fill_value=5, dtype="int32")
        out = paddle.linspace(start, stop, num, dtype="float32")
        np_res = np.linspace(0, 10, 5, dtype="float32")
        self.assertEqual((out.numpy() == np_res).all(), True)
        paddle.enable_static()

    def test_imperative(self):
        paddle.disable_static(paddle.CustomPlace("npu", 0))
        out1 = paddle.linspace(0, 10, 5, dtype="float32")
        np_out1 = np.linspace(0, 10, 5, dtype="float32")
        out2 = paddle.linspace(0, 10.1, 5, dtype="float64")
        np_out2 = np.linspace(0, 10.1, 5, dtype="float64")
        out3 = paddle.linspace(0, 10, 6, dtype="int32")
        np_out3 = np.linspace(0, 10, 6, dtype="int32")
        out4 = paddle.linspace(0, 10, 7, dtype="int64")
        np_out4 = np.linspace(0, 10, 7, dtype="int64")
        out5 = paddle.linspace(0, 10, 200, dtype="int32")
        np_out5 = np.linspace(0, 10, 200, dtype="int32")
        paddle.enable_static()
        self.assertEqual((out1.numpy() == np_out1).all(), True)
        self.assertEqual((out2.numpy() == np_out2).all(), True)
        self.assertEqual((out3.numpy() == np_out3).all(), True)
        self.assertEqual((out4.numpy() == np_out4).all(), True)
        self.assertEqual((out5.numpy() == np_out5).all(), True)


if __name__ == "__main__":
    unittest.main()
