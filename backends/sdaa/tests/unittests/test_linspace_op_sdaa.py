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

import numpy as np
import unittest

from op_test import OpTest, paddle_static_guard
import paddle
from paddle import base
from paddle.base import Program, core, program_guard

paddle.enable_static()
SEED = 1024


class TestLinspaceOpCommonCase(OpTest):
    def setUp(self):
        self.op_type = "linspace"
        self.python_api = paddle.linspace
        self.place = paddle.CustomPlace("sdaa", 0)
        self._set_dtype()
        self._set_data()
        self.attrs = {"dtype": self.attr_dtype}

    def _set_dtype(self):
        self.dtype = "float32"
        self.attr_dtype = int(core.VarDesc.VarType.FP32)

    def _set_data(self):
        self.outputs = {"Out": np.arange(0, 11).astype(self.dtype)}
        self.inputs = {
            "Start": np.array([0]).astype(self.dtype),
            "Stop": np.array([10]).astype(self.dtype),
            "Num": np.array([11]).astype("int32"),
        }

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestLinspaceOpReverseCase(TestLinspaceOpCommonCase):
    def _set_data(self):
        self.inputs = {
            "Start": np.array([10]).astype(self.dtype),
            "Stop": np.array([0]).astype(self.dtype),
            "Num": np.array([11]).astype("int32"),
        }
        self.outputs = {"Out": np.arange(10, -1, -1).astype(self.dtype)}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestLinspaceOpNumOneCase(TestLinspaceOpCommonCase):
    def _set_data(self):
        self.inputs = {
            "Start": np.array([10]).astype(self.dtype),
            "Stop": np.array([0]).astype(self.dtype),
            "Num": np.array([1]).astype("int32"),
        }
        self.outputs = {"Out": np.array([10], dtype=self.dtype)}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestLinspaceOpCommonCaseFP16(TestLinspaceOpCommonCase):
    def _set_dtype(self):
        self.dtype = np.float16
        self.attr_dtype = int(core.VarDesc.VarType.FP16)


class TestLinspaceOpReverseCaseFP16(TestLinspaceOpReverseCase):
    def _set_dtype(self):
        self.dtype = np.float16
        self.attr_dtype = int(core.VarDesc.VarType.FP16)


class TestLinspaceOpNumOneCaseFP16(TestLinspaceOpNumOneCase):
    def _set_dtype(self):
        self.dtype = np.float16
        self.attr_dtype = int(core.VarDesc.VarType.FP16)


class TestLinspaceOpCommonCaseDouble(TestLinspaceOpCommonCase):
    def _set_dtype(self):
        self.dtype = np.double
        self.attr_dtype = int(core.VarDesc.VarType.FP64)


class TestLinspaceOpReverseCaseDouble(TestLinspaceOpReverseCase):
    def _set_dtype(self):
        self.dtype = np.double
        self.attr_dtype = int(core.VarDesc.VarType.FP64)


class TestLinspaceOpNumOneCaseDouble(TestLinspaceOpNumOneCase):
    def _set_dtype(self):
        self.dtype = np.double
        self.attr_dtype = int(core.VarDesc.VarType.FP64)


class TestLinspaceOpCommonCaseInt32(TestLinspaceOpCommonCase):
    def _set_dtype(self):
        self.dtype = np.int32
        self.attr_dtype = int(core.VarDesc.VarType.INT32)


class TestLinspaceOpReverseCaseInt32(TestLinspaceOpReverseCase):
    def _set_dtype(self):
        self.dtype = np.int32
        self.attr_dtype = int(core.VarDesc.VarType.INT32)


class TestLinspaceOpNumOneCaseInt32(TestLinspaceOpNumOneCase):
    def _set_dtype(self):
        self.dtype = np.int32
        self.attr_dtype = int(core.VarDesc.VarType.INT32)


class TestLinspaceOpCommonCaseInt64(TestLinspaceOpCommonCase):
    def _set_dtype(self):
        self.dtype = np.int64
        self.attr_dtype = int(core.VarDesc.VarType.INT64)


class TestLinspaceOpReverseCaseInt64(TestLinspaceOpReverseCase):
    def _set_dtype(self):
        self.dtype = np.int64
        self.attr_dtype = int(core.VarDesc.VarType.INT64)


class TestLinspaceOpNumOneCaseInt32(TestLinspaceOpNumOneCase):
    def _set_dtype(self):
        self.dtype = np.int64
        self.attr_dtype = int(core.VarDesc.VarType.INT64)


class TestLinspaceAPI(unittest.TestCase):
    def test_variable_input1(self):
        with paddle_static_guard():
            start = paddle.full(shape=[1], fill_value=0, dtype="float32")
            stop = paddle.full(shape=[1], fill_value=10, dtype="float32")
            num = paddle.full(shape=[1], fill_value=5, dtype="int32")
            out = paddle.linspace(start, stop, num, dtype="float32")
            exe = base.Executor(place=paddle.CustomPlace("sdaa", 0))
            res = exe.run(base.default_main_program(), fetch_list=[out])


class TestLinspaceAPI(unittest.TestCase):
    def test_variable_input1(self):
        with paddle_static_guard():
            start = paddle.full(shape=[1], fill_value=0, dtype="float32")
            stop = paddle.full(shape=[1], fill_value=10, dtype="float32")
            num = paddle.full(shape=[1], fill_value=5, dtype="int32")
            out = paddle.linspace(start, stop, num, dtype="float32")
            exe = base.Executor(place=paddle.CustomPlace("sdaa", 0))
            res = exe.run(base.default_main_program(), fetch_list=[out])


class TestLinspaceAPI(unittest.TestCase):
    def test_variable_input1(self):
        with paddle_static_guard():
            start = paddle.full(shape=[1], fill_value=0, dtype="float32")
            stop = paddle.full(shape=[1], fill_value=10, dtype="float32")
            num = paddle.full(shape=[1], fill_value=5, dtype="int32")
            out = paddle.linspace(start, stop, num, dtype="float32")
            exe = base.Executor(place=paddle.CustomPlace("sdaa", 0))
            res = exe.run(base.default_main_program(), fetch_list=[out])
            np_res = np.linspace(0, 10, 5, dtype="float32")
            self.assertEqual((res == np_res).all(), True)

    def test_variable_input2(self):
        paddle.device.set_device("sdaa")
        start = paddle.full(shape=[1], fill_value=0, dtype="float32")
        stop = paddle.full(shape=[1], fill_value=10, dtype="float32")
        num = paddle.full(shape=[1], fill_value=5, dtype="int32")
        out = paddle.linspace(start, stop, num, dtype="float32")
        np_res = np.linspace(0, 10, 5, dtype="float32")
        self.assertEqual((out.numpy() == np_res).all(), True)

    def test_dtype(self):
        with paddle_static_guard():
            out_1 = paddle.linspace(0, 10, 5, dtype="float32")
            out_2 = paddle.linspace(0, 10, 5, dtype=np.float32)
            out_3 = paddle.linspace(0, 10, 5, dtype=core.VarDesc.VarType.FP32)
            exe = base.Executor(place=paddle.CustomPlace("sdaa", 0))
            res_1, res_2, res_3 = exe.run(
                base.default_main_program(), fetch_list=[out_1, out_2, out_3]
            )
            assert np.array_equal(res_1, res_2)

    def test_name(self):
        paddle.device.set_device("sdaa")
        with paddle_static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                out = paddle.linspace(0, 10, 5, dtype="float32", name="linspace_res")
                assert "linspace_res" in out.name

    def test_imperative(self):
        paddle.disable_static()
        paddle.device.set_device("sdaa")
        out1 = paddle.linspace(0, 10, 5, dtype="float32")
        np_out1 = np.linspace(0, 10, 5, dtype="float32")
        out2 = paddle.linspace(0, 10, 5, dtype="int32")
        np_out2 = np.linspace(0, 10, 5, dtype="int32")
        out3 = paddle.linspace(0, 10, 200, dtype="int32")
        np_out3 = np.linspace(0, 10, 200, dtype="int32")
        self.assertEqual((out1.numpy() == np_out1).all(), True)
        self.assertEqual((out2.numpy() == np_out2).all(), True)
        self.assertEqual((out3.numpy() == np_out3).all(), True)


class TestLinspaceOpError(unittest.TestCase):
    def test_errors(self):
        paddle.device.set_device("sdaa")
        with paddle_static_guard():
            with program_guard(Program(), Program()):

                def test_dtype():
                    paddle.linspace(0, 10, 1, dtype="int8")

                self.assertRaises(TypeError, test_dtype)

                def test_dtype1():
                    paddle.linspace(0, 10, 1.33, dtype="int32")

                self.assertRaises(TypeError, test_dtype1)

                def test_start_type():
                    paddle.linspace([0], 10, 1, dtype="float32")

                self.assertRaises(TypeError, test_start_type)

                def test_end_type():
                    paddle.linspace(0, [10], 1, dtype="float32")

                self.assertRaises(TypeError, test_end_type)

                def test_step_dtype():
                    paddle.linspace(0, 10, [0], dtype="float32")

                self.assertRaises(TypeError, test_step_dtype)

                def test_start_dtype():
                    start = paddle.static.data(shape=[1], dtype="float64", name="start")
                    paddle.linspace(start, 10, 1, dtype="float32")

                self.assertRaises(ValueError, test_start_dtype)

                def test_end_dtype():
                    end = paddle.static.data(shape=[1], dtype="float64", name="end")
                    paddle.linspace(0, end, 1, dtype="float32")

                self.assertRaises(ValueError, test_end_dtype)

                def test_num_dtype():
                    num = paddle.static.data(shape=[1], dtype="int32", name="step")
                    paddle.linspace(0, 10, num, dtype="float32")

                self.assertRaises(TypeError, test_step_dtype)


if __name__ == "__main__":
    unittest.main()
